"""Unified model resolution across Ollama, HuggingFace, and Kaggle.

Resolves human-friendly model names to local file paths, downloading
from the best available source if needed.

Source aliases are derived from the v2 catalog manifest via
:class:`~octomil.models.catalog_client.CatalogClientV2`.

Supports:
- Explicit sources: ``hf:org/model``, ``ollama:name:tag``, ``kaggle:org/model``
- Known aliases: ``phi-4-mini``, ``gemma-1b``, ``llama-3.2-3b``
- Direct HuggingFace repo IDs: ``microsoft/Phi-4-mini-instruct``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import click

from ..models.catalog import _parse_hf_uri
from ..models.catalog_client import CatalogClientV2
from .base import SourceResult
from .huggingface import HuggingFaceSource
from .kaggle import KaggleSource
from .ollama import OllamaSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V2 manifest → source aliases
# ---------------------------------------------------------------------------

_v2_client: Optional[CatalogClientV2] = None


def _get_v2_client() -> CatalogClientV2:
    """Return the module-level CatalogClientV2 singleton."""
    global _v2_client
    if _v2_client is None:
        _v2_client = CatalogClientV2()
    return _v2_client


def _manifest_to_aliases(manifest: dict) -> Dict[str, Dict[str, str]]:
    """Convert a v2 manifest to source alias mappings.

    For each model, extracts download sources from packages:
    - Packages with ``runtime_executor=ollama`` → ``ollama`` alias
    - Packages with ``hf://`` URIs → ``hf`` alias
    - Packages with ``runtime_executor in (onnxruntime, ort)`` and ``hf://`` → ``hf_onnx`` alias
    """
    aliases: Dict[str, Dict[str, str]] = {}

    for model in manifest.get("models", []):
        model_id: str = model.get("id", "")
        if not model_id:
            continue

        model_aliases: Dict[str, str] = {}

        for pkg in model.get("packages", []):
            executor: str = pkg.get("runtime_executor", "")

            # Find weights resource
            weights = None
            for res in pkg.get("resources", []):
                if res.get("kind") == "weights":
                    weights = res
                    break
            if weights is None:
                continue

            uri: str = weights.get("uri", "")

            if executor == "ollama" and "ollama" not in model_aliases:
                model_aliases["ollama"] = uri
            elif executor in ("onnxruntime", "ort") and uri.startswith("hf://"):
                repo, _ = _parse_hf_uri(uri)
                if "hf_onnx" not in model_aliases:
                    model_aliases["hf_onnx"] = repo
            elif uri.startswith("hf://") and "hf" not in model_aliases:
                repo, _ = _parse_hf_uri(uri)
                model_aliases["hf"] = repo

        if model_aliases:
            aliases[model_id] = model_aliases

    return aliases


def _get_model_aliases() -> Dict[str, Dict[str, str]]:
    """Fetch source-level model aliases from the v2 manifest (cached)."""
    manifest = _get_v2_client().get_manifest()
    return _manifest_to_aliases(manifest)


_ollama = OllamaSource()
_hf = HuggingFaceSource()
_kaggle = KaggleSource()


def _parse_explicit_source(name: str) -> Optional[tuple[str, str]]:
    """Parse ``hf:org/model`` -> ``("hf", "org/model")``. Returns None if no prefix."""
    for prefix in ("hf:", "huggingface:", "ollama:", "kaggle:"):
        if name.startswith(prefix):
            source = prefix.rstrip(":")
            if source == "huggingface":
                source = "hf"
            return source, name[len(prefix) :]
    return None


_SOURCE_HINTS: Dict[str, str] = {
    "ollama": "Install Ollama from https://ollama.com",
    "hf": "Install the Python package: pip install huggingface_hub",
    "kaggle": "Install the Kaggle CLI: pip install kaggle",
}


def _try_source(source: str, ref: str) -> Optional[SourceResult]:
    """Attempt a single source. Returns None on failure (with a logged hint)."""
    backends = {"ollama": _ollama, "hf": _hf, "kaggle": _kaggle}
    backend = backends.get(source)
    if backend is None:
        return None
    try:
        if not backend.is_available():
            hint = _SOURCE_HINTS.get(source, "")
            logger.debug("Source %s unavailable for %s", source, ref)
            click.echo(f"  {source} backend unavailable. {hint}", err=True)
            return None
        return backend.resolve(ref)
    except Exception as exc:
        logger.debug("Source %s failed for %s: %s", source, ref, exc)
    return None


def resolve_and_download(name: str) -> str:
    """Resolve a model name and return a local file path.

    Parameters
    ----------
    name:
        Model name, alias, or explicit source reference.

    Returns
    -------
    str
        Local path to the downloaded model.

    Raises
    ------
    RuntimeError
        If the model cannot be resolved from any source.
    """
    # ── Explicit source ───────────────────────────────────────────────────
    explicit = _parse_explicit_source(name)
    if explicit:
        source, ref = explicit
        click.echo(f"  Downloading from {source}: {ref}")
        result = _try_source(source, ref)
        if result:
            if result.cached:
                click.echo(f"  Using cache: {result.path}")
            return result.path
        raise RuntimeError(f"Could not download '{ref}' from {source}")

    # ── Resolve catalog alias (phi-4-mini → phi-mini, etc.) ─────────────
    from ..models.catalog import _resolve_alias

    canonical = _resolve_alias(name)

    # ── Alias lookup (manifest-derived) ─────────────────────────────────
    model_aliases = _get_model_aliases()
    aliases = model_aliases.get(canonical) or model_aliases.get(name)
    if aliases:
        # Try Ollama first (fastest — local cache), then HuggingFace
        if "ollama" in aliases:
            click.echo(f"  Checking Ollama for {aliases['ollama']}...")
            result = _try_source("ollama", aliases["ollama"])
            if result:
                if result.cached:
                    click.echo("  Using Ollama cache")
                else:
                    click.echo("  Downloaded from Ollama")
                return result.path

        if "hf" in aliases:
            click.echo(f"  Downloading from HuggingFace: {aliases['hf']}")
            result = _try_source("hf", aliases["hf"])
            if result:
                return result.path

        if "kaggle" in aliases:
            result = _try_source("kaggle", aliases["kaggle"])
            if result:
                return result.path

        raise RuntimeError(f"Could not download '{name}' from any source. Tried: {', '.join(aliases.keys())}")

    # ── Direct HuggingFace repo ID (org/model format) ─────────────────────
    if "/" in name:
        click.echo(f"  Downloading from HuggingFace: {name}")
        result = _try_source("hf", name)
        if result:
            return result.path
        raise RuntimeError(f"Could not download '{name}' from HuggingFace")

    # ── Unknown ───────────────────────────────────────────────────────────
    known = ", ".join(sorted(model_aliases.keys()))
    raise RuntimeError(
        f"Unknown model: '{name}'\n  Known models: {known}\n  Or use: hf:<org>/<model>, ollama:<name>, kaggle:<path>"
    )


def resolve_hf_repo(name: str, *, prefer_onnx: bool = True) -> Optional[str]:
    """Resolve a model name to a HuggingFace repo ID without downloading.

    When *prefer_onnx* is ``True`` (the default), returns the ``hf_onnx``
    repo if one is registered for this alias — these repos already contain
    ``.onnx`` files and skip server-side conversion entirely.

    Returns the HF repo string (e.g. ``microsoft/Phi-4-mini-instruct``) or
    ``None`` if the name is not a known alias or explicit HF reference.
    """
    # Explicit hf: prefix — user chose a specific repo, honour it
    explicit = _parse_explicit_source(name)
    if explicit:
        source, ref = explicit
        return ref if source == "hf" else None

    # Alias lookup — prefer ONNX variant for server-side import
    from ..models.catalog import _resolve_alias

    canonical = _resolve_alias(name)
    model_aliases = _get_model_aliases()
    aliases = model_aliases.get(canonical) or model_aliases.get(name)
    if aliases:
        if prefer_onnx and "hf_onnx" in aliases:
            return aliases["hf_onnx"]
        if "hf" in aliases:
            return aliases["hf"]

    # Direct org/model format
    if "/" in name:
        return name

    return None
