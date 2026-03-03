"""Unified model resolution across Ollama, HuggingFace, and Kaggle.

Resolves human-friendly model names to local file paths, downloading
from the best available source if needed.

Model alias data is fetched from the Octomil server at runtime and
cached locally. An empty fallback is used when the server is
unreachable — explicit source prefixes and direct repo IDs still work.

Supports:
- Explicit sources: ``hf:org/model``, ``ollama:name:tag``, ``kaggle:org/model``
- Known aliases: ``phi-4-mini``, ``gemma-1b``, ``llama-3.2-3b``
- Direct HuggingFace repo IDs: ``microsoft/Phi-4-mini-instruct``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import click

from ..models.catalog_client import SourceAliasesClient
from .base import SourceResult
from .huggingface import HuggingFaceSource
from .kaggle import KaggleSource
from .ollama import OllamaSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server-fetched model aliases (singleton)
# ---------------------------------------------------------------------------

_aliases_client: Optional[SourceAliasesClient] = None


def _get_aliases_client() -> SourceAliasesClient:
    """Return the module-level SourceAliasesClient singleton."""
    global _aliases_client
    if _aliases_client is None:
        _aliases_client = SourceAliasesClient()
    return _aliases_client


def _get_model_aliases() -> Dict[str, Dict[str, str]]:
    """Fetch source-level model aliases from the server (cached) or fallback."""
    return _get_aliases_client().get_aliases()


# Backward-compatible module-level name — always empty at import time.
# Runtime code should call _get_model_aliases() instead.
_MODEL_ALIASES: Dict[str, Dict[str, str]] = {}

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

    # ── Alias lookup (server-fetched) ────────────────────────────────────
    model_aliases = _get_model_aliases()
    aliases = model_aliases.get(name)
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
    model_aliases = _get_model_aliases()
    aliases = model_aliases.get(name)
    if aliases:
        if prefer_onnx and "hf_onnx" in aliases:
            return aliases["hf_onnx"]
        if "hf" in aliases:
            return aliases["hf"]

    # Direct org/model format
    if "/" in name:
        return name

    return None
