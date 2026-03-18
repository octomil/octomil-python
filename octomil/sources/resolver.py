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
from typing import Any, Dict, Optional, Union

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


def _manifest_to_aliases(manifest: dict) -> Dict[str, Dict[str, Union[str, Dict[str, Any]]]]:
    """Convert a v2 nested manifest to source alias mappings.

    Walks the canonical nested manifest format (family → variants → versions
    → packages) and extracts download sources for each variant:
    - Packages with ``runtime_executor=ollama`` → ``ollama`` alias (bare string)
    - Packages with ``hf://`` URIs → ``hf`` alias (dict with resolution context)
    - Packages with ``runtime_executor in (onnxruntime, ort)`` and ``hf://`` → ``hf_onnx`` alias
    """
    aliases: Dict[str, Dict[str, Union[str, Dict[str, Any]]]] = {}

    for family_name, family_data in manifest.items():
        if not isinstance(family_data, dict) or "variants" not in family_data:
            continue

        for variant_name, variant_data in family_data["variants"].items():
            if not variant_name:
                continue

            model_aliases: Dict[str, Union[str, Dict[str, Any]]] = {}

            # Collect packages from all versions
            for ver_data in variant_data.get("versions", {}).values():
                for pkg in ver_data.get("packages", []):
                    executor: str = pkg.get("runtime_executor", "")
                    artifact_format: str = pkg.get("artifact_format", "")
                    quantization: str = pkg.get("quantization", "")

                    # Find weights resource
                    weights = None
                    for res in pkg.get("resources", []):
                        if res.get("kind") == "weights":
                            weights = res
                            break

                    if weights is None:
                        # Multi-resource package (e.g. sherpa encoder/decoder/joiner/tokens).
                        # Extract the common HF repo from the first hf:// resource.
                        if "hf" not in model_aliases:
                            hf_resources = [r for r in pkg.get("resources", []) if r.get("uri", "").startswith("hf://")]
                            if hf_resources:
                                repo, _ = _parse_hf_uri(hf_resources[0]["uri"])
                                model_aliases["hf"] = {
                                    "repo_id": repo,
                                    "filename": None,
                                    "revision": None,
                                    "quantization_hint": quantization.lower() if quantization else None,
                                    "artifact_format": artifact_format or None,
                                    "uri_type": "directory",
                                }
                        continue

                    uri: str = weights.get("uri", "")
                    meta: dict = weights.get("metadata") or {}
                    uri_type: str = meta.get("uri_type", "file")
                    revision: Optional[str] = meta.get("revision")

                    if executor == "ollama" and "ollama" not in model_aliases:
                        model_aliases["ollama"] = uri
                    elif executor in ("onnxruntime", "ort") and uri.startswith("hf://"):
                        repo, _ = _parse_hf_uri(uri)
                        if "hf_onnx" not in model_aliases:
                            model_aliases["hf_onnx"] = repo
                    elif uri.startswith("hf://") and "hf" not in model_aliases:
                        repo, filename = _parse_hf_uri(uri)
                        hf_alias: Dict[str, Any] = {
                            "repo_id": repo,
                            "filename": filename or None,
                            "revision": revision,
                            "quantization_hint": quantization.lower() if quantization else None,
                            "artifact_format": artifact_format or None,
                            "uri_type": uri_type,
                        }
                        # Include projector resource info if present
                        for res in pkg.get("resources", []):
                            if res.get("kind") == "projector":
                                proj_uri = res.get("uri", "")
                                if proj_uri.startswith("hf://"):
                                    pr, pf = _parse_hf_uri(proj_uri)
                                    hf_alias["projector_repo"] = pr if pr != repo else None
                                    hf_alias["projector_filename"] = pf or None
                                break
                        model_aliases["hf"] = hf_alias

            if model_aliases:
                aliases[variant_name] = model_aliases

    return aliases


def _get_model_aliases() -> Dict[str, Dict[str, Union[str, Dict[str, Any]]]]:
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


def _try_source(
    source: str,
    ref: str,
    *,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    quantization_hint: Optional[str] = None,
    artifact_format: Optional[str] = None,
) -> Optional[SourceResult]:
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
        return backend.resolve(
            ref,
            filename=filename,
            revision=revision,
            quantization_hint=quantization_hint,
            artifact_format=artifact_format,
        )
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
            ollama_ref = str(aliases["ollama"])
            click.echo(f"  Checking Ollama for {ollama_ref}...")
            result = _try_source("ollama", ollama_ref)
            if result:
                if result.cached:
                    click.echo("  Using Ollama cache")
                else:
                    click.echo("  Downloaded from Ollama")
                return result.path

        if "hf" in aliases:
            hf_info = aliases["hf"]
            if isinstance(hf_info, dict):
                click.echo(f"  Downloading from HuggingFace: {hf_info['repo_id']}")
                result = _try_source(
                    "hf",
                    hf_info["repo_id"],
                    filename=hf_info.get("filename"),
                    revision=hf_info.get("revision"),
                    quantization_hint=hf_info.get("quantization_hint"),
                    artifact_format=hf_info.get("artifact_format"),
                )
                # Download projector (mmproj) alongside if present
                if result and hf_info.get("projector_filename"):
                    proj_repo = hf_info.get("projector_repo") or hf_info["repo_id"]
                    proj_file = hf_info["projector_filename"]
                    click.echo(f"  Downloading projector: {proj_file}")
                    _try_source("hf", proj_repo, filename=proj_file)
            else:
                # Backward compat: bare string repo ID
                click.echo(f"  Downloading from HuggingFace: {hf_info}")
                result = _try_source("hf", hf_info)
            if result:
                return result.path

        if "kaggle" in aliases:
            result = _try_source("kaggle", str(aliases["kaggle"]))
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
            return str(aliases["hf_onnx"])
        if "hf" in aliases:
            hf_info = aliases["hf"]
            if isinstance(hf_info, dict):
                return hf_info["repo_id"]
            return hf_info

    # Direct org/model format
    if "/" in name:
        return name

    return None
