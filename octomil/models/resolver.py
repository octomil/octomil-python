"""Model resolution — maps ``model:variant`` to engine-specific artifacts.

Takes a parsed model specifier and resolves it to a concrete HuggingFace
repo ID (and optional filename) for a given engine.

Engine priority and alias data are fetched from the Octomil server at
runtime and cached locally. A minimal fallback (``["auto"]``) is embedded
for offline bootstrap only.

Usage::

    from octomil.models import resolve

    r = resolve("gemma-3b:4bit", available_engines=["mlx-lm", "llama.cpp"])
    print(r.engine, r.hf_repo, r.filename)
"""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from .catalog import CATALOG, ModelEntry, MoEMetadata, _resolve_alias
from .catalog_client import EnginePriorityClient
from .parser import normalize_variant, parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedModel:
    """Result of resolving a model specifier to engine-specific artifacts."""

    family: Optional[str]
    quant: str
    engine: Optional[str]
    hf_repo: str
    filename: Optional[str] = None
    mlx_repo: Optional[str] = None
    source_repo: Optional[str] = None
    raw: str = ""
    architecture: str = "dense"
    moe: Optional[MoEMetadata] = None

    @property
    def is_gguf(self) -> bool:
        return self.filename is not None and self.filename.endswith(".gguf")

    @property
    def is_moe(self) -> bool:
        """True if this model uses Mixture of Experts architecture."""
        return self.architecture == "moe" and self.moe is not None


class ModelResolutionError(ValueError):
    """Raised when a model specifier cannot be resolved."""


# Engine name normalization for matching — these are just string aliases,
# not proprietary intelligence.  Kept client-side for fast normalization.
_ENGINE_ALIASES: dict[str, str] = {
    "mlx": "mlx-lm",
    "mlx-lm": "mlx-lm",
    "gguf": "llama.cpp",
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "mnn": "mnn",
    "mnn-llm": "mnn",
    "executorch": "executorch",
    "onnxruntime": "onnxruntime",
    "onnx": "onnxruntime",
    "ort": "onnxruntime",
    "mlc-llm": "mlc-llm",
    "mlc": "mlc-llm",
    "mlcllm": "mlc-llm",
    "whisper.cpp": "whisper.cpp",
    "whisper": "whisper.cpp",
    "whispercpp": "whisper.cpp",
    "ollama": "ollama",
    "echo": "echo",
}

# ---------------------------------------------------------------------------
# Server-fetched engine priority (singleton)
# ---------------------------------------------------------------------------

_priority_client: Optional[EnginePriorityClient] = None


def _get_priority_client() -> EnginePriorityClient:
    """Return the module-level EnginePriorityClient singleton."""
    global _priority_client
    if _priority_client is None:
        _priority_client = EnginePriorityClient()
    return _priority_client


def _get_engine_priority() -> list[str]:
    """Return the engine priority list from server (cached) or fallback."""
    return _get_priority_client().get_priority()


# Backward-compatible module-level name for direct imports.
# Tests that do ``from octomil.models.resolver import _ENGINE_PRIORITY``
# will get the fallback value.  Runtime code should call _get_engine_priority().
_ENGINE_PRIORITY: list[str] = ["auto"]


def _normalize_engine(engine: str) -> str:
    """Normalize an engine name to its canonical form."""
    return _ENGINE_ALIASES.get(engine.lower(), engine.lower())


# ---------------------------------------------------------------------------
# Server-side resolution fallback
# ---------------------------------------------------------------------------


def _resolve_via_server(name: str, engine: str = "") -> Optional[ResolvedModel]:
    """Resolve a model via the Octomil API when local catalog has no variants.

    Calls ``POST /api/v1/resolve_model`` on the server, which has the full
    unscrubbed catalog. Returns None on any failure (network, auth, etc.).
    """
    try:
        import httpx
    except ImportError:
        return None

    api_base = os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")
    headers: dict[str, str] = {}
    api_key = os.environ.get("OCTOMIL_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{api_base.rstrip('/')}/resolve_model",
                json={"name": name, "engine": engine},
                headers=headers,
            )
        if resp.status_code != 200:
            logger.debug("Server resolve returned HTTP %d for '%s'", resp.status_code, name)
            return None

        data: dict[str, Any] = resp.json()
        return ResolvedModel(
            family=data.get("family"),
            quant=data.get("quant", "unknown"),
            engine=data.get("engine"),
            hf_repo=data.get("hf_repo", ""),
            filename=data.get("filename"),
            mlx_repo=data.get("mlx_repo"),
            source_repo=data.get("source_repo"),
            raw=name,
            architecture=data.get("architecture", "dense"),
        )
    except Exception:
        logger.debug("Server resolve failed for '%s'", name, exc_info=True)
        return None


def _suggest_models(name: str, n: int = 3) -> list[str]:
    """Return close matches for typo suggestions."""
    return list(difflib.get_close_matches(name, CATALOG.keys(), n=n, cutoff=0.4))


def _pick_engine(
    entry: ModelEntry,
    quant: str,
    available_engines: list[str],
) -> Optional[str]:
    """Pick the best available engine for a model + quant combination.

    Prefers engines in priority order that both:
    1. Are in the model's ``engines`` set
    2. Are in the ``available_engines`` list
    3. Have an artifact for the requested quant
    """
    variant = entry.variants.get(quant)
    if variant is None:
        return None

    normalized_available = [_normalize_engine(e) for e in available_engines]
    engine_priority = _get_engine_priority()

    # If server returned the minimal fallback ["auto"], use available_engines
    # order directly — let the caller's ordering decide.
    if engine_priority == ["auto"]:
        engine_priority = list(dict.fromkeys(normalized_available))

    for engine in engine_priority:
        if engine not in normalized_available:
            continue

        # Ollama support is derived from catalog tags, not entry.engines
        if engine == "ollama" and variant.ollama:
            return engine

        if engine not in entry.engines:
            continue

        # Check that this engine has an artifact for this quant
        if engine == "mlx-lm" and variant.mlx:
            return engine
        if engine == "mlc-llm" and (variant.mlc or variant.source_repo):
            return engine
        if engine == "llama.cpp" and variant.gguf:
            return engine
        if engine in ("mnn", "executorch") and (variant.gguf or variant.source_repo):
            return engine
        if engine == "onnxruntime" and (variant.ort or variant.source_repo):
            return engine
        if engine == "echo":
            return engine

    return None


def resolve(
    name: str,
    *,
    available_engines: Optional[list[str]] = None,
    engine: Optional[str] = None,
) -> ResolvedModel:
    """Resolve a model specifier to an engine-specific artifact.

    Parameters
    ----------
    name:
        Model specifier (e.g. ``"gemma-3b:4bit"``, ``"phi-mini"``,
        ``"mlx-community/model-4bit"``).
    available_engines:
        List of engine names available on this system. Used to pick
        the best engine when ``engine`` is not specified.
    engine:
        Force a specific engine. Overrides ``available_engines``.

    Returns
    -------
    ResolvedModel
        Resolved artifact with ``hf_repo``, ``filename`` (for GGUF),
        ``engine``, and ``quant``.

    Raises
    ------
    ModelResolutionError
        If the model family or variant is not found, or no engine
        can serve the requested combination.
    """
    parsed = parse(name)

    # --- Passthrough: local files and full repo IDs ---
    if parsed.is_passthrough:
        resolved_engine = _normalize_engine(engine) if engine else None
        return ResolvedModel(
            family=None,
            quant="unknown",
            engine=resolved_engine,
            hf_repo=parsed.raw,
            filename=parsed.raw if parsed.is_local_file else None,
            raw=name,
        )

    # --- Catalog lookup (with alias resolution) ---
    assert parsed.family is not None
    canonical = _resolve_alias(parsed.family)
    entry = CATALOG.get(canonical)
    if entry is None:
        suggestions = _suggest_models(parsed.family)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ModelResolutionError(
            f"Unknown model '{parsed.family}'. Available: {', '.join(sorted(CATALOG.keys()))}.{hint}"
        )

    # Determine quant level
    if parsed.variant is not None:
        quant = normalize_variant(parsed.variant)
    else:
        quant = entry.default_quant

    # Check variant exists
    variant = entry.variants.get(quant)
    if variant is None:
        available_quants = ", ".join(sorted(entry.variants.keys()))
        if available_quants:
            raise ModelResolutionError(
                f"Unknown variant '{parsed.variant or quant}' for model '{parsed.family}'. "
                f"Available: {available_quants}"
            )
        # No variants in local catalog — try server-side resolution
        server_result = _resolve_via_server(name, engine=engine or "")
        if server_result is not None:
            logger.info("Resolved '%s' via server (local catalog has no variants)", name)
            return server_result
        raise ModelResolutionError(
            f"Model '{parsed.family}' has no downloadable variants. "
            f"Try a full HuggingFace repo ID (e.g. 'mlx-community/gemma-2-2b-it-4bit')."
        )

    # Determine engine
    if engine:
        resolved_engine = _normalize_engine(engine)
    elif available_engines:
        resolved_engine = _pick_engine(entry, quant, available_engines)
    else:
        # No engine info — pick based on artifact availability
        resolved_engine = _pick_engine(entry, quant, list(_ENGINE_ALIASES.values()))

    # Build result based on engine
    hf_repo: str
    filename: Optional[str] = None
    mlx_repo: Optional[str] = variant.mlx

    if resolved_engine == "mlx-lm" and variant.mlx:
        hf_repo = variant.mlx
    elif resolved_engine == "mlc-llm" and variant.mlc:
        hf_repo = variant.mlc
    elif resolved_engine == "llama.cpp" and variant.gguf:
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif resolved_engine == "onnxruntime" and variant.ort:
        hf_repo = variant.ort
    elif resolved_engine == "ollama" and variant.ollama:
        # For Ollama, hf_repo stores the ollama tag
        hf_repo = variant.ollama
    elif variant.gguf:
        # Fallback to GGUF for engines that can consume GGUF (mnn, etc.)
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif variant.mlx:
        hf_repo = variant.mlx
    elif variant.source_repo:
        hf_repo = variant.source_repo
    else:
        # No local artifact — try server-side resolution
        server_result = _resolve_via_server(name, engine=engine or "")
        if server_result is not None:
            logger.info("Resolved '%s' via server (no local artifact for engine '%s')", name, resolved_engine)
            return server_result
        raise ModelResolutionError(
            f"No artifact found for '{parsed.family}:{quant}' "
            f"on engine '{resolved_engine}'. "
            f"Try a different variant or pass a full HuggingFace repo ID."
        )

    return ResolvedModel(
        family=canonical,
        quant=quant,
        engine=resolved_engine,
        hf_repo=hf_repo,
        filename=filename,
        mlx_repo=mlx_repo,
        source_repo=variant.source_repo,
        raw=name,
        architecture=entry.architecture,
        moe=entry.moe,
    )
