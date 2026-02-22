"""Model resolution — maps ``model:variant`` to engine-specific artifacts.

Takes a parsed model specifier and resolves it to a concrete HuggingFace
repo ID (and optional filename) for a given engine.

Usage::

    from edgeml.models import resolve

    r = resolve("gemma-3b:4bit", available_engines=["mlx-lm", "llama.cpp"])
    print(r.engine, r.hf_repo, r.filename)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

from .catalog import CATALOG, GGUFSource, ModelEntry
from .parser import ParsedModel, normalize_variant, parse


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

    @property
    def is_gguf(self) -> bool:
        return self.filename is not None and self.filename.endswith(".gguf")


class ModelResolutionError(ValueError):
    """Raised when a model specifier cannot be resolved."""


# Engine name normalization for matching
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
    "whisper.cpp": "whisper.cpp",
    "whisper": "whisper.cpp",
    "whispercpp": "whisper.cpp",
    "echo": "echo",
}

# Engine priority order — used when picking the best engine automatically.
# Lower index = higher priority.
_ENGINE_PRIORITY = ["mlx-lm", "mnn", "llama.cpp", "executorch", "onnxruntime", "whisper.cpp", "echo"]


def _normalize_engine(engine: str) -> str:
    """Normalize an engine name to its canonical form."""
    return _ENGINE_ALIASES.get(engine.lower(), engine.lower())


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

    for engine in _ENGINE_PRIORITY:
        if engine not in normalized_available:
            continue
        if engine not in entry.engines:
            continue

        # Check that this engine has an artifact for this quant
        if engine == "mlx-lm" and variant.mlx:
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

    # --- Catalog lookup ---
    assert parsed.family is not None
    entry = CATALOG.get(parsed.family)
    if entry is None:
        suggestions = _suggest_models(parsed.family)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ModelResolutionError(
            f"Unknown model '{parsed.family}'. "
            f"Available: {', '.join(sorted(CATALOG.keys()))}.{hint}"
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
        raise ModelResolutionError(
            f"Unknown variant '{parsed.variant or quant}' for model "
            f"'{parsed.family}'. Available: {available_quants}"
        )

    # Determine engine
    if engine:
        resolved_engine = _normalize_engine(engine)
    elif available_engines:
        resolved_engine = _pick_engine(entry, quant, available_engines)
    else:
        # No engine info — pick based on artifact availability
        resolved_engine = _pick_engine(
            entry, quant, list(_ENGINE_ALIASES.values())
        )

    # Build result based on engine
    hf_repo: str
    filename: Optional[str] = None
    mlx_repo: Optional[str] = variant.mlx

    if resolved_engine == "mlx-lm" and variant.mlx:
        hf_repo = variant.mlx
    elif resolved_engine == "llama.cpp" and variant.gguf:
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif resolved_engine == "onnxruntime" and variant.ort:
        hf_repo = variant.ort
    elif variant.gguf:
        # Fallback to GGUF for engines that can consume GGUF (mnn, etc.)
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif variant.mlx:
        hf_repo = variant.mlx
    elif variant.source_repo:
        hf_repo = variant.source_repo
    else:
        raise ModelResolutionError(
            f"No artifact found for '{parsed.family}:{quant}' "
            f"on engine '{resolved_engine}'. "
            f"Try a different variant or pass a full HuggingFace repo ID."
        )

    return ResolvedModel(
        family=parsed.family,
        quant=quant,
        engine=resolved_engine,
        hf_repo=hf_repo,
        filename=filename,
        mlx_repo=mlx_repo,
        source_repo=variant.source_repo,
        raw=name,
    )
