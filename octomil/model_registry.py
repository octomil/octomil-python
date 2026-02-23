"""Model registry with tag syntax and multi-source backends.

Provides structured model metadata, tag-based resolution (``name:tag``),
and multi-source download paths for HuggingFace, Ollama, and Kaggle.

Usage::

    from octomil.model_registry import parse_model_tag, resolve_model, MODEL_FAMILIES

    family, tag = parse_model_tag("gemma-4b:q8_0")
    result = resolve_model("gemma-4b:q8_0", backend="mlx")
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    """A single downloadable source for a model variant."""

    type: str  # "huggingface", "ollama", "kaggle"
    ref: str  # repo ID, ollama ref, or kaggle model path
    file: Optional[str] = None  # specific file in repo (for GGUF)
    trust: str = "community"  # "official", "curated", "community"


@dataclass
class ModelVariant:
    """A specific quantization/format variant of a model family."""

    quantization_family: str  # "4bit", "8bit", "16bit", "32bit"
    sources: list[ModelSource] = field(default_factory=list)
    mlx: Optional[str] = None  # MLX repo for Apple Silicon


@dataclass
class ModelFamily:
    """A named model family with multiple quantization variants."""

    default_tag: str
    publisher: str
    params: str  # e.g. "4B", "7B"
    variants: dict[str, ModelVariant] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag syntax parser
# ---------------------------------------------------------------------------

DEFAULT_TAG = "q4_k_m"


def parse_model_tag(name: str) -> tuple[Optional[str], str]:
    """Parse ``name:tag`` into (family_key, tag).

    Returns ``(None, name)`` for full repo paths (containing ``/``).

    Examples::

        parse_model_tag("gemma-4b")         -> ("gemma-4b", "q4_k_m")
        parse_model_tag("gemma-4b:q8_0")    -> ("gemma-4b", "q8_0")
        parse_model_tag("mlx-community/foo") -> (None, "mlx-community/foo")
        parse_model_tag("user/repo:branch")  -> (None, "user/repo:branch")
    """
    # Full repo paths pass through unchanged
    if "/" in name:
        return None, name

    if ":" in name:
        family, tag = name.split(":", 1)
        return family, tag

    return name, DEFAULT_TAG


@dataclass
class ResolvedModel:
    """Result of resolving a model name to a downloadable reference."""

    family: Optional[str]
    tag: str
    source: Optional[ModelSource]
    mlx_repo: Optional[str]
    variant: Optional[ModelVariant]
    raw_name: str  # original input


class ModelResolutionError(ValueError):
    """Raised when a model name cannot be resolved."""


# ---------------------------------------------------------------------------
# Trust priority ordering
# ---------------------------------------------------------------------------

REDACTED


def _sort_sources_by_trust(sources: list[ModelSource]) -> list[ModelSource]:
    """Sort sources by trust level (official first)."""
    return sorted(sources, key=lambda s: TRUST_PRIORITY.get(s.trust, 99))


# ---------------------------------------------------------------------------
# Model registry — all known model families and their variants
# ---------------------------------------------------------------------------

MODEL_FAMILIES: dict[str, ModelFamily] = {
    # -----------------------------------------------------------------------
    # Google Gemma
    # -----------------------------------------------------------------------
    "gemma-1b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Google",
        params="1B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:1b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "gemma-4b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Google",
        params="4B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:4b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "gemma-12b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Google",
        params="12B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:12b",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "gemma-27b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Google",
        params="27B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:27b",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Meta Llama
    # -----------------------------------------------------------------------
    "llama-1b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Meta",
        params="1B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.2:1b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "llama-3b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Meta",
        params="3B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.2:3b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "llama-8b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Meta",
        params="8B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.1:8b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Microsoft Phi
    # -----------------------------------------------------------------------
    "phi-4": ModelFamily(
        default_tag="q4_k_m",
        publisher="Microsoft",
        params="14B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="phi4",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "phi-mini": ModelFamily(
        default_tag="q4_k_m",
        publisher="Microsoft",
        params="3.8B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="phi3.5",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Alibaba Qwen
    # -----------------------------------------------------------------------
    "qwen-1.5b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="1.5B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:1.5b",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "qwen-3b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="3B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:3b",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    "qwen-7b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="7B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:7b",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Mistral AI
    # -----------------------------------------------------------------------
    "mistral-7b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Mistral AI",
        params="7B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="ollama",
                        ref="mistral:7b",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # HuggingFace SmolLM
    # -----------------------------------------------------------------------
    "smollm-360m": ModelFamily(
        default_tag="q4_k_m",
        publisher="HuggingFace",
        params="360M",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="REDACTED",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        REDACTED,
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="REDACTED",
                        file="REDACTED",
                        REDACTED,
                    ),
                ],
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------


def _suggest_families(name: str, n: int = 3) -> list[str]:
    """Find the closest matching family names for typo suggestions."""
    matches = difflib.get_close_matches(name, MODEL_FAMILIES.keys(), n=n, cutoff=0.4)
    return list(matches)


def resolve_model(
    name: str,
    *,
    backend: str = "auto",
    prefer_local: bool = True,
) -> ResolvedModel:
    """Resolve a model name (with optional tag) to a downloadable reference.

    Parameters
    ----------
    name:
        Model specifier. Can be ``"gemma-4b"``, ``"gemma-4b:q8_0"``,
        or a full repo path like ``"REDACTED"``.
    backend:
        Target backend: ``"mlx"``, ``"gguf"``, or ``"auto"``.
    prefer_local:
        When True, check for local Ollama cache before remote sources.

    Returns
    -------
    ResolvedModel
        Contains the resolved source, MLX repo (if applicable), and variant info.

    Raises
    ------
    ModelResolutionError
        If the model family or tag is not found in the registry.
    """
    family_key, tag = parse_model_tag(name)

    # Full repo path — pass through unchanged
    if family_key is None:
        return ResolvedModel(
            family=None,
            tag=tag,
            source=ModelSource(type="huggingface", ref=tag, REDACTED),
            mlx_repo=None,
            variant=None,
            raw_name=name,
        )

    # Look up family
    family = MODEL_FAMILIES.get(family_key)
    if family is None:
        suggestions = _suggest_families(family_key)
        suggestion_str = ""
        if suggestions:
            suggestion_str = f" Did you mean: {', '.join(suggestions)}?"
        raise ModelResolutionError(
            f"Unknown model '{family_key}'. "
            f"Available: {', '.join(sorted(MODEL_FAMILIES))}.{suggestion_str}"
        )

    # Use family default tag if the parsed tag matches the default sentinel
    effective_tag = tag if tag != DEFAULT_TAG else family.default_tag

    # Look up variant
    variant = family.variants.get(effective_tag)
    if variant is None:
        available_tags = ", ".join(sorted(family.variants.keys()))
        raise ModelResolutionError(
            f"Unknown tag '{effective_tag}' for model '{family_key}'. "
            f"Available tags: {available_tags}"
        )

    # Select best source by trust priority
    sources = _sort_sources_by_trust(variant.sources)

    # For MLX backend, prefer the mlx field
    mlx_repo = variant.mlx if backend in ("mlx", "auto") else None

    # Filter sources by backend preference
    if backend == "gguf":
        # Prefer sources with GGUF files
        gguf_sources = [s for s in sources if s.file and s.file.endswith(".gguf")]
        if gguf_sources:
            sources = gguf_sources
    elif backend == "mlx":
        # For MLX, the mlx repo is the primary; sources are fallback
        pass

    best_source = sources[0] if sources else None

    return ResolvedModel(
        family=family_key,
        tag=effective_tag,
        source=best_source,
        mlx_repo=mlx_repo,
        variant=variant,
        raw_name=name,
    )


def list_families() -> dict[str, ModelFamily]:
    """Return all registered model families."""
    return dict(MODEL_FAMILIES)


def get_family(name: str) -> Optional[ModelFamily]:
    """Return a specific model family by name, or None."""
    return MODEL_FAMILIES.get(name)
