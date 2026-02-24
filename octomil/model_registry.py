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

TRUST_PRIORITY = {"official": 0, "curated": 1, "community": 2}


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
                mlx="mlx-community/gemma-3-1b-it-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-1b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:1b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/google_gemma-3-1b-it-GGUF",
                        file="google_gemma-3-1b-it-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/gemma-3-1b-it-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-1b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/google_gemma-3-1b-it-GGUF",
                        file="google_gemma-3-1b-it-Q8_0.gguf",
                        trust="community",
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-1b-it",
                        trust="official",
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
                mlx="mlx-community/gemma-3-4b-it-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-4b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:4b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/google_gemma-3-4b-it-GGUF",
                        file="google_gemma-3-4b-it-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/gemma-3-4b-it-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-4b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/google_gemma-3-4b-it-GGUF",
                        file="google_gemma-3-4b-it-Q8_0.gguf",
                        trust="community",
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-4b-it",
                        trust="official",
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
                mlx="mlx-community/gemma-3-12b-it-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-12b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:12b",
                        trust="curated",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/gemma-3-12b-it-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-12b-it",
                        trust="official",
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
                mlx="mlx-community/gemma-3-27b-it-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="google/gemma-3-27b-it",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="gemma3:27b",
                        trust="curated",
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
                mlx="mlx-community/Llama-3.2-1B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Llama-3.2-1B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.2:1b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Llama-3.2-1B-Instruct-GGUF",
                        file="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Llama-3.2-1B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Llama-3.2-1B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Llama-3.2-1B-Instruct-GGUF",
                        file="Llama-3.2-1B-Instruct-Q8_0.gguf",
                        trust="community",
                    ),
                ],
            ),
            "fp16": ModelVariant(
                quantization_family="16bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Llama-3.2-1B-Instruct",
                        trust="official",
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
                mlx="mlx-community/Llama-3.2-3B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Llama-3.2-3B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.2:3b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Llama-3.2-3B-Instruct-GGUF",
                        file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Llama-3.2-3B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Llama-3.2-3B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Llama-3.2-3B-Instruct-GGUF",
                        file="Llama-3.2-3B-Instruct-Q8_0.gguf",
                        trust="community",
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
                mlx="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="llama3.1:8b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                        file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                        file="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                        trust="community",
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
                mlx="mlx-community/phi-4-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="microsoft/phi-4",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="phi4",
                        trust="curated",
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
                mlx="mlx-community/Phi-3.5-mini-instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="microsoft/Phi-3.5-mini-instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="phi3.5",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Phi-3.5-mini-instruct-GGUF",
                        file="Phi-3.5-mini-instruct-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Phi-3.5-mini-instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="microsoft/Phi-3.5-mini-instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Phi-3.5-mini-instruct-GGUF",
                        file="Phi-3.5-mini-instruct-Q8_0.gguf",
                        trust="community",
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
                mlx="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        file="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:1.5b",
                        trust="curated",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Qwen2.5-1.5B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        file="qwen2.5-1.5b-instruct-q8_0.gguf",
                        trust="official",
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
                mlx="mlx-community/Qwen2.5-3B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-3B-Instruct-GGUF",
                        file="qwen2.5-3b-instruct-q4_k_m.gguf",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:3b",
                        trust="curated",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Qwen2.5-3B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-3B-Instruct-GGUF",
                        file="qwen2.5-3b-instruct-q8_0.gguf",
                        trust="official",
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
                mlx="mlx-community/Qwen2.5-7B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-7B-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="qwen2.5:7b",
                        trust="curated",
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Alibaba Qwen — Coder (code-specialized)
    # -----------------------------------------------------------------------
    "qwen-coder-1.5b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="1.5B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                        file="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
                        trust="official",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                        file="qwen2.5-coder-1.5b-instruct-q8_0.gguf",
                        trust="official",
                    ),
                ],
            ),
        },
    ),
    "qwen-coder-3b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="3B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                        file="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                        trust="official",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Qwen2.5-Coder-3B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                        file="qwen2.5-coder-3b-instruct-q8_0.gguf",
                        trust="official",
                    ),
                ],
            ),
        },
    ),
    "qwen-coder-7b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Qwen",
        params="7B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        file="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                        trust="official",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Qwen2.5-Coder-7B-Instruct-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                        file="qwen2.5-coder-7b-instruct-q8_0.gguf",
                        trust="official",
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
                mlx="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="mistralai/Mistral-7B-Instruct-v0.3",
                        trust="official",
                    ),
                    ModelSource(
                        type="ollama",
                        ref="mistral:7b",
                        trust="curated",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                        file="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="mistralai/Mistral-7B-Instruct-v0.3",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                        file="Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                        trust="community",
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
                mlx="mlx-community/SmolLM-360M-Instruct-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="HuggingFaceTB/SmolLM2-360M-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/SmolLM2-360M-Instruct-GGUF",
                        file="SmolLM2-360M-Instruct-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="HuggingFaceTB/SmolLM2-360M-Instruct",
                        trust="official",
                    ),
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/SmolLM2-360M-Instruct-GGUF",
                        file="SmolLM2-360M-Instruct-Q8_0.gguf",
                        trust="community",
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Zhipu AI GLM
    # -----------------------------------------------------------------------
    "glm-flash": ModelFamily(
        default_tag="q4_k_m",
        publisher="Zhipu AI",
        params="30B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/GLM-4.7-Flash-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/zai-org_GLM-4.7-Flash-GGUF",
                        file="zai-org_GLM-4.7-Flash-Q4_K_M.gguf",
                        trust="community",
                    ),
                ],
            ),
            "q8_0": ModelVariant(
                quantization_family="8bit",
                mlx="mlx-community/GLM-4.7-Flash-8bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="bartowski/zai-org_GLM-4.7-Flash-GGUF",
                        file="zai-org_GLM-4.7-Flash-Q8_0.gguf",
                        trust="community",
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # Mistral AI — Devstral (agentic coding)
    # -----------------------------------------------------------------------
    "devstral-123b": ModelFamily(
        default_tag="q4_k_m",
        publisher="Mistral AI",
        params="123B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/Devstral-2-123B-Instruct-2512-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="unsloth/Devstral-2-123B-Instruct-2512-GGUF",
                        file="Devstral-2-123B-Instruct-2512-Q4_K_M-00001-of-00002.gguf",
                        trust="community",
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # DeepSeek V3.2
    # -----------------------------------------------------------------------
    "deepseek-v3.2": ModelFamily(
        default_tag="q4_k_m",
        publisher="DeepSeek",
        params="685B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/DeepSeek-V3.2-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="unsloth/DeepSeek-V3.2-GGUF",
                        file="DeepSeek-V3.2-Q4_K_M-00001-of-00009.gguf",
                        trust="community",
                    ),
                ],
            ),
        },
    ),
    # -----------------------------------------------------------------------
    # MiniMax M2.1
    # -----------------------------------------------------------------------
    "minimax-m2.1": ModelFamily(
        default_tag="q4_k_m",
        publisher="MiniMax",
        params="229B",
        variants={
            "q4_k_m": ModelVariant(
                quantization_family="4bit",
                mlx="mlx-community/MiniMax-M2.1-4bit",
                sources=[
                    ModelSource(
                        type="huggingface",
                        ref="unsloth/MiniMax-M2.1-GGUF",
                        file="MiniMax-M2.1-Q4_K_M-00001-of-00003.gguf",
                        trust="community",
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
        or a full repo path like ``"mlx-community/gemma-3-4b-it-4bit"``.
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
            source=ModelSource(type="huggingface", ref=tag, trust="community"),
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
