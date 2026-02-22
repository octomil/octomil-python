"""Unified model catalog — single source of truth for all engines.

Replaces per-engine ``_MLX_CATALOG``, ``_GGUF_CATALOG``, ``_MNN_CATALOG``,
``_ET_CATALOG`` sets and the ``_MLX_MODELS`` / ``_GGUF_MODELS`` dicts in
``serve.py``.

Each model entry maps quant variants to engine-specific artifacts:

- ``mlx``: HuggingFace MLX repo ID
- ``gguf``: Tuple of (HF repo, filename)
- ``source``: Original model HF repo (for engines that download and convert)

Engine-specific quant names are handled here so the resolver doesn't need
to know about GGUF's ``Q4_K_M`` vs MLX's ``4bit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GGUFSource:
    """GGUF model source — HuggingFace repo + filename."""

    repo: str
    filename: str


@dataclass(frozen=True)
class VariantSpec:
    """Artifact locations for a single quantization variant across engines."""

    mlx: Optional[str] = None
    gguf: Optional[GGUFSource] = None
    ort: Optional[str] = None  # ONNX Runtime model repo ID
    source_repo: Optional[str] = None  # original (fp16/bf16) repo


@dataclass
class ModelEntry:
    """A model family with its per-engine, per-quant variants."""

    publisher: str
    params: str
    default_quant: str = "4bit"
    variants: dict[str, VariantSpec] = field(default_factory=dict)
    engines: frozenset[str] = frozenset()  # engines this model is known to work on


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, ModelEntry] = {
    # -------------------------------------------------------------------
    # Google Gemma
    # -------------------------------------------------------------------
    "gemma-1b": ModelEntry(
        publisher="Google",
        params="1B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-1b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-1b-it-GGUF",
                    "google_gemma-3-1b-it-Q4_K_M.gguf",
                ),
                source_repo="google/gemma-3-1b-it",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/gemma-3-1b-it-8bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-1b-it-GGUF",
                    "google_gemma-3-1b-it-Q8_0.gguf",
                ),
                source_repo="google/gemma-3-1b-it",
            ),
            "fp16": VariantSpec(
                source_repo="google/gemma-3-1b-it",
            ),
        },
    ),
    "gemma-4b": ModelEntry(
        publisher="Google",
        params="4B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-4b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-4b-it-GGUF",
                    "google_gemma-3-4b-it-Q4_K_M.gguf",
                ),
                source_repo="google/gemma-3-4b-it",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/gemma-3-4b-it-8bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-4b-it-GGUF",
                    "google_gemma-3-4b-it-Q8_0.gguf",
                ),
                source_repo="google/gemma-3-4b-it",
            ),
            "fp16": VariantSpec(
                source_repo="google/gemma-3-4b-it",
            ),
        },
    ),
    "gemma-12b": ModelEntry(
        publisher="Google",
        params="12B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-12b-it-4bit",
                source_repo="google/gemma-3-12b-it",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/gemma-3-12b-it-8bit",
                source_repo="google/gemma-3-12b-it",
            ),
        },
    ),
    "gemma-27b": ModelEntry(
        publisher="Google",
        params="27B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-27b-it-4bit",
                source_repo="google/gemma-3-27b-it",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Meta Llama
    # -------------------------------------------------------------------
    "llama-1b": ModelEntry(
        publisher="Meta",
        params="1B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-1B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="meta-llama/Llama-3.2-1B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-1B-Instruct-8bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q8_0.gguf",
                ),
                source_repo="meta-llama/Llama-3.2-1B-Instruct",
            ),
            "fp16": VariantSpec(
                source_repo="meta-llama/Llama-3.2-1B-Instruct",
            ),
        },
    ),
    "llama-3b": ModelEntry(
        publisher="Meta",
        params="3B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-3B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="meta-llama/Llama-3.2-3B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-3B-Instruct-8bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "Llama-3.2-3B-Instruct-Q8_0.gguf",
                ),
                source_repo="meta-llama/Llama-3.2-3B-Instruct",
            ),
        },
    ),
    "llama-8b": ModelEntry(
        publisher="Meta",
        params="8B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
                gguf=GGUFSource(
                    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                ),
                source_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Microsoft Phi
    # -------------------------------------------------------------------
    "phi-4": ModelEntry(
        publisher="Microsoft",
        params="14B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/phi-4-4bit",
                source_repo="microsoft/phi-4",
            ),
        },
    ),
    "phi-mini": ModelEntry(
        publisher="Microsoft",
        params="3.8B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Phi-3.5-mini-instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Phi-3.5-mini-instruct-GGUF",
                    "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                ),
                source_repo="microsoft/Phi-3.5-mini-instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Phi-3.5-mini-instruct-8bit",
                gguf=GGUFSource(
                    "bartowski/Phi-3.5-mini-instruct-GGUF",
                    "Phi-3.5-mini-instruct-Q8_0.gguf",
                ),
                source_repo="microsoft/Phi-3.5-mini-instruct",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Alibaba Qwen
    # -------------------------------------------------------------------
    "qwen-1.5b": ModelEntry(
        publisher="Qwen",
        params="1.5B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen2.5-1.5B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-1.5B-Instruct-8bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "qwen2.5-1.5b-instruct-q8_0.gguf",
                ),
                source_repo="Qwen/Qwen2.5-1.5B-Instruct",
            ),
        },
    ),
    "qwen-3b": ModelEntry(
        publisher="Qwen",
        params="3B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-3B-Instruct-4bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "qwen2.5-3b-instruct-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen2.5-3B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-3B-Instruct-8bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "qwen2.5-3b-instruct-q8_0.gguf",
                ),
                source_repo="Qwen/Qwen2.5-3B-Instruct",
            ),
        },
    ),
    "qwen-7b": ModelEntry(
        publisher="Qwen",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-7B-Instruct-4bit",
                source_repo="Qwen/Qwen2.5-7B-Instruct",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Mistral AI
    # -------------------------------------------------------------------
    "mistral-7b": ModelEntry(
        publisher="Mistral AI",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                gguf=GGUFSource(
                    "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                ),
                source_repo="mistralai/Mistral-7B-Instruct-v0.3",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                gguf=GGUFSource(
                    "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                    "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                ),
                source_repo="mistralai/Mistral-7B-Instruct-v0.3",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # HuggingFace SmolLM
    # -------------------------------------------------------------------
    "smollm-360m": ModelEntry(
        publisher="HuggingFace",
        params="360M",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp", "mnn"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/SmolLM-360M-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/SmolLM2-360M-Instruct-GGUF",
                    "SmolLM2-360M-Instruct-Q4_K_M.gguf",
                ),
                source_repo="HuggingFaceTB/SmolLM2-360M-Instruct",
            ),
            "8bit": VariantSpec(
                gguf=GGUFSource(
                    "bartowski/SmolLM2-360M-Instruct-GGUF",
                    "SmolLM2-360M-Instruct-Q8_0.gguf",
                ),
                source_repo="HuggingFaceTB/SmolLM2-360M-Instruct",
            ),
        },
    ),
}


def list_models() -> list[str]:
    """Return sorted list of all known model family names."""
    return sorted(CATALOG.keys())


def get_model(name: str) -> Optional[ModelEntry]:
    """Look up a model entry by family name."""
    return CATALOG.get(name.lower())


def supports_engine(name: str, engine: str) -> bool:
    """Check if a model family is known to work on a given engine."""
    entry = CATALOG.get(name.lower())
    if entry is None:
        return False
    return engine in entry.engines
