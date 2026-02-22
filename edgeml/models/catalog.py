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
    mlc: Optional[str] = None  # MLC-LLM pre-compiled model repo ID
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-1b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-1b-it-GGUF",
                    "google_gemma-3-1b-it-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/gemma-2b-it-q4f16_1-MLC",
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/gemma-3-4b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/google_gemma-3-4b-it-GGUF",
                    "google_gemma-3-4b-it-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/gemma-2b-it-q4f16_1-MLC",
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-1B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Llama-3.2-3B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/Meta-Llama-3.1-8B-Instruct-q4f16_1-MLC",
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
        engines=frozenset(
            {"mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"}
        ),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Phi-3.5-mini-instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Phi-3.5-mini-instruct-GGUF",
                    "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                ),
                mlc="mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
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
    # -------------------------------------------------------------------
    # OpenAI Whisper (Speech-to-Text)
    # -------------------------------------------------------------------
    "whisper-tiny": ModelEntry(
        publisher="OpenAI",
        params="39M",
        default_quant="fp16",
        engines=frozenset({"whisper.cpp"}),
        variants={
            "fp16": VariantSpec(
                source_repo="openai/whisper-tiny",
            ),
        },
    ),
    "whisper-base": ModelEntry(
        publisher="OpenAI",
        params="74M",
        default_quant="fp16",
        engines=frozenset({"whisper.cpp"}),
        variants={
            "fp16": VariantSpec(
                source_repo="openai/whisper-base",
            ),
        },
    ),
    "whisper-small": ModelEntry(
        publisher="OpenAI",
        params="244M",
        default_quant="fp16",
        engines=frozenset({"whisper.cpp"}),
        variants={
            "fp16": VariantSpec(
                source_repo="openai/whisper-small",
            ),
        },
    ),
    "whisper-medium": ModelEntry(
        publisher="OpenAI",
        params="769M",
        default_quant="fp16",
        engines=frozenset({"whisper.cpp"}),
        variants={
            "fp16": VariantSpec(
                source_repo="openai/whisper-medium",
            ),
        },
    ),
    "whisper-large-v3": ModelEntry(
        publisher="OpenAI",
        params="1.55B",
        default_quant="fp16",
        engines=frozenset({"whisper.cpp"}),
        variants={
            "fp16": VariantSpec(
                source_repo="openai/whisper-large-v3",
            ),
        },
    ),
}


# User-friendly aliases -> canonical catalog key.
# Allows docs to reference full model names (phi-4-mini, llama-3.2-1b)
# while the catalog uses shorter keys.
MODEL_ALIASES: dict[str, str] = {
    "phi-4-mini": "phi-mini",
    "phi-mini-3.8b": "phi-mini",
    "phi-3.5-mini": "phi-mini",
    "llama-3.2-1b": "llama-1b",
    "llama-3.2-3b": "llama-3b",
    "llama-3.1-8b": "llama-8b",
    "gemma-3-1b": "gemma-1b",
    "gemma-3-4b": "gemma-4b",
    "gemma-3-12b": "gemma-12b",
    "gemma-3-27b": "gemma-27b",
    "gemma-3b": "gemma-4b",
    "qwen-2.5-1.5b": "qwen-1.5b",
    "qwen-2.5-3b": "qwen-3b",
    "qwen-2.5-7b": "qwen-7b",
    "phi4": "phi-4",
    "phi4-mini": "phi-mini",
}


def _resolve_alias(name: str) -> str:
    """Resolve a model name alias to its canonical catalog key."""
    return MODEL_ALIASES.get(name.lower(), name.lower())


def list_models() -> list[str]:
    """Return sorted list of all known model family names."""
    return sorted(CATALOG.keys())


def get_model(name: str) -> Optional[ModelEntry]:
    """Look up a model entry by family name, checking aliases."""
    key = _resolve_alias(name)
    return CATALOG.get(key)


def supports_engine(name: str, engine: str) -> bool:
    """Check if a model family is known to work on a given engine."""
    entry = CATALOG.get(name.lower())
    if entry is None:
        return False
    return engine in entry.engines
