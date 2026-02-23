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


@dataclass(frozen=True)
class MoEMetadata:
    """Mixture of Experts model metadata.

    Captures the sparse activation pattern that defines MoE models:
    only ``active_experts`` out of ``num_experts`` are activated per token,
    meaning RAM usage is closer to ``active_params`` than ``total_params``.
    """

    num_experts: int
    active_experts: int
    expert_size: str  # human-readable, e.g. "7B"
    total_params: str  # total parameter count, e.g. "46.7B"
    active_params: str  # params active per token, e.g. "12.9B"


@dataclass
class ModelEntry:
    """A model family with its per-engine, per-quant variants."""

    publisher: str
    params: str
    default_quant: str = "4bit"
    variants: dict[str, VariantSpec] = field(default_factory=dict)
    engines: frozenset[str] = frozenset()  # engines this model is known to work on
    architecture: str = "dense"  # "dense" or "moe"
    moe: Optional[MoEMetadata] = None  # populated when architecture == "moe"


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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "fp16": VariantSpec(
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "fp16": VariantSpec(
                source_repo="REDACTED",
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
                mlx="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                source_repo="REDACTED",
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
                mlx="REDACTED",
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "fp16": VariantSpec(
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                mlc="REDACTED",
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Mistral AI — MoE models
    # -------------------------------------------------------------------
    "mixtral-8x7b": ModelEntry(
        publisher="Mistral AI",
        params="46.7B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=8,
            active_experts=2,
            expert_size="7B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    "mixtral-8x22b": ModelEntry(
        publisher="Mistral AI",
        params="141B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=8,
            active_experts=2,
            expert_size="22B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Databricks DBRX
    # -------------------------------------------------------------------
    "dbrx": ModelEntry(
        publisher="Databricks",
        params="132B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=16,
            active_experts=4,
            expert_size="8B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # DeepSeek MoE models
    # -------------------------------------------------------------------
    "deepseek-v3": ModelEntry(
        publisher="DeepSeek",
        params="671B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=256,
            active_experts=8,
            expert_size="2B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-v2-lite": ModelEntry(
        publisher="DeepSeek",
        params="15.7B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=64,
            active_experts=6,
            expert_size="0.2B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Qwen MoE models
    # -------------------------------------------------------------------
    "qwen-moe-14b": ModelEntry(
        publisher="Qwen",
        params="14.3B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=60,
            active_experts=4,
            expert_size="0.2B",
            REDACTED,
            REDACTED,
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
            ),
            "8bit": VariantSpec(
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                source_repo="REDACTED",
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
                source_repo="REDACTED",
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
                source_repo="REDACTED",
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
                source_repo="REDACTED",
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
                source_repo="REDACTED",
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
                source_repo="REDACTED",
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
    # MoE aliases
    "mixtral": "mixtral-8x7b",
    "mixtral-instruct": "mixtral-8x7b",
    "mixtral-8x22b-instruct": "mixtral-8x22b",
    "dbrx-instruct": "dbrx",
    "deepseek-v2-lite-chat": "deepseek-v2-lite",
    "qwen-1.5-moe": "qwen-moe-14b",
    "qwen-moe": "qwen-moe-14b",
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


def is_moe_model(name: str) -> bool:
    """Check if a model uses Mixture of Experts architecture."""
    entry = get_model(name)
    if entry is None:
        return False
    return entry.architecture == "moe"


def list_moe_models() -> list[str]:
    """Return sorted list of all MoE model family names."""
    return sorted(
        name for name, entry in CATALOG.items() if entry.architecture == "moe"
    )


def get_moe_metadata(name: str) -> Optional[MoEMetadata]:
    """Get MoE metadata for a model, or None if not an MoE model."""
    entry = get_model(name)
    if entry is None:
        return None
    return entry.moe
