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
    ollama: Optional[str] = None  # Ollama model tag (e.g. "gemma3:1b")
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
                ollama="gemma3:1b",
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
                ollama="gemma3:4b",
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
                ollama="gemma3:12b",
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
                ollama="gemma3:27b",
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
                ollama="llama3.2:1b",
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
                ollama="llama3.2:3b",
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
                ollama="llama3.1:8b",
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
                ollama="phi4",
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
                ollama="phi4-mini",
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
                ollama="qwen2.5:1.5b",
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
                ollama="qwen2.5:3b",
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
                ollama="qwen2.5:7b",
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
                ollama="mistral:7b",
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
                ollama="mixtral:8x7b",
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
                ollama="smollm2:360m",
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
    # DeepSeek R1 — distilled dense models
    # -------------------------------------------------------------------
    "deepseek-r1-1.5b": ModelEntry(
        publisher="DeepSeek",
        params="1.5B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:1.5b",
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-r1-7b": ModelEntry(
        publisher="DeepSeek",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-r1-8b": ModelEntry(
        publisher="DeepSeek",
        params="8B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:8b",
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-r1-14b": ModelEntry(
        publisher="DeepSeek",
        params="14B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:14b",
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-r1-32b": ModelEntry(
        publisher="DeepSeek",
        params="32B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:32b",
                source_repo="REDACTED",
            ),
        },
    ),
    "deepseek-r1-70b": ModelEntry(
        publisher="DeepSeek",
        params="70B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="deepseek-r1:70b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Alibaba Qwen 3
    # -------------------------------------------------------------------
    "qwen3-0.6b": ModelEntry(
        publisher="Qwen",
        params="0.6B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:0.6b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen3-1.7b": ModelEntry(
        publisher="Qwen",
        params="1.7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:1.7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen3-4b": ModelEntry(
        publisher="Qwen",
        params="4B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:4b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen3-8b": ModelEntry(
        publisher="Qwen",
        params="8B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:8b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen3-14b": ModelEntry(
        publisher="Qwen",
        params="14B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:14b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen3-32b": ModelEntry(
        publisher="Qwen",
        params="32B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen3:32b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Alibaba Qwen 2.5 Coder
    # -------------------------------------------------------------------
    "qwen2.5-coder-1.5b": ModelEntry(
        publisher="Qwen",
        params="1.5B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen2.5-coder:1.5b",
                source_repo="REDACTED",
            ),
        },
    ),
    "qwen2.5-coder-7b": ModelEntry(
        publisher="Qwen",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="qwen2.5-coder:7b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Google Gemma 2
    # -------------------------------------------------------------------
    "gemma2-2b": ModelEntry(
        publisher="Google",
        params="2B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="gemma2:2b",
                source_repo="REDACTED",
            ),
        },
    ),
    "gemma2-9b": ModelEntry(
        publisher="Google",
        params="9B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="gemma2:9b",
                source_repo="REDACTED",
            ),
        },
    ),
    "gemma2-27b": ModelEntry(
        publisher="Google",
        params="27B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="gemma2:27b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Meta Llama 3.1 / 3.3
    # -------------------------------------------------------------------
    "llama-3.1-70b": ModelEntry(
        publisher="Meta",
        params="70B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="llama3.1:70b",
                source_repo="REDACTED",
            ),
        },
    ),
    "llama-3.3-70b": ModelEntry(
        publisher="Meta",
        params="70B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="llama3.3:70b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Mistral AI — expanded
    # -------------------------------------------------------------------
    "mistral-nemo": ModelEntry(
        publisher="Mistral AI",
        params="12B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="mistral-nemo",
                source_repo="REDACTED",
            ),
        },
    ),
    "mistral-small": ModelEntry(
        publisher="Mistral AI",
        params="24B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="mistral-small",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Meta CodeLlama
    # -------------------------------------------------------------------
    "codellama-7b": ModelEntry(
        publisher="Meta",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="codellama:7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "codellama-13b": ModelEntry(
        publisher="Meta",
        params="13B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="codellama:13b",
                source_repo="REDACTED",
            ),
        },
    ),
    "codellama-34b": ModelEntry(
        publisher="Meta",
        params="34B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="codellama:34b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # BigCode StarCoder2
    # -------------------------------------------------------------------
    "starcoder2-3b": ModelEntry(
        publisher="BigCode",
        params="3B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="starcoder2:3b",
                source_repo="REDACTED",
            ),
        },
    ),
    "starcoder2-7b": ModelEntry(
        publisher="BigCode",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="starcoder2:7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "starcoder2-15b": ModelEntry(
        publisher="BigCode",
        params="15B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="starcoder2:15b",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # 01.AI Yi 1.5
    # -------------------------------------------------------------------
    "yi-6b": ModelEntry(
        publisher="01.AI",
        params="6B",
        default_quant="4bit",
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
    "yi-9b": ModelEntry(
        publisher="01.AI",
        params="9B",
        default_quant="4bit",
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
    "yi-34b": ModelEntry(
        publisher="01.AI",
        params="34B",
        default_quant="4bit",
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
    # TII Falcon 3
    # -------------------------------------------------------------------
    "falcon3-1b": ModelEntry(
        publisher="TII",
        params="1B",
        default_quant="4bit",
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
    "falcon3-7b": ModelEntry(
        publisher="TII",
        params="7B",
        default_quant="4bit",
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
    "falcon3-10b": ModelEntry(
        publisher="TII",
        params="10B",
        default_quant="4bit",
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
    # Cohere Command-R
    # -------------------------------------------------------------------
    "command-r": ModelEntry(
        publisher="Cohere",
        params="35B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="command-r",
                source_repo="REDACTED",
            ),
        },
    ),
    "command-r-plus": ModelEntry(
        publisher="Cohere",
        params="104B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="command-r-plus",
                source_repo="REDACTED",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Other popular models
    # -------------------------------------------------------------------
    "tinyllama": ModelEntry(
        publisher="TinyLlama",
        params="1.1B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="tinyllama",
                source_repo="REDACTED",
            ),
        },
    ),
    "smollm2-1.7b": ModelEntry(
        publisher="HuggingFace",
        params="1.7B",
        default_quant="4bit",
        engines=frozenset({"llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="smollm2:1.7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "internlm2-7b": ModelEntry(
        publisher="Shanghai AI Lab",
        params="7B",
        default_quant="4bit",
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
    "codegemma-7b": ModelEntry(
        publisher="Google",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="codegemma:7b",
                source_repo="REDACTED",
            ),
        },
    ),
    "stable-code-3b": ModelEntry(
        publisher="Stability AI",
        params="3B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="stable-code:3b",
                source_repo="REDACTED",
            ),
        },
    ),
    "mathstral-7b": ModelEntry(
        publisher="Mistral AI",
        params="7B",
        default_quant="4bit",
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
    "glm4-9b": ModelEntry(
        publisher="Tsinghua",
        params="9B",
        default_quant="4bit",
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
    "granite-3.2-2b": ModelEntry(
        publisher="IBM",
        params="2B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="granite3.2:2b",
                source_repo="REDACTED",
            ),
        },
    ),
    "granite-3.2-8b": ModelEntry(
        publisher="IBM",
        params="8B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="granite3.2:8b",
                source_repo="REDACTED",
            ),
        },
    ),
    "olmo2-7b": ModelEntry(
        publisher="AI2",
        params="7B",
        default_quant="4bit",
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
    "zephyr-7b": ModelEntry(
        publisher="HuggingFace",
        params="7B",
        default_quant="4bit",
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
    "solar-10.7b": ModelEntry(
        publisher="Upstage",
        params="10.7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="REDACTED",
                gguf=GGUFSource(
                    "REDACTED",
                    "REDACTED",
                ),
                ollama="solar",
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
    # DeepSeek R1 aliases
    "deepseek-r1": "deepseek-r1-7b",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1-1.5b",
    "deepseek-r1-distill-qwen-7b": "deepseek-r1-7b",
    "deepseek-r1-distill-llama-8b": "deepseek-r1-8b",
    "deepseek-r1-distill-qwen-14b": "deepseek-r1-14b",
    "deepseek-r1-distill-qwen-32b": "deepseek-r1-32b",
    "deepseek-r1-distill-llama-70b": "deepseek-r1-70b",
    # Qwen 3 aliases
    "qwen3": "qwen3-4b",
    # Gemma 2 aliases
    "gemma2": "gemma2-9b",
    # Llama 3.1 / 3.3 aliases
    "llama3.1": "llama-8b",
    "llama-3.1-8b": "llama-8b",
    "llama3.3": "llama-3.3-70b",
    # Mistral expanded aliases
    "mistral-nemo-12b": "mistral-nemo",
    "mistral-small-24b": "mistral-small",
    # Code model aliases
    "codellama": "codellama-7b",
    "starcoder2": "starcoder2-15b",
    "codegemma": "codegemma-7b",
    "stable-code": "stable-code-3b",
    # Other family aliases
    "falcon3": "falcon3-7b",
    "yi": "yi-6b",
    "internlm2": "internlm2-7b",
    "mathstral": "mathstral-7b",
    "glm4": "glm4-9b",
    "granite-3.2": "granite-3.2-8b",
    "granite": "granite-3.2-8b",
    "olmo2": "olmo2-7b",
    "zephyr": "zephyr-7b",
    "solar": "solar-10.7b",
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
