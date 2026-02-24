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
    # Alibaba Qwen — Coder (code-specialized)
    # -------------------------------------------------------------------
    "qwen-coder-1.5b": ModelEntry(
        publisher="Qwen",
        params="1.5B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-8bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                    "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            ),
        },
    ),
    "qwen-coder-3b": ModelEntry(
        publisher="Qwen",
        params="3B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                    "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-3B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-3B-Instruct-8bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                    "qwen2.5-coder-3b-instruct-q8_0.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-3B-Instruct",
            ),
        },
    ),
    "qwen-coder-7b": ModelEntry(
        publisher="Qwen",
        params="7B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Qwen2.5-Coder-7B-Instruct-8bit",
                gguf=GGUFSource(
                    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    "qwen2.5-coder-7b-instruct-q8_0.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
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
            total_params="46.7B",
            active_params="12.9B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
                gguf=GGUFSource(
                    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                    "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
                ),
                source_repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/Mixtral-8x7B-Instruct-v0.1-8bit",
                gguf=GGUFSource(
                    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                    "mixtral-8x7b-instruct-v0.1.Q8_0.gguf",
                ),
                source_repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
            total_params="141B",
            active_params="39B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit",
                gguf=GGUFSource(
                    "bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF",
                    "Mixtral-8x22B-Instruct-v0.1-Q4_K_M.gguf",
                ),
                source_repo="mistralai/Mixtral-8x22B-Instruct-v0.1",
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
            total_params="132B",
            active_params="36B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/DBRX-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/dbrx-instruct-GGUF",
                    "dbrx-instruct-Q4_K_M.gguf",
                ),
                source_repo="databricks/dbrx-instruct",
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
            total_params="671B",
            active_params="37B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/DeepSeek-V3-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-V3-GGUF",
                    "DeepSeek-V3-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-V3",
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
            total_params="15.7B",
            active_params="2.4B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                gguf=GGUFSource(
                    "bartowski/DeepSeek-V2-Lite-Chat-GGUF",
                    "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-V2-Lite-Chat",
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
            total_params="14.3B",
            active_params="2.7B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                gguf=GGUFSource(
                    "Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF",
                    "qwen1_5-moe-a2.7b-chat-q4_k_m.gguf",
                ),
                source_repo="Qwen/Qwen1.5-MoE-A2.7B-Chat",
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
    # -------------------------------------------------------------------
    # Zhipu AI GLM
    # -------------------------------------------------------------------
    "glm-flash": ModelEntry(
        publisher="Zhipu AI",
        params="30B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=160,
            active_experts=8,
            expert_size="0.2B",
            total_params="30B",
            active_params="3B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/GLM-4.7-Flash-4bit",
                gguf=GGUFSource(
                    "bartowski/zai-org_GLM-4.7-Flash-GGUF",
                    "zai-org_GLM-4.7-Flash-Q4_K_M.gguf",
                ),
                source_repo="zai-org/GLM-4.7-Flash",
            ),
            "8bit": VariantSpec(
                mlx="mlx-community/GLM-4.7-Flash-8bit",
                gguf=GGUFSource(
                    "bartowski/zai-org_GLM-4.7-Flash-GGUF",
                    "zai-org_GLM-4.7-Flash-Q8_0.gguf",
                ),
                source_repo="zai-org/GLM-4.7-Flash",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # Mistral AI — Devstral (agentic coding)
    # -------------------------------------------------------------------
    "devstral-123b": ModelEntry(
        publisher="Mistral AI",
        params="123B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/Devstral-2-123B-Instruct-2512-4bit",
                gguf=GGUFSource(
                    "unsloth/Devstral-2-123B-Instruct-2512-GGUF",
                    "Devstral-2-123B-Instruct-2512-Q4_K_M-00001-of-00002.gguf",
                ),
                source_repo="mistralai/Devstral-2-123B-Instruct-2512",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # DeepSeek V3.2
    # -------------------------------------------------------------------
    "deepseek-v3.2": ModelEntry(
        publisher="DeepSeek",
        params="685B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=256,
            active_experts=8,
            expert_size="2B",
            total_params="685B",
            active_params="37B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/DeepSeek-V3.2-4bit",
                gguf=GGUFSource(
                    "unsloth/DeepSeek-V3.2-GGUF",
                    "DeepSeek-V3.2-Q4_K_M-00001-of-00009.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-V3.2",
            ),
        },
    ),
    # -------------------------------------------------------------------
    # MiniMax M2.1
    # -------------------------------------------------------------------
    "minimax-m2.1": ModelEntry(
        publisher="MiniMax",
        params="229B",
        default_quant="4bit",
        architecture="moe",
        moe=MoEMetadata(
            num_experts=8,
            active_experts=2,
            expert_size="10B",
            total_params="229B",
            active_params="10B",
        ),
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/MiniMax-M2.1-4bit",
                gguf=GGUFSource(
                    "unsloth/MiniMax-M2.1-GGUF",
                    "MiniMax-M2.1-Q4_K_M-00001-of-00003.gguf",
                ),
                source_repo="MiniMaxAI/MiniMax-M2.1",
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
    # Coder aliases
    "qwen-2.5-coder-1.5b": "qwen-coder-1.5b",
    "qwen-2.5-coder-3b": "qwen-coder-3b",
    "qwen-2.5-coder-7b": "qwen-coder-7b",
    "qwen-coder": "qwen-coder-7b",
    # MoE aliases
    "mixtral": "mixtral-8x7b",
    "mixtral-instruct": "mixtral-8x7b",
    "mixtral-8x22b-instruct": "mixtral-8x22b",
    "dbrx-instruct": "dbrx",
    "deepseek-v2-lite-chat": "deepseek-v2-lite",
    "qwen-1.5-moe": "qwen-moe-14b",
    "qwen-moe": "qwen-moe-14b",
    # Large model aliases
    "glm-4.7-flash": "glm-flash",
    "glm-4.7": "glm-flash",
    "devstral": "devstral-123b",
    "devstral-2": "devstral-123b",
    "minimax": "minimax-m2.1",
    "minimax-m2": "minimax-m2.1",
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
