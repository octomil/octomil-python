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
    # DeepSeek R1 — distilled dense models
    # -------------------------------------------------------------------
    "deepseek-r1-1.5b": ModelEntry(
        publisher="DeepSeek",
        params="1.5B",
        default_quant="4bit",
        engines=frozenset({"mlx-lm", "llama.cpp"}),
        variants={
            "4bit": VariantSpec(
                mlx="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
                    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
                mlx="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
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
                mlx="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
                    "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
                mlx="mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
                    "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
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
                mlx="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                    "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
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
                mlx="mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
                gguf=GGUFSource(
                    "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
                    "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
                ),
                source_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
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
                mlx="mlx-community/Qwen3-0.6B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-0.6B-GGUF",
                    "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-0.6B",
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
                mlx="mlx-community/Qwen3-1.7B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-1.7B-GGUF",
                    "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-1.7B",
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
                mlx="mlx-community/Qwen3-4B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-4B-GGUF",
                    "Qwen_Qwen3-4B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-4B",
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
                mlx="mlx-community/Qwen3-8B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-8B-GGUF",
                    "Qwen_Qwen3-8B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-8B",
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
                mlx="mlx-community/Qwen3-14B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-14B-GGUF",
                    "Qwen_Qwen3-14B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-14B",
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
                mlx="mlx-community/Qwen3-32B-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen_Qwen3-32B-GGUF",
                    "Qwen_Qwen3-32B-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen3-32B",
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
                mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                    "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
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
                mlx="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
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
                mlx="mlx-community/gemma-2-2b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/gemma-2-2b-it-GGUF",
                    "gemma-2-2b-it-Q4_K_M.gguf",
                ),
                source_repo="google/gemma-2-2b-it",
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
                mlx="mlx-community/gemma-2-9b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/gemma-2-9b-it-GGUF",
                    "gemma-2-9b-it-Q4_K_M.gguf",
                ),
                source_repo="google/gemma-2-9b-it",
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
                mlx="mlx-community/gemma-2-27b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/gemma-2-27b-it-GGUF",
                    "gemma-2-27b-it-Q4_K_M.gguf",
                ),
                source_repo="google/gemma-2-27b-it",
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
                mlx="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
                    "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="meta-llama/Meta-Llama-3.1-70B-Instruct",
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
                mlx="mlx-community/Llama-3.3-70B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Llama-3.3-70B-Instruct-GGUF",
                    "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="meta-llama/Llama-3.3-70B-Instruct",
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
                mlx="mlx-community/Mistral-Nemo-Instruct-2407-4bit",
                gguf=GGUFSource(
                    "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
                    "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
                ),
                source_repo="mistralai/Mistral-Nemo-Instruct-2407",
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
                mlx="mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
                gguf=GGUFSource(
                    "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
                    "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
                ),
                source_repo="mistralai/Mistral-Small-24B-Instruct-2501",
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
                mlx="mlx-community/CodeLlama-7b-Instruct-hf-4bit-MLX",
                gguf=GGUFSource(
                    "TheBloke/CodeLlama-7B-Instruct-GGUF",
                    "codellama-7b-instruct.Q4_K_M.gguf",
                ),
                source_repo="meta-llama/CodeLlama-7b-Instruct-hf",
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
                mlx="mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
                gguf=GGUFSource(
                    "TheBloke/CodeLlama-13B-Instruct-GGUF",
                    "codellama-13b-instruct.Q4_K_M.gguf",
                ),
                source_repo="meta-llama/CodeLlama-13b-Instruct-hf",
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
                mlx="mlx-community/CodeLlama-34b-Instruct-hf-4bit",
                gguf=GGUFSource(
                    "TheBloke/CodeLlama-34B-Instruct-GGUF",
                    "codellama-34b-instruct.Q4_K_M.gguf",
                ),
                source_repo="meta-llama/CodeLlama-34b-Instruct-hf",
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
                mlx="mlx-community/starcoder2-3b-4bit",
                gguf=GGUFSource(
                    "QuantFactory/starcoder2-3b-GGUF",
                    "starcoder2-3b.Q4_K_M.gguf",
                ),
                source_repo="bigcode/starcoder2-3b",
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
                mlx="mlx-community/starcoder2-7b-4bit",
                gguf=GGUFSource(
                    "QuantFactory/starcoder2-7b-GGUF",
                    "starcoder2-7b.Q4_K_M.gguf",
                ),
                source_repo="bigcode/starcoder2-7b",
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
                mlx="mlx-community/starcoder2-15b-4bit",
                gguf=GGUFSource(
                    "bartowski/starcoder2-15b-instruct-v0.1-GGUF",
                    "starcoder2-15b-instruct-v0.1-Q4_K_M.gguf",
                ),
                source_repo="bigcode/starcoder2-15b-instruct-v0.1",
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
                mlx="mlx-community/Yi-1.5-6B-Chat-4bit",
                gguf=GGUFSource(
                    "bartowski/Yi-1.5-6B-Chat-GGUF",
                    "Yi-1.5-6B-Chat-Q4_K_M.gguf",
                ),
                source_repo="01-ai/Yi-1.5-6B-Chat",
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
                mlx="mlx-community/Yi-1.5-9B-Chat-4bit",
                gguf=GGUFSource(
                    "bartowski/Yi-1.5-9B-Chat-GGUF",
                    "Yi-1.5-9B-Chat-Q4_K_M.gguf",
                ),
                source_repo="01-ai/Yi-1.5-9B-Chat",
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
                mlx="mlx-community/Yi-1.5-34B-Chat-4bit",
                gguf=GGUFSource(
                    "bartowski/Yi-1.5-34B-Chat-GGUF",
                    "Yi-1.5-34B-Chat-Q4_K_M.gguf",
                ),
                source_repo="01-ai/Yi-1.5-34B-Chat",
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
                mlx="mlx-community/Falcon3-1B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Falcon3-1B-Instruct-GGUF",
                    "Falcon3-1B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="tiiuae/Falcon3-1B-Instruct",
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
                mlx="mlx-community/Falcon3-7B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Falcon3-7B-Instruct-GGUF",
                    "Falcon3-7B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="tiiuae/Falcon3-7B-Instruct",
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
                mlx="mlx-community/Falcon3-10B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/Falcon3-10B-Instruct-GGUF",
                    "Falcon3-10B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="tiiuae/Falcon3-10B-Instruct",
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
                mlx="mlx-community/c4ai-command-r-v01-4bit",
                gguf=GGUFSource(
                    "bartowski/c4ai-command-r-v01-GGUF",
                    "c4ai-command-r-v01-Q4_K_M.gguf",
                ),
                source_repo="CohereForAI/c4ai-command-r-v01",
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
                mlx="mlx-community/c4ai-command-r-plus-4bit",
                gguf=GGUFSource(
                    "bartowski/c4ai-command-r-plus-GGUF",
                    "c4ai-command-r-plus-Q4_K_M.gguf",
                ),
                source_repo="CohereForAI/c4ai-command-r-plus",
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
                mlx="mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
                gguf=GGUFSource(
                    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                ),
                source_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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
                    "bartowski/SmolLM2-1.7B-Instruct-GGUF",
                    "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
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
                mlx="mlx-community/internlm2_5-7b-chat-4bit",
                gguf=GGUFSource(
                    "internlm/internlm2_5-7b-chat-gguf",
                    "internlm2_5-7b-chat-q4_k_m.gguf",
                ),
                source_repo="internlm/internlm2_5-7b-chat",
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
                mlx="mlx-community/codegemma-1.1-7b-it-4bit",
                gguf=GGUFSource(
                    "bartowski/codegemma-7b-it-GGUF",
                    "codegemma-7b-it-Q4_K_M.gguf",
                ),
                source_repo="google/codegemma-7b-it",
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
                mlx="mlx-community/stable-code-instruct-3b-4bit",
                gguf=GGUFSource(
                    "bartowski/stable-code-instruct-3b-GGUF",
                    "stable-code-instruct-3b-Q4_K_M.gguf",
                ),
                source_repo="stabilityai/stable-code-instruct-3b",
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
                mlx="mlx-community/mathstral-7B-v0.1-4bit",
                gguf=GGUFSource(
                    "bartowski/mathstral-7B-v0.1-GGUF",
                    "mathstral-7B-v0.1-Q4_K_M.gguf",
                ),
                source_repo="mistralai/mathstral-7B-v0.1",
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
                mlx="mlx-community/glm-4-9b-chat-1m-4bit",
                gguf=GGUFSource(
                    "bartowski/THUDM_GLM-4-9B-0414-GGUF",
                    "THUDM_GLM-4-9B-0414-Q4_K_M.gguf",
                ),
                source_repo="THUDM/glm-4-9b-chat",
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
                mlx="mlx-community/IBM-granite-3.2-2b-instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/ibm-granite_granite-3.2-2b-instruct-GGUF",
                    "ibm-granite_granite-3.2-2b-instruct-Q4_K_M.gguf",
                ),
                source_repo="ibm-granite/granite-3.2-2b-instruct",
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
                mlx="mlx-community/granite-3.2-8b-instruct-preview-4bit",
                gguf=GGUFSource(
                    "lmstudio-community/granite-3.2-8b-instruct-GGUF",
                    "granite-3.2-8b-instruct-Q4_K_M.gguf",
                ),
                source_repo="ibm-granite/granite-3.2-8b-instruct",
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
                mlx="mlx-community/OLMo-2-1124-7B-Instruct-4bit",
                gguf=GGUFSource(
                    "bartowski/OLMo-2-1124-7B-Instruct-GGUF",
                    "OLMo-2-1124-7B-Instruct-Q4_K_M.gguf",
                ),
                source_repo="allenai/OLMo-2-1124-7B-Instruct",
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
                mlx="mlx-community/zephyr-7b-beta-4bit",
                gguf=GGUFSource(
                    "TheBloke/zephyr-7B-beta-GGUF",
                    "zephyr-7b-beta.Q4_K_M.gguf",
                ),
                source_repo="HuggingFaceH4/zephyr-7b-beta",
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
                mlx="mlx-community/SOLAR-10.7B-Instruct-v1.0-4bit",
                gguf=GGUFSource(
                    "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF",
                    "solar-10.7b-instruct-v1.0.Q4_K_M.gguf",
                ),
                source_repo="upstage/SOLAR-10.7B-Instruct-v1.0",
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
