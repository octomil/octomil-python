"""Shared test fixtures and helpers for the octomil test suite.

Provides mock server response data so tests run without a live Octomil API.
The fixture data mirrors a minimal subset of the old hardcoded catalogs —
just enough to satisfy the models referenced in the test suite.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# OTLP telemetry helpers (used across multiple test files)
# ---------------------------------------------------------------------------


def parse_otlp_kv(kv_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert an OTLP KeyValue array to a flat dict with native types."""
    result: dict[str, Any] = {}
    for kv in kv_list:
        key = kv["key"]
        val = kv["value"]
        if "stringValue" in val:
            result[key] = val["stringValue"]
        elif "intValue" in val:
            result[key] = int(val["intValue"])
        elif "doubleValue" in val:
            result[key] = val["doubleValue"]
        elif "boolValue" in val:
            result[key] = val["boolValue"]
        else:
            result[key] = str(val)
    return result


def extract_otlp_records(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all LogRecords from an OTLP ExportLogsServiceRequest."""
    records: list[dict[str, Any]] = []
    for rl in envelope.get("resourceLogs", []):
        for sl in rl.get("scopeLogs", []):
            records.extend(sl.get("logRecords", []))
    return records


# ---------------------------------------------------------------------------
# Mock catalog data (returned as if from GET /api/v1/models/catalog)
# ---------------------------------------------------------------------------

_MOCK_CATALOG: dict = {
    "gemma-1b": {
        "publisher": "Google",
        "params": "1B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "gemma3:1b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
            "fp16": {
                "source_repo": "REDACTED",
            },
        },
    },
    "gemma-4b": {
        "publisher": "Google",
        "params": "4B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "gemma3:4b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED-8bit",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
            "fp16": {
                "source_repo": "REDACTED",
            },
        },
    },
    "gemma-12b": {
        "publisher": "Google",
        "params": "12B",
        "default_quant": "4bit",
        "engines": ["mlx-lm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "ollama": "gemma3:12b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "source_repo": "REDACTED",
            },
        },
    },
    "gemma-27b": {
        "publisher": "Google",
        "params": "27B",
        "default_quant": "4bit",
        "engines": ["mlx-lm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "ollama": "gemma3:27b",
                "source_repo": "REDACTED",
            },
        },
    },
    "llama-1b": {
        "publisher": "Meta",
        "params": "1B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "llama3.2:1b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
            "fp16": {
                "source_repo": "REDACTED",
            },
        },
    },
    "llama-3b": {
        "publisher": "Meta",
        "params": "3B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "llama3.2:3b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "llama-8b": {
        "publisher": "Meta",
        "params": "8B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "llama3.1:8b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "phi-4": {
        "publisher": "Microsoft",
        "params": "14B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "bartowski/phi-4-GGUF",
                    "filename": "phi-4-Q4_K_M.gguf",
                },
                "ollama": "phi4",
                "source_repo": "REDACTED",
            },
        },
    },
    "phi-mini": {
        "publisher": "Microsoft",
        "params": "3.8B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime", "mlc-llm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "mlc": "mlc-ai/REDACTED-MLC",
                "ollama": "phi3.5",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "qwen-1.5b": {
        "publisher": "Qwen",
        "params": "1.5B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "ollama": "qwen2.5:1.5b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "qwen-3b": {
        "publisher": "Qwen",
        "params": "3B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "ollama": "qwen2.5:3b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "qwen-7b": {
        "publisher": "Qwen",
        "params": "7B",
        "default_quant": "4bit",
        "engines": ["mlx-lm"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "ollama": "qwen2.5:7b",
                "source_repo": "REDACTED",
            },
        },
    },
    "mistral-7b": {
        "publisher": "Mistral AI",
        "params": "7B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "ollama": "mistral:7b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "smollm-360m": {
        "publisher": "HuggingFace",
        "params": "360M",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
            "8bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
                "source_repo": "REDACTED",
            },
        },
    },
    "whisper-tiny": {
        "publisher": "OpenAI",
        "params": "39M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "REDACTED"}},
    },
    "whisper-base": {
        "publisher": "OpenAI",
        "params": "74M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "REDACTED"}},
    },
    "whisper-small": {
        "publisher": "OpenAI",
        "params": "244M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "REDACTED"}},
    },
    "whisper-medium": {
        "publisher": "OpenAI",
        "params": "769M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "REDACTED"}},
    },
    "whisper-large-v3": {
        "publisher": "OpenAI",
        "params": "1.55B",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "REDACTED"}},
    },
    # ---- MoE models ----
    "mixtral-8x7b": {
        "publisher": "Mistral AI",
        "params": "46.7B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 8,
            "active_experts": 2,
            "expert_size": "7B",
            "total_params": "46.7B",
            "active_params": "12.9B",
        },
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
            "8bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
    "mixtral-8x22b": {
        "publisher": "Mistral AI",
        "params": "141B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 8,
            "active_experts": 2,
            "expert_size": "22B",
            "total_params": "141B",
            "active_params": "39B",
        },
        "variants": {
            "4bit": {
                "mlx": "mlx-community/REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
    "dbrx": {
        "publisher": "Databricks",
        "params": "132B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 16,
            "active_experts": 4,
            "expert_size": "8B",
            "total_params": "132B",
            "active_params": "36B",
        },
        "variants": {
            "4bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
    "deepseek-v3": {
        "publisher": "DeepSeek",
        "params": "671B",
        "default_quant": "4bit",
        "engines": ["llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 256,
            "active_experts": 8,
            "expert_size": "2.4B",
            "total_params": "671B",
            "active_params": "37B",
        },
        "variants": {
            "4bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
    "deepseek-v2-lite": {
        "publisher": "DeepSeek",
        "params": "16B",
        "default_quant": "4bit",
        "engines": ["llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 64,
            "active_experts": 6,
            "expert_size": "0.25B",
            "total_params": "16B",
            "active_params": "2.4B",
        },
        "variants": {
            "4bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
    "qwen-moe-14b": {
        "publisher": "Qwen",
        "params": "14B",
        "default_quant": "4bit",
        "engines": ["llama.cpp"],
        "architecture": "moe",
        "moe": {
            "num_experts": 60,
            "active_experts": 4,
            "expert_size": "0.25B",
            "total_params": "14B",
            "active_params": "2.7B",
        },
        "variants": {
            "4bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED.gguf",
                },
            },
        },
    },
}

# Mock catalog aliases
_MOCK_CATALOG_ALIASES: dict[str, str] = {
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
    "mixtral": "mixtral-8x7b",
    "mixtral-instruct": "mixtral-8x7b",
    "dbrx-instruct": "dbrx",
    "qwen-moe": "qwen-moe-14b",
}

# Mock engine priority (same as old hardcoded)
_MOCK_ENGINE_PRIORITY: list[str] = [
    "mlx-lm",
    "mnn",
    "mlc-llm",
    "llama.cpp",
    "executorch",
    "onnxruntime",
    "whisper.cpp",
    "ollama",
    "echo",
]

# Mock model families for model_registry.py (key subset)
_MOCK_MODEL_FAMILIES: dict = {
    "gemma-1b": {
        "default_tag": "q4_k_m",
        "publisher": "Google",
        "params": "1B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "fp16": {
                "quantization_family": "16bit",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                ],
            },
        },
    },
    "gemma-4b": {
        "default_tag": "q4_k_m",
        "publisher": "Google",
        "params": "4B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:4b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "fp16": {
                "quantization_family": "16bit",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                ],
            },
        },
    },
    "llama-1b": {
        "default_tag": "q4_k_m",
        "publisher": "Meta",
        "params": "1B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
    "llama-3b": {
        "default_tag": "q4_k_m",
        "publisher": "Meta",
        "params": "3B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:3b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
    "llama-8b": {
        "default_tag": "q4_k_m",
        "publisher": "Meta",
        "params": "8B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.1:8b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
    "phi-4": {
        "default_tag": "q4_k_m",
        "publisher": "Microsoft",
        "params": "14B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "phi4", "trust": "curated"},
                ],
            },
        },
    },
    "phi-mini": {
        "default_tag": "q4_k_m",
        "publisher": "Microsoft",
        "params": "3.8B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "phi3.5", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
    "qwen-1.5b": {
        "default_tag": "q4_k_m",
        "publisher": "Qwen",
        "params": "1.5B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "official",
                    },
                    {"type": "ollama", "ref": "qwen2.5:1.5b", "trust": "curated"},
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "official",
                    },
                ],
            },
        },
    },
    "qwen-3b": {
        "default_tag": "q4_k_m",
        "publisher": "Qwen",
        "params": "3B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "official",
                    },
                    {"type": "ollama", "ref": "qwen2.5:3b", "trust": "curated"},
                ],
            },
        },
    },
    "qwen-7b": {
        "default_tag": "q4_k_m",
        "publisher": "Qwen",
        "params": "7B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "qwen2.5:7b", "trust": "curated"},
                ],
            },
        },
    },
    "mistral-7b": {
        "default_tag": "q4_k_m",
        "publisher": "Mistral AI",
        "params": "7B",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "mistral:7b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
    "smollm-360m": {
        "default_tag": "q4_k_m",
        "publisher": "HuggingFace",
        "params": "360M",
        "variants": {
            "q4_k_m": {
                "quantization_family": "4bit",
                "mlx": "mlx-community/REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED.gguf",
                        "trust": "community",
                    },
                ],
            },
        },
    },
}

# Mock source aliases for sources/resolver.py
_MOCK_SOURCE_ALIASES: dict[str, dict[str, str]] = {
    "phi-4-mini": {"hf": "microsoft/Phi-4-mini-instruct-onnx", "ollama": "phi4-mini"},
    "phi-mini": {
        "hf": "REDACTED",
        "hf_onnx": "REDACTED-onnx",
        "ollama": "phi3.5",
    },
    "gemma-1b": {"hf": "REDACTED", "hf_onnx": "onnx-community/gemma-3-1b-it-ONNX", "ollama": "gemma3:1b"},
    "gemma-4b": {"hf": "REDACTED", "ollama": "gemma3:4b"},
    "llama-1b": {
        "hf": "REDACTED",
        "hf_onnx": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
        "ollama": "llama3.2:1b",
    },
    "llama-3b": {"hf": "REDACTED", "ollama": "llama3.2:3b"},
}


# ---------------------------------------------------------------------------
# Test-only hydration helpers (moved from catalog.py during v1 removal)
# ---------------------------------------------------------------------------


def _gguf_from_dict(d: Any) -> Any:
    """Hydrate a GGUFSource from a server dict or None."""
    from octomil.models.catalog import GGUFSource

    if d is None:
        return None
    if isinstance(d, dict):
        return GGUFSource(repo=d.get("repo", ""), filename=d.get("filename", ""))
    return None


def _variant_from_dict(d: dict) -> Any:
    """Hydrate a VariantSpec from a test fixture dict."""
    from octomil.models.catalog import VariantSpec

    return VariantSpec(
        mlx=d.get("mlx"),
        gguf=_gguf_from_dict(d.get("gguf")),
        ort=d.get("ort"),
        mlc=d.get("mlc"),
        ollama=d.get("ollama"),
        source_repo=d.get("source_repo"),
    )


def _moe_from_dict(d: Any) -> Any:
    """Hydrate MoEMetadata from a test fixture dict or None."""
    from octomil.models.catalog import MoEMetadata

    if d is None or not isinstance(d, dict):
        return None
    return MoEMetadata(
        num_experts=d.get("num_experts", 0),
        active_experts=d.get("active_experts", 0),
        expert_size=d.get("expert_size", ""),
        total_params=d.get("total_params", ""),
        active_params=d.get("active_params", ""),
    )


def _entry_from_dict(d: dict) -> Any:
    """Hydrate a ModelEntry from a test fixture dict."""
    from octomil.models.catalog import ModelEntry

    variants_raw = d.get("variants", {})
    variants = {k: _variant_from_dict(v) for k, v in variants_raw.items()}
    engines_raw = d.get("engines", [])
    engines = frozenset(engines_raw) if isinstance(engines_raw, list) else frozenset()
    return ModelEntry(
        publisher=d.get("publisher", ""),
        params=d.get("params", ""),
        default_quant=d.get("default_quant", "4bit"),
        variants=variants,
        engines=engines,
        architecture=d.get("architecture", "dense"),
        moe=_moe_from_dict(d.get("moe")),
        download_size=d.get("download_size"),
    )


def _hydrate_test_catalog(raw: dict) -> dict:
    """Convert v1-format test fixture data to typed ModelEntry dict."""
    from octomil.models.catalog import ModelEntry

    result = {}
    for name, entry_data in raw.items():
        if isinstance(entry_data, ModelEntry):
            result[name] = entry_data
        elif isinstance(entry_data, dict):
            result[name] = _entry_from_dict(entry_data)
    return result


def _source_from_dict(d: dict) -> Any:
    """Hydrate a model_registry.ModelSource from a test fixture dict."""
    from octomil.model_registry import ModelSource

    return ModelSource(
        type=d.get("type", "huggingface"),
        ref=d.get("ref", ""),
        file=d.get("file"),
        trust=d.get("trust", "community"),
    )


def _registry_variant_from_dict(d: dict) -> Any:
    """Hydrate a model_registry.ModelVariant from a test fixture dict."""
    from octomil.model_registry import ModelVariant

    sources_raw = d.get("sources", [])
    sources = [_source_from_dict(s) for s in sources_raw if isinstance(s, dict)]
    return ModelVariant(
        quantization_family=d.get("quantization_family", "4bit"),
        sources=sources,
        mlx=d.get("mlx"),
    )


def _hydrate_test_families(raw: dict) -> dict:
    """Convert test fixture data to typed model_registry.ModelFamily dict."""
    from octomil.model_registry import DEFAULT_TAG, ModelFamily

    result = {}
    for name, family_data in raw.items():
        if isinstance(family_data, dict):
            variants_raw = family_data.get("variants", {})
            variants = {k: _registry_variant_from_dict(v) for k, v in variants_raw.items() if isinstance(v, dict)}
            result[name] = ModelFamily(
                default_tag=family_data.get("default_tag", DEFAULT_TAG),
                publisher=family_data.get("publisher", ""),
                params=family_data.get("params", ""),
                variants=variants,
            )
    return result


# ---------------------------------------------------------------------------
# Auto-use fixture: patch all server-fetched singletons
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_model_routing_clients(monkeypatch):
    """Patch server-fetched model routing data for all tests.

    This ensures tests never hit the real Octomil API and always get
    deterministic, known-good test data.
    """
    _hydrated_mock_catalog = _hydrate_test_catalog(_MOCK_CATALOG)
    _hydrated_mock_families = _hydrate_test_families(_MOCK_MODEL_FAMILIES)

    class _MockCatalogClientV2:
        """Mock v2 client — prevents real HTTP requests."""

        def get_manifest(self, platform=None):
            return {"version": "mock-v1", "generated_at": "2026-01-01T00:00:00Z", "models": []}

        def get_models(self, platform=None):
            return []

        def invalidate_cache(self):
            pass

    # Patch DeviceConfigClient — inject the original hardcoded values
    # so existing tests continue to pass without hitting the server.
    from octomil.device_config import DeviceConfig, EarlyExitPresetConfig, RoutingOffsets, SmartRouterDefaults

    _mock_device_config = DeviceConfig(
        quant_speed_factors={
            "Q2_K": 1.4,
            "Q3_K_S": 1.25,
            "Q3_K_M": 1.15,
            "Q4_0": 1.1,
            "Q4_K_S": 1.05,
            "Q4_K_M": 1.0,
            "Q5_0": 0.95,
            "Q5_K_S": 0.9,
            "Q5_K_M": 0.85,
            "Q6_K": 0.8,
            "Q8_0": 0.7,
            "F16": 0.5,
            "F32": 0.25,
        },
        quant_preference_order=[
            "Q8_0",
            "Q6_K",
            "Q5_K_M",
            "Q5_K_S",
            "Q5_0",
            "Q4_K_M",
            "Q4_K_S",
            "Q4_0",
            "Q3_K_M",
            "Q3_K_S",
            "Q2_K",
        ],
        early_exit_presets={
            "quality": EarlyExitPresetConfig(threshold=0.1, min_layers_fraction=0.75),
            "balanced": EarlyExitPresetConfig(threshold=0.3, min_layers_fraction=0.5),
            "fast": EarlyExitPresetConfig(threshold=0.5, min_layers_fraction=0.25),
        },
        routing_offsets=RoutingOffsets(
            quality_score_offset=0.5,
            balanced_score_offset=0.25,
            latency_offset=0.5,
            throughput_offset=0.25,
        ),
        smart_router=SmartRouterDefaults(
            long_gen_threshold=512,
            concurrency_threshold=2,
            prefer_throughput_engine="mlx-lm",
            prefer_latency_engine="llama.cpp",
        ),
    )

    class _MockDeviceConfigClient:
        def get_config(self):
            return _mock_device_config

    import octomil.device_config as dc_mod

    monkeypatch.setattr(dc_mod, "_client", _MockDeviceConfigClient())

    # Reset and immediately repopulate the lazy PRESET dicts in early_exit
    # so tests that read them directly (e.g. PRESET_THRESHOLDS[...]) get
    # the mocked values without needing to call _ensure_presets_loaded().
    import octomil.early_exit as ee_mod

    ee_mod.PRESET_THRESHOLDS.clear()
    ee_mod.PRESET_MIN_LAYERS_FRACTION.clear()
    ee_mod._ensure_presets_loaded()

    # Reset the singletons in all modules that cache them
    import octomil.model_registry as reg_mod
    import octomil.models.catalog as cat_mod
    import octomil.models.resolver as res_mod
    import octomil.sources.resolver as src_mod

    # Inject mock v2 client into all modules to prevent real server requests.
    _mock_v2 = _MockCatalogClientV2()
    monkeypatch.setattr(cat_mod, "_client", _mock_v2)
    monkeypatch.setattr(res_mod, "_v2_client", _mock_v2)
    monkeypatch.setattr(reg_mod, "_v2_client", _mock_v2)
    monkeypatch.setattr(src_mod, "_v2_client", _mock_v2)

    # Reset the existing lazy dicts so they re-load from mocked loaders.
    # We must mutate the existing objects rather than replacing them, because
    # test modules import CATALOG/MODEL_ALIASES/MODEL_FAMILIES at module level
    # and hold direct references to the original dict objects.
    #
    # Inject pre-hydrated test data directly via _loader callables,
    # bypassing the v2 manifest conversion path.
    cat_mod.CATALOG.clear()
    cat_mod.CATALOG._loaded = False  # type: ignore[attr-defined]
    cat_mod.CATALOG._loader = lambda: _hydrated_mock_catalog  # type: ignore[attr-defined]
    cat_mod.MODEL_ALIASES.clear()
    cat_mod.MODEL_ALIASES._loaded = False  # type: ignore[attr-defined]
    cat_mod.MODEL_ALIASES._loader = lambda: _MOCK_CATALOG_ALIASES  # type: ignore[attr-defined]
    reg_mod.MODEL_FAMILIES.clear()
    reg_mod.MODEL_FAMILIES._loaded = False  # type: ignore[attr-defined]
    reg_mod.MODEL_FAMILIES._loader = lambda: _hydrated_mock_families  # type: ignore[attr-defined]

    # Force CATALOG to reload so we can recompute derived module-level globals.
    cat_mod.CATALOG._ensure_loaded()  # type: ignore[attr-defined]

    # Recompute _ORT_CATALOG (set computed at import time from CATALOG).
    import octomil.runtime.engines.ort.engine as ort_mod

    ort_mod._ORT_CATALOG.clear()
    ort_mod._ORT_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "onnxruntime" in entry.engines)

    # Recompute _GGUF_MODELS (dict computed at import time from CATALOG).
    import octomil.serve as serve_mod

    serve_mod._GGUF_MODELS.clear()
    for _name, _entry in cat_mod.CATALOG.items():
        _variant = _entry.variants.get(_entry.default_quant)
        if _variant is None:
            continue
        if _variant.gguf is not None:
            serve_mod._GGUF_MODELS[_name] = (_variant.gguf.repo, _variant.gguf.filename)

    # Recompute _MOE_MODELS and _MLX_CATALOG/_GGUF_CATALOG in engine plugins.
    import octomil.runtime.engines.experimental.cactus.engine as cactus_mod
    import octomil.runtime.engines.experimental.executorch.engine as et_mod
    import octomil.runtime.engines.llamacpp.engine as llama_mod
    import octomil.runtime.engines.mlx.engine as mlx_mod

    llama_mod._MOE_MODELS.clear()
    llama_mod._MOE_MODELS.update(
        name for name, entry in cat_mod.CATALOG.items() if entry.architecture == "moe" and "llama.cpp" in entry.engines
    )

    llama_mod._GGUF_CATALOG.clear()
    llama_mod._GGUF_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "llama.cpp" in entry.engines)

    mlx_mod._MOE_MODELS.clear()
    mlx_mod._MOE_MODELS.update(
        name for name, entry in cat_mod.CATALOG.items() if entry.architecture == "moe" and "mlx-lm" in entry.engines
    )

    mlx_mod._MLX_CATALOG.clear()
    mlx_mod._MLX_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "mlx-lm" in entry.engines)

    cactus_mod._CACTUS_CATALOG.clear()
    cactus_mod._CACTUS_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "llama.cpp" in entry.engines)

    et_mod._ET_CATALOG.clear()
    et_mod._ET_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "executorch" in entry.engines)

    # Recompute _MLC_CATALOG (set computed at import time from CATALOG).
    import octomil.runtime.engines.experimental.mlc.engine as mlc_mod

    mlc_mod._MLC_CATALOG.clear()
    mlc_mod._MLC_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "mlc-llm" in entry.engines)

    # Recompute _MNN_CATALOG (set computed at import time from CATALOG).
    import octomil.runtime.engines.experimental.mnn.engine as mnn_mod

    mnn_mod._MNN_CATALOG.clear()
    mnn_mod._MNN_CATALOG.update(name for name, entry in cat_mod.CATALOG.items() if "mnn" in entry.engines)

    # Patch _ENGINE_PRIORITY in-place (tests import the list reference directly).
    res_mod._ENGINE_PRIORITY[:] = _MOCK_ENGINE_PRIORITY
