"""Shared test fixtures for model catalog, aliases, and registry tests.

Provides mock server response data so tests run without a live Octomil API.
The fixture data mirrors a minimal subset of the old hardcoded catalogs —
just enough to satisfy the models referenced in the test suite.
"""

from __future__ import annotations

import pytest

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
                "mlx": "mlx-community/gemma-3-1b-it-4bit",
                "gguf": {
                    "repo": "bartowski/google_gemma-3-1b-it-GGUF",
                    "filename": "google_gemma-3-1b-it-Q4_K_M.gguf",
                },
                "mlc": "mlc-ai/gemma-2b-it-q4f16_1-MLC",
                "ollama": "gemma3:1b",
                "source_repo": "google/gemma-3-1b-it",
            },
            "8bit": {
                "mlx": "mlx-community/gemma-3-1b-it-8bit",
                "gguf": {
                    "repo": "bartowski/google_gemma-3-1b-it-GGUF",
                    "filename": "google_gemma-3-1b-it-Q8_0.gguf",
                },
                "source_repo": "google/gemma-3-1b-it",
            },
            "fp16": {
                "source_repo": "google/gemma-3-1b-it",
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
                "mlx": "mlx-community/gemma-3-4b-it-4bit",
                "gguf": {
                    "repo": "bartowski/google_gemma-3-4b-it-GGUF",
                    "filename": "google_gemma-3-4b-it-Q4_K_M.gguf",
                },
                "mlc": "mlc-ai/gemma-2b-it-q4f16_1-MLC",
                "ollama": "gemma3:4b",
                "source_repo": "google/gemma-3-4b-it",
            },
            "8bit": {
                "mlx": "mlx-community/gemma-3-4b-it-8bit",
                "gguf": {
                    "repo": "bartowski/google_gemma-3-4b-it-GGUF",
                    "filename": "google_gemma-3-4b-it-Q8_0.gguf",
                },
                "source_repo": "google/gemma-3-4b-it",
            },
            "fp16": {
                "source_repo": "google/gemma-3-4b-it",
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
                "mlx": "mlx-community/gemma-3-12b-it-4bit",
                "ollama": "gemma3:12b",
                "source_repo": "google/gemma-3-12b-it",
            },
            "8bit": {
                "mlx": "mlx-community/gemma-3-12b-it-8bit",
                "source_repo": "google/gemma-3-12b-it",
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
                "mlx": "mlx-community/gemma-3-27b-it-4bit",
                "ollama": "gemma3:27b",
                "source_repo": "google/gemma-3-27b-it",
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
                "mlx": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "gguf": {
                    "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                },
                "ollama": "llama3.2:1b",
                "source_repo": "meta-llama/Llama-3.2-1B-Instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Llama-3.2-1B-Instruct-8bit",
                "gguf": {
                    "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "filename": "Llama-3.2-1B-Instruct-Q8_0.gguf",
                },
                "source_repo": "meta-llama/Llama-3.2-1B-Instruct",
            },
            "fp16": {
                "source_repo": "meta-llama/Llama-3.2-1B-Instruct",
            },
        },
    },
    "llama-3b": {
        "publisher": "Meta",
        "params": "3B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "gguf": {
                    "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                },
                "ollama": "llama3.2:3b",
                "source_repo": "meta-llama/Llama-3.2-3B-Instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Llama-3.2-3B-Instruct-8bit",
                "gguf": {
                    "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                    "filename": "Llama-3.2-3B-Instruct-Q8_0.gguf",
                },
                "source_repo": "meta-llama/Llama-3.2-3B-Instruct",
            },
        },
    },
    "llama-8b": {
        "publisher": "Meta",
        "params": "8B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                "gguf": {
                    "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                },
                "ollama": "llama3.1:8b",
                "source_repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
                "gguf": {
                    "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "filename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                },
                "source_repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
                "mlx": "mlx-community/phi-4-4bit",
                "gguf": {
                    "repo": "bartowski/phi-4-GGUF",
                    "filename": "phi-4-Q4_K_M.gguf",
                },
                "ollama": "phi4",
                "source_repo": "microsoft/phi-4",
            },
        },
    },
    "phi-mini": {
        "publisher": "Microsoft",
        "params": "3.8B",
        "default_quant": "4bit",
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"],
        "variants": {
            "4bit": {
                "mlx": "mlx-community/Phi-3.5-mini-instruct-4bit",
                "gguf": {
                    "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
                    "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                },
                "ollama": "phi3.5",
                "source_repo": "microsoft/Phi-3.5-mini-instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Phi-3.5-mini-instruct-8bit",
                "gguf": {
                    "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
                    "filename": "Phi-3.5-mini-instruct-Q8_0.gguf",
                },
                "source_repo": "microsoft/Phi-3.5-mini-instruct",
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
                "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "gguf": {
                    "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                },
                "ollama": "qwen2.5:1.5b",
                "source_repo": "Qwen/Qwen2.5-1.5B-Instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
                "gguf": {
                    "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
                },
                "source_repo": "Qwen/Qwen2.5-1.5B-Instruct",
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
                "mlx": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                "gguf": {
                    "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
                },
                "ollama": "qwen2.5:3b",
                "source_repo": "Qwen/Qwen2.5-3B-Instruct",
            },
            "8bit": {
                "mlx": "mlx-community/Qwen2.5-3B-Instruct-8bit",
                "gguf": {
                    "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "filename": "qwen2.5-3b-instruct-q8_0.gguf",
                },
                "source_repo": "Qwen/Qwen2.5-3B-Instruct",
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
                "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "ollama": "qwen2.5:7b",
                "source_repo": "Qwen/Qwen2.5-7B-Instruct",
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
                "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "gguf": {
                    "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                    "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                },
                "ollama": "mistral:7b",
                "source_repo": "mistralai/Mistral-7B-Instruct-v0.3",
            },
            "8bit": {
                "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                "gguf": {
                    "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                    "filename": "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                },
                "source_repo": "mistralai/Mistral-7B-Instruct-v0.3",
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
                "mlx": "mlx-community/SmolLM-360M-Instruct-4bit",
                "gguf": {
                    "repo": "bartowski/SmolLM2-360M-Instruct-GGUF",
                    "filename": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
                },
                "source_repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
            },
            "8bit": {
                "gguf": {
                    "repo": "bartowski/SmolLM2-360M-Instruct-GGUF",
                    "filename": "SmolLM2-360M-Instruct-Q8_0.gguf",
                },
                "source_repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
            },
        },
    },
    "whisper-tiny": {
        "publisher": "OpenAI",
        "params": "39M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "openai/whisper-tiny"}},
    },
    "whisper-base": {
        "publisher": "OpenAI",
        "params": "74M",
        "default_quant": "fp16",
        "engines": ["whisper.cpp"],
        "variants": {"fp16": {"source_repo": "openai/whisper-base"}},
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
                "mlx": "mlx-community/gemma-3-1b-it-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-1b-it", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/google_gemma-3-1b-it-GGUF",
                        "file": "google_gemma-3-1b-it-Q4_K_M.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/gemma-3-1b-it-8bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-1b-it", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/google_gemma-3-1b-it-GGUF",
                        "file": "google_gemma-3-1b-it-Q8_0.gguf",
                        "trust": "community",
                    },
                ],
            },
            "fp16": {
                "quantization_family": "16bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-1b-it", "trust": "official"},
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
                "mlx": "mlx-community/gemma-3-4b-it-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-4b-it", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:4b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/google_gemma-3-4b-it-GGUF",
                        "file": "google_gemma-3-4b-it-Q4_K_M.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/gemma-3-4b-it-8bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-4b-it", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/google_gemma-3-4b-it-GGUF",
                        "file": "google_gemma-3-4b-it-Q8_0.gguf",
                        "trust": "community",
                    },
                ],
            },
            "fp16": {
                "quantization_family": "16bit",
                "sources": [
                    {"type": "huggingface", "ref": "google/gemma-3-4b-it", "trust": "official"},
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
                "mlx": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "meta-llama/Llama-3.2-1B-Instruct", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
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
                "mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "meta-llama/Llama-3.2-3B-Instruct", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:3b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
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
                "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "meta-llama/Meta-Llama-3.1-8B-Instruct", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.1:8b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
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
                "mlx": "mlx-community/phi-4-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "microsoft/phi-4", "trust": "official"},
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
                "mlx": "mlx-community/Phi-3.5-mini-instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "microsoft/Phi-3.5-mini-instruct", "trust": "official"},
                    {"type": "ollama", "ref": "phi3.5", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Phi-3.5-mini-instruct-GGUF",
                        "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/Phi-3.5-mini-instruct-8bit",
                "sources": [
                    {"type": "huggingface", "ref": "microsoft/Phi-3.5-mini-instruct", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Phi-3.5-mini-instruct-GGUF",
                        "file": "Phi-3.5-mini-instruct-Q8_0.gguf",
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
                "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                        "trust": "official",
                    },
                    {"type": "ollama", "ref": "qwen2.5:1.5b", "trust": "curated"},
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        "file": "qwen2.5-1.5b-instruct-q8_0.gguf",
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
                "mlx": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
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
                "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "Qwen/Qwen2.5-7B-Instruct", "trust": "official"},
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
                "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "mistralai/Mistral-7B-Instruct-v0.3", "trust": "official"},
                    {"type": "ollama", "ref": "mistral:7b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                "sources": [
                    {"type": "huggingface", "ref": "mistralai/Mistral-7B-Instruct-v0.3", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                        "file": "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
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
                "mlx": "mlx-community/SmolLM-360M-Instruct-4bit",
                "sources": [
                    {"type": "huggingface", "ref": "HuggingFaceTB/SmolLM2-360M-Instruct", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/SmolLM2-360M-Instruct-GGUF",
                        "file": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "sources": [
                    {"type": "huggingface", "ref": "HuggingFaceTB/SmolLM2-360M-Instruct", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "bartowski/SmolLM2-360M-Instruct-GGUF",
                        "file": "SmolLM2-360M-Instruct-Q8_0.gguf",
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
        "hf": "microsoft/Phi-3.5-mini-instruct",
        "hf_onnx": "microsoft/Phi-3.5-mini-instruct-onnx",
        "ollama": "phi3.5",
    },
    "gemma-1b": {"hf": "google/gemma-3-1b-it", "hf_onnx": "onnx-community/gemma-3-1b-it-ONNX", "ollama": "gemma3:1b"},
    "gemma-4b": {"hf": "google/gemma-3-4b-it", "ollama": "gemma3:4b"},
    "llama-1b": {
        "hf": "meta-llama/Llama-3.2-1B-Instruct",
        "hf_onnx": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
        "ollama": "llama3.2:1b",
    },
    "llama-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "ollama": "llama3.2:3b"},
}


# ---------------------------------------------------------------------------
# Auto-use fixture: patch all server-fetched singletons
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_model_routing_clients(monkeypatch):
    """Patch server-fetched model routing data for all tests.

    This ensures tests never hit the real Octomil API and always get
    deterministic, known-good test data.
    """

    # Patch CatalogClient
    class _MockCatalogClient:
        def get_catalog(self):
            return _MOCK_CATALOG

        def get_aliases(self):
            return _MOCK_CATALOG_ALIASES

    # Patch EnginePriorityClient
    class _MockEnginePriorityClient:
        def get_priority(self):
            return _MOCK_ENGINE_PRIORITY

    # Patch ModelFamiliesClient
    class _MockModelFamiliesClient:
        def get_families(self):
            return _MOCK_MODEL_FAMILIES

    # Patch SourceAliasesClient
    class _MockSourceAliasesClient:
        def get_aliases(self):
            return _MOCK_SOURCE_ALIASES

    # Reset the singletons in all modules that cache them
    import octomil.model_registry as reg_mod
    import octomil.models.catalog as cat_mod
    import octomil.models.resolver as res_mod
    import octomil.sources.resolver as src_mod

    # Directly inject mock client instances into the module-level singletons.
    # This bypasses the CatalogClient() constructor which was already imported.
    monkeypatch.setattr(cat_mod, "_client", _MockCatalogClient())
    monkeypatch.setattr(res_mod, "_priority_client", _MockEnginePriorityClient())
    monkeypatch.setattr(reg_mod, "_families_client", _MockModelFamiliesClient())
    monkeypatch.setattr(src_mod, "_aliases_client", _MockSourceAliasesClient())

    # Reset the existing lazy dicts so they re-load from the mocked clients.
    # We must mutate the existing objects rather than replacing them, because
    # test modules import CATALOG/MODEL_ALIASES/MODEL_FAMILIES at module level
    # and hold direct references to the original dict objects.
    cat_mod.CATALOG.clear()
    cat_mod.CATALOG._loaded = False  # type: ignore[attr-defined]
    cat_mod.MODEL_ALIASES.clear()
    cat_mod.MODEL_ALIASES._loaded = False  # type: ignore[attr-defined]
    reg_mod.MODEL_FAMILIES.clear()
    reg_mod.MODEL_FAMILIES._loaded = False  # type: ignore[attr-defined]
