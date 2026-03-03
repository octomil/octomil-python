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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "mlc": "REDACTED",
                "ollama": "gemma3:1b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "mlc": "REDACTED",
                "ollama": "gemma3:4b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "ollama": "gemma3:12b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
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
                "mlx": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "llama3.2:1b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"],
        "variants": {
            "4bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "llama3.2:3b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "source_repo": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "llama3.1:8b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
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
        "engines": ["mlx-lm", "llama.cpp", "mnn", "executorch", "onnxruntime"],
        "variants": {
            "4bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "phi3.5",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "qwen2.5:1.5b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "qwen2.5:3b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "ollama": "mistral:7b",
                "source_repo": "REDACTED",
            },
            "8bit": {
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
                },
                "source_repo": "REDACTED",
            },
            "8bit": {
                "gguf": {
                    "repo": "REDACTED",
                    "filename": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "gemma3:4b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:1b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.2:3b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "llama3.1:8b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "phi3.5", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
                        "trust": "official",
                    },
                    {"type": "ollama", "ref": "qwen2.5:1.5b", "trust": "curated"},
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {"type": "ollama", "ref": "mistral:7b", "trust": "curated"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
                        "trust": "community",
                    },
                ],
            },
            "q8_0": {
                "quantization_family": "8bit",
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                "mlx": "REDACTED",
                "sources": [
                    {"type": "huggingface", "ref": "REDACTED", "trust": "official"},
                    {
                        "type": "huggingface",
                        "ref": "REDACTED",
                        "file": "REDACTED",
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
                        "file": "REDACTED",
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
# Auto-use fixture: patch all server-fetched singletons
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_model_routing_clients(monkeypatch):
    """Patch server-fetched model routing data for all tests.

    This ensures tests never hit the real Octomil API and always get
    deterministic, known-good test data.
    """

    # Consolidated mock matching SdkConfigClient API
    class _MockSdkConfigClient:
        def get_catalog(self):
            return _MOCK_CATALOG

        def get_aliases(self):
            return _MOCK_CATALOG_ALIASES

        def get_priority(self):
            return _MOCK_ENGINE_PRIORITY

        def get_families(self):
            return _MOCK_MODEL_FAMILIES

        def get_source_aliases(self):
            return _MOCK_SOURCE_ALIASES

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

    # Inject the consolidated mock client into catalog._client.
    # resolver, model_registry, and sources/resolver all delegate to it.
    _mock_sdk_client = _MockSdkConfigClient()
    monkeypatch.setattr(cat_mod, "_client", _mock_sdk_client)

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
