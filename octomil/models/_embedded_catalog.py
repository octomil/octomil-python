"""Embedded minimal catalog — offline fallback when server and disk cache are unavailable.

Contains a small set of blessed models so the SDK can bootstrap without
network access. This data is intentionally minimal: just enough to resolve
the most common model names to downloadable artifacts.

Format matches the server's ``GET /api/v2/catalog/manifest`` response:
nested ``{family_name: {variants: {variant_name: {versions: ...}}}}``.

This file is auto-generated from the server manifest. Do not edit by hand.
"""

from __future__ import annotations

EMBEDDED_MANIFEST: dict = {
    "gemma-2": {
        "id": "gemma-2",
        "vendor": "Google",
        "description": "Google Gemma 2 family of lightweight open models",
        "modalities": ["text"],
        "license": "gemma",
        "homepage_url": "https://ai.google.dev/gemma",
        "variants": {
            "gemma-2-2b": {
                "id": "gemma-2-2b",
                "parameter_count": "2B",
                "context_length": 8192,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M", "Q4"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma_2_2b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma_2_2b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-2b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma_2_2b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/gemma-2-2b-it-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma_2_2b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-2b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "gemma-2-9b": {
                "id": "gemma-2-9b",
                "parameter_count": "9B",
                "context_length": 8192,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma_2_9b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma_2_9b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-9b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma_2_9b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-9b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "gemma-2-27b": {
                "id": "gemma-2-27b",
                "parameter_count": "27B",
                "context_length": 8192,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma_2_27b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma_2_27b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-27b-it-GGUF/gemma-2-27b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-27b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma_2_27b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/gemma-2-27b-it-GGUF/gemma-2-27b-it-Q4_K_M.gguf",
                                        "path": "gemma-2-27b-it-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "gemma-3": {
        "id": "gemma-3",
        "vendor": "Google",
        "description": "Google Gemma 3 multimodal models with vision support",
        "modalities": ["text", "vision"],
        "license": "gemma",
        "homepage_url": "https://ai.google.dev/gemma",
        "variants": {
            "gemma3-1b": {
                "id": "gemma3-1b",
                "parameter_count": "1B",
                "context_length": 32768,
                "modalities": ["text", "vision"],
                "quantizations": ["Q4_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma3_1b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma3_1b_gguf_q4_0",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
                                        "path": "gemma-3-1b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_1b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/gemma-3-1b-it-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_1b_gguf_q4_0",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
                                        "path": "gemma-3-1b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "gemma3-4b": {
                "id": "gemma3-4b",
                "parameter_count": "4B",
                "context_length": 128000,
                "modalities": ["text", "vision"],
                "quantizations": ["Q4_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma3_4b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma3_4b_gguf_q4_0",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf",
                                        "path": "gemma-3-4b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_4b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/gemma-3-4b-it-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_4b_gguf_q4_0",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf",
                                        "path": "gemma-3-4b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "gemma3-12b": {
                "id": "gemma3-12b",
                "parameter_count": "12B",
                "context_length": 128000,
                "modalities": ["text", "vision"],
                "quantizations": ["Q4_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma3_12b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma3_12b_gguf_q4_0",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-12b-it-qat-q4_0-gguf/gemma-3-12b-it-q4_0.gguf",
                                        "path": "gemma-3-12b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_12b_gguf_q4_0",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-12b-it-qat-q4_0-gguf/gemma-3-12b-it-q4_0.gguf",
                                        "path": "gemma-3-12b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "gemma3-27b": {
                "id": "gemma3-27b",
                "parameter_count": "27B",
                "context_length": 128000,
                "modalities": ["text", "vision"],
                "quantizations": ["Q4_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-gemma3_27b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma3_27b_gguf_q4_0",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-27b-it-qat-q4_0-gguf/gemma-3-27b-it-q4_0.gguf",
                                        "path": "gemma-3-27b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_gemma3_27b_gguf_q4_0",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://google/gemma-3-27b-it-qat-q4_0-gguf/gemma-3-27b-it-q4_0.gguf",
                                        "path": "gemma-3-27b-it-q4_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "llama-3.2": {
        "id": "llama-3.2",
        "vendor": "Meta",
        "description": "Meta Llama 3.2 lightweight instruction-tuned models",
        "modalities": ["text"],
        "license": "llama3.2",
        "homepage_url": "https://llama.meta.com/",
        "variants": {
            "llama-3.2-1b": {
                "id": "llama-3.2-1b",
                "parameter_count": "1B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-llama_3_2_1b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_llama_3_2_1b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                                        "path": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_llama_3_2_1b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Llama-3.2-1B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_llama_3_2_1b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                                        "path": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "llama-3.2-3b": {
                "id": "llama-3.2-3b",
                "parameter_count": "3B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-llama_3_2_3b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_llama_3_2_3b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                                        "path": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_llama_3_2_3b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Llama-3.2-3B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_llama_3_2_3b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                                        "path": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "phi-4": {
        "id": "phi-4",
        "vendor": "Microsoft",
        "description": "Microsoft Phi 4 state-of-the-art 14B open model",
        "modalities": ["text"],
        "license": "mit",
        "homepage_url": "https://azure.microsoft.com/en-us/products/phi",
        "variants": {
            "phi4": {
                "id": "phi4",
                "parameter_count": "14B",
                "context_length": 16384,
                "modalities": ["text"],
                "quantizations": ["Q4_K"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-phi4-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_phi4_gguf_q4_k",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://microsoft/phi-4-gguf/phi-4-Q4_K.gguf",
                                        "path": "phi-4-Q4_K.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_phi4_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/phi-4-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_phi4_gguf_q4_k",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://microsoft/phi-4-gguf/phi-4-Q4_K.gguf",
                                        "path": "phi-4-Q4_K.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "qwen2.5": {
        "id": "qwen2.5",
        "vendor": "Alibaba",
        "description": "Qwen 2.5 instruction-tuned models with 128K context",
        "modalities": ["text"],
        "license": "apache-2.0",
        "homepage_url": "https://qwenlm.github.io/",
        "variants": {
            "qwen2.5-0.5b": {
                "id": "qwen2.5-0.5b",
                "parameter_count": "0.5B",
                "context_length": 32768,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_0_5b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_0_5b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_0_5b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-1.5b": {
                "id": "qwen2.5-1.5b",
                "parameter_count": "1.5B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_1_5b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_1_5b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_1_5b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_1_5b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-3b": {
                "id": "qwen2.5-3b",
                "parameter_count": "3B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_3b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_3b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-3b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_3b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Qwen2.5-3B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_3b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf",
                                        "path": "qwen2.5-3b-instruct-q4_k_m.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-7b": {
                "id": "qwen2.5-7b",
                "parameter_count": "7B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_7b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_7b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-7B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_7b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Qwen2.5-7B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_7b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-7B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-14b": {
                "id": "qwen2.5-14b",
                "parameter_count": "14B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_14b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_14b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-14B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_14b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "Q4",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/Qwen2.5-14B-Instruct-4bit",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_14b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-14B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-32b": {
                "id": "qwen2.5-32b",
                "parameter_count": "32B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_32b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_32b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-32B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_32b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-32B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "qwen2.5-72b": {
                "id": "qwen2.5-72b",
                "parameter_count": "72B",
                "context_length": 131072,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen2_5_72b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen2_5_72b_gguf_q4_k_m",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-72B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_qwen2_5_72b_gguf_q4_k_m",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://Qwen/Qwen2.5-72B-Instruct-GGUF",
                                        "path": ".",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "whisper": {
        "id": "whisper",
        "vendor": "OpenAI",
        "description": "OpenAI Whisper automatic speech recognition models",
        "modalities": ["audio"],
        "license": "mit",
        "homepage_url": "https://github.com/openai/whisper",
        "variants": {
            "whisper-tiny": {
                "id": "whisper-tiny",
                "parameter_count": "39M",
                "context_length": None,
                "modalities": ["audio"],
                "quantizations": ["fp16", "Q8_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-whisper_tiny-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-16T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_whisper_tiny_gguf_fp16",
                                "platform": "android",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-tiny.bin",
                                        "path": "ggml-tiny.bin",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_whisper_tiny_gguf_fp16_macos",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-tiny.bin",
                                        "path": "ggml-tiny.bin",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_whisper_tiny_gguf_fp16_linux",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-tiny.bin",
                                        "path": "ggml-tiny.bin",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "whisper-base": {
                "id": "whisper-base",
                "parameter_count": "74M",
                "context_length": None,
                "modalities": ["audio"],
                "quantizations": ["fp16", "Q8_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-whisper_base-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-16T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_whisper_base_gguf_fp16",
                                "platform": "android",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-base.bin",
                                        "path": "ggml-base.bin",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_whisper_base_gguf_fp16_macos",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-base.bin",
                                        "path": "ggml-base.bin",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_whisper_base_gguf_fp16_linux",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-base.bin",
                                        "path": "ggml-base.bin",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "whisper-large-v3": {
                "id": "whisper-large-v3",
                "parameter_count": "1.5B",
                "context_length": None,
                "modalities": ["audio"],
                "quantizations": ["fp16"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-whisper_large_v3-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_whisper_large_v3_gguf_fp16",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-large-v3.bin",
                                        "path": "ggml-large-v3.bin",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_whisper_large_v3_gguf_fp16_linux",
                                "platform": "linux",
                                "artifact_format": "gguf",
                                "runtime_executor": "whisper",
                                "quantization": "fp16",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggerganov/whisper.cpp/ggml-large-v3.bin",
                                        "path": "ggml-large-v3.bin",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "smolvlm2": {
        "id": "smolvlm2",
        "vendor": "HuggingFace",
        "description": "HuggingFace SmolVLM2 multimodal vision-language models",
        "modalities": ["multimodal"],
        "license": "apache-2.0",
        "homepage_url": "https://huggingface.co/collections/HuggingFaceTB/smolvlm2",
        "variants": {
            "smolvlm2-500m": {
                "id": "smolvlm2-500m",
                "parameter_count": "500M",
                "context_length": 2048,
                "modalities": ["multimodal"],
                "quantizations": ["Q8_0"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-smolvlm2_500m-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-16T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_smolvlm2_500m_gguf_q8_0",
                                "platform": "android",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q8_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                                        "path": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                                        "required": True,
                                    },
                                    {
                                        "kind": "projector",
                                        "uri": "hf://ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                                        "path": "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
            "smolvlm2-2.2b": {
                "id": "smolvlm2-2.2b",
                "parameter_count": "2.2B",
                "context_length": 2048,
                "modalities": ["multimodal"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-smolvlm2_2_2b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-16T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_smolvlm2_2_2b_gguf_q4_k_m",
                                "platform": "android",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q4_K_M",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://ggml-org/SmolVLM2-2.2B-Instruct-GGUF/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                                        "path": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                                        "required": True,
                                    },
                                    {
                                        "kind": "projector",
                                        "uri": "hf://ggml-org/SmolVLM2-2.2B-Instruct-GGUF/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                                        "path": "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
    "smollm2": {
        "id": "smollm2",
        "vendor": "HuggingFace",
        "description": "HuggingFace SmolLM2 compact language models for on-device text generation",
        "modalities": ["text"],
        "license": "apache-2.0",
        "homepage_url": "https://huggingface.co/collections/HuggingFaceTB/smollm2",
        "variants": {
            "smollm2-135m": {
                "id": "smollm2-135m",
                "parameter_count": "135M",
                "context_length": 2048,
                "modalities": ["text"],
                "quantizations": ["Q8_0", "Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-smollm2_135m-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-16T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_smollm2_135m_gguf_q8_0",
                                "platform": "android",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q8_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
                                        "path": "SmolLM2-135M-Instruct-Q8_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                            {
                                "id": "pkg_smollm2_135m_gguf_q8_0_macos",
                                "platform": "macos",
                                "artifact_format": "gguf",
                                "runtime_executor": "llamacpp",
                                "quantization": "Q8_0",
                                "support_tier": "blessed",
                                "is_default": True,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
                                        "path": "SmolLM2-135M-Instruct-Q8_0.gguf",
                                        "required": True,
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    },
}
