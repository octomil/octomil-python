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
                        "id": "embedded-gemma2-2b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_gemma2_2b_gguf_q4km",
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
                                    }
                                ],
                            },
                            {
                                "id": "pkg_gemma2_2b_mlx_q4",
                                "platform": "macos",
                                "artifact_format": "mlx",
                                "runtime_executor": "mlx",
                                "quantization": "4bit",
                                "support_tier": "blessed",
                                "is_default": False,
                                "resources": [
                                    {
                                        "kind": "weights",
                                        "uri": "hf://mlx-community/gemma-2-2b-it-4bit",
                                        "path": ".",
                                        "required": True,
                                    }
                                ],
                            },
                        ],
                    }
                },
            }
        },
    },
    "qwen2.5": {
        "id": "qwen2.5",
        "vendor": "Qwen",
        "description": "Qwen 2.5 series of instruction-tuned language models",
        "modalities": ["text"],
        "license": "apache-2.0",
        "homepage_url": "https://qwenlm.github.io/",
        "variants": {
            "qwen2.5-3b": {
                "id": "qwen2.5-3b",
                "parameter_count": "3B",
                "context_length": 32768,
                "modalities": ["text"],
                "quantizations": ["Q4_K_M"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-qwen25-3b-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_qwen25_3b_gguf_q4km",
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
                                    }
                                ],
                            }
                        ],
                    }
                },
            }
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
            "whisper-large-v3": {
                "id": "whisper-large-v3",
                "parameter_count": "1.5B",
                "modalities": ["audio"],
                "quantizations": ["fp16"],
                "versions": {
                    "1.0.0": {
                        "id": "embedded-whisper-v3-v1",
                        "version": "1.0.0",
                        "lifecycle": "active",
                        "released_at": "2026-03-12T00:00:00Z",
                        "min_sdk_version": None,
                        "packages": [
                            {
                                "id": "pkg_whisper_large_v3_gguf",
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
                                    }
                                ],
                            }
                        ],
                    }
                },
            }
        },
    },
}
