"""Embedded minimal catalog — offline fallback when server and disk cache are unavailable.

Contains a small set of blessed models so the SDK can bootstrap without
network access. This data is intentionally minimal: just enough to resolve
the most common model names to downloadable artifacts.

This file is auto-generated from the server manifest. Do not edit by hand.
"""

from __future__ import annotations

EMBEDDED_MANIFEST: dict = {
    "version": "embedded-v1",
    "generated_at": "2026-03-12T00:00:00Z",
    "models": [
        {
            "id": "gemma-2-2b",
            "family": "gemma-2",
            "name": "Gemma 2 2B",
            "parameter_count": "2B",
            "modalities": ["text-generation"],
            "default_quantization": "q4_k_m",
            "packages": [
                {
                    "id": "pkg_gemma2_2b_gguf_q4km",
                    "artifact_format": "gguf",
                    "runtime_executor": "llamacpp",
                    "platform": "macos",
                    "quantization": "q4_k_m",
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
                    "artifact_format": "mlx",
                    "runtime_executor": "mlx",
                    "platform": "macos",
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
        },
        {
            "id": "qwen2.5-3b",
            "family": "qwen2.5",
            "name": "Qwen2.5 3B",
            "parameter_count": "3B",
            "modalities": ["text-generation"],
            "default_quantization": "q4_k_m",
            "packages": [
                {
                    "id": "pkg_qwen25_3b_gguf_q4km",
                    "artifact_format": "gguf",
                    "runtime_executor": "llamacpp",
                    "platform": "macos",
                    "quantization": "q4_k_m",
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
        },
        {
            "id": "whisper-large-v3",
            "family": "whisper",
            "name": "Whisper Large V3",
            "parameter_count": "1.5B",
            "modalities": ["audio-transcription"],
            "default_quantization": "fp16",
            "packages": [
                {
                    "id": "pkg_whisper_large_v3_gguf",
                    "artifact_format": "gguf",
                    "runtime_executor": "whisper",
                    "platform": "macos",
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
        },
    ],
}
