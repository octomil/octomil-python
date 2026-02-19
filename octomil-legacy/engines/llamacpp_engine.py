"""llama.cpp engine plugin — cross-platform GGUF inference.

Works on any platform with CPU, and supports Metal (macOS) and CUDA (NVIDIA)
GPU offloading. Uses GGUF quantized models from HuggingFace.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with llama.cpp (GGUF format)
_GGUF_CATALOG = {
    "gemma-1b",
    "gemma-4b",
    "llama-1b",
    "llama-3b",
    "llama-8b",
    "phi-mini",
    "qwen-1.5b",
    "qwen-3b",
    "mistral-7b",
    "smollm-360m",
}


class LlamaCppEngine(EnginePlugin):
    """Cross-platform engine using llama-cpp-python."""

    @property
    def name(self) -> str:
        return "llama.cpp"

    @property
    def display_name(self) -> str:
        sys = platform.system()
        machine = platform.machine()
        accel = "Metal" if sys == "Darwin" else "CUDA" if self._has_cuda() else "CPU"
        return f"llama.cpp ({accel}, {machine})"

    @property
    def priority(self) -> int:
        return 20  # Second priority after mlx-lm

    @staticmethod
    def _has_cuda() -> bool:
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]

            # llama.cpp with CUBLAS support reports n_gpu_layers capability
            return True  # Heuristic — actual CUDA check happens at load time
        except ImportError:
            return False

    def detect(self) -> bool:
        try:
            import llama_cpp  # type: ignore[import-untyped]  # noqa: F401

            return True
        except ImportError:
            return False

    def detect_info(self) -> str:
        sys = platform.system()
        if sys == "Darwin":
            return "CPU + Metal"
        return "CPU" + (" + CUDA" if self._has_cuda() else "")

    def supports_model(self, model_name: str) -> bool:
        # Supports catalog names, .gguf files, and HuggingFace repo IDs
        return (
            model_name in _GGUF_CATALOG
            or model_name.endswith(".gguf")
            or "/" in model_name
        )

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        try:
            from ..serve import LlamaCppBackend

            backend = LlamaCppBackend(cache_enabled=False)
            backend.load_model(model_name)

            from ..serve import GenerationRequest

            req = GenerationRequest(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=n_tokens,
            )

            start = time.monotonic()
            _text, metrics = backend.generate(req)
            elapsed = time.monotonic() - start

            tps = metrics.tokens_per_second
            if tps == 0 and metrics.total_tokens > 0 and elapsed > 0:
                tps = metrics.total_tokens / elapsed

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=metrics.ttfc_ms,
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        from ..serve import LlamaCppBackend

        backend = LlamaCppBackend(
            cache_size_mb=kwargs.get("cache_size_mb", 2048),
            cache_enabled=kwargs.get("cache_enabled", True),
        )
        backend.load_model(model_name)
        return backend
