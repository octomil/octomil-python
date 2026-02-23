"""llama.cpp engine plugin — cross-platform GGUF inference.

Works on any platform with CPU, and supports Metal (macOS), CUDA (NVIDIA),
and ROCm/hipBLAS (AMD) GPU offloading. Uses GGUF quantized models from
HuggingFace.

MoE models (Mixtral, DBRX, DeepSeek) are handled natively by llama.cpp
via GGUF format. No special configuration is needed — the expert routing
is embedded in the GGUF weights and executed by the llama.cpp runtime.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import time
from typing import Any

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with llama.cpp — derived from the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_GGUF_CATALOG = {
    name for name, entry in _UNIFIED_CATALOG.items() if "llama.cpp" in entry.engines
}

# MoE models in the catalog that llama.cpp supports natively
_MOE_MODELS = {
    name
    for name, entry in _UNIFIED_CATALOG.items()
    if entry.architecture == "moe" and "llama.cpp" in entry.engines
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
        if sys == "Darwin":
            accel = "Metal"
        elif self._has_cuda():
            accel = "CUDA"
        elif self._has_rocm():
            accel = "ROCm"
        else:
            accel = "CPU"
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

    @staticmethod
    def _has_rocm() -> bool:
        """Detect AMD ROCm/hipBLAS availability on Linux.

        Checks three indicators (any one is sufficient):
        1. HIP_VISIBLE_DEVICES env var is set
        2. /opt/rocm directory exists
        3. rocminfo CLI tool is on PATH
        """
        if os.environ.get("HIP_VISIBLE_DEVICES"):
            return True
        if os.path.isdir("/opt/rocm"):
            return True
        if shutil.which("rocminfo") is not None:
            return True
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
        if self._has_cuda():
            return "CPU + CUDA"
        if self._has_rocm():
            return "CPU + ROCm"
        return "CPU"

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

    def is_moe_model(self, model_name: str) -> bool:
        """Check if the model is a known MoE model.

        llama.cpp handles MoE natively via GGUF — Mixtral, DBRX,
        and DeepSeek expert routing is baked into the weights.
        """
        return model_name in _MOE_MODELS

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        from ..serve import LlamaCppBackend

        if self.is_moe_model(model_name):
            logger.info(
                "MoE model '%s' detected — llama.cpp handles expert "
                "routing natively via GGUF",
                model_name,
            )

        backend = LlamaCppBackend(
            cache_size_mb=kwargs.get("cache_size_mb", 2048),
            cache_enabled=kwargs.get("cache_enabled", True),
        )
        backend.load_model(model_name)
        return backend
