"""MLX-LM engine plugin — Apple Silicon inference via mlx-lm.

Highest-performance backend on Apple Silicon Macs. Uses unified memory
for zero-copy GPU access and quantized HuggingFace models.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with mlx-lm (subset for quick reference)
_MLX_CATALOG = {
    "gemma-1b",
    "gemma-4b",
    "gemma-12b",
    "gemma-27b",
    "llama-1b",
    "llama-3b",
    "llama-8b",
    "phi-4",
    "phi-mini",
    "qwen-1.5b",
    "qwen-3b",
    "qwen-7b",
    "mistral-7b",
    "smollm-360m",
}


class MLXEngine(EnginePlugin):
    """Apple Silicon engine using mlx-lm."""

    @property
    def name(self) -> str:
        return "mlx-lm"

    @property
    def display_name(self) -> str:
        return "mlx-lm (Apple Silicon)"

    @property
    def priority(self) -> int:
        return 10  # Highest priority on Apple Silicon

    def detect(self) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            import mlx_lm  # type: ignore[import-untyped]  # noqa: F401

            return True
        except ImportError:
            return False

    def detect_info(self) -> str:
        try:
            chip = platform.processor() or platform.machine()
            return f"Apple Silicon {chip}"
        except Exception:
            return "Apple Silicon"

    def supports_model(self, model_name: str) -> bool:
        # Supports catalog names and any HuggingFace repo ID
        return model_name in _MLX_CATALOG or "/" in model_name

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        try:
            import mlx_lm  # type: ignore[import-untyped]

            from ..serve import MLXBackend, resolve_model_name

            repo_id = resolve_model_name(model_name, "mlx")
            model, tokenizer = mlx_lm.load(repo_id)

            # Build a simple prompt
            prompt = "Hello, how are you?"
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted = f"user: {prompt}\nassistant:"

            from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

            sampler = make_sampler(temp=0.7)

            # Warm up
            start = time.monotonic()
            tokens_generated = 0
            first_token_time = None

            for response in mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=n_tokens,
                sampler=sampler,
            ):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                tokens_generated += 1
                if response.finish_reason:
                    break

            elapsed = time.monotonic() - start
            ttft = ((first_token_time or start) - start) * 1000
            tps = tokens_generated / elapsed if elapsed > 0 else 0

            # Clean up to free memory
            del model, tokenizer

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=ttft,
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        from ..serve import MLXBackend

        backend = MLXBackend(
            cache_size_mb=kwargs.get("cache_size_mb", 2048),
            cache_enabled=kwargs.get("cache_enabled", True),
        )
        backend.load_model(model_name)
        return backend
