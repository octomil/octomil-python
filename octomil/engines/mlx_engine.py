"""MLX-LM engine plugin — Apple Silicon inference via mlx-lm.

Highest-performance backend on Apple Silicon Macs. Uses unified memory
for zero-copy GPU access and quantized HuggingFace models.

Attention: MLX uses Metal fused attention automatically on Apple Silicon.
No explicit ``flash_attn`` flag is needed — the Metal Performance Shaders
backend fuses multi-head attention into a single GPU kernel.  Reported as
``metal_fused`` in telemetry.

MoE models (Mixtral, DBRX, DeepSeek) are handled natively by MLX — the
framework loads all expert weights into unified memory and routes tokens
through the sparse gating network automatically.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with mlx-lm — derived from the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_MLX_CATALOG = {
    name for name, entry in _UNIFIED_CATALOG.items() if "mlx-lm" in entry.engines
}

# MoE models in the catalog that MLX supports natively
_MOE_MODELS = {
    name
    for name, entry in _UNIFIED_CATALOG.items()
    if entry.architecture == "moe" and "mlx-lm" in entry.engines
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

    @property
    def manages_own_download(self) -> bool:
        return True  # mlx_lm.load() handles HuggingFace download + caching

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
        # Supports catalog names (with alias resolution) and any HuggingFace repo ID
        from ..models.catalog import _resolve_alias

        canonical = _resolve_alias(model_name)
        return canonical in _MLX_CATALOG or "/" in model_name

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        try:
            import mlx_lm  # type: ignore[import-untyped]

            from ..serve import resolve_model_name

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

            # Warmup: JIT-compile Metal shaders and warm GPU caches.
            # A 1-token warmup compiles kernels but doesn't warm the decode
            # loop, so generate a few tokens to match Ollama's always-warm state.
            for response in mlx_lm.stream_generate(
                model, tokenizer, prompt=formatted, max_tokens=8, sampler=sampler
            ):
                if response.finish_reason:
                    break

            start = time.monotonic()
            tokens_generated = 0
            first_token_time = None
            generation_tps: float = 0.0

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
                generation_tps = response.generation_tps
                if response.finish_reason:
                    break

            ttft = ((first_token_time or start) - start) * 1000
            # Use MLX's generation_tps (decode-only, excludes prompt processing)
            # to match Ollama's eval_duration measurement for fair comparison.
            tps = generation_tps if generation_tps > 0 else (
                tokens_generated / (time.monotonic() - start)
            )

            # Clean up to free GPU memory
            del model, tokenizer, sampler
            import gc

            gc.collect()

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=ttft,
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def is_moe_model(self, model_name: str) -> bool:
        """Check if the model is a known MoE model.

        MLX handles MoE natively — expert routing is part of the model's
        computation graph in unified memory.
        """
        return model_name in _MOE_MODELS

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        from ..serve import MLXBackend

        if self.is_moe_model(model_name):
            logger.info(
                "MoE model '%s' detected — MLX handles expert "
                "routing natively in unified memory",
                model_name,
            )

        backend = MLXBackend(
            cache_size_mb=kwargs.get("cache_size_mb", 2048),
            cache_enabled=kwargs.get("cache_enabled", True),
        )
        backend.load_model(model_name)
        return backend
