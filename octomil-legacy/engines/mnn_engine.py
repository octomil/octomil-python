"""MNN-LLM engine plugin — Alibaba's high-performance inference runtime.

MNN-LLM supports GGUF models and provides Metal (macOS/iOS), Vulkan (Android/Linux),
OpenCL, and CUDA backends. Benchmarks show up to 25x faster than llama.cpp on
some Metal GPUs.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import time
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with MNN-LLM
_MNN_CATALOG = {
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


def _find_mnn_cli() -> Optional[str]:
    """Find the mnn-llm CLI binary."""
    for name in ("mnn-llm", "MNN-LLM", "mnnllm"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _has_pymnn() -> bool:
    """Check if MNN Python bindings are available."""
    try:
        import MNN  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


class MNNEngine(EnginePlugin):
    """MNN-LLM inference engine — Metal/Vulkan/CUDA accelerated."""

    @property
    def name(self) -> str:
        return "mnn"

    @property
    def display_name(self) -> str:
        sys = platform.system()
        if sys == "Darwin":
            accel = "Metal"
        elif sys == "Linux":
            accel = "Vulkan/CUDA"
        else:
            accel = "CPU"
        return f"MNN-LLM ({accel})"

    @property
    def priority(self) -> int:
        return 15  # Higher priority than llama.cpp (20), lower than mlx-lm (10)

    def detect(self) -> bool:
        return _has_pymnn() or _find_mnn_cli() is not None

    def detect_info(self) -> str:
        parts: list[str] = []
        if _has_pymnn():
            parts.append("Python bindings")
        cli = _find_mnn_cli()
        if cli:
            parts.append(f"CLI: {cli}")

        sys = platform.system()
        if sys == "Darwin":
            parts.append("Metal")
        elif sys == "Linux":
            parts.append("Vulkan")

        return ", ".join(parts) if parts else ""

    def supports_model(self, model_name: str) -> bool:
        return (
            model_name in _MNN_CATALOG
            or model_name.endswith(".gguf")
            or model_name.endswith(".mnn")
            or "/" in model_name
        )

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        # Try Python bindings first, then CLI
        if _has_pymnn():
            return self._benchmark_python(model_name, n_tokens)
        cli = _find_mnn_cli()
        if cli:
            return self._benchmark_cli(cli, model_name, n_tokens)
        return BenchmarkResult(engine_name=self.name, error="MNN not available")

    def _benchmark_python(self, model_name: str, n_tokens: int) -> BenchmarkResult:
        try:
            import MNN.llm as mnn_llm  # type: ignore[import-untyped]

            model = mnn_llm.create(model_name)
            model.load()

            prompt = "Hello, how are you?"
            start = time.monotonic()
            _output = model.generate(prompt, max_new_tokens=n_tokens)
            elapsed = time.monotonic() - start

            tps = n_tokens / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                metadata={"method": "python"},
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def _benchmark_cli(
        self, cli_path: str, model_name: str, n_tokens: int
    ) -> BenchmarkResult:
        try:
            result = subprocess.run(
                [
                    cli_path,
                    "--model",
                    model_name,
                    "--benchmark",
                    "--max_tokens",
                    str(n_tokens),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return BenchmarkResult(
                    engine_name=self.name,
                    error=f"CLI exited {result.returncode}: {result.stderr[:200]}",
                )

            # Parse tok/s from CLI output (format: "X.XX tok/s")
            tps = _parse_tps_from_output(result.stdout)
            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                metadata={"method": "cli"},
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                engine_name=self.name, error="Benchmark timed out (120s)"
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        if _has_pymnn():
            return _MNNBackend(model_name, **kwargs)
        raise RuntimeError(
            "MNN Python bindings required for serving. Install with: pip install MNN"
        )


def _parse_tps_from_output(output: str) -> float:
    """Parse tokens/second from MNN-LLM CLI output."""
    import re

    # Look for patterns like "XX.XX tok/s" or "XX.XX tokens/s"
    match = re.search(r"([\d.]+)\s*tok(?:en)?s?/s", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


class _MNNBackend:
    """Thin wrapper around MNN.llm for the InferenceBackend interface."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        import MNN.llm as mnn_llm  # type: ignore[import-untyped]

        self._model = mnn_llm.create(model_name)
        self._model.load()
        self.model_name = model_name

    def generate(self, request: Any) -> tuple[str, Any]:
        if self._model is None:
            self.load_model(self.model_name)

        prompt = request.messages[-1]["content"] if request.messages else ""
        max_tokens = getattr(request, "max_tokens", 512)

        start = time.monotonic()
        output = self._model.generate(prompt, max_new_tokens=max_tokens)
        elapsed = time.monotonic() - start

        token_count = len(output.split()) if isinstance(output, str) else 0
        tps = token_count / elapsed if elapsed > 0 else 0.0

        from dataclasses import dataclass

        @dataclass
        class Metrics:
            total_tokens: int = token_count
            tokens_per_second: float = tps
            ttfc_ms: float = 0.0

        return str(output), Metrics()

    def list_models(self) -> list[str]:
        return [self.model_name] if self.model_name else []
