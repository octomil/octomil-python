"""Echo engine plugin — testing fallback when no real engine is available.

Always available, always last priority. Echoes input back as output.
Useful for testing the API layer without any inference backend.
"""

from __future__ import annotations

from typing import Any

from .base import BenchmarkResult, EnginePlugin


class EchoEngine(EnginePlugin):
    """Fallback engine that echoes input — for testing the API layer."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def display_name(self) -> str:
        return "echo (testing fallback)"

    @property
    def priority(self) -> int:
        return 999  # Always last

    def detect(self) -> bool:
        return True  # Always available

    def detect_info(self) -> str:
        return "no real inference — echoes input"

    def supports_model(self, model_name: str) -> bool:
        return True  # Supports any model name

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        # Echo is instant but not real inference
        return BenchmarkResult(
            engine_name=self.name,
            tokens_per_second=0.0,
            error="echo backend — no real inference",
        )

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        from ..serve import EchoBackend

        backend = EchoBackend()
        backend.load_model(model_name)
        return backend
