"""Engine registry — detect, benchmark, and select the fastest runtime.

Usage::

    from octomil.engines import get_registry

    registry = get_registry()
    available = registry.detect_all()
    results = registry.benchmark_all("gemma-2b", n_tokens=32)
    best = registry.select_best(results)
    backend = best.engine.create_backend("gemma-2b")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of engine detection."""

    engine: EnginePlugin
    available: bool
    info: str = ""


@dataclass
class RankedEngine:
    """Engine with its benchmark result, for selection."""

    engine: EnginePlugin
    result: BenchmarkResult


class EngineRegistry:
    """Registry of available inference engines.

    Engines register themselves via ``register()``. On startup,
    ``detect_all()`` finds which are available, ``benchmark_all()``
    runs a quick test on each, and ``select_best()`` picks the winner.

    ``auto_select()`` caches its result per model name so the benchmark
    only runs once per process lifetime.
    """

    def __init__(self) -> None:
        self._engines: list[EnginePlugin] = []
        self._auto_select_cache: dict[str, tuple[EnginePlugin, list[RankedEngine]]] = {}

    def register(self, engine: EnginePlugin) -> None:
        """Register an engine plugin."""
        # Avoid duplicate registration
        for existing in self._engines:
            if existing.name == engine.name:
                return
        self._engines.append(engine)

    @property
    def engines(self) -> list[EnginePlugin]:
        """All registered engines."""
        return list(self._engines)

    def get_engine(self, name: str) -> Optional[EnginePlugin]:
        """Get a specific engine by name."""
        for engine in self._engines:
            if engine.name == name:
                return engine
        return None

    def detect_all(self, model_name: Optional[str] = None) -> list[DetectionResult]:
        """Detect which engines are available on this system.

        If model_name is provided, also checks model support.
        Resolves model name aliases before checking engine support.
        """
        # Resolve alias once, pass canonical name to all engines
        canonical_name = model_name
        if model_name:
            from ..models.catalog import _resolve_alias

            canonical_name = _resolve_alias(model_name)

        results: list[DetectionResult] = []
        for engine in self._engines:
            try:
                available = engine.detect()
                if available and canonical_name:
                    available = engine.supports_model(canonical_name)
                info = engine.detect_info() if available else ""
                results.append(
                    DetectionResult(engine=engine, available=available, info=info)
                )
            except Exception as exc:
                logger.debug("Engine %s detection failed: %s", engine.name, exc)
                results.append(
                    DetectionResult(engine=engine, available=False, info=str(exc))
                )
        return results

    def benchmark_all(
        self,
        model_name: str,
        n_tokens: int = 32,
        engines: Optional[list[EnginePlugin]] = None,
    ) -> list[RankedEngine]:
        """Benchmark all available engines (or a specific subset).

        Returns engines ranked by tokens_per_second (highest first).
        """
        targets = engines or [
            d.engine for d in self.detect_all(model_name) if d.available
        ]

        ranked: list[RankedEngine] = []
        for engine in targets:
            logger.info("Benchmarking %s...", engine.name)
            start = time.monotonic()
            try:
                result = engine.benchmark(model_name, n_tokens=n_tokens)
            except Exception as exc:
                logger.warning("Benchmark failed for %s: %s", engine.name, exc)
                result = BenchmarkResult(engine_name=engine.name, error=str(exc))
            elapsed = time.monotonic() - start
            result.metadata["benchmark_duration_s"] = round(elapsed, 2)
            ranked.append(RankedEngine(engine=engine, result=result))

        # Sort: successful benchmarks first (by tok/s desc), then by priority
        ranked.sort(
            key=lambda r: (
                0 if r.result.ok else 1,
                -r.result.tokens_per_second,
                r.engine.priority,
            )
        )
        return ranked

    def select_best(
        self,
        ranked: list[RankedEngine],
    ) -> Optional[RankedEngine]:
        """Select the best engine from benchmark results."""
        for r in ranked:
            if r.result.ok:
                return r
        # No successful benchmarks — return the first engine that's not echo
        for r in ranked:
            if r.engine.name != "echo":
                return r
        # Absolute fallback
        return ranked[0] if ranked else None

    def auto_select(
        self,
        model_name: str,
        n_tokens: int = 32,
        engine_override: Optional[str] = None,
    ) -> tuple[EnginePlugin, list[RankedEngine]]:
        """Detect, benchmark, and return the best engine + all results.

        If engine_override is specified, skip benchmark and use that engine.
        Results are cached per model name — the benchmark only runs once
        per process lifetime.
        Raises ValueError if no engine is available.
        """
        if engine_override:
            engine = self.get_engine(engine_override)
            if engine is None:
                available = [e.name for e in self._engines]
                raise ValueError(
                    f"Unknown engine '{engine_override}'. "
                    f"Available: {', '.join(available)}"
                )
            if not engine.detect():
                raise ValueError(
                    f"Engine '{engine_override}' is not available on this system. "
                    f"Check that the required libraries are installed."
                )
            return engine, []

        # Return cached result if available
        cache_key = model_name
        if cache_key in self._auto_select_cache:
            return self._auto_select_cache[cache_key]

        ranked = self.benchmark_all(model_name, n_tokens=n_tokens)
        if not ranked:
            raise ValueError(
                "No inference engines available. Install one of:\n"
                "  pip install 'octomil-sdk[mlx]'     # Apple Silicon\n"
                "  pip install 'octomil-sdk[llama]'    # Cross-platform\n"
                "  pip install 'octomil-sdk[onnx]'     # ONNX Runtime"
            )

        best = self.select_best(ranked)
        if best is None:
            raise ValueError("All engine benchmarks failed.")

        result = (best.engine, ranked)
        self._auto_select_cache[cache_key] = result
        return result


# ---------------------------------------------------------------------------
# Global registry singleton
# ---------------------------------------------------------------------------

_registry: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    """Get the global engine registry, auto-registering built-in engines."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
        _auto_register(_registry)
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None


def _auto_register(registry: EngineRegistry) -> None:
    """Register all built-in engines."""
    from .cactus_engine import CactusEngine
    from .echo_engine import EchoEngine
    from .executorch_engine import ExecuTorchEngine
    from .llamacpp_engine import LlamaCppEngine
    from .mlc_engine import MLCEngine
    from .mlx_engine import MLXEngine
    from .mnn_engine import MNNEngine
    from .ollama_engine import OllamaEngine
    from .ort_engine import ONNXRuntimeEngine
    from .samsung_one_engine import SamsungOneEngine
    from .whisper_engine import WhisperCppEngine

    registry.register(MLXEngine())
    registry.register(MNNEngine())
    registry.register(MLCEngine())
    registry.register(SamsungOneEngine())  # NPU-accelerated on Samsung/Exynos
    registry.register(LlamaCppEngine())
    registry.register(CactusEngine())
    registry.register(ExecuTorchEngine())
    registry.register(ONNXRuntimeEngine())
    registry.register(WhisperCppEngine())
    registry.register(OllamaEngine())  # Zero-pip fallback, before echo
    registry.register(EchoEngine())
