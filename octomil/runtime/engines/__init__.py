"""Engine plugin system — detect, benchmark, pick fastest runtime.

Usage::

    from octomil.runtime.engines import get_registry

    registry = get_registry()
    engine, results = registry.auto_select("gemma-2b")
    backend = engine.create_backend("gemma-2b")
"""

from octomil.runtime.core.base import BenchmarkResult, EnginePlugin
from octomil.runtime.engines.registry import (
    DetectionResult,
    EngineRegistry,
    RankedEngine,
    get_registry,
    reset_registry,
)

__all__ = [
    "BenchmarkResult",
    "DetectionResult",
    "EnginePlugin",
    "EngineRegistry",
    "RankedEngine",
    "get_registry",
    "reset_registry",
]
