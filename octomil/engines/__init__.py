"""Octomil engine plugin system — detect, benchmark, pick fastest runtime.

Usage::

    from octomil.engines import get_registry

    registry = get_registry()
    engine, results = registry.auto_select("gemma-2b")
    backend = engine.create_backend("gemma-2b")
"""

from .base import BenchmarkResult, EnginePlugin
from .registry import (
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
