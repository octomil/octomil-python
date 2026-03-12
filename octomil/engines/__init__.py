"""Backward-compatible re-exports — use octomil.runtime.engines instead."""

from octomil.runtime.core.base import BenchmarkResult, EnginePlugin
from octomil.runtime.engines.registry import (
    DetectionResult,
    EngineRegistry,
    RankedEngine,
    _auto_register,
    _register_experimental,
    get_registry,
    reset_registry,
)

__all__ = [
    "BenchmarkResult",
    "DetectionResult",
    "EnginePlugin",
    "EngineRegistry",
    "RankedEngine",
    "_auto_register",
    "_register_experimental",
    "get_registry",
    "reset_registry",
]
