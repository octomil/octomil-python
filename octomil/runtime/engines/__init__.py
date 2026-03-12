"""Engine plugin system — detect, benchmark, pick fastest runtime.

Public API::

    from octomil.runtime.engines import EnginePlugin, get_registry

    registry = get_registry()
    engine, results = registry.auto_select("gemma-2b")
    backend = engine.create_backend("gemma-2b")

For internal types (EngineRegistry, DetectionResult, etc.) import directly
from ``octomil.runtime.engines.registry``.
"""

from octomil.runtime.core.base import EnginePlugin
from octomil.runtime.engines.registry import get_registry

__all__ = [
    "EnginePlugin",
    "get_registry",
]
