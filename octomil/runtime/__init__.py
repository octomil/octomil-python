"""Octomil runtime — unified engine + model runtime layer.

Public API::

    from octomil.runtime import ModelRuntime, RuntimeFactory
    from octomil.runtime.engines import EnginePlugin, get_registry

For internal types (RuntimeRequest, RuntimeChunk, etc.) import directly
from ``octomil.runtime.core``.
"""

from __future__ import annotations

import os

from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory

__all__ = [
    "ModelRuntime",
    "RuntimeFactory",
]


def _connect_engines() -> None:
    """Wire EngineRegistry as the default factory for ModelRuntimeRegistry."""
    try:
        from octomil.runtime.core.engine_bridge import engine_registry_factory
        from octomil.runtime.core.registry import ModelRuntimeRegistry
        from octomil.runtime.engines.registry import (
            _register_experimental,
            get_registry,
        )

        # Always register stable engines
        registry = get_registry()

        # Register experimental engines if env var is set
        if os.environ.get("OCTOMIL_EXPERIMENTAL_ENGINES"):
            _register_experimental(registry)

        ModelRuntimeRegistry.shared().default_factory = engine_registry_factory
    except Exception:
        pass  # engines module may not be available in all environments


_connect_engines()
