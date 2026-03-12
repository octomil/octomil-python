"""Octomil runtime — unified engine + model runtime layer.

Public API::

    from octomil.runtime.engines import get_registry
    from octomil.runtime.core import ModelRuntime, ModelRuntimeRegistry
"""

from __future__ import annotations

import os

from octomil.runtime.core import (
    CloudModelRuntime,
    InferenceBackendAdapter,
    ModelRuntime,
    ModelRuntimeRegistry,
    RouterModelRuntime,
    RoutingPolicy,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeFactory,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
    RuntimeToolCallDelta,
    RuntimeToolDef,
    RuntimeUsage,
)

__all__ = [
    "CloudModelRuntime",
    "InferenceBackendAdapter",
    "ModelRuntime",
    "ModelRuntimeRegistry",
    "RouterModelRuntime",
    "RoutingPolicy",
    "RuntimeCapabilities",
    "RuntimeChunk",
    "RuntimeFactory",
    "RuntimeRequest",
    "RuntimeResponse",
    "RuntimeToolCall",
    "RuntimeToolCallDelta",
    "RuntimeToolDef",
    "RuntimeUsage",
    "_connect_engines",
]


def _connect_engines() -> None:
    """Wire EngineRegistry as the default factory for ModelRuntimeRegistry."""
    try:
        from octomil.runtime.core.engine_bridge import engine_registry_factory
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
