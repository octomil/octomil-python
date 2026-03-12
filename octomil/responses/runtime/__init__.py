"""Layer 1: ModelRuntime — typed per-engine inference interface."""

from __future__ import annotations

from .adapter import InferenceBackendAdapter
from .cloud_runtime import CloudModelRuntime
from .model_runtime import ModelRuntime, RuntimeFactory
from .policy import RoutingPolicy
from .registry import ModelRuntimeRegistry
from .router import RouterModelRuntime
from .types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
    RuntimeToolCallDelta,
    RuntimeToolDef,
    RuntimeUsage,
)

__all__ = [
    "CloudModelRuntime",
    "ModelRuntime",
    "RuntimeFactory",
    "ModelRuntimeRegistry",
    "InferenceBackendAdapter",
    "RouterModelRuntime",
    "RoutingPolicy",
    "RuntimeCapabilities",
    "RuntimeRequest",
    "RuntimeResponse",
    "RuntimeChunk",
    "RuntimeToolCall",
    "RuntimeToolCallDelta",
    "RuntimeToolDef",
    "RuntimeUsage",
]


def _connect_engines() -> None:
    """Wire EngineRegistry as the default factory for ModelRuntimeRegistry."""
    try:
        from .engine_bridge import engine_registry_factory

        ModelRuntimeRegistry.shared().default_factory = engine_registry_factory
    except Exception:
        pass  # engines module may not be available in all environments


_connect_engines()
