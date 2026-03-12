"""Layer 1: ModelRuntime — typed per-engine inference interface."""

from __future__ import annotations

from .adapter import InferenceBackendAdapter
from .model_runtime import ModelRuntime, RuntimeFactory
from .registry import ModelRuntimeRegistry
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
    "ModelRuntime",
    "RuntimeFactory",
    "ModelRuntimeRegistry",
    "InferenceBackendAdapter",
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
