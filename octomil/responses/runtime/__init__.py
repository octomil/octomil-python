"""Backward-compatible re-exports — use octomil.runtime.core instead."""

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.cloud_runtime import CloudModelRuntime
from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.router import RouterModelRuntime
from octomil.runtime.core.types import (
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
]


def _connect_engines() -> None:
    """Wire EngineRegistry as the default factory for ModelRuntimeRegistry."""
    from octomil.runtime import _connect_engines as _real_connect

    _real_connect()
