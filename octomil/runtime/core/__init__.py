"""Runtime core — protocols, types, registries, and adapters.

This package is **internal**. Import from ``octomil.runtime`` for the public API.
Direct imports from ``octomil.runtime.core`` are allowed within the SDK but
are not part of the stable public API surface.
"""

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
    ToolCallTier,
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
    "ToolCallTier",
]
