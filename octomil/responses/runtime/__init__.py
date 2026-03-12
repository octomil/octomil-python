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
