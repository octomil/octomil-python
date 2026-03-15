"""Bridge EngineRegistry into ModelRuntimeRegistry as the default factory."""

from __future__ import annotations

from typing import Optional

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import RuntimeCapabilities, ToolCallTier

_runtime_cache: dict[str, ModelRuntime] = {}

# Models known to follow the tool-call JSON protocol reliably.
# Extend this list as models are evaluated.
_TEXT_JSON_FAMILIES: frozenset[str] = frozenset(
    {
        "gemma",
        "llama-3",
        "qwen2",
        "phi-3",
        "mistral",
    }
)


def _infer_tool_call_tier(model_id: str) -> ToolCallTier:
    """Infer tool-call tier from model family. Defaults to NONE."""
    model_lower = model_id.lower()
    for family in _TEXT_JSON_FAMILIES:
        if family in model_lower:
            return ToolCallTier.TEXT_JSON
    return ToolCallTier.NONE


def engine_registry_factory(model_id: str) -> Optional[ModelRuntime]:
    """RuntimeFactory that uses EngineRegistry to auto-select an engine."""
    if model_id in _runtime_cache:
        return _runtime_cache[model_id]

    try:
        from octomil.runtime.engines import get_registry

        registry = get_registry()
        engine, _ = registry.auto_select(model_id, n_tokens=0)
        backend = engine.create_backend(model_id)
        adapter = InferenceBackendAdapter(
            backend=backend,
            model_name=model_id,
            capabilities=RuntimeCapabilities(
                tool_call_tier=_infer_tool_call_tier(model_id),
                supports_streaming=True,
            ),
        )
        _runtime_cache[model_id] = adapter
        return adapter
    except (ValueError, RuntimeError, ImportError):
        return None
