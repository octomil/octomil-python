"""Bridge EngineRegistry into ModelRuntimeRegistry as the default factory."""

from __future__ import annotations

from typing import Optional

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory
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
        detections = registry.detect_all(model_id)
        real_engines = [d.engine for d in detections if d.available and d.engine.name != "echo"]
        if not real_engines:
            return None

        engine = sorted(real_engines, key=lambda e: e.priority)[0]
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


def cloud_runtime_factory(base_url: str, api_key: str, model: str) -> RuntimeFactory:
    """RuntimeFactory that creates CloudModelRuntime for cloud inference."""
    _cached: Optional[ModelRuntime] = None

    def factory(model_id: str) -> Optional[ModelRuntime]:
        nonlocal _cached
        if _cached is None:
            from octomil.runtime.core.cloud_runtime import CloudModelRuntime

            _cached = CloudModelRuntime(base_url, api_key, model)
        return _cached

    return factory
