"""Bridge EngineRegistry into ModelRuntimeRegistry as the default factory."""

from __future__ import annotations

from typing import Optional

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.model_runtime import ModelRuntime

_runtime_cache: dict[str, ModelRuntime] = {}


def engine_registry_factory(model_id: str) -> Optional[ModelRuntime]:
    """RuntimeFactory that uses EngineRegistry to auto-select an engine."""
    if model_id in _runtime_cache:
        return _runtime_cache[model_id]

    try:
        from octomil.runtime.engines import get_registry

        registry = get_registry()
        engine, _ = registry.auto_select(model_id, n_tokens=0)
        backend = engine.create_backend(model_id)
        adapter = InferenceBackendAdapter(backend=backend, model_name=model_id)
        _runtime_cache[model_id] = adapter
        return adapter
    except (ValueError, RuntimeError, ImportError):
        return None
