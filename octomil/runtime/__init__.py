"""Octomil runtime — unified engine + model runtime layer.

Public API::

    from octomil.runtime import ModelRuntime, RuntimeFactory
    from octomil.runtime.engines import EnginePlugin, get_registry

For internal types (RuntimeRequest, RuntimeChunk, etc.) import directly
from ``octomil.runtime.core``.
"""

from __future__ import annotations

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
        from octomil.runtime.engines.registry import get_registry

        # Initialize stable engines and any env-gated experimental engines.
        get_registry()

        ModelRuntimeRegistry.shared().default_factory = engine_registry_factory
    except Exception:
        pass  # engines module may not be available in all environments


def _connect_native_embeddings() -> None:
    """Register the native embeddings factory for embedding-capable
    model families.

    Runs after ``_connect_engines`` so prefix matches (which the
    embedding factory relies on) take precedence over the
    chat-oriented default factory. Best-effort: a missing native
    runtime / lifecycle module yields a silent no-op rather than
    blocking SDK import.
    """
    try:
        from octomil.runtime.native.embeddings_runtime import register_native_embeddings_factory

        register_native_embeddings_factory()
    except Exception:
        pass  # native module unavailable on this platform / build


_connect_engines()
_connect_native_embeddings()
