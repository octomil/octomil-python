"""NativeEmbeddingsRuntime — ModelRuntime adapter around
:class:`NativeEmbeddingsBackend`.

Bridges the kernel's runtime resolution path
(``ModelRuntimeRegistry.resolve(model_id) -> ModelRuntime``) to the
native embeddings backend. The kernel's embedding dispatcher
(``ExecutionKernel._local_embed``) duck-types on ``runtime.embed(inputs)``
plus ``result.embeddings`` and ``result.usage.total_tokens``; this
adapter satisfies that contract by wrapping the backend's flat
``EmbeddingsResult`` into the public
:class:`octomil.embeddings.EmbeddingResult` shape (which has
``EmbeddingUsage`` nested under ``.usage``).

Hard-cutover discipline:
  1. ``run()`` and ``stream()`` raise ``UNSUPPORTED_MODALITY`` —
     embedding runtimes never serve chat. Callers MUST NOT silently
     fall through to a Python embedder.
  2. ``embed()`` errors propagate as bounded ``OctomilError`` from the
     backend; no swallow / no fallback.
  3. Lazy load — the backend is constructed on first ``embed()`` so a
     factory can register without paying load cost up-front. Once
     loaded, the cached backend lives until ``close()``.

Use ``native_embeddings_factory(model_id, *, model_dir)`` to construct
adapters from a ``ModelRuntimeRegistry`` factory site. The factory
resolves a PrepareManager-materialized GGUF model_dir and hands it
to the backend.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Optional

from ...embeddings import EmbeddingResult, EmbeddingUsage
from ...errors import OctomilError, OctomilErrorCode
from ..core.model_runtime import ModelRuntime
from ..core.types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse, ToolCallTier
from .embeddings_backend import EmbeddingsResult, NativeEmbeddingsBackend

logger = logging.getLogger(__name__)


# Embedding-capable model-family prefixes recognized by the factory.
# Any prefix match here routes through the native llama.cpp embeddings
# path; anything else falls through to the registry's default factory
# (which only knows how to build chat backends and will return None
# for embedding GGUFs without an llama.cpp engine selection).
_EMBEDDING_FAMILY_PREFIXES: tuple[str, ...] = (
    "nomic-embed",
    "bge-",
    "bge_",
    "e5-",
    "gte-",
    "mxbai-embed",
    "snowflake-arctic-embed",
    "all-minilm",
    "jina-embed",
)


def is_embedding_model(model_id: str) -> bool:
    """Return True iff ``model_id`` matches a known embedding family.

    Used by the factory and the local-runner server to gate the
    embedding code path. Match is case-insensitive prefix.
    """
    lowered = model_id.lower()
    return any(lowered.startswith(prefix) for prefix in _EMBEDDING_FAMILY_PREFIXES)


class NativeEmbeddingsRuntime(ModelRuntime):
    """ModelRuntime that owns a :class:`NativeEmbeddingsBackend`.

    The kernel's embeddings dispatcher reads ``runtime.embed(inputs)``
    and expects ``result.embeddings`` + ``result.usage.total_tokens``.
    Returning :class:`octomil.embeddings.EmbeddingResult` keeps the
    public-API shape identical between cloud and local routes.
    """

    def __init__(self, *, model_name: str, model_dir: Optional[str] = None) -> None:
        self._model_name = model_name
        self._model_dir = model_dir
        self._backend: Optional[NativeEmbeddingsBackend] = None

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            tool_call_tier=ToolCallTier.NONE,
            supports_streaming=False,
        )

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        del request
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                f"NativeEmbeddingsRuntime for {self._model_name!r} is an embeddings-only "
                f"runtime; chat/completion (run) is not supported. Route this model via "
                f"client.embeddings.create(...) instead."
            ),
        )

    def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        del request
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                f"NativeEmbeddingsRuntime for {self._model_name!r} is an embeddings-only "
                f"runtime; streaming chat (stream) is not supported."
            ),
        )

    def _ensure_backend(self) -> NativeEmbeddingsBackend:
        if self._backend is None:
            backend = NativeEmbeddingsBackend(model_dir=self._model_dir)
            backend.load_model(self._model_name)
            self._backend = backend
        return self._backend

    def embed(self, inputs: list[str]) -> EmbeddingResult:
        """Run an embeddings batch. Returns the public-API shape."""
        backend = self._ensure_backend()
        native_result: EmbeddingsResult = backend.embed(inputs)
        return EmbeddingResult(
            embeddings=native_result.embeddings,
            model=native_result.model or self._model_name,
            usage=EmbeddingUsage(
                prompt_tokens=native_result.prompt_tokens,
                total_tokens=native_result.total_tokens,
            ),
        )

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None


def _resolve_prepared_model_dir(model_id: str) -> Optional[str]:
    """Return a PrepareManager-materialized model_dir if one exists.

    Best-effort: returns ``None`` when no static recipe is registered
    for ``(model_id, "embedding")``, when PrepareManager hasn't yet
    materialized the artifact, or when the lifecycle modules are not
    importable in this interpreter.

    The factory uses this so a registered embedding family without a
    prepared artifact yields ``None`` (kernel surfaces a bounded error
    rather than silently falling through to a Python embedder).
    """
    try:
        from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    except Exception:  # noqa: BLE001
        return None
    try:
        manager = PrepareManager()
        artifact_dir = manager.artifact_dir_for(model_id)
    except Exception:  # noqa: BLE001
        return None
    if not artifact_dir.is_dir():
        return None
    return str(artifact_dir)


def native_embeddings_factory(model_id: str) -> Optional[ModelRuntime]:
    """RuntimeFactory: build a :class:`NativeEmbeddingsRuntime` for
    embedding-capable model families.

    Returns ``None`` for non-embedding model ids so the registry can
    fall through to the default chat factory. Returns a constructed
    :class:`NativeEmbeddingsRuntime` for recognized families. Backend
    construction (GGUF load) is deferred to first ``embed()`` so the
    factory remains cheap.
    """
    if not is_embedding_model(model_id):
        return None

    # Operator override: explicit GGUF path via env. Same shape as the
    # backend's own model_dir resolver so dev workflows don't need a
    # PrepareManager-materialized layout.
    explicit_dir = os.environ.get("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR")
    model_dir = explicit_dir or _resolve_prepared_model_dir(model_id)
    return NativeEmbeddingsRuntime(model_name=model_id, model_dir=model_dir)


def register_native_embeddings_factory() -> None:
    """Register :func:`native_embeddings_factory` against every known
    embedding family prefix in the global :class:`ModelRuntimeRegistry`.

    Idempotent — calling twice is a no-op (the registry just overwrites
    with the same factory). Called from a module-level import side
    effect so callers that import :mod:`octomil` automatically get
    embedding routing.
    """
    from ..core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    for prefix in _EMBEDDING_FAMILY_PREFIXES:
        registry.register(prefix, native_embeddings_factory)
