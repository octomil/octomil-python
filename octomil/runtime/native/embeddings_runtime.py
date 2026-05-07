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
     embedding runtimes never serve chat. ``stream()`` is an async
     generator so the rejection is observable via ``async for`` (not
     a sync raise at call site).
  2. ``embed()`` errors propagate as bounded ``OctomilError`` from the
     backend; no swallow / no fallback.
  3. Lazy load with a per-instance lock — concurrent first ``embed()``
     calls won't double-load the backend or leak the loser's instance.
  4. Factory caches runtime instances by model_id (mirrors
     ``engine_bridge._runtime_cache``) so repeat resolutions reuse the
     warmed backend instead of re-loading the GGUF on every call.

Capability honesty: :func:`native_embeddings_factory` returns ``None``
when no PrepareManager-materialized artifact AND no env override are
available. Without a real artifact the runtime cannot serve a real
embedding request, so we MUST NOT advertise local availability —
otherwise ``ExecutionKernel._can_local`` reports True and configured
cloud fallback is silently suppressed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import AsyncIterator, Optional

from ...embeddings import EmbeddingResult, EmbeddingUsage
from ...errors import OctomilError, OctomilErrorCode
from ..core.model_runtime import ModelRuntime
from ..core.types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse, ToolCallTier
from .embeddings_backend import EmbeddingsResult, NativeEmbeddingsBackend

logger = logging.getLogger(__name__)


# Embedding-capable model-family prefixes recognized by the factory.
# Match is case-insensitive prefix on the model id with any
# org-prefix stripped (so HuggingFace-style 'BAAI/bge-base-en-v1.5'
# matches the same way as 'bge-base-en-v1.5').
#
# Prefixes are intentionally narrow — 'e5-' alone is too broad
# (could match a future 'e5-coder' chat model), so we list the
# concrete e5 embedding shapes instead.
_EMBEDDING_FAMILY_PREFIXES: tuple[str, ...] = (
    "nomic-embed",
    "bge-",
    "bge_",
    "e5-mistral",
    "e5-base",
    "e5-large",
    "e5-small",
    "gte-",
    "mxbai-embed",
    "snowflake-arctic-embed",
    "all-minilm",
    "jina-embed",
)


def _strip_org_prefix(model_id: str) -> str:
    """Drop a single leading 'org/' segment from an HF-style id.

    The catalog and CLI accept both ``bge-base-en-v1.5`` and
    ``BAAI/bge-base-en-v1.5``; the embedding-family gate must
    recognize both. We strip exactly one slash-segment — paths with
    multiple slashes (e.g. user-supplied filesystem paths) fall
    through unchanged so the gate rejects them as non-embedding ids.
    """
    parts = model_id.split("/", 1)
    if len(parts) != 2:
        return model_id
    if "/" in parts[1]:  # filesystem path, leave alone
        return model_id
    return parts[1]


def is_embedding_model(model_id: str) -> bool:
    """Return True iff ``model_id`` matches a known embedding family.

    Used by the factory and the local-runner server to gate the
    embedding code path. Match is case-insensitive prefix against
    the model id with at most one ``org/`` prefix stripped.
    """
    candidate = _strip_org_prefix(model_id).lower()
    return any(candidate.startswith(prefix) for prefix in _EMBEDDING_FAMILY_PREFIXES)


class NativeEmbeddingsRuntime(ModelRuntime):
    """ModelRuntime that owns a :class:`NativeEmbeddingsBackend`.

    The kernel's embeddings dispatcher reads ``runtime.embed(inputs)``
    and expects ``result.embeddings`` + ``result.usage.total_tokens``.
    Returning :class:`octomil.embeddings.EmbeddingResult` keeps the
    public-API shape identical between cloud and local routes.

    Concurrency: ``embed()`` is safe to call from multiple threads /
    asyncio tasks. The first call wins the lazy-load lock; concurrent
    callers wait and reuse the same backend instance.
    """

    def __init__(self, *, model_name: str, model_dir: Optional[str] = None) -> None:
        self._model_name = model_name
        self._model_dir = model_dir
        self._backend: Optional[NativeEmbeddingsBackend] = None
        self._load_lock = threading.Lock()

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

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        """Reject chat streaming. Implemented as an async generator
        (raises on first ``async for`` iteration, NOT at call time)
        so callers wrapping ``async for chunk in runtime.stream(...)``
        in try/except observe the rejection where they expect it."""
        del request
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                f"NativeEmbeddingsRuntime for {self._model_name!r} is an embeddings-only "
                f"runtime; streaming chat (stream) is not supported."
            ),
        )
        # Unreachable, but required to make this an async generator
        # rather than a coroutine returning AsyncIterator.
        yield  # type: ignore[unreachable]

    def _ensure_backend(self) -> NativeEmbeddingsBackend:
        """Lazy-load the backend under a lock.

        Two threads racing through ``self._backend is None`` won't
        both pay the multi-second GGUF load cost or leak a backend.
        Re-checks inside the critical section; common case (already
        loaded) hits the lock briefly then returns.
        """
        if self._backend is not None:
            return self._backend
        with self._load_lock:
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
        with self._load_lock:
            if self._backend is not None:
                try:
                    self._backend.close()
                finally:
                    self._backend = None


def _resolve_prepared_model_dir(model_id: str) -> Optional[str]:
    """Return a PrepareManager-materialized model_dir if one exists.

    Best-effort: returns ``None`` when no artifact has been
    materialized, when no static recipe is registered, or when the
    lifecycle modules are not importable in this interpreter. The
    factory uses ``None`` as a hard signal that local serving is
    NOT available for this model.
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


# Process-wide cache of constructed runtimes, keyed by canonical
# model id. Mirrors ``engine_bridge._runtime_cache`` so repeat
# resolutions reuse the warmed backend rather than paying GGUF load
# cost on every call. Concurrent factory calls are safe because dict
# assignment is atomic in CPython; a benign double-construct on a
# rare race wastes a Runtime object but doesn't leak a backend
# (lazy-load only happens on .embed()).
_runtime_cache: dict[str, ModelRuntime] = {}


def native_embeddings_factory(model_id: str) -> Optional[ModelRuntime]:
    """RuntimeFactory: build a :class:`NativeEmbeddingsRuntime` for
    embedding-capable model families when a prepared artifact (or
    explicit env override) is available.

    Returns ``None`` for non-embedding model ids OR when no artifact
    can be resolved. The latter case is critical for capability
    honesty: ``ExecutionKernel._can_local`` reports local available
    iff ``registry.resolve(model)`` returns a runtime, so falsely
    returning a runtime here suppresses configured cloud fallback.

    Cached by model_id so repeat resolutions reuse the same
    (warm-on-first-embed) runtime instead of re-loading the GGUF on
    every call. Cache scope is process-wide (matches the chat path).
    """
    if not is_embedding_model(model_id):
        return None

    cached = _runtime_cache.get(model_id)
    if cached is not None:
        return cached

    explicit_dir = os.environ.get("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR")
    model_dir = explicit_dir or _resolve_prepared_model_dir(model_id)
    if model_dir is None:
        # No artifact => no real local capability. Returning None
        # here keeps _can_local honest so cloud fallback engages.
        return None

    runtime = NativeEmbeddingsRuntime(model_name=model_id, model_dir=model_dir)
    _runtime_cache[model_id] = runtime
    return runtime


def reset_runtime_cache() -> None:
    """Clear the process-wide runtime cache. Test-only helper —
    closes any cached backends so the next factory call starts
    cold. Tests use this in fixtures to keep state from leaking
    across test functions."""
    for runtime in list(_runtime_cache.values()):
        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass
    _runtime_cache.clear()


# Known HuggingFace orgs that publish embedding models for each
# family. Registered as ``<org>/<family>`` prefix variants so the
# registry's literal-prefix matching reaches the factory for canonical
# HF ids (BAAI/bge-base-en-v1.5, intfloat/e5-mistral-7b-instruct,
# etc.) without changing the registry's matching algorithm.
#
# This is the round-3 fix for the "registry prefix-matches the raw
# model id but the factory's family gate strips org prefix" gap
# Codex flagged in R2: ``is_embedding_model('BAAI/bge-...')`` was
# True, yet ``registry.resolve('BAAI/bge-...')`` never reached the
# factory because no registered prefix matched 'baai/'.
_KNOWN_HF_ORG_PREFIXES: dict[str, tuple[str, ...]] = {
    "bge-": ("baai/",),
    "bge_": ("baai/",),
    "nomic-embed": ("nomic-ai/",),
    "e5-mistral": ("intfloat/",),
    "e5-base": ("intfloat/",),
    "e5-large": ("intfloat/",),
    "e5-small": ("intfloat/",),
    "gte-": ("thenlper/", "alibaba-nlp/"),
    "mxbai-embed": ("mixedbread-ai/",),
    "snowflake-arctic-embed": ("snowflake/",),
    "all-minilm": ("sentence-transformers/",),
    "jina-embed": ("jinaai/",),
}


def register_native_embeddings_factory() -> None:
    """Register :func:`native_embeddings_factory` against every known
    embedding family prefix AND the common ``<org>/<family>`` HuggingFace
    variants in the global :class:`ModelRuntimeRegistry`.

    Idempotent — calling twice is a no-op (the registry just overwrites
    with the same factory). Called from a module-level import side
    effect so callers that import :mod:`octomil` automatically get
    embedding routing.

    Why register org/family combos: ``ModelRuntimeRegistry.resolve``
    matches the lowercased raw model id against registered prefix
    keys with ``str.startswith``. ``BAAI/bge-base-en-v1.5`` lowers
    to ``baai/bge-base-en-v1.5`` which does NOT start with ``bge-``,
    so the factory is never reached without the ``baai/bge-`` entry.
    """
    from ..core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    for prefix in _EMBEDDING_FAMILY_PREFIXES:
        registry.register(prefix, native_embeddings_factory)
        for org_prefix in _KNOWN_HF_ORG_PREFIXES.get(prefix, ()):
            registry.register(org_prefix + prefix, native_embeddings_factory)
