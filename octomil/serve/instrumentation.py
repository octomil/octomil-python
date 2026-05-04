"""Transparent instrumentation wrapper for InferenceBackend.

When verbose mode is enabled, ``InstrumentedBackend`` wraps any backend and
emits structured ``backend.*`` lifecycle events (load, generate, stream)
without requiring per-backend manual instrumentation.

Backends can provide engine-specific metadata by overriding
``get_verbose_metadata()`` on the base class.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from .types import (
    GenerationChunk,
    GenerationRequest,
    InferenceBackend,
    InferenceMetrics,
    resolve_backend_capabilities,
)
from .verbose_events import VerboseEventEmitter


class InstrumentedBackend(InferenceBackend):
    """Transparent wrapper that emits verbose lifecycle events for any backend.

    Applied automatically by ``_detect_backend()`` when verbose mode is on.
    Delegates all real work to the inner backend.
    """

    def __init__(self, backend: InferenceBackend, emitter: VerboseEventEmitter) -> None:
        # Intentionally skip super().__init__() — the inner backend already
        # owns the ThreadPoolExecutor; creating a second one wastes resources.
        self._inner = backend
        self._emitter = emitter
        # Shadow class-level attrs so __getattr__ isn't needed for these.
        # Cutover follow-up #71 (R4 Codex): explicitly delegate `capabilities`
        # because Python class lookup finds the base `InferenceBackend.capabilities`
        # default before `__getattr__` runs — without this, callers who don't
        # `unwrap_backend()` first would see the conservative default
        # (grammar=False) even when the wrapped backend declares grammar=True.
        self.name = backend.name  # type: ignore[assignment]
        self.attention_backend = backend.attention_backend  # type: ignore[assignment]
        # Cutover follow-up #71 (R5 Codex): use the defensive helper so
        # wrapping a duck-typed backend (no `capabilities` class attr) falls
        # through to conservative defaults instead of AttributeError'ing
        # at construction time.
        self.capabilities = resolve_backend_capabilities(backend)  # type: ignore[misc]

    # --- Proxy properties ---------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the inner backend.

        Allows debug endpoints (e.g. _last_timings) to work transparently
        through the wrapper without explicit forwarding.
        """
        return getattr(self._inner, name)

    # --- Lifecycle interception ----------------------------------------------

    def load_model(self, model_name: str) -> None:
        extra = self._inner.get_verbose_metadata("backend.load_started")
        self._emitter.emit("backend.load_started", model=model_name, engine=self._inner.name, **extra)
        start = time.monotonic()
        try:
            self._inner.load_model(model_name)
        except Exception as exc:
            self._emitter.emit("backend.load_failed", model=model_name, engine=self._inner.name, error=str(exc))
            raise
        elapsed_ms = (time.monotonic() - start) * 1000
        extra = self._inner.get_verbose_metadata("backend.load_completed")
        self._emitter.emit(
            "backend.load_completed",
            model=model_name,
            engine=self._inner.name,
            load_time_ms=round(elapsed_ms, 1),
            **extra,
        )

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        extra = self._inner.get_verbose_metadata("backend.generate_started", request=request)
        self._emitter.emit(
            "backend.generate_started",
            engine=self._inner.name,
            model=request.model,
            max_tokens=request.max_tokens,
            **extra,
        )
        start = time.monotonic()
        try:
            text, metrics = self._inner.generate(request)
        except Exception as exc:
            self._emitter.emit("backend.generate_failed", engine=self._inner.name, error=str(exc))
            raise
        elapsed_ms = (time.monotonic() - start) * 1000
        tps = metrics.total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        extra = self._inner.get_verbose_metadata("backend.generate_completed", request=request, metrics=metrics)
        self._emitter.emit(
            "backend.generate_completed",
            engine=self._inner.name,
            duration_ms=round(elapsed_ms, 1),
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.total_tokens,
            tokens_per_second=round(tps, 1),
            **extra,
        )
        return text, metrics

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        extra = self._inner.get_verbose_metadata("backend.stream_started", request=request)
        self._emitter.emit(
            "backend.stream_started",
            engine=self._inner.name,
            model=request.model,
            max_tokens=request.max_tokens,
            **extra,
        )
        start = time.monotonic()
        tokens_generated = 0
        try:
            async for chunk in self._inner.generate_stream(request):
                tokens_generated += 1
                yield chunk
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._emitter.emit(
                "backend.stream_failed",
                engine=self._inner.name,
                error=str(exc),
                tokens_generated=tokens_generated,
                duration_ms=round(elapsed_ms, 1),
            )
            raise
        else:
            elapsed_ms = (time.monotonic() - start) * 1000
            tps = tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            extra = self._inner.get_verbose_metadata("backend.stream_completed", request=request, metrics=None)
            self._emitter.emit(
                "backend.stream_completed",
                engine=self._inner.name,
                tokens_generated=tokens_generated,
                duration_ms=round(elapsed_ms, 1),
                tokens_per_second=round(tps, 1),
                **extra,
            )

    def warmup(self) -> None:
        self._inner.warmup()

    def list_models(self) -> list[str]:
        return self._inner.list_models()


def unwrap_backend(backend: InferenceBackend) -> InferenceBackend:
    """Unwrap InstrumentedBackend to get the inner backend for capability checks."""
    if isinstance(backend, InstrumentedBackend):
        return backend._inner
    return backend
