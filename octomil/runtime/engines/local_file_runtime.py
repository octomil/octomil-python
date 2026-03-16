"""LocalFileModelRuntime — wraps a local model file on disk."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
)


class LocalFileModelRuntime(ModelRuntime):
    """ModelRuntime backed by a local file on disk.

    Delegates actual engine creation to EngineRegistry based on
    the file extension. The engine is lazily created on first use.
    """

    def __init__(
        self,
        model_id: str,
        file_path: Path,
        capabilities: Optional[RuntimeCapabilities] = None,
    ) -> None:
        self._model_id = model_id
        self._file_path = file_path
        self._caps = capabilities or RuntimeCapabilities(supports_streaming=True)
        self._engine: Any = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._caps

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        backend = self._resolve_engine()
        tokens: list[str] = []

        for chunk in backend.generate_stream(request.prompt, max_tokens=request.max_tokens):
            if isinstance(chunk, str):
                tokens.append(chunk)
            elif hasattr(chunk, "token"):
                tokens.append(chunk.token)

        text = "".join(tokens)
        prompt_tokens = len(request.prompt.split())
        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=RuntimeUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(tokens),
                total_tokens=prompt_tokens + len(tokens),
            ),
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        backend = self._resolve_engine()

        for chunk in backend.generate_stream(request.prompt, max_tokens=request.max_tokens):
            text = chunk if isinstance(chunk, str) else getattr(chunk, "token", str(chunk))
            yield RuntimeChunk(text=text)

    def close(self) -> None:
        with self._lock:
            self._engine = None

    def _resolve_engine(self) -> Any:
        with self._lock:
            if self._engine is not None:
                return self._engine

        from octomil.runtime.engines import get_registry

        registry = get_registry()
        engine, _ = registry.auto_select(self._model_id, n_tokens=0)
        backend = engine.create_backend(self._model_id, model_path=str(self._file_path))

        with self._lock:
            self._engine = backend
        return backend
