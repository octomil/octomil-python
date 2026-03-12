"""InferenceBackendAdapter — bridges existing InferenceBackend to ModelRuntime."""

from __future__ import annotations

from typing import AsyncIterator

from .model_runtime import ModelRuntime
from .types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
)


class InferenceBackendAdapter(ModelRuntime):
    """Bridges an existing InferenceBackend to the ModelRuntime protocol.

    Zero changes to existing backend implementations.
    """

    def __init__(
        self,
        backend: object,
        model_name: str,
        capabilities: RuntimeCapabilities | None = None,
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._capabilities = capabilities or RuntimeCapabilities(
            supports_tool_calls=False,
            supports_structured_output=False,
            supports_multimodal_input=False,
            supports_streaming=True,
        )

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        gen_request = self._to_generation_request(request)
        text, metrics = self._backend.generate(gen_request)  # type: ignore[attr-defined]
        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=RuntimeUsage(
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.total_tokens - metrics.prompt_tokens,
                total_tokens=metrics.total_tokens,
            ),
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:  # type: ignore[override]
        gen_request = self._to_generation_request(request)
        async for chunk in self._backend.generate_stream(gen_request):  # type: ignore[attr-defined]
            yield RuntimeChunk(
                text=chunk.text if chunk.text else None,
                finish_reason=chunk.finish_reason,
            )

    def close(self) -> None:
        pass

    def _to_generation_request(self, request: RuntimeRequest) -> object:
        from octomil.serve import GenerationRequest

        return GenerationRequest(
            model=self._model_name,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
