"""CloudModelRuntime — wraps cloud SSE inference as a ModelRuntime."""

from __future__ import annotations

from typing import AsyncIterator

from .model_runtime import ModelRuntime
from .types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse


class CloudModelRuntime(ModelRuntime):
    """ModelRuntime that delegates to a cloud inference endpoint.

    Uses octomil.streaming under the hood for SSE token streaming.
    """

    def __init__(self, server_url: str, api_key: str) -> None:
        self._server_url = server_url
        self._api_key = api_key

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_tool_calls=True,
            supports_structured_output=True,
            supports_streaming=True,
        )

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        from octomil.streaming import stream_inference

        tokens = []
        for token in stream_inference(
            server_url=self._server_url,
            api_key=self._api_key,
            model_id="default",
            input_data=request.prompt,
            parameters={"max_tokens": request.max_tokens, "temperature": request.temperature},
        ):
            tokens.append(token.token)
        return RuntimeResponse(text="".join(tokens), finish_reason="stop")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        from octomil.streaming import stream_inference_async

        async for token in stream_inference_async(
            server_url=self._server_url,
            api_key=self._api_key,
            model_id="default",
            input_data=request.prompt,
            parameters={"max_tokens": request.max_tokens, "temperature": request.temperature},
        ):
            yield RuntimeChunk(
                text=token.token if not token.done else None,
                finish_reason="stop" if token.done else None,
            )

    def close(self) -> None:
        pass
