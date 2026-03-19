"""CloudModelRuntime — wraps cloud SSE inference as a ModelRuntime."""

from __future__ import annotations

from typing import AsyncIterator

from octomil.runtime.core.chatml_renderer import render_chatml
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse


class CloudModelRuntime(ModelRuntime):
    """ModelRuntime that delegates to a cloud inference endpoint.

    Uses octomil.streaming under the hood for SSE token streaming.
    """

    def __init__(self, server_url: str, api_key: str) -> None:
        self._server_url = server_url
        self._api_key = api_key

    @property
    def capabilities(self) -> RuntimeCapabilities:
        from octomil.runtime.core.types import ToolCallTier

        return RuntimeCapabilities(
            tool_call_tier=ToolCallTier.NATIVE,
            supports_structured_output=True,
            supports_streaming=True,
        )

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        from octomil.streaming import stream_inference

        prompt = render_chatml(request)
        gc = request.generation_config
        tokens = []
        for token in stream_inference(
            server_url=self._server_url,
            api_key=self._api_key,
            model_id="default",
            input_data=prompt,
            parameters={"max_tokens": gc.max_tokens, "temperature": gc.temperature},
        ):
            tokens.append(token.token)
        return RuntimeResponse(text="".join(tokens), finish_reason="stop")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        from octomil.streaming import stream_inference_async

        prompt = render_chatml(request)
        gc = request.generation_config
        async for token in stream_inference_async(
            server_url=self._server_url,
            api_key=self._api_key,
            model_id="default",
            input_data=prompt,
            parameters={"max_tokens": gc.max_tokens, "temperature": gc.temperature},
        ):
            yield RuntimeChunk(
                text=token.token if not token.done else None,
                finish_reason="stop" if token.done else None,
            )

    def close(self) -> None:
        pass
