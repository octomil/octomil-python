"""Cloud inference backend — delegates to an OpenAI-compatible API via CloudClient."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from octomil.runtime.core.cloud_client import CloudClient

from ..types import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


class CloudInferenceBackend(InferenceBackend):
    """Serve-layer backend that proxies requests to a cloud provider.

    Reuses CloudClient for HTTP transport — same SSE parser, retry
    policy, and error model as CloudModelRuntime.
    """

    name = "cloud"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        super().__init__()
        self._cloud_client = CloudClient(base_url, api_key, model)
        self._model = model

    def load_model(self, model_name: str) -> None:
        # No-op for cloud — model is specified by cloud_model config
        logger.info("Cloud backend ready for model '%s' via %s", self._model, self._cloud_client._base_url)

    async def generate_async(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        """Async generation — safe to call from an already-running event loop."""
        messages = _to_openai_messages(request)
        result = await self._cloud_client.chat(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        choice = result.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content") or ""

        usage = result.get("usage", {})
        metrics = InferenceMetrics(
            prompt_tokens=usage.get("prompt_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=0.0,
        )
        return text, metrics

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        """Sync generation — safe for both running and non-running event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an async context (e.g. FastAPI handler).
            # Run in a worker thread to avoid "This event loop is already running".
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate_async(request)).result()

        return asyncio.run(self.generate_async(request))

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        messages = _to_openai_messages(request)
        async for chunk in self._cloud_client.chat_stream(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            text = delta.get("content") or ""
            if text or finish_reason:
                yield GenerationChunk(
                    text=text,
                    finish_reason=finish_reason,
                )

    def list_models(self) -> list[str]:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._cloud_client.list_models())
        except Exception:
            return [self._model]


def _to_openai_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    """Convert GenerationRequest messages to OpenAI format."""
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in request.messages]
