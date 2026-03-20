"""Chat namespace -- ``client.chat.create()`` / ``client.chat.stream()``.

**Tier: Core Contract (MUST)**

Wraps the existing ``OctomilClient`` chat methods behind an OpenAI-style
``client.chat`` sub-API so callers can write::

    response = client.chat.create(model="phi-4-mini", messages=[...])
    async for chunk in client.chat.stream(model="phi-4-mini", messages=[...]):
        print(chunk)

Contract: ``chat.completions.create`` MUST delegate to ``responses.create``
internally, not to a direct HTTP call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from .client import OctomilClient


@dataclass
class ChatCompletion:
    """Result of a non-streaming chat completion."""

    message: dict[str, str]
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatChunk:
    """Single chunk from a streaming chat completion."""

    index: int
    content: str
    done: bool
    role: str = "assistant"


def _messages_to_input(messages: list[dict[str, str]]) -> list[Any]:
    """Convert OpenAI-style chat messages to Response API InputItems."""
    from .responses.types import AssistantInput, SystemInput, TextContent, UserInput

    items: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            items.append(SystemInput(content=content))
        elif role == "assistant":
            items.append(AssistantInput(content=[TextContent(text=content)]))
        else:
            items.append(UserInput(content=[TextContent(text=content)]))
    return items


class ChatClient:
    """OpenAI-style ``client.chat`` namespace.

    Exposes ``create()`` for non-streaming and ``stream()`` for
    streaming chat completions.  Both delegate to ``OctomilResponses``
    per the SDK contract -- NOT to direct HTTP calls.
    """

    def __init__(self, client: OctomilClient) -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> ChatCompletion:
        """Non-streaming chat completion.

        Delegates to ``OctomilResponses.create()`` per the SDK contract.

        Args:
            model: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds (reserved for future use).
            **parameters: Additional generation parameters (unused).

        Returns:
            :class:`ChatCompletion` with message, latency, and optional usage.
        """
        import time as _time

        from .responses.types import ResponseRequest, TextOutput

        input_items = _messages_to_input(messages)
        request = ResponseRequest(
            model=model,
            input=input_items,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        start = _time.monotonic()

        async def _run() -> Any:
            return await self._client.responses.create(request)

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(_run())
        finally:
            loop.close()

        latency_ms = (_time.monotonic() - start) * 1000

        text = "".join(item.text for item in response.output if isinstance(item, TextOutput))

        usage: dict[str, Any] = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatCompletion(
            message={"role": "assistant", "content": text},
            latency_ms=latency_ms,
            usage=usage,
        )

    async def stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> AsyncIterator[ChatChunk]:
        """Streaming chat completion.

        Delegates to ``OctomilResponses.stream()`` per the SDK contract.
        Yields :class:`ChatChunk` objects as they arrive.

        Args:
            model: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds (reserved for future use).
            **parameters: Additional generation parameters (unused).

        Yields:
            :class:`ChatChunk` with ``index``, ``content``, ``done``, ``role``.
        """
        from .responses.types import DoneEvent, ResponseRequest, TextDeltaEvent

        input_items = _messages_to_input(messages)
        request = ResponseRequest(
            model=model,
            input=input_items,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        idx = 0
        async for event in self._client.responses.stream(request):
            if isinstance(event, TextDeltaEvent):
                yield ChatChunk(
                    index=idx,
                    content=event.delta,
                    done=False,
                )
                idx += 1
            elif isinstance(event, DoneEvent):
                yield ChatChunk(
                    index=idx,
                    content="",
                    done=True,
                )
