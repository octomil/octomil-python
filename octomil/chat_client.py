"""Chat namespace -- ``client.chat.create()`` / ``client.chat.stream()``.

**Tier: Core Contract (MUST)**

Wraps the existing ``OctomilClient`` chat methods behind an OpenAI-style
``client.chat`` sub-API so callers can write::

    response = client.chat.create(model="phi-4-mini", messages=[...])
    async for chunk in client.chat.stream(model="phi-4-mini", messages=[...]):
        print(chunk)

Backward compatibility: ``client.chat(...)`` still works because
``ChatClient.__call__`` delegates to ``create()`` and returns a raw dict.
"""

from __future__ import annotations

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


class ChatClient:
    """OpenAI-style ``client.chat`` namespace.

    Exposes ``create()`` for non-streaming and ``stream()`` for
    streaming chat completions.  Both delegate to the parent
    ``OctomilClient``'s internal ``_chat_create`` / ``_chat_stream``
    methods.

    Also callable directly (``client.chat(...)``) for backward
    compatibility — returns the raw dict that the old ``chat()``
    method returned.
    """

    def __init__(self, client: OctomilClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Backward-compat: client.chat(model_id, messages, ...) still works
    # ------------------------------------------------------------------

    def __call__(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Backward-compatible callable.

        Returns the same raw dict that the old ``OctomilClient.chat()``
        returned, so existing code like ``result = client.chat(...)``
        continues to work.
        """
        return self._client._chat_create(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            **parameters,
        )

    # ------------------------------------------------------------------
    # New facade API
    # ------------------------------------------------------------------

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

        Args:
            model: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Returns:
            :class:`ChatCompletion` with message, latency, and optional usage.
        """
        result = self._client._chat_create(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            **parameters,
        )
        return ChatCompletion(
            message=result.get("message", {"role": "assistant", "content": ""}),
            latency_ms=result.get("latency_ms", 0.0),
            usage=result.get("usage", {}),
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

        Yields :class:`ChatChunk` objects as they arrive.

        Args:
            model: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Yields:
            :class:`ChatChunk` with ``index``, ``content``, ``done``, ``role``.
        """
        async for raw_chunk in self._client._chat_stream(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            **parameters,
        ):
            yield ChatChunk(
                index=raw_chunk.get("index", 0),
                content=raw_chunk.get("content", ""),
                done=raw_chunk.get("done", False),
                role=raw_chunk.get("role", "assistant"),
            )
