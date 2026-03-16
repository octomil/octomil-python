"""Regression tests for ChatClient delegation to OctomilResponses.

Contract: chat.completions.create MUST delegate to responses.create internally.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from octomil.chat_client import ChatClient, ChatCompletion
from octomil.responses.types import (
    DoneEvent,
    Response,
    ResponseUsage,
    TextDeltaEvent,
    TextOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client(responses_mock: Any) -> MagicMock:
    """Build a minimal OctomilClient mock with a .responses property."""
    client = MagicMock()
    client.responses = responses_mock
    return client


def _make_response(text: str = "hello world", usage: ResponseUsage | None = None) -> Response:
    return Response(
        id="resp_test123",
        model="test-model",
        output=[TextOutput(text=text)],
        finish_reason="stop",
        usage=usage,
    )


# ---------------------------------------------------------------------------
# ChatClient.create -- delegates to OctomilResponses.create
# ---------------------------------------------------------------------------


class TestChatClientCreate:
    def test_delegates_to_responses_create(self) -> None:
        """create() must call OctomilResponses.create, not _chat_create."""
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response("hi there"))
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chat.create("test-model", [{"role": "user", "content": "hello"}])

        responses_mock.create.assert_called_once()
        assert not client._chat_create.called

    def test_returns_chat_completion(self) -> None:
        """create() returns a ChatCompletion with correct message."""
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response("hi there"))
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        result = chat.create("test-model", [{"role": "user", "content": "hello"}])

        assert isinstance(result, ChatCompletion)
        assert result.message == {"role": "assistant", "content": "hi there"}

    def test_passes_model_to_response_request(self) -> None:
        """create() passes the model name in the ResponseRequest."""
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response())
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chat.create("phi-4-mini", [{"role": "user", "content": "test"}])

        call_args = responses_mock.create.call_args
        request = call_args[0][0]
        assert request.model == "phi-4-mini"

    def test_converts_messages_to_input_items(self) -> None:
        """create() converts chat messages to ResponseRequest input items."""
        from octomil.responses.types import SystemInput, UserInput

        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response())
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chat.create(
            "test-model",
            [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ],
        )

        request = responses_mock.create.call_args[0][0]
        assert len(request.input) == 2
        assert isinstance(request.input[0], SystemInput)
        assert request.input[0].content == "you are helpful"
        assert isinstance(request.input[1], UserInput)

    def test_passes_temperature_and_max_tokens(self) -> None:
        """create() forwards temperature and max_tokens to ResponseRequest."""
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response())
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chat.create("test-model", [{"role": "user", "content": "test"}], temperature=0.3, max_tokens=256)

        request = responses_mock.create.call_args[0][0]
        assert request.temperature == 0.3
        assert request.max_output_tokens == 256

    def test_usage_mapped_to_dict(self) -> None:
        """create() maps ResponseUsage to usage dict in ChatCompletion."""
        usage = ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response("ok", usage=usage))
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        result = chat.create("test-model", [{"role": "user", "content": "x"}])

        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_latency_ms_is_positive(self) -> None:
        """create() populates latency_ms with a non-negative value."""
        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=_make_response())
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        result = chat.create("test-model", [{"role": "user", "content": "hi"}])

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# ChatClient.stream -- delegates to OctomilResponses.stream
# ---------------------------------------------------------------------------


class TestChatClientStream:
    @pytest.mark.asyncio
    async def test_delegates_to_responses_stream(self) -> None:
        """stream() must call OctomilResponses.stream, not _chat_stream."""

        async def _fake_stream(request: Any) -> AsyncIterator:  # type: ignore[return]
            yield TextDeltaEvent(delta="hello")
            yield TextDeltaEvent(delta=" world")
            yield DoneEvent(response=_make_response("hello world"))

        responses_mock = MagicMock()
        responses_mock.stream = _fake_stream
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chunks = []
        async for chunk in chat.stream("test-model", [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert not client._chat_stream.called
        # Two text chunks + one done chunk
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_yields_chat_chunks_from_text_deltas(self) -> None:
        """stream() yields ChatChunk for each TextDeltaEvent."""

        async def _fake_stream(request: Any) -> AsyncIterator:  # type: ignore[return]
            yield TextDeltaEvent(delta="hello")
            yield TextDeltaEvent(delta=" world")
            yield DoneEvent(response=_make_response("hello world"))

        responses_mock = MagicMock()
        responses_mock.stream = _fake_stream
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chunks = [c async for c in chat.stream("test-model", [{"role": "user", "content": "hi"}])]

        text_chunks = [c for c in chunks if not c.done]
        assert len(text_chunks) == 2
        assert text_chunks[0].content == "hello"
        assert text_chunks[1].content == " world"
        assert text_chunks[0].done is False

    @pytest.mark.asyncio
    async def test_done_chunk_emitted_on_done_event(self) -> None:
        """stream() yields a ChatChunk with done=True for the DoneEvent."""

        async def _fake_stream(request: Any) -> AsyncIterator:  # type: ignore[return]
            yield DoneEvent(response=_make_response(""))

        responses_mock = MagicMock()
        responses_mock.stream = _fake_stream
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chunks = [c async for c in chat.stream("test-model", [{"role": "user", "content": "hi"}])]

        assert len(chunks) == 1
        assert chunks[0].done is True

    @pytest.mark.asyncio
    async def test_passes_model_in_request(self) -> None:
        """stream() passes the model name in the ResponseRequest."""
        captured_request: dict[str, Any] = {}

        async def _fake_stream(request: Any) -> AsyncIterator:  # type: ignore[return]
            captured_request["req"] = request
            yield DoneEvent(response=_make_response(""))

        responses_mock = MagicMock()
        responses_mock.stream = _fake_stream
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        async for _ in chat.stream("phi-4-mini", [{"role": "user", "content": "test"}]):
            pass

        assert captured_request["req"].model == "phi-4-mini"
        assert captured_request["req"].stream is True

    @pytest.mark.asyncio
    async def test_index_increments_per_chunk(self) -> None:
        """stream() increments the chunk index for each text delta."""

        async def _fake_stream(request: Any) -> AsyncIterator:  # type: ignore[return]
            yield TextDeltaEvent(delta="a")
            yield TextDeltaEvent(delta="b")
            yield TextDeltaEvent(delta="c")
            yield DoneEvent(response=_make_response("abc"))

        responses_mock = MagicMock()
        responses_mock.stream = _fake_stream
        client = _make_mock_client(responses_mock)

        chat = ChatClient(client)
        chunks = [c async for c in chat.stream("m", [{"role": "user", "content": "x"}])]

        text_chunks = [c for c in chunks if not c.done]
        assert [c.index for c in text_chunks] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Backward compat: __call__ still routes through _chat_create
# ---------------------------------------------------------------------------


class TestChatClientBackwardCompat:
    def test_call_still_uses_chat_create(self) -> None:
        """client.chat(model, messages) still calls _chat_create for compat."""
        responses_mock = MagicMock()
        client = _make_mock_client(responses_mock)
        client._chat_create.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "latency_ms": 10.0,
        }

        chat = ChatClient(client)
        result = chat("test-model", [{"role": "user", "content": "hi"}])

        client._chat_create.assert_called_once()
        responses_mock.create.assert_not_called()
        assert result["message"]["content"] == "ok"
