"""Tests for the Anthropic Messages API translation layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from octomil.serve_anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesBody,
    _anthropic_to_openai_messages,
    _stream_anthropic,
)

# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiMessages:
    def test_simple_string_content(self):
        body = AnthropicMessagesBody(
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        result = _anthropic_to_openai_messages(body)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_prompt_prepended(self):
        body = AnthropicMessagesBody(
            system="You are helpful.",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        result = _anthropic_to_openai_messages(body)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hi"}

    def test_content_blocks(self):
        body = AnthropicMessagesBody(
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicContentBlock(type="text", text="Part 1. "),
                        AnthropicContentBlock(type="text", text="Part 2."),
                    ],
                )
            ],
        )
        result = _anthropic_to_openai_messages(body)
        assert result == [{"role": "user", "content": "Part 1. Part 2."}]

    def test_mixed_content_blocks_skips_non_text(self):
        body = AnthropicMessagesBody(
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicContentBlock(type="text", text="Hello"),
                        AnthropicContentBlock(type="image", text=""),
                    ],
                )
            ],
        )
        result = _anthropic_to_openai_messages(body)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_multi_turn_conversation(self):
        body = AnthropicMessagesBody(
            messages=[
                AnthropicMessage(role="user", content="What is 2+2?"),
                AnthropicMessage(role="assistant", content="4"),
                AnthropicMessage(role="user", content="And 3+3?"),
            ],
        )
        result = _anthropic_to_openai_messages(body)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_empty_messages(self):
        body = AnthropicMessagesBody(messages=[])
        result = _anthropic_to_openai_messages(body)
        assert result == []

    def test_no_system(self):
        body = AnthropicMessagesBody(
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        result = _anthropic_to_openai_messages(body)
        assert len(result) == 1
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamAnthropic:
    @pytest.mark.asyncio
    async def test_stream_event_sequence(self):
        """Verify the correct SSE event sequence is emitted."""
        mock_chunk1 = MagicMock(text="Hello", finish_reason=None)
        mock_chunk2 = MagicMock(text=" world", finish_reason="stop")

        mock_backend = MagicMock()

        async def fake_stream(req):
            yield mock_chunk1
            yield mock_chunk2

        mock_backend.generate_stream = fake_stream

        state = MagicMock()
        state.backend = mock_backend

        gen_req = MagicMock()
        gen_req.model = "qwen-7b"

        events = []
        async for event in _stream_anthropic(state, gen_req, "msg_test123"):
            events.append(event)

        # Parse event types
        event_types = []
        for e in events:
            if e.startswith("event: "):
                event_type = e.split("\n")[0].replace("event: ", "")
                event_types.append(event_type)

        assert event_types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    @pytest.mark.asyncio
    async def test_stream_message_start_format(self):
        """Verify message_start event contains correct structure."""
        mock_backend = MagicMock()

        async def fake_stream(req):
            return
            yield  # make it an async generator

        mock_backend.generate_stream = fake_stream

        state = MagicMock()
        state.backend = mock_backend

        gen_req = MagicMock()
        gen_req.model = "qwen-7b"

        events = []
        async for event in _stream_anthropic(state, gen_req, "msg_abc"):
            events.append(event)

        # First event should be message_start
        first_data = json.loads(events[0].split("data: ")[1])
        assert first_data["type"] == "message_start"
        assert first_data["message"]["id"] == "msg_abc"
        assert first_data["message"]["role"] == "assistant"
        assert first_data["message"]["model"] == "qwen-7b"

    @pytest.mark.asyncio
    async def test_stream_delta_contains_text(self):
        """Verify content_block_delta events contain the text."""
        mock_chunk = MagicMock(text="Hi there")
        mock_backend = MagicMock()

        async def fake_stream(req):
            yield mock_chunk

        mock_backend.generate_stream = fake_stream

        state = MagicMock()
        state.backend = mock_backend

        gen_req = MagicMock()
        gen_req.model = "test"

        events = []
        async for event in _stream_anthropic(state, gen_req, "msg_x"):
            events.append(event)

        # Find the content_block_delta event
        delta_events = [e for e in events if "content_block_delta" in e]
        assert len(delta_events) == 1
        data = json.loads(delta_events[0].split("data: ")[1])
        assert data["delta"]["text"] == "Hi there"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestAnthropicModels:
    def test_messages_body_defaults(self):
        body = AnthropicMessagesBody(
            messages=[AnthropicMessage(role="user", content="test")],
        )
        assert body.max_tokens == 4096
        assert body.stream is False
        assert body.system is None
        assert body.temperature is None

    def test_content_block_defaults(self):
        block = AnthropicContentBlock(text="hello")
        assert block.type == "text"
