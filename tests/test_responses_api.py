"""Tests for OctomilResponses (Layer 2 create + stream)."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from octomil.responses import OctomilResponses
from octomil.responses.runtime import (
    ModelRuntime,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
    RuntimeUsage,
)
from octomil.responses.types import (
    DoneEvent,
    ResponseRequest,
    TextDeltaEvent,
    TextOutput,
    ToolCallOutput,
    text_input,
)


class MockRuntime(ModelRuntime):
    def __init__(self, response: RuntimeResponse) -> None:
        self._response = response

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return self._response

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        return
        yield  # pragma: no cover


class StreamingMockRuntime(ModelRuntime):
    def __init__(self, chunks: list[RuntimeChunk]) -> None:
        self._chunks = chunks

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text="")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_create_returns_text():
    runtime = MockRuntime(RuntimeResponse(text="Hello world"))
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    response = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))

    assert len(response.output) == 1
    assert isinstance(response.output[0], TextOutput)
    assert response.output[0].text == "Hello world"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_create_returns_tool_calls():
    runtime = MockRuntime(
        RuntimeResponse(
            text="",
            tool_calls=[RuntimeToolCall(id="call_1", name="get_weather", arguments='{"city":"NYC"}')],
        )
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    response = await responses.create(ResponseRequest(model="test", input=[text_input("Weather?")]))

    tool_calls = [i for i in response.output if isinstance(i, ToolCallOutput)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call.name == "get_weather"
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_stream_emits_text_and_done():
    runtime = StreamingMockRuntime(
        [
            RuntimeChunk(text="Hello"),
            RuntimeChunk(text=" world"),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    events = []
    async for event in responses.stream(ResponseRequest(model="test", input=[text_input("Hi")])):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_deltas) == 2
    assert text_deltas[0].delta == "Hello"
    assert text_deltas[1].delta == " world"

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_create_includes_usage():
    runtime = MockRuntime(
        RuntimeResponse(
            text="result",
            usage=RuntimeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    response = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))

    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_create_throws_when_no_runtime():
    responses = OctomilResponses(runtime_resolver=lambda _: None)
    with pytest.raises(RuntimeError, match="No ModelRuntime"):
        await responses.create(ResponseRequest(model="unknown", input=[text_input("Hi")]))
