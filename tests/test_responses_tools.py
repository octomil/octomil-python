"""Tests for ToolRunner (Layer 3)."""

from __future__ import annotations

import uuid
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
)
from octomil.responses.tools import ToolExecutor, ToolResult, ToolRunner
from octomil.responses.types import (
    ResponseRequest,
    TextOutput,
    text_input,
)
from octomil.responses.types import (
    ResponseToolCall as RTC,
)


class SequentialRuntime(ModelRuntime):
    def __init__(self, responses: list[RuntimeResponse]) -> None:
        self._responses = iter(responses)

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return next(self._responses, RuntimeResponse(text=""))

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        return
        yield  # pragma: no cover


class AlwaysToolCallRuntime(ModelRuntime):
    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(
            text="",
            tool_calls=[
                RuntimeToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name="loop",
                    arguments="{}",
                )
            ],
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        return
        yield  # pragma: no cover


class CountingExecutor(ToolExecutor):
    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, call: RTC) -> ToolResult:
        self.call_count += 1
        return ToolResult(tool_call_id=call.id, content="ok")


class MapExecutor(ToolExecutor):
    def __init__(self, results: dict[str, str]) -> None:
        self._results = results

    async def execute(self, call: RTC) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            content=self._results.get(call.name, "unknown"),
        )


class FailingExecutor(ToolExecutor):
    async def execute(self, call: RTC) -> ToolResult:
        raise RuntimeError("Network error")


@pytest.mark.asyncio
async def test_returns_immediately_no_tool_calls():
    runtime = SequentialRuntime([RuntimeResponse(text="Hello world")])
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    executor = CountingExecutor()
    runner = ToolRunner(responses, executor)

    response = await runner.run(ResponseRequest(model="test", input=[text_input("Hi")]))

    assert isinstance(response.output[0], TextOutput)
    assert response.output[0].text == "Hello world"
    assert executor.call_count == 0


@pytest.mark.asyncio
async def test_executes_tool_and_feeds_back():
    runtime = SequentialRuntime(
        [
            RuntimeResponse(
                text="",
                tool_calls=[
                    RuntimeToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments='{"city":"NYC"}',
                    )
                ],
            ),
            RuntimeResponse(text="It's 72\u00b0F in NYC"),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    executor = MapExecutor({"get_weather": "72\u00b0F, sunny"})
    runner = ToolRunner(responses, executor)

    response = await runner.run(ResponseRequest(model="test", input=[text_input("Weather?")]))

    texts = [i.text for i in response.output if isinstance(i, TextOutput)]
    assert "".join(texts) == "It's 72\u00b0F in NYC"


@pytest.mark.asyncio
async def test_respects_max_iterations():
    runtime = AlwaysToolCallRuntime()
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    executor = CountingExecutor()
    runner = ToolRunner(responses, executor, max_iterations=3)

    await runner.run(ResponseRequest(model="test", input=[text_input("Loop")]))

    assert executor.call_count == 3


@pytest.mark.asyncio
async def test_handles_tool_error():
    runtime = SequentialRuntime(
        [
            RuntimeResponse(
                text="",
                tool_calls=[RuntimeToolCall(id="call_1", name="failing_tool", arguments="{}")],
            ),
            RuntimeResponse(text="Sorry, that didn't work"),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    executor = FailingExecutor()
    runner = ToolRunner(responses, executor)

    response = await runner.run(ResponseRequest(model="test", input=[text_input("Try this")]))

    texts = [i.text for i in response.output if isinstance(i, TextOutput)]
    assert "".join(texts) == "Sorry, that didn't work"
