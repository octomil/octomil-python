"""Tests for WorkflowRunner."""

from __future__ import annotations

import pytest

from octomil.responses.responses import OctomilResponses
from octomil.responses.tools.executor import ToolExecutor, ToolResult
from octomil.responses.types import TextOutput
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeResponse,
    RuntimeToolCall,
)
from octomil.workflows import (
    InferenceStep,
    ToolRoundStep,
    TransformStep,
    Workflow,
    WorkflowRunner,
)


class EchoRuntime(ModelRuntime):
    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request):
        return RuntimeResponse(text=f"echo: {request.prompt[:30]}")

    async def stream(self, request):
        yield RuntimeChunk(text=f"echo: {request.prompt[:30]}")


class ToolCallingRuntime(ModelRuntime):
    """First call returns a tool call, second call returns text."""

    def __init__(self):
        self._call_count = 0

    @property
    def capabilities(self):
        return RuntimeCapabilities(supports_tool_calls=True)

    async def run(self, request):
        self._call_count += 1
        if self._call_count == 1:
            return RuntimeResponse(
                text="",
                tool_calls=[RuntimeToolCall(id="tc1", name="greet", arguments='{"name":"world"}')],
                finish_reason="tool_calls",
            )
        return RuntimeResponse(text="Hello world!")

    async def stream(self, request):
        yield RuntimeChunk(text="stream")


class StubExecutor(ToolExecutor):
    async def execute(self, call):
        return ToolResult(tool_call_id=call.id, content="executed")


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRuntimeRegistry.shared().clear()
    yield
    ModelRuntimeRegistry.shared().clear()


@pytest.mark.asyncio
async def test_single_inference_step():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: EchoRuntime()
    runner = WorkflowRunner(OctomilResponses())
    result = await runner.run(
        Workflow(name="test", steps=[InferenceStep(model="echo")]),
        input="hello",
    )
    assert len(result.outputs) == 1
    assert isinstance(result.outputs[0].output[0], TextOutput)


@pytest.mark.asyncio
async def test_multi_step_pipeline():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: EchoRuntime()
    runner = WorkflowRunner(OctomilResponses())
    result = await runner.run(
        Workflow(
            name="pipeline",
            steps=[
                InferenceStep(model="echo"),
                TransformStep(name="upper", transform=lambda text: text.upper()),
                InferenceStep(model="echo"),
            ],
        ),
        input="start",
    )
    assert len(result.outputs) == 2


@pytest.mark.asyncio
async def test_async_transform_step():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: EchoRuntime()
    runner = WorkflowRunner(OctomilResponses())

    async def async_upper(text: str) -> str:
        return text.upper()

    result = await runner.run(
        Workflow(
            name="async-transform",
            steps=[
                InferenceStep(model="echo"),
                TransformStep(name="upper", transform=async_upper),
                InferenceStep(model="echo"),
            ],
        ),
        input="start",
    )
    assert len(result.outputs) == 2


@pytest.mark.asyncio
async def test_tool_round_step():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: ToolCallingRuntime()
    runner = WorkflowRunner(OctomilResponses(), executor=StubExecutor())
    result = await runner.run(
        Workflow(
            name="tools",
            steps=[
                ToolRoundStep(
                    tools=[{"type": "function", "function": {"name": "greet", "description": "Greet"}}],
                    model="tool-model",
                ),
            ],
        ),
        input="greet someone",
    )
    assert len(result.outputs) == 1


@pytest.mark.asyncio
async def test_tool_round_step_without_executor_raises():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: EchoRuntime()
    runner = WorkflowRunner(OctomilResponses())
    with pytest.raises(RuntimeError, match="ToolExecutor required"):
        await runner.run(
            Workflow(
                name="no-executor",
                steps=[
                    ToolRoundStep(
                        tools=[{"type": "function", "function": {"name": "f", "description": "d"}}], model="m"
                    ),
                ],
            ),
            input="test",
        )


@pytest.mark.asyncio
async def test_empty_workflow():
    runner = WorkflowRunner(OctomilResponses())
    result = await runner.run(Workflow(name="empty", steps=[]), input="nothing")
    assert len(result.outputs) == 0
    assert result.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_workflow_result_latency():
    ModelRuntimeRegistry.shared().default_factory = lambda mid: EchoRuntime()
    runner = WorkflowRunner(OctomilResponses())
    result = await runner.run(
        Workflow(name="latency", steps=[InferenceStep(model="echo")]),
        input="test",
    )
    assert result.total_latency_ms > 0
