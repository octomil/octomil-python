"""Integration tests for responses.create() with registered runtimes."""

from __future__ import annotations

import pytest

from octomil.responses.responses import OctomilResponses
from octomil.responses.runtime.model_runtime import ModelRuntime
from octomil.responses.runtime.registry import ModelRuntimeRegistry
from octomil.responses.runtime.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeResponse,
)
from octomil.responses.types import (
    ResponseRequest,
    TextOutput,
    text_input,
)


class MockRuntime(ModelRuntime):
    def __init__(self):
        self.last_prompt = None

    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request):
        self.last_prompt = request.prompt
        return RuntimeResponse(text=f"Reply to: {request.prompt[:50]}")

    async def stream(self, request):
        self.last_prompt = request.prompt
        yield RuntimeChunk(text="chunk1")
        yield RuntimeChunk(text="chunk2", finish_reason="stop")


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRuntimeRegistry.shared().clear()
    yield
    ModelRuntimeRegistry.shared().clear()


@pytest.mark.asyncio
async def test_create_with_default_factory():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    response = await responses.create(ResponseRequest(model="test", input=[text_input("Hello")]))
    assert len(response.output) > 0
    assert isinstance(response.output[0], TextOutput)


@pytest.mark.asyncio
async def test_stream_with_default_factory():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    events = []
    async for event in responses.stream(ResponseRequest(model="test", input=[text_input("Hello")])):
        events.append(event)
    assert len(events) >= 2  # at least chunks + done


@pytest.mark.asyncio
async def test_string_input_shorthand():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    response = await responses.create(ResponseRequest(model="test", input="Hello string"))
    assert isinstance(response.output[0], TextOutput)


@pytest.mark.asyncio
async def test_instructions_prepended():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    await responses.create(
        ResponseRequest(model="test", input=[text_input("question")], instructions="You are helpful")
    )
    assert "<|system|>" in mock.last_prompt
    assert "You are helpful" in mock.last_prompt


@pytest.mark.asyncio
async def test_previous_response_id_chains():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    r1 = await responses.create(ResponseRequest(model="test", input=[text_input("first")]))
    await responses.create(ResponseRequest(model="test", input=[text_input("second")], previous_response_id=r1.id))
    # The second prompt should contain the first response's text
    assert "Reply to:" in mock.last_prompt


@pytest.mark.asyncio
async def test_text_convenience_constructor():
    mock = MockRuntime()
    ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock
    responses = OctomilResponses()
    request = ResponseRequest.text("test", "Hello convenience")
    response = await responses.create(request)
    assert isinstance(response.output[0], TextOutput)
