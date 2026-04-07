"""Tests for OctomilResponses (Layer 2 create + stream)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from octomil.responses import OctomilResponses
from octomil.responses.types import (
    DoneEvent,
    ResponseRequest,
    TextDeltaEvent,
    TextOutput,
    ToolCallOutput,
    text_input,
)
from octomil.runtime.core import (
    ModelRuntime,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
    RuntimeUsage,
)
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.router import RouterModelRuntime


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
    with pytest.raises((RuntimeError, Exception)):
        await responses.create(ResponseRequest(model="unknown", input=[text_input("Hi")]))


@pytest.mark.asyncio
async def test_tool_schema_serialized_with_json_dumps():
    """Verify that tool schemas are serialized with json.dumps, not str()."""
    captured_request: list[RuntimeRequest] = []

    class CapturingRuntime(ModelRuntime):
        @property
        def capabilities(self) -> RuntimeCapabilities:
            return RuntimeCapabilities()

        async def run(self, request: RuntimeRequest) -> RuntimeResponse:
            captured_request.append(request)
            return RuntimeResponse(text="ok")

        async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
            return
            yield  # pragma: no cover

    responses = OctomilResponses(runtime_resolver=lambda _: CapturingRuntime())
    await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            tools=[
                {
                    "name": "fn",
                    "description": "desc",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                }
            ],
        )
    )

    assert len(captured_request) == 1
    tool_def = captured_request[0].tool_definitions[0]
    # Must be valid JSON (json.dumps output), not Python repr (str() output)
    parsed = json.loads(tool_def.parameters_schema)
    assert parsed["type"] == "object"
    assert "properties" in parsed


@pytest.mark.asyncio
async def test_input_schema_preferred_over_parameters():
    """input_schema takes precedence when both are present."""
    captured_request: list[RuntimeRequest] = []

    class CapturingRuntime(ModelRuntime):
        @property
        def capabilities(self) -> RuntimeCapabilities:
            return RuntimeCapabilities()

        async def run(self, request: RuntimeRequest) -> RuntimeResponse:
            captured_request.append(request)
            return RuntimeResponse(text="ok")

        async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
            return
            yield  # pragma: no cover

    responses = OctomilResponses(runtime_resolver=lambda _: CapturingRuntime())
    await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            tools=[
                {
                    "name": "fn",
                    "description": "desc",
                    "parameters": {"type": "object", "properties": {"old": {"type": "string"}}},
                    "input_schema": {"type": "object", "properties": {"new": {"type": "string"}}},
                }
            ],
        )
    )

    tool_def = captured_request[0].tool_definitions[0]
    parsed = json.loads(tool_def.parameters_schema)
    assert "new" in parsed["properties"]
    assert "old" not in parsed["properties"]


# ---------------------------------------------------------------------------
# default_routing_policy integration
# ---------------------------------------------------------------------------


class StubRuntime(ModelRuntime):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text=f"from-{self.name}")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text=f"from-{self.name}")


def _make_router(*, prefer_local: bool = True) -> RouterModelRuntime:
    return RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )


@pytest.mark.asyncio
async def test_default_routing_policy_quality_routes_to_cloud():
    """default_routing_policy with quality (prefer_local=False) routes to cloud."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=RoutingPolicy.auto(prefer_local=False),
    )

    result = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result.output[0].text == "from-cloud"
    assert result.locality == "cloud"


@pytest.mark.asyncio
async def test_default_routing_policy_balanced_routes_to_local():
    """default_routing_policy with balanced (prefer_local=True) routes to local."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=RoutingPolicy.auto(prefer_local=True),
    )

    result = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result.output[0].text == "from-local"
    assert result.locality == "on_device"


@pytest.mark.asyncio
async def test_metadata_routing_overrides_default():
    """Per-request metadata routing takes precedence over default_routing_policy."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=RoutingPolicy.auto(prefer_local=True),
    )

    # Metadata says cloud_only — should override default (balanced/local)
    result = await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            metadata={"routing.policy": "cloud_only"},
        )
    )
    assert result.output[0].text == "from-cloud"
    assert result.locality == "cloud"


@pytest.mark.asyncio
async def test_default_routing_policy_local_only():
    """default_routing_policy=local_only prevents cloud fallback."""
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=RoutingPolicy.local_only(),
    )

    result = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result.output[0].text == "from-local"


@pytest.mark.asyncio
async def test_default_routing_policy_stream():
    """default_routing_policy works with stream() too."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=RoutingPolicy.auto(prefer_local=False),
    )

    events = []
    async for event in responses.stream(ResponseRequest(model="test", input=[text_input("Hi")])):
        events.append(event)

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].response.locality == "cloud"


@pytest.mark.asyncio
async def test_from_desired_state_entry_to_responses():
    """End-to-end: from_desired_state_entry → OctomilResponses with RouterModelRuntime."""
    entry = {"routing_preference": "quality", "cloud_fallback": {"enabled": True}}
    policy = RoutingPolicy.from_desired_state_entry(entry)
    assert policy is not None
    assert policy.prefer_local is False

    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=policy,
    )

    result = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result.output[0].text == "from-cloud"
    assert result.locality == "cloud"


@pytest.mark.asyncio
async def test_from_desired_state_entry_local_first():
    """End-to-end: local_first preset routes to local."""
    entry = {"routing_preference": "local", "cloud_fallback": {"enabled": True}}
    policy = RoutingPolicy.from_desired_state_entry(entry)
    assert policy is not None

    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        default_routing_policy=policy,
    )

    result = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result.output[0].text == "from-local"
    assert result.locality == "on_device"


@pytest.mark.asyncio
async def test_per_deployment_routing_policy():
    """Per-deployment routing resolves by metadata deployment_id."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        routing_policies={
            "dep_local": RoutingPolicy.local_only(),
            "dep_cloud": RoutingPolicy.auto(prefer_local=False),
        },
        default_routing_policy=RoutingPolicy.auto(prefer_local=True),
    )

    # Request with dep_cloud deployment_id → cloud routing
    result_cloud = await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            metadata={"deployment_id": "dep_cloud"},
        )
    )
    assert result_cloud.output[0].text == "from-cloud"
    assert result_cloud.locality == "cloud"

    # Request with dep_local deployment_id → local routing
    result_local = await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            metadata={"deployment_id": "dep_local"},
        )
    )
    assert result_local.output[0].text == "from-local"
    assert result_local.locality == "on_device"

    # Request with no deployment_id → default (balanced/local)
    result_default = await responses.create(ResponseRequest(model="test", input=[text_input("Hi")]))
    assert result_default.output[0].text == "from-local"
    assert result_default.locality == "on_device"

    # Request with unknown deployment_id → default fallback
    result_unknown = await responses.create(
        ResponseRequest(
            model="test",
            input=[text_input("Hi")],
            metadata={"deployment_id": "dep_unknown"},
        )
    )
    assert result_unknown.output[0].text == "from-local"
    assert result_unknown.locality == "on_device"


@pytest.mark.asyncio
async def test_model_based_auto_routing():
    """Model name automatically resolves to deployment routing — no metadata needed."""
    router = _make_router()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        routing_policies={
            "dep_local": RoutingPolicy.local_only(),
            "dep_cloud": RoutingPolicy.auto(prefer_local=False),
        },
        model_deployment_map={
            "chat-model": "dep_cloud",
            "private-model": "dep_local",
        },
        default_routing_policy=RoutingPolicy.auto(prefer_local=True),
    )

    # "chat-model" → dep_cloud → cloud routing (no metadata)
    result_cloud = await responses.create(ResponseRequest(model="chat-model", input=[text_input("Hi")]))
    assert result_cloud.output[0].text == "from-cloud"
    assert result_cloud.locality == "cloud"

    # "private-model" → dep_local → local routing (no metadata)
    result_local = await responses.create(ResponseRequest(model="private-model", input=[text_input("Hi")]))
    assert result_local.output[0].text == "from-local"
    assert result_local.locality == "on_device"

    # Unknown model → default fallback (balanced/local)
    result_default = await responses.create(ResponseRequest(model="unknown-model", input=[text_input("Hi")]))
    assert result_default.output[0].text == "from-local"
    assert result_default.locality == "on_device"

    # Explicit metadata still takes precedence over model-based lookup
    result_override = await responses.create(
        ResponseRequest(
            model="private-model",
            input=[text_input("Hi")],
            metadata={"routing.policy": "cloud_only"},
        )
    )
    assert result_override.output[0].text == "from-cloud"
    assert result_override.locality == "cloud"


@pytest.mark.asyncio
async def test_ambiguous_model_falls_through_to_default():
    """Model not in model_deployment_map falls through to default routing."""
    router = _make_router()
    # Simulate: "shared-model" was excluded from model map due to ambiguity,
    # but per-deployment policies still exist for explicit lookup.
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        routing_policies={
            "dep_a": RoutingPolicy.local_only(),
            "dep_b": RoutingPolicy.auto(prefer_local=False),
        },
        model_deployment_map={},  # shared-model excluded (ambiguous)
        default_routing_policy=RoutingPolicy.auto(prefer_local=True),
    )

    # No model map entry → falls through to default (balanced/local)
    result = await responses.create(ResponseRequest(model="shared-model", input=[text_input("Hi")]))
    assert result.output[0].text == "from-local"
    assert result.locality == "on_device"

    # Explicit deployment_id still works for disambiguation
    result_a = await responses.create(
        ResponseRequest(
            model="shared-model",
            input=[text_input("Hi")],
            metadata={"deployment_id": "dep_a"},
        )
    )
    assert result_a.output[0].text == "from-local"  # local_only
    assert result_a.locality == "on_device"

    result_b = await responses.create(
        ResponseRequest(
            model="shared-model",
            input=[text_input("Hi")],
            metadata={"deployment_id": "dep_b"},
        )
    )
    assert result_b.output[0].text == "from-cloud"  # quality/cloud
    assert result_b.locality == "cloud"
