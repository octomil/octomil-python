"""Tests for locality telemetry in OctomilResponses and RouterModelRuntime."""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from octomil.responses import OctomilResponses
from octomil.responses.types import DoneEvent, ResponseRequest, text_input
from octomil.runtime.core import (
    ModelRuntime,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)
from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.cloud_runtime import CloudModelRuntime
from octomil.runtime.core.router import LOCALITY_CLOUD, LOCALITY_ON_DEVICE, RouterModelRuntime
from octomil.runtime.planner.schemas import (
    AppResolution,
    RuntimeCandidatePlan,
    RuntimeSelection,
)


class _StubRuntime(ModelRuntime):
    def __init__(self, name: str = "stub") -> None:
        self.name = name

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text=f"from-{self.name}")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text=f"from-{self.name}")


# ---------------------------------------------------------------------------
# _determine_locality helper tests
# ---------------------------------------------------------------------------


def test_locality_cloud_runtime():
    from octomil.responses.responses import _determine_locality

    runtime = MagicMock(spec=CloudModelRuntime)
    locality, is_fallback = _determine_locality(runtime, "some-model")
    assert locality == LOCALITY_CLOUD
    assert is_fallback is False


def test_locality_inference_backend_adapter():
    from octomil.responses.responses import _determine_locality

    runtime = MagicMock(spec=InferenceBackendAdapter)
    locality, is_fallback = _determine_locality(runtime, "some-model")
    assert locality == LOCALITY_ON_DEVICE
    assert is_fallback is False


def test_locality_router_local():
    from octomil.responses.responses import _determine_locality

    router = RouterModelRuntime(
        local_factory=lambda mid: _StubRuntime("local"),
        cloud_factory=lambda mid: _StubRuntime("cloud"),
    )
    locality, is_fallback = _determine_locality(router, "some-model")
    assert locality == LOCALITY_ON_DEVICE
    assert is_fallback is False


def test_locality_router_cloud_fallback():
    from octomil.responses.responses import _determine_locality

    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: _StubRuntime("cloud"),
    )
    locality, is_fallback = _determine_locality(router, "some-model")
    assert locality == LOCALITY_CLOUD
    assert is_fallback is True


def test_locality_unknown_runtime_defaults_on_device():
    from octomil.responses.responses import _determine_locality

    locality, is_fallback = _determine_locality(_StubRuntime(), "some-model")
    assert locality == LOCALITY_ON_DEVICE
    assert is_fallback is False


# ---------------------------------------------------------------------------
# OctomilResponses.create() locality propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sets_locality_on_device():
    runtime = _StubRuntime("local")
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    response = await responses.create(ResponseRequest(model="test", input=[text_input("hi")]))
    assert response.locality == LOCALITY_ON_DEVICE


@pytest.mark.asyncio
async def test_create_sets_locality_cloud():
    cloud = MagicMock(spec=CloudModelRuntime)
    cloud.run = _StubRuntime("cloud").run
    cloud.capabilities = RuntimeCapabilities()
    responses = OctomilResponses(runtime_resolver=lambda _: cloud)
    response = await responses.create(ResponseRequest(model="test", input=[text_input("hi")]))
    assert response.locality == LOCALITY_CLOUD


@pytest.mark.asyncio
async def test_create_emits_fallback_cloud_telemetry():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: _StubRuntime("cloud"),
    )
    mock_telemetry = MagicMock()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        telemetry_reporter=mock_telemetry,
    )
    await responses.create(ResponseRequest(model="mymodel", input=[text_input("hi")]))
    mock_telemetry.report_fallback_cloud.assert_called_once_with(
        model_id="mymodel",
        fallback_reason="local_unavailable",
    )


@pytest.mark.asyncio
async def test_create_no_fallback_telemetry_when_local_available():
    router = RouterModelRuntime(
        local_factory=lambda mid: _StubRuntime("local"),
        cloud_factory=lambda mid: _StubRuntime("cloud"),
    )
    mock_telemetry = MagicMock()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        telemetry_reporter=mock_telemetry,
    )
    await responses.create(ResponseRequest(model="mymodel", input=[text_input("hi")]))
    mock_telemetry.report_fallback_cloud.assert_not_called()


# ---------------------------------------------------------------------------
# OctomilResponses.stream() locality propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_sets_locality_in_done_event():
    runtime = _StubRuntime("local")
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)
    done_event = None
    async for event in responses.stream(ResponseRequest(model="test", input=[text_input("hi")])):
        if isinstance(event, DoneEvent):
            done_event = event
    assert done_event is not None
    assert done_event.response.locality == LOCALITY_ON_DEVICE


@pytest.mark.asyncio
async def test_stream_emits_fallback_cloud_telemetry():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: _StubRuntime("cloud"),
    )
    mock_telemetry = MagicMock()
    responses = OctomilResponses(
        runtime_resolver=lambda _: router,
        telemetry_reporter=mock_telemetry,
    )
    async for _ in responses.stream(ResponseRequest(model="mymodel", input=[text_input("hi")])):
        pass
    mock_telemetry.report_fallback_cloud.assert_called_once_with(
        model_id="mymodel",
        fallback_reason="local_unavailable",
    )


# ---------------------------------------------------------------------------
# TelemetryReporter.report_fallback_cloud
# ---------------------------------------------------------------------------


def test_report_fallback_cloud_enqueues_event():
    from octomil.telemetry import TelemetryReporter

    reporter = TelemetryReporter(api_key="test-key")
    reporter.close()  # drain before testing
    # Re-create to capture enqueue
    reporter2 = TelemetryReporter(api_key="test-key")
    captured: list = []
    original_enqueue = reporter2._enqueue

    def capture(**kwargs):
        captured.append(kwargs)
        original_enqueue(**kwargs)

    reporter2._enqueue = capture  # type: ignore[method-assign]
    reporter2.report_fallback_cloud(model_id="llama-3", fallback_reason="local_unavailable")
    reporter2.close()
    assert len(captured) == 1
    assert captured[0]["name"] == "octomil.fallback.cloud"
    assert captured[0]["attributes"]["model.id"] == "llama-3"
    assert captured[0]["attributes"]["fallback.reason"] == "local_unavailable"


def test_report_inference_started_includes_locality():
    from octomil.telemetry import TelemetryReporter

    reporter = TelemetryReporter(api_key="test-key")
    captured: list = []
    original_enqueue = reporter._enqueue

    def capture(**kwargs):
        captured.append(kwargs)
        original_enqueue(**kwargs)

    reporter._enqueue = capture  # type: ignore[method-assign]
    reporter.report_inference_started(
        model_id="llama-3",
        version="1.0",
        session_id="sess-1",
        locality="on_device",
    )
    reporter.close()
    assert captured[0]["attributes"]["locality"] == "on_device"


def test_report_inference_completed_includes_locality():
    from octomil.telemetry import TelemetryReporter

    reporter = TelemetryReporter(api_key="test-key")
    captured: list = []
    original_enqueue = reporter._enqueue

    def capture(**kwargs):
        captured.append(kwargs)
        original_enqueue(**kwargs)

    reporter._enqueue = capture  # type: ignore[method-assign]
    reporter.report_inference_completed(
        session_id="sess-1",
        model_id="llama-3",
        version="1.0",
        total_chunks=10,
        total_duration_ms=200.0,
        ttfc_ms=50.0,
        throughput=5.0,
        locality="cloud",
    )
    reporter.close()
    assert captured[0]["attributes"]["locality"] == "cloud"


@pytest.mark.asyncio
async def test_create_emits_route_decision_telemetry():
    runtime = _StubRuntime("local")
    mock_telemetry = MagicMock()
    responses = OctomilResponses(
        runtime_resolver=lambda _: runtime,
        telemetry_reporter=mock_telemetry,
        planner_enabled=False,
    )

    response = await responses.create(
        ResponseRequest(model="mymodel", input=[text_input("hi")]),
    )

    mock_telemetry._enqueue.assert_called_once()
    kwargs = mock_telemetry._enqueue.call_args.kwargs
    assert kwargs["name"] == "route.decision"
    assert "route.id" in kwargs["attributes"]
    assert kwargs["attributes"]["route.request_id"] == response.id
    assert kwargs["attributes"]["route.model_ref"] == "mymodel"
    assert kwargs["attributes"]["route.final_locality"] == "local"
    assert kwargs["attributes"]["route.final_mode"] == "sdk_runtime"


@pytest.mark.asyncio
async def test_app_ref_route_telemetry_includes_app_context():
    runtime = _StubRuntime("local")
    mock_telemetry = MagicMock()
    responses = OctomilResponses(
        runtime_resolver=lambda _: runtime,
        telemetry_reporter=mock_telemetry,
    )
    selection = RuntimeSelection(
        locality="local",
        engine="mlx-lm",
        source="server_plan",
        candidates=[
            RuntimeCandidatePlan(
                locality="local",
                priority=1,
                confidence=0.9,
                reason="local available",
                engine=None,
            )
        ],
        app_resolution=AppResolution(
            app_id="app-123",
            app_slug="test-limits",
            capability="chat",
            routing_policy="private",
            selected_model="gemma3-1b",
        ),
    )

    with patch(
        "octomil.responses.responses._try_resolve_planner_selection",
        return_value=selection,
    ):
        await responses.create(
            ResponseRequest(model="@app/test-limits/chat", input=[text_input("hi")]),
        )

    kwargs = mock_telemetry._enqueue.call_args.kwargs
    assert kwargs["name"] == "route.decision"
    assert "route.id" in kwargs["attributes"]
    assert kwargs["attributes"]["route.app_id"] == "app-123"
    assert kwargs["attributes"]["route.app_slug"] == "test-limits"
    assert kwargs["attributes"]["route.policy"] == "private"
    assert kwargs["attributes"]["route.model_ref_kind"] == "app"
