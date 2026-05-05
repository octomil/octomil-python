"""Tests for InferenceBackendAdapter — tier-aware tool call extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from octomil._generated.message_role import MessageRole
from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeToolDef,
    ToolCallTier,
)


def _text_request(text: str = "test", **kwargs) -> RuntimeRequest:
    """Build a minimal text-only RuntimeRequest for tests."""
    return RuntimeRequest(
        messages=[RuntimeMessage(role=MessageRole.USER, parts=[RuntimeContentPart.text_part(text)])],
        **kwargs,
    )


@dataclass
class FakeMetrics:
    prompt_tokens: int = 10
    total_tokens: int = 20
    # Mirrors the non-zero portion of ``InferenceMetrics`` that the
    # real backends populate: time-to-first-chunk and steady-state
    # throughput. ``InferenceBackendAdapter`` reads these onto the
    # ``RuntimeResponse`` so the route-event emitter can include them
    # in dashboard telemetry.
    ttfc_ms: float = 0.0
    tokens_per_second: float = 0.0


class FakeBackend:
    """Minimal backend stub for testing."""

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, request: object) -> tuple[str, FakeMetrics]:
        return self._text, FakeMetrics()


def _tool_defs() -> list[RuntimeToolDef]:
    return [
        RuntimeToolDef(
            name="get_weather",
            description="Get weather",
            parameters_schema=json.dumps(
                {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            ),
        )
    ]


@pytest.mark.asyncio
async def test_text_json_tier_extracts_tool_call():
    backend = FakeBackend('{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}')
    adapter = InferenceBackendAdapter(
        backend=backend,
        model_name="test",
        capabilities=RuntimeCapabilities(tool_call_tier=ToolCallTier.TEXT_JSON),
    )
    request = _text_request(tool_definitions=_tool_defs())
    response = await adapter.run(request)

    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.finish_reason == "tool_calls"
    assert response.raw_text is not None


@pytest.mark.asyncio
async def test_none_tier_does_not_extract():
    backend = FakeBackend('{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}')
    adapter = InferenceBackendAdapter(
        backend=backend,
        model_name="test",
        capabilities=RuntimeCapabilities(tool_call_tier=ToolCallTier.NONE),
    )
    request = _text_request(tool_definitions=_tool_defs())
    response = await adapter.run(request)

    assert response.tool_calls is None
    assert response.finish_reason == "stop"
    assert response.text == '{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}'


@pytest.mark.asyncio
async def test_raw_text_set_when_tools_present():
    backend = FakeBackend("Just a text response")
    adapter = InferenceBackendAdapter(
        backend=backend,
        model_name="test",
        capabilities=RuntimeCapabilities(tool_call_tier=ToolCallTier.TEXT_JSON),
    )
    request = _text_request(tool_definitions=_tool_defs())
    response = await adapter.run(request)

    assert response.tool_calls is None
    assert response.raw_text == "Just a text response"


@pytest.mark.asyncio
async def test_raw_text_none_without_tools():
    backend = FakeBackend("Just a text response")
    adapter = InferenceBackendAdapter(
        backend=backend,
        model_name="test",
        capabilities=RuntimeCapabilities(tool_call_tier=ToolCallTier.TEXT_JSON),
    )
    request = _text_request()
    response = await adapter.run(request)

    assert response.raw_text is None


@pytest.mark.asyncio
async def test_default_capabilities_are_none_tier():
    adapter = InferenceBackendAdapter(backend=FakeBackend("hi"), model_name="test")
    assert adapter.capabilities.tool_call_tier == ToolCallTier.NONE
    assert adapter.capabilities.supports_tool_calls is False


class _BackendWithMetrics:
    """Backend that returns non-zero ``ttfc_ms`` + ``tokens_per_second`` so we can
    verify ``InferenceBackendAdapter.run`` surfaces them onto the response."""

    def __init__(
        self,
        text: str,
        *,
        ttfc_ms: float,
        tokens_per_second: float,
    ) -> None:
        self._text = text
        self._ttfc_ms = ttfc_ms
        self._tps = tokens_per_second

    def generate(self, request: object) -> tuple[str, FakeMetrics]:
        return self._text, FakeMetrics(
            ttfc_ms=self._ttfc_ms,
            tokens_per_second=self._tps,
        )


@pytest.mark.asyncio
async def test_adapter_surfaces_ttft_and_throughput_to_runtime_response():
    """Backend latency telemetry (``ttfc_ms`` = TTFT for non-streaming;
    ``tokens_per_second``) must propagate into ``RuntimeResponse`` so the
    Layer-2 route-event emit path can ship it to the dashboard. Pre-fix
    the adapter dropped these fields; ``Avg TTFT`` / ``Avg throughput``
    rendered em-dashes."""
    backend = _BackendWithMetrics("hello", ttfc_ms=42.0, tokens_per_second=128.5)
    adapter = InferenceBackendAdapter(backend=backend, model_name="test")
    response = await adapter.run(_text_request())

    assert response.ttft_ms == 42.0
    assert response.tokens_per_second == 128.5


@pytest.mark.asyncio
async def test_adapter_returns_none_when_backend_reports_zero_latency():
    """Real backends sometimes return 0.0 for one or both fields when
    they couldn't measure (e.g., a synchronous one-shot path that
    didn't probe first-chunk time). Treat 0.0 as 'no signal' and
    surface ``None`` rather than reporting fake-perfect 0ms latency
    that would skew the dashboard's averages."""
    backend = _BackendWithMetrics("hello", ttfc_ms=0.0, tokens_per_second=0.0)
    adapter = InferenceBackendAdapter(backend=backend, model_name="test")
    response = await adapter.run(_text_request())

    assert response.ttft_ms is None
    assert response.tokens_per_second is None
