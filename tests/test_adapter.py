"""Tests for InferenceBackendAdapter — tier-aware tool call extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeRequest,
    RuntimeToolDef,
    ToolCallTier,
)


@dataclass
class FakeMetrics:
    prompt_tokens: int = 10
    total_tokens: int = 20


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
    request = RuntimeRequest(prompt="test", tool_definitions=_tool_defs())
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
    request = RuntimeRequest(prompt="test", tool_definitions=_tool_defs())
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
    request = RuntimeRequest(prompt="test", tool_definitions=_tool_defs())
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
    request = RuntimeRequest(prompt="test")
    response = await adapter.run(request)

    assert response.raw_text is None


@pytest.mark.asyncio
async def test_default_capabilities_are_none_tier():
    adapter = InferenceBackendAdapter(backend=FakeBackend("hi"), model_name="test")
    assert adapter.capabilities.tool_call_tier == ToolCallTier.NONE
    assert adapter.capabilities.supports_tool_calls is False
