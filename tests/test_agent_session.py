"""Tests for AgentSession."""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from octomil.agents.session import AgentResult, AgentSession
from octomil.responses import OctomilResponses
from octomil.runtime.core import (
    ModelRuntime,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
)

# ---------------------------------------------------------------------------
# Test runtime that returns canned responses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HTTP mock helpers
# ---------------------------------------------------------------------------

_CREATE_SESSION_RESPONSE = {
    "session_id": "sess-test-1",
    "tools": [
        {
            "name": "get_model",
            "description": "Retrieve model details",
            "input_schema": {
                "type": "object",
                "properties": {"model_id": {"type": "string"}},
                "required": ["model_id"],
            },
        }
    ],
    "system_prompt": "You are a deployment advisor.",
}


def _mock_post(url: str, **kwargs):
    """Route mock HTTP calls by URL path."""
    if "/sessions" in url and "/execute" in url:
        return httpx.Response(
            200,
            json={
                "tool_call_id": kwargs.get("json", {}).get("tool_call_id", "tc_1"),
                "content": '{"name":"phi-mini","framework":"pytorch"}',
                "is_error": False,
            },
            request=httpx.Request("POST", url),
        )
    elif "/sessions" in url and "/complete" in url:
        return httpx.Response(
            200,
            json={"session_id": "sess-test-1", "status": "completed"},
            request=httpx.Request("POST", url),
        )
    elif "/sessions" in url:
        return httpx.Response(
            200,
            json=_CREATE_SESSION_RESPONSE,
            request=httpx.Request("POST", url),
        )
    return httpx.Response(404, request=httpx.Request("POST", url))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_session_lifecycle_no_tools():
    """Session with a model that returns text immediately (no tool calls)."""
    runtime = SequentialRuntime(
        [
            RuntimeResponse(text="## Summary\nDeployment looks good.\n\n## Risk Assessment\nConfidence: 0.9"),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    session = AgentSession(
        base_url="https://api.example.com",
        auth_token="tok-abc",
        responses=responses,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_mock_post):
        result = await session.run("deployment_advisor", "Deploy phi-mini")

    assert isinstance(result, AgentResult)
    assert result.session_id == "sess-test-1"
    assert "Deployment looks good" in result.summary
    assert result.confidence == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_session_with_tool_round_trip():
    """Session where model emits a tool call, gets result, then responds."""
    runtime = SequentialRuntime(
        [
            RuntimeResponse(
                text="",
                tool_calls=[
                    RuntimeToolCall(
                        id="call_1",
                        name="get_model",
                        arguments='{"model_id":"m-123"}',
                    )
                ],
            ),
            RuntimeResponse(text="## Summary\nModel m-123 found and ready.\n\n## Risk Assessment\nConfidence: 0.85"),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    session = AgentSession(
        base_url="https://api.example.com",
        auth_token="tok-abc",
        responses=responses,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_mock_post):
        result = await session.run("deployment_advisor", "Check model m-123")

    assert result.session_id == "sess-test-1"
    assert "m-123" in result.summary
    assert result.confidence == pytest.approx(0.85, abs=0.01)


@pytest.mark.asyncio
async def test_session_completes_even_on_completion_failure():
    """If the /complete call fails, the result is still returned."""
    runtime = SequentialRuntime(
        [
            RuntimeResponse(text="## Summary\nAll good."),
        ]
    )
    responses = OctomilResponses(runtime_resolver=lambda _: runtime)

    session = AgentSession(
        base_url="https://api.example.com",
        auth_token="tok-abc",
        responses=responses,
    )

    call_count = [0]

    async def _mock_post_with_failure(url: str, **kwargs):
        call_count[0] += 1
        if "/complete" in url:
            raise httpx.ConnectError("Connection refused")
        return _mock_post(url, **kwargs)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_mock_post_with_failure):
        result = await session.run("deployment_advisor", "Quick test")

    assert result.summary == "All good."
    assert call_count[0] >= 2  # At least create + complete attempt


@pytest.mark.asyncio
async def test_agent_result_dataclass():
    """AgentResult holds all expected fields."""
    result = AgentResult(
        session_id="sess-1",
        summary="Test summary",
        confidence=0.75,
        evidence=["Evidence A"],
        next_steps=["Step 1"],
    )
    assert result.session_id == "sess-1"
    assert result.confidence == 0.75
    assert result.evidence == ["Evidence A"]
    assert result.next_steps == ["Step 1"]
