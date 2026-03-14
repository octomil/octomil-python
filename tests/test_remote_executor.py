"""Tests for RemoteToolExecutor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from octomil.responses.tools.remote_executor import RemoteToolExecutor
from octomil.responses.types import ResponseToolCall


def _make_call(name: str = "get_model", args: dict | None = None) -> ResponseToolCall:
    return ResponseToolCall(
        id="tc_test_1",
        name=name,
        arguments=json.dumps(args or {"model_id": "m-123"}),
    )


@pytest.mark.asyncio
async def test_successful_execution():
    """Tool execution returns ToolResult with server response."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com",
        session_id="sess-1",
        auth_token="tok-abc",
    )
    mock_response = httpx.Response(
        200,
        json={
            "tool_call_id": "tc_test_1",
            "content": '{"name":"phi-mini"}',
            "is_error": False,
        },
        request=httpx.Request("POST", "https://api.example.com/api/v1/agents/sessions/sess-1/execute"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await executor.execute(_make_call())

    assert result.tool_call_id == "tc_test_1"
    assert result.content == '{"name":"phi-mini"}'
    assert result.is_error is False


@pytest.mark.asyncio
async def test_http_error_returns_error_result():
    """HTTP 500 produces ToolResult with is_error=True."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com",
        session_id="sess-1",
        auth_token="tok-abc",
    )
    mock_response = httpx.Response(
        500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.example.com/api/v1/agents/sessions/sess-1/execute"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await executor.execute(_make_call())

    assert result.tool_call_id == "tc_test_1"
    assert result.is_error is True
    assert "500" in result.content


@pytest.mark.asyncio
async def test_auth_token_passed_in_headers():
    """Authorization header includes the auth token."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com",
        session_id="sess-1",
        auth_token="my-secret-token",
    )
    mock_response = httpx.Response(
        200,
        json={"tool_call_id": "tc_test_1", "content": "ok", "is_error": False},
        request=httpx.Request("POST", "https://api.example.com/api/v1/agents/sessions/sess-1/execute"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        await executor.execute(_make_call())

    # Verify headers
    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["Authorization"] == "Bearer my-secret-token"


@pytest.mark.asyncio
async def test_request_error_returns_graceful_error():
    """Network errors produce ToolResult with is_error=True."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com",
        session_id="sess-1",
        auth_token="tok-abc",
    )

    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await executor.execute(_make_call())

    assert result.tool_call_id == "tc_test_1"
    assert result.is_error is True
    assert "failed" in result.content.lower()


@pytest.mark.asyncio
async def test_json_parse_error_returns_graceful_error():
    """Invalid JSON in server response produces ToolResult with is_error=True."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com",
        session_id="sess-1",
        auth_token="tok-abc",
    )
    mock_response = httpx.Response(
        200,
        content=b"not json",
        request=httpx.Request("POST", "https://api.example.com/api/v1/agents/sessions/sess-1/execute"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await executor.execute(_make_call())

    assert result.tool_call_id == "tc_test_1"
    assert result.is_error is True


@pytest.mark.asyncio
async def test_url_construction():
    """URL is correctly built from base_url and session_id."""
    executor = RemoteToolExecutor(
        base_url="https://api.example.com/",  # trailing slash
        session_id="sess-42",
        auth_token="tok",
    )
    mock_response = httpx.Response(
        200,
        json={"tool_call_id": "tc_test_1", "content": "ok", "is_error": False},
        request=httpx.Request("POST", "https://api.example.com/api/v1/agents/sessions/sess-42/execute"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        await executor.execute(_make_call())

    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
    assert url == "https://api.example.com/api/v1/agents/sessions/sess-42/execute"
