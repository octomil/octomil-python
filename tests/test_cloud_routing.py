"""Tests for cloud routing: CloudClient, CloudModelRuntime, CloudInferenceBackend."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from octomil.errors import OctomilError
from octomil.runtime.core.cloud_client import CloudClient, _raise_for_status
from octomil.runtime.core.cloud_runtime import CloudModelRuntime, _messages_to_openai, _tools_to_openai
from octomil.runtime.core.engine_bridge import cloud_runtime_factory
from octomil.runtime.core.router import RouterModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolDef,
    ToolCallTier,
)
from octomil.serve.backends.cloud import CloudInferenceBackend
from octomil.serve.types import GenerationRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime_request(content: str = "hello") -> RuntimeRequest:
    from octomil._generated.message_role import MessageRole
    from octomil.runtime.core.types import RuntimeContentPart

    return RuntimeRequest(
        messages=[
            RuntimeMessage(
                role=MessageRole.USER,
                parts=[RuntimeContentPart.text_part(content)],
            )
        ],
        generation_config=GenerationConfig(max_tokens=100, temperature=0.5),
    )


def _make_generation_request(content: str = "hello") -> GenerationRequest:
    return GenerationRequest(
        model="test-model",
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0.5,
    )


def _mock_chat_response(content: str = "hi there", usage: dict | None = None) -> dict:
    resp: dict = {
        "choices": [{"message": {"content": content, "role": "assistant"}, "finish_reason": "stop"}],
    }
    if usage:
        resp["usage"] = usage
    return resp


def _mock_stream_chunks(content: str = "hi there") -> list[str]:
    """Build SSE lines for a streaming response."""
    words = content.split()
    lines = []
    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        chunk = {
            "choices": [
                {
                    "delta": {"content": word + ("" if is_last else " ")},
                    "finish_reason": "stop" if is_last else None,
                }
            ]
        }
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


# ---------------------------------------------------------------------------
# CloudClient tests
# ---------------------------------------------------------------------------


class TestCloudClient:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        expected = _mock_chat_response(
            "test response", usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = expected

        client = CloudClient("https://api.example.com/v1", "test-key", "test-model")
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.chat([{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "test response"
        assert result["usage"]["total_tokens"] == 7
        await client.close()

    @pytest.mark.asyncio
    async def test_chat_auth_error(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.text = "Invalid API key"

        client = CloudClient("https://api.example.com/v1", "bad-key", "test-model")
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(OctomilError) as exc_info:
                await client.chat([{"role": "user", "content": "hi"}])
            assert "authentication" in str(exc_info.value).lower()
        await client.close()

    @pytest.mark.asyncio
    async def test_chat_rate_limit_retry(self):
        rate_limited = MagicMock(spec=httpx.Response)
        rate_limited.status_code = 429
        rate_limited.is_success = False
        rate_limited.headers = {"retry-after": "0.01"}
        rate_limited.json.return_value = {"error": {"message": "Rate limited"}}
        rate_limited.text = "Rate limited"

        success = MagicMock(spec=httpx.Response)
        success.status_code = 200
        success.is_success = True
        success.json.return_value = _mock_chat_response("retried ok")

        client = CloudClient("https://api.example.com/v1", "key", "model")
        with patch.object(client._client, "post", new_callable=AsyncMock, side_effect=[rate_limited, success]):
            result = await client.chat([{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "retried ok"
        await client.close()

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        lines = _mock_stream_chunks("hello world")

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.aclose = AsyncMock()

        client = CloudClient("https://api.example.com/v1", "key", "model")
        with patch.object(client, "_post_stream", new_callable=AsyncMock, return_value=mock_response):
            chunks = []
            async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "hello "
        assert chunks[1]["choices"][0]["delta"]["content"] == "world"
        await client.close()


# ---------------------------------------------------------------------------
# _raise_for_status tests
# ---------------------------------------------------------------------------


class TestRaiseForStatus:
    def test_success(self):
        resp = MagicMock(spec=httpx.Response)
        resp.is_success = True
        _raise_for_status(resp)  # no exception

    def test_401(self):
        resp = MagicMock(spec=httpx.Response)
        resp.is_success = False
        resp.status_code = 401
        resp.json.return_value = {"error": {"message": "bad key"}}
        with pytest.raises(OctomilError) as exc_info:
            _raise_for_status(resp)
        assert "authentication_failed" in repr(exc_info.value)

    def test_429(self):
        resp = MagicMock(spec=httpx.Response)
        resp.is_success = False
        resp.status_code = 429
        resp.json.return_value = {"error": {"message": "rate limited"}}
        with pytest.raises(OctomilError) as exc_info:
            _raise_for_status(resp)
        assert "rate_limited" in repr(exc_info.value)

    def test_500(self):
        resp = MagicMock(spec=httpx.Response)
        resp.is_success = False
        resp.status_code = 500
        resp.json.return_value = {"error": {"message": "server error"}}
        with pytest.raises(OctomilError) as exc_info:
            _raise_for_status(resp)
        assert "server_error" in repr(exc_info.value)


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_messages_to_openai(self):
        request = _make_runtime_request("what is 2+2")
        messages = _messages_to_openai(request)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "what is 2+2"

    def test_tools_to_openai_none(self):
        request = _make_runtime_request()
        assert _tools_to_openai(request) is None

    def test_tools_to_openai(self):
        request = _make_runtime_request()
        request.tool_definitions = [
            RuntimeToolDef(
                name="get_weather",
                description="Get weather for a city",
                parameters_schema='{"type":"object","properties":{"city":{"type":"string"}}}',
            )
        ]
        tools = _tools_to_openai(request)
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert "city" in json.dumps(tools[0]["function"]["parameters"])


# ---------------------------------------------------------------------------
# CloudModelRuntime tests
# ---------------------------------------------------------------------------


class TestCloudModelRuntime:
    def test_capabilities(self):
        runtime = CloudModelRuntime("https://api.example.com/v1", "key", "model")
        caps = runtime.capabilities
        assert caps.tool_call_tier == ToolCallTier.NATIVE
        assert caps.supports_streaming is True
        assert caps.supports_structured_output is True

    @pytest.mark.asyncio
    async def test_run(self):
        runtime = CloudModelRuntime("https://api.example.com/v1", "key", "gpt-4o-mini")
        mock_result = _mock_chat_response(
            "answer", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        with patch.object(runtime._client, "chat", new_callable=AsyncMock, return_value=mock_result):
            response = await runtime.run(_make_runtime_request("question"))

        assert isinstance(response, RuntimeResponse)
        assert response.text == "answer"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_run_with_tool_calls(self):
        runtime = CloudModelRuntime("https://api.example.com/v1", "key", "gpt-4o-mini")
        mock_result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        with patch.object(runtime._client, "chat", new_callable=AsyncMock, return_value=mock_result):
            response = await runtime.run(_make_runtime_request())

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].id == "call_123"

    @pytest.mark.asyncio
    async def test_stream(self):
        runtime = CloudModelRuntime("https://api.example.com/v1", "key", "gpt-4o-mini")

        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "hello "}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}

        with patch.object(runtime._client, "chat_stream", side_effect=mock_stream):
            chunks = []
            async for chunk in runtime.stream(_make_runtime_request()):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "hello "
        assert chunks[1].text == "world"
        assert chunks[1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# CloudInferenceBackend tests
# ---------------------------------------------------------------------------


class TestCloudInferenceBackend:
    def test_name(self):
        backend = CloudInferenceBackend("https://api.example.com/v1", "key", "model")
        assert backend.name == "cloud"

    def test_load_model_noop(self):
        backend = CloudInferenceBackend("https://api.example.com/v1", "key", "model")
        backend.load_model("anything")  # should not raise


# ---------------------------------------------------------------------------
# cloud_runtime_factory tests
# ---------------------------------------------------------------------------


class TestCloudRuntimeFactory:
    def test_creates_runtime(self):
        factory = cloud_runtime_factory("https://api.example.com/v1", "key", "model")
        runtime = factory("any-model-id")
        assert runtime is not None
        assert isinstance(runtime, CloudModelRuntime)

    def test_caches_instance(self):
        factory = cloud_runtime_factory("https://api.example.com/v1", "key", "model")
        r1 = factory("model-a")
        r2 = factory("model-b")
        assert r1 is r2  # same cached instance


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------


class TestRouterWithCloud:
    def test_cloud_only_mode(self):
        factory = cloud_runtime_factory("https://api.example.com/v1", "key", "model")
        from octomil.runtime.core.policy import RoutingPolicy

        router = RouterModelRuntime(cloud_factory=factory, default_policy=RoutingPolicy.cloud_only())
        locality, is_fallback = router.resolve_locality()
        assert locality == "cloud"
        assert is_fallback is False

    def test_auto_mode_falls_back_to_cloud(self):
        factory = cloud_runtime_factory("https://api.example.com/v1", "key", "model")
        from octomil.runtime.core.policy import RoutingPolicy

        router = RouterModelRuntime(
            local_factory=lambda mid: None,  # no local available
            cloud_factory=factory,
            default_policy=RoutingPolicy.auto(),
        )
        locality, is_fallback = router.resolve_locality()
        assert locality == "cloud"
        assert is_fallback is True
