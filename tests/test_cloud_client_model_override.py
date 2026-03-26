"""Tests for per-request model override in CloudClient and CloudModelRuntime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from octomil._generated.message_role import MessageRole
from octomil.runtime.core.cloud_client import CloudClient
from octomil.runtime.core.cloud_runtime import CloudModelRuntime
from octomil.runtime.core.types import (
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
)


class TestCloudClientModelOverride:
    """CloudClient._build_body respects model override."""

    def test_build_body_uses_override_model(self):
        client = CloudClient("http://test:8080", "key", "default-model")
        body = client._build_body(
            [{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.5,
            top_p=1.0,
            tools=None,
            stream=False,
            model="gpt-4o",
        )
        assert body["model"] == "gpt-4o"

    def test_build_body_falls_back_to_instance_model(self):
        client = CloudClient("http://test:8080", "key", "default-model")
        body = client._build_body(
            [{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.5,
            top_p=1.0,
            tools=None,
            stream=False,
            model=None,
        )
        assert body["model"] == "default-model"

    def test_build_body_falls_back_when_no_model_param(self):
        client = CloudClient("http://test:8080", "key", "default-model")
        body = client._build_body(
            [{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.5,
            top_p=1.0,
            tools=None,
            stream=False,
        )
        assert body["model"] == "default-model"

    @pytest.mark.asyncio
    async def test_chat_passes_model_through(self):
        client = CloudClient("http://test:8080", "key", "default-model")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.chat(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o",
        )

        call_args = client._client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["model"] == "gpt-4o"


class TestRuntimeRequestModel:
    """RuntimeRequest.model field works correctly."""

    def test_model_defaults_to_none(self):
        req = RuntimeRequest(messages=[])
        assert req.model is None

    def test_model_can_be_set(self):
        req = RuntimeRequest(messages=[], model="gpt-4o")
        assert req.model == "gpt-4o"


class TestCloudModelRuntimeOverride:
    """CloudModelRuntime passes request.model to CloudClient."""

    @pytest.mark.asyncio
    async def test_run_passes_model_to_client(self):
        runtime = CloudModelRuntime("http://test:8080", "key", "default-model")

        mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        runtime._client.chat = AsyncMock(return_value=mock_response)

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part("hello")],
                )
            ],
            model="gpt-4o",
        )

        await runtime.run(request)

        runtime._client.chat.assert_called_once()
        call_kwargs = runtime._client.chat.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs[1].get("model") == "gpt-4o"

    @pytest.mark.asyncio
    async def test_run_passes_none_model_when_not_set(self):
        runtime = CloudModelRuntime("http://test:8080", "key", "default-model")

        mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        runtime._client.chat = AsyncMock(return_value=mock_response)

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part("hello")],
                )
            ],
        )

        await runtime.run(request)

        call_kwargs = runtime._client.chat.call_args
        # model should be None (not set on request), so CloudClient falls back to self._model
        assert call_kwargs.kwargs.get("model") is None or call_kwargs[1].get("model") is None
