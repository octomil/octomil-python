"""Tests for edgeml.serve — OpenAI-compatible inference server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.serve import (
    ChatCompletionBody,
    ChatMessage,
    EchoBackend,
    GenerationRequest,
    _detect_backend,
    create_app,
    resolve_model_name,
)


# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    def test_full_repo_id_passthrough(self):
        assert resolve_model_name("mlx-community/gemma-3-1b-it-4bit", "mlx") == (
            "mlx-community/gemma-3-1b-it-4bit"
        )

    def test_full_repo_id_passthrough_gguf(self):
        assert resolve_model_name("user/repo", "gguf") == "user/repo"

    def test_mlx_short_name(self):
        result = resolve_model_name("gemma-1b", "mlx")
        assert result == "mlx-community/gemma-3-1b-it-4bit"

    def test_mlx_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model 'nonexistent'"):
            resolve_model_name("nonexistent", "mlx")

    def test_gguf_short_name(self):
        result = resolve_model_name("gemma-1b", "gguf")
        assert result == "gemma-1b"  # gguf resolves at download time

    def test_gguf_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model 'nonexistent'"):
            resolve_model_name("nonexistent", "gguf")

    def test_unknown_backend_passthrough(self):
        assert resolve_model_name("anything", "other") == "anything"


# ---------------------------------------------------------------------------
# EchoBackend
# ---------------------------------------------------------------------------


class TestEchoBackend:
    def setup_method(self):
        self.backend = EchoBackend()
        self.backend.load_model("test-model")

    def test_name(self):
        assert self.backend.name == "echo"

    def test_list_models(self):
        assert self.backend.list_models() == ["test-model"]

    def test_list_models_empty_before_load(self):
        b = EchoBackend()
        assert b.list_models() == []

    def test_generate(self):
        req = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello world"}],
        )
        text, metrics = self.backend.generate(req)
        assert "[echo:test-model]" in text
        assert "hello world" in text
        assert metrics.total_tokens > 0

    def test_generate_empty_messages(self):
        req = GenerationRequest(model="test-model", messages=[])
        text, metrics = self.backend.generate(req)
        assert "[echo:test-model]" in text

    def test_generate_stream(self):
        req = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        chunks = asyncio.run(_collect_stream(self.backend.generate_stream(req)))
        assert len(chunks) > 0
        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"
        # Reconstructed text should contain the input
        full_text = "".join(c.text for c in chunks)
        assert "hi" in full_text


async def _collect_stream(gen):
    result = []
    async for chunk in gen:
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# _detect_backend fallback to echo
# ---------------------------------------------------------------------------


class TestDetectBackend:
    @patch("edgeml.serve.platform")
    def test_falls_back_to_echo_when_no_backends(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"

        with patch.dict("sys.modules", {"mlx_lm": None, "llama_cpp": None}):
            backend = _detect_backend("some-model")
        assert backend.name == "echo"

    @patch("edgeml.serve.platform")
    def test_echo_for_unknown_model_name(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        backend = _detect_backend("totally-unknown-model")
        assert backend.name == "echo"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_chat_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_chat_completion_body_defaults(self):
        body = ChatCompletionBody()
        assert body.model == ""
        assert body.messages == []
        assert body.max_tokens == 512
        assert body.temperature == 0.7
        assert body.top_p == 1.0
        assert body.stream is False

    def test_chat_completion_body_with_values(self):
        body = ChatCompletionBody(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )
        assert body.model == "test"
        assert len(body.messages) == 1
        assert body.stream is True


# ---------------------------------------------------------------------------
# FastAPI app (using EchoBackend via create_app)
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app():
    """Create a FastAPI app with EchoBackend for testing."""
    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model")

        # Trigger lifespan startup manually
        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.mark.asyncio
async def test_health_endpoint(echo_app):
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert data["backend"] == "echo"


@pytest.mark.asyncio
async def test_list_models(echo_app):
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"
    assert data["data"][0]["owned_by"] == "edgeml"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(echo_app):
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 50,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "hello" in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data


@pytest.mark.asyncio
async def test_chat_completions_streaming(echo_app):
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200

    # Parse SSE events
    lines = resp.text.strip().split("\n\n")
    events = []
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))

    assert len(events) > 0
    # Each event should be a chat.completion.chunk
    for evt in events:
        assert evt["object"] == "chat.completion.chunk"
        assert len(evt["choices"]) == 1

    # Last real event should have finish_reason
    last_event = events[-1]
    assert last_event["choices"][0]["finish_reason"] == "stop"

    # Final line should be [DONE]
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_chat_completions_empty_messages(echo_app):
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_model_default(echo_app):
    """When model is omitted, should use the server's model name."""
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "test-model"
