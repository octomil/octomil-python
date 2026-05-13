"""Tests for octomil.serve — OpenAI-compatible inference server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.errors import OctomilError
from octomil.serve import (
    ChatCompletionBody,
    ChatMessage,
    EchoBackend,
    GenerationRequest,
    InferenceMetrics,
    _detect_backend,
    create_app,
    create_multi_model_app,
    resolve_model_name,
)
from octomil.serve.json_mode import JsonModeConfig, coerce_json_mode_output, resolve_json_mode_config

# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    def test_full_repo_id_passthrough(self):
        assert resolve_model_name("some-org/some-model", "mlx") == "some-org/some-model"

    def test_full_repo_id_passthrough_gguf(self):
        assert resolve_model_name("user/repo", "gguf") == "user/repo"

    def test_mlx_short_name(self):
        result = resolve_model_name("gemma-1b", "mlx")
        assert result == "mlx-community/REDACTED"

    def test_mlx_unknown_name_raises(self):
        from octomil.errors import OctomilError

        with pytest.raises(OctomilError, match="Unknown model 'nonexistent'"):
            resolve_model_name("nonexistent", "mlx")

    def test_gguf_short_name(self):
        result = resolve_model_name("gemma-1b", "gguf")
        assert result == "gemma-1b"  # gguf resolves at download time

    def test_gguf_unknown_name_raises(self):
        from octomil.errors import OctomilError

        with pytest.raises(OctomilError, match="Unknown model 'nonexistent'"):
            resolve_model_name("nonexistent", "gguf")

    def test_unknown_backend_unknown_model_raises(self):
        from octomil.errors import OctomilError

        # With the model registry, unknown model names raise even with
        # unknown backends (stricter validation than before).
        with pytest.raises(OctomilError, match="Unknown model 'anything'"):
            resolve_model_name("anything", "other")

    def test_unknown_backend_known_model_passthrough(self):
        # Known models still resolve even with unknown backend names
        result = resolve_model_name("gemma-1b", "other")
        assert result == "gemma-1b"


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
    def test_falls_back_to_echo_when_no_backends(self):
        """With no real engines installed, _detect_backend returns echo."""
        from octomil.runtime.engines.registry import EngineRegistry

        with patch.object(EngineRegistry, "detect_all", return_value=[]):
            backend = _detect_backend("gemma-1b")
        assert backend.name == "echo"

    def test_echo_for_unknown_model_name(self):
        """Unknown model name still gets a backend (echo fallback)."""
        from octomil.runtime.engines.registry import EngineRegistry

        with patch.object(EngineRegistry, "detect_all", return_value=[]):
            backend = _detect_backend("totally-unknown-model")
        assert backend.name == "echo"

    def test_detect_backend_returns_backend(self):
        """_detect_backend with echo override returns an InferenceBackend."""
        result = _detect_backend("gemma-1b", engine_override="echo")
        assert hasattr(result, "name")
        assert hasattr(result, "generate")

    def test_detect_backend_with_engine_override(self):
        """_detect_backend with engine_override='echo' returns echo."""
        backend = _detect_backend("gemma-1b", engine_override="echo")
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
        assert body.max_completion_tokens is None
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
    with patch("octomil.serve.app._detect_backend") as mock_detect:
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
    assert data["data"][0]["owned_by"] == "octomil"


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


class _StaticTextBackend(EchoBackend):
    """Test backend that returns scripted text responses."""

    def __init__(self, *responses: str) -> None:
        super().__init__()
        self._responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        self.requests.append(request)
        text = self._responses.pop(0) if self._responses else ""
        return text, InferenceMetrics(total_tokens=max(1, len(text.split())))


@pytest.mark.asyncio
async def test_json_mode_repairs_fenced_json_before_returning():
    backend = _StaticTextBackend('```json\n{"ok": true}\n```')
    backend.load_model("test-model")

    with patch("octomil.serve.app._detect_backend", return_value=backend):
        app = create_app("test-model", max_queue_depth=0)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "json please"}],
                    "response_format": {"type": "json_object"},
                },
            )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == '{"ok": true}'
    assert data["usage"]["json_validation"]["status"] == "repaired_fenced"


def test_json_mode_helpers_validate_defaults_and_bad_schema_wrappers():
    assert resolve_json_mode_config(None) == JsonModeConfig()
    assert resolve_json_mode_config({"type": "json_object", "strict": True}).strict is True

    with pytest.raises(OctomilError, match="response_format.json_schema must be an object"):
        resolve_json_mode_config({"type": "json_schema", "json_schema": ["not", "an", "object"]})

    with pytest.raises(OctomilError, match="response_format.json_schema.schema must be an object"):
        resolve_json_mode_config({"type": "json_schema", "json_schema": {"schema": ["not", "an", "object"]}})

    with pytest.raises(OctomilError, match="Invalid JSON schema"):
        resolve_json_mode_config({"type": "json_schema", "json_schema": {"schema": {"type": 123}}})


def test_json_mode_helpers_repair_prose_and_reject_non_objects():
    repaired = coerce_json_mode_output('Result: {"ok": true}', JsonModeConfig())
    assert repaired.text == '{"ok": true}'
    assert repaired.status == "repaired_extracted"

    with pytest.raises(OctomilError, match="JSON mode validation failed"):
        coerce_json_mode_output("[1, 2, 3]", JsonModeConfig())

    with pytest.raises(OctomilError, match="JSON mode validation failed"):
        coerce_json_mode_output("", JsonModeConfig())


@pytest.mark.asyncio
async def test_json_mode_rejects_unrepairable_text_after_retry():
    backend = _StaticTextBackend("not json", "still not json")
    backend.load_model("test-model")

    with patch("octomil.serve.app._detect_backend", return_value=backend):
        app = create_app("test-model", max_queue_depth=0)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "json please"}],
                    "response_format": {"type": "json_object"},
                },
            )

    assert resp.status_code == 503
    data = resp.json()
    assert data["code"] == "inference_failed"
    assert "JSON mode validation failed" in data["message"]


@pytest.mark.asyncio
async def test_json_mode_strict_rejects_prose_wrapped_json():
    backend = _StaticTextBackend('Here is JSON: {"ok": true}', 'Still prose: {"ok": true}')
    backend.load_model("test-model")

    with patch("octomil.serve.app._detect_backend", return_value=backend):
        app = create_app("test-model", max_queue_depth=0)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "json please"}],
                    "response_format": {"type": "json_object", "strict": True},
                },
            )

    assert resp.status_code == 503
    assert "JSON mode validation failed" in resp.json()["message"]


@pytest.mark.asyncio
async def test_multi_model_json_mode_retries_and_reports_validation_status():
    backend = _StaticTextBackend("not json", '{"ok": true}')
    backend.load_model("small-model")

    with patch("octomil.serve.multi_model._detect_backend", return_value=backend):
        app = create_multi_model_app(["small-model"])
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "small-model",
                    "messages": [{"role": "user", "content": "json please"}],
                    "response_format": {"type": "json_object"},
                },
            )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == '{"ok": true}'
    assert data["usage"]["json_validation"] == {
        "mode": "json_object",
        "status": "valid",
        "strict": False,
        "schema": False,
    }
    assert len(backend.requests) == 2
    assert backend.requests[0].json_mode is False
    assert backend.requests[1].json_mode is False
    assert backend.requests[1].messages[0]["role"] == "system"
    assert "previous response failed JSON validation" in backend.requests[1].messages[0]["content"]


@pytest.mark.asyncio
async def test_json_schema_mode_validates_schema_after_retry():
    backend = _StaticTextBackend('{"ok": true}', '{"ok": false}')
    backend.load_model("test-model")

    with patch("octomil.serve.app._detect_backend", return_value=backend):
        app = create_app("test-model", max_queue_depth=0)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "schema please"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ticker_payload",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"ticker": {"type": "string"}},
                                "required": ["ticker"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            )

    assert resp.status_code == 503
    assert "JSON schema validation failed" in resp.json()["message"]


@pytest.mark.asyncio
async def test_chat_completions_honors_max_completion_tokens_alias():
    backend = _StaticTextBackend('{"ok": true}')
    backend.load_model("test-model")

    with patch("octomil.serve.app._detect_backend", return_value=backend):
        app = create_app("test-model", max_queue_depth=0)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "short"}],
                    "max_tokens": 100,
                    "max_completion_tokens": 7,
                },
            )

    assert resp.status_code == 200, resp.text
    assert backend.requests[0].max_tokens == 7
