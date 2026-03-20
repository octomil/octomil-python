"""Tests for verbose runtime event emitter (octomil serve -v)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.serve import (
    EchoBackend,
    create_app,
)
from octomil.serve.instrumentation import InstrumentedBackend, unwrap_backend
from octomil.serve.types import GenerationRequest
from octomil.serve.verbose_events import RuntimeEvent, VerboseEventEmitter

# Mock target: _detect_backend is imported in app.py from .detection,
# so we need to patch the reference where it's used, not the re-export.
_DETECT_PATCH = "octomil.serve.app._detect_backend"


# ---------------------------------------------------------------------------
# VerboseEventEmitter unit tests
# ---------------------------------------------------------------------------


class TestRuntimeEvent:
    def test_to_dict(self):
        event = RuntimeEvent(event_name="test.event", timestamp=1234.5, metadata={"key": "val"})
        d = event.to_dict()
        assert d["event_name"] == "test.event"
        assert d["timestamp"] == 1234.5
        assert d["metadata"]["key"] == "val"


class TestVerboseEventEmitter:
    def test_emit_stores_in_buffer(self):
        emitter = VerboseEventEmitter()
        emitter.emit("test.event", foo="bar", count=42)
        events = emitter.recent_events()
        assert len(events) == 1
        assert events[0]["event_name"] == "test.event"
        assert events[0]["metadata"]["foo"] == "bar"
        assert events[0]["metadata"]["count"] == 42

    def test_buffer_max_size(self):
        emitter = VerboseEventEmitter(buffer_size=5)
        for i in range(10):
            emitter.emit(f"event.{i}")
        events = emitter.recent_events()
        assert len(events) == 5
        assert events[0]["event_name"] == "event.5"

    def test_no_server_posting_without_api_key(self):
        """When api_key is None, no background poster is started."""
        emitter = VerboseEventEmitter()
        assert emitter._poster is None

    def test_close_without_poster(self):
        """close() should not raise when no poster is running."""
        emitter = VerboseEventEmitter()
        emitter.close()

    def test_emit_with_no_metadata(self):
        emitter = VerboseEventEmitter()
        emitter.emit("simple.event")
        events = emitter.recent_events()
        assert len(events) == 1
        assert events[0]["metadata"] == {}


# ---------------------------------------------------------------------------
# InstrumentedBackend unit tests
# ---------------------------------------------------------------------------


class TestInstrumentedBackend:
    def test_load_model_emits_events(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        wrapped = InstrumentedBackend(echo, emitter)
        wrapped.load_model("test-model")

        events = emitter.recent_events()
        names = [e["event_name"] for e in events]
        assert "backend.load_started" in names
        assert "backend.load_completed" in names

        started = next(e for e in events if e["event_name"] == "backend.load_started")
        assert started["metadata"]["model"] == "test-model"
        assert started["metadata"]["engine"] == "echo"

        completed = next(e for e in events if e["event_name"] == "backend.load_completed")
        assert completed["metadata"]["model"] == "test-model"
        assert "load_time_ms" in completed["metadata"]

    def test_generate_emits_events(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        echo.load_model("test-model")
        wrapped = InstrumentedBackend(echo, emitter)
        emitter._buffer.clear()

        req = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=50,
        )
        text, metrics = wrapped.generate(req)
        assert "hello" in text

        events = emitter.recent_events()
        names = [e["event_name"] for e in events]
        assert "backend.generate_started" in names
        assert "backend.generate_completed" in names

        completed = next(e for e in events if e["event_name"] == "backend.generate_completed")
        assert "duration_ms" in completed["metadata"]
        assert "tokens_per_second" in completed["metadata"]
        assert completed["metadata"]["completion_tokens"] == metrics.total_tokens

    @pytest.mark.asyncio
    async def test_stream_emits_events(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        echo.load_model("test-model")
        wrapped = InstrumentedBackend(echo, emitter)
        emitter._buffer.clear()

        req = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=50,
        )
        chunks = []
        async for chunk in wrapped.generate_stream(req):
            chunks.append(chunk)
        assert len(chunks) > 0

        events = emitter.recent_events()
        names = [e["event_name"] for e in events]
        assert "backend.stream_started" in names
        assert "backend.stream_completed" in names

        completed = next(e for e in events if e["event_name"] == "backend.stream_completed")
        assert completed["metadata"]["tokens_generated"] > 0
        assert "duration_ms" in completed["metadata"]
        assert "tokens_per_second" in completed["metadata"]

    def test_unwrap_backend_returns_inner(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        wrapped = InstrumentedBackend(echo, emitter)
        assert unwrap_backend(wrapped) is echo

    def test_unwrap_backend_passthrough(self):
        echo = EchoBackend()
        assert unwrap_backend(echo) is echo

    def test_getattr_delegates(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        echo.load_model("test-model")
        wrapped = InstrumentedBackend(echo, emitter)
        assert wrapped._model_name == "test-model"

    def test_list_models_delegates(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        echo.load_model("test-model")
        wrapped = InstrumentedBackend(echo, emitter)
        assert wrapped.list_models() == ["test-model"]

    def test_name_property(self):
        emitter = VerboseEventEmitter()
        echo = EchoBackend()
        wrapped = InstrumentedBackend(echo, emitter)
        assert wrapped.name == "echo"

    def test_get_verbose_metadata_hook(self):
        class CustomBackend(EchoBackend):
            def get_verbose_metadata(self, event_name, *, request=None, metrics=None):
                if event_name == "backend.generate_completed":
                    return {"custom_field": "custom_value"}
                return {}

        emitter = VerboseEventEmitter()
        backend = CustomBackend()
        backend.load_model("test-model")
        wrapped = InstrumentedBackend(backend, emitter)
        emitter._buffer.clear()

        req = GenerationRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        wrapped.generate(req)

        events = emitter.recent_events()
        completed = next(e for e in events if e["event_name"] == "backend.generate_completed")
        assert completed["metadata"]["custom_field"] == "custom_value"

    def test_load_failed_emits_on_error(self):
        class FailingBackend(EchoBackend):
            def load_model(self, model_name):
                raise RuntimeError("load failed")

        emitter = VerboseEventEmitter()
        backend = FailingBackend()
        wrapped = InstrumentedBackend(backend, emitter)

        with pytest.raises(RuntimeError, match="load failed"):
            wrapped.load_model("bad-model")

        events = emitter.recent_events()
        names = [e["event_name"] for e in events]
        assert "backend.load_started" in names
        assert "backend.load_failed" in names
        failed = next(e for e in events if e["event_name"] == "backend.load_failed")
        assert "load failed" in failed["metadata"]["error"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_app(*, verbose: bool = False):
    """Create a FastAPI app with EchoBackend and optionally verbose mode."""
    with patch(_DETECT_PATCH) as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", verbose=verbose)

        async def _trigger():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger())

    return app


# ---------------------------------------------------------------------------
# CLI flag propagation tests
# ---------------------------------------------------------------------------


class TestVerboseFlagPropagation:
    """Verify that --verbose/-v flag propagates to ServerState."""

    def test_create_app_verbose_off(self):
        app = _make_echo_app(verbose=False)
        transport = ASGITransport(app=app)

        async def _check():
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/debug/runtime-events")
            return resp.json()

        result = asyncio.run(_check())
        assert result["enabled"] is False
        assert result["events"] == []

    def test_create_app_verbose_on(self):
        app = _make_echo_app(verbose=True)
        transport = ASGITransport(app=app)

        async def _check():
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/debug/runtime-events")
            return resp.json()

        result = asyncio.run(_check())
        assert result["enabled"] is True
        assert len(result["events"]) >= 1
        event_names = [e["event_name"] for e in result["events"]]
        assert "server.started" in event_names


# ---------------------------------------------------------------------------
# Verbose events during inference
# ---------------------------------------------------------------------------


@pytest.fixture
def verbose_echo_app():
    return _make_echo_app(verbose=True)


@pytest.mark.asyncio
async def test_verbose_events_emitted_on_inference(verbose_echo_app):
    transport = ASGITransport(app=verbose_echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 50,
            },
        )
        resp = await client.get("/v1/debug/runtime-events")
        data = resp.json()

    assert data["enabled"] is True
    event_names = [e["event_name"] for e in data["events"]]
    assert "inference.request_received" in event_names
    assert "inference.completed" in event_names


@pytest.fixture
def non_verbose_echo_app():
    return _make_echo_app(verbose=False)


@pytest.mark.asyncio
async def test_verbose_off_no_extra_events(non_verbose_echo_app):
    transport = ASGITransport(app=non_verbose_echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        resp = await client.get("/v1/debug/runtime-events")
        data = resp.json()

    assert data["enabled"] is False
    assert data["events"] == []
