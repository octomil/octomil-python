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
        # Oldest events should be evicted
        assert events[0]["event_name"] == "event.5"

    def test_no_server_posting_without_api_key(self):
        """When api_key is None, no background poster is started."""
        emitter = VerboseEventEmitter()
        assert emitter._poster is None

    def test_close_without_poster(self):
        """close() should not raise when no poster is running."""
        emitter = VerboseEventEmitter()
        emitter.close()  # Should not raise

    def test_emit_with_no_metadata(self):
        emitter = VerboseEventEmitter()
        emitter.emit("simple.event")
        events = emitter.recent_events()
        assert len(events) == 1
        assert events[0]["metadata"] == {}


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
        """create_app without verbose creates app with no verbose emitter."""
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
        """create_app with verbose=True creates a verbose emitter."""
        app = _make_echo_app(verbose=True)
        transport = ASGITransport(app=app)

        async def _check():
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/debug/runtime-events")
            return resp.json()

        result = asyncio.run(_check())
        assert result["enabled"] is True
        # Should have at least the server.started event
        assert len(result["events"]) >= 1
        event_names = [e["event_name"] for e in result["events"]]
        assert "server.started" in event_names


# ---------------------------------------------------------------------------
# Verbose events during inference
# ---------------------------------------------------------------------------


@pytest.fixture
def verbose_echo_app():
    """Create a FastAPI app with EchoBackend and verbose mode enabled."""
    return _make_echo_app(verbose=True)


@pytest.mark.asyncio
async def test_verbose_events_emitted_on_inference(verbose_echo_app):
    """When verbose is on, inference requests emit runtime events."""
    transport = ASGITransport(app=verbose_echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Make an inference request
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 50,
            },
        )

        # Check runtime events
        resp = await client.get("/v1/debug/runtime-events")
        data = resp.json()

    assert data["enabled"] is True
    event_names = [e["event_name"] for e in data["events"]]

    # Should have request_received and completed events
    assert "inference.request_received" in event_names
    assert "inference.completed" in event_names


@pytest.fixture
def non_verbose_echo_app():
    """Create a FastAPI app with EchoBackend and verbose mode disabled."""
    return _make_echo_app(verbose=False)


@pytest.mark.asyncio
async def test_verbose_off_no_extra_events(non_verbose_echo_app):
    """When verbose is off, no runtime events are emitted."""
    transport = ASGITransport(app=non_verbose_echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Make an inference request
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        # Check runtime events endpoint
        resp = await client.get("/v1/debug/runtime-events")
        data = resp.json()

    assert data["enabled"] is False
    assert data["events"] == []
