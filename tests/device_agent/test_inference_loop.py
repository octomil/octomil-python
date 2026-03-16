"""Tests for InferenceLoop — request processing, session pinning, telemetry."""

from __future__ import annotations

import json

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.inference_session_manager import InferenceSessionManager
from octomil.device_agent.loops.inference_loop import InferenceLoop, InferenceRequest
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore


@pytest.fixture
def components(tmp_path):
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    registry = DeviceModelRegistry(db, models_dir=tmp_path / "models")
    session_mgr = InferenceSessionManager()
    tel_store = TelemetryStore(db, device_id="dev1", boot_id="boot1")
    # Set up an active model
    registry.set_active_model("m1", "v1")
    yield db, registry, session_mgr, tel_store
    db.close()


class TestProcessRequestInline:
    """Tests for process_request when the loop is NOT running (inline execution)."""

    def test_inline_stub_inference(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        request = InferenceRequest(model_id="m1", prompt="hello")
        result = loop.process_request(request)
        assert result["model_id"] == "m1"
        assert result["version"] == "v1"
        assert "stub" in result["output"]

    def test_inline_custom_inference_fn(self, components) -> None:
        db, registry, session_mgr, tel_store = components

        def my_fn(model_id, version, model_path, prompt, **kw):
            return {"answer": f"reply to {prompt}", "model_id": model_id}

        loop = InferenceLoop(session_mgr, registry, tel_store, inference_fn=my_fn)
        request = InferenceRequest(model_id="m1", prompt="test prompt")
        result = loop.process_request(request)
        assert result["answer"] == "reply to test prompt"

    def test_inline_no_active_model_raises(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        request = InferenceRequest(model_id="nonexistent", prompt="hello")
        with pytest.raises(ValueError, match="No active model"):
            loop.process_request(request)

    def test_session_released_after_request(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        request = InferenceRequest(model_id="m1", prompt="hello")
        loop.process_request(request)
        assert session_mgr.get_refcount("m1", "v1") == 0

    def test_session_released_on_error(self, components) -> None:
        db, registry, session_mgr, tel_store = components

        def failing_fn(**kw):
            raise RuntimeError("boom")

        loop = InferenceLoop(session_mgr, registry, tel_store, inference_fn=failing_fn)
        request = InferenceRequest(model_id="m1", prompt="hello")
        with pytest.raises(RuntimeError, match="boom"):
            loop.process_request(request)
        assert session_mgr.get_refcount("m1", "v1") == 0

    def test_telemetry_event_logged(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        request = InferenceRequest(model_id="m1", prompt="hello")
        loop.process_request(request)
        events = tel_store.get_unsent(batch_size=10)
        assert len(events) >= 1
        assert events[0]["event_type"] == "serving.request.completed"

    def test_telemetry_on_error(self, components) -> None:
        db, registry, session_mgr, tel_store = components

        def failing_fn(**kw):
            raise RuntimeError("boom")

        loop = InferenceLoop(session_mgr, registry, tel_store, inference_fn=failing_fn)
        request = InferenceRequest(model_id="m1", prompt="hello")
        with pytest.raises(RuntimeError):
            loop.process_request(request)
        events = tel_store.get_unsent(batch_size=10)
        assert len(events) >= 1
        payload = json.loads(events[0]["payload_json"]) if events[0]["payload_json"] else {}
        assert "error" in payload


class TestLoopLifecycle:
    """Tests for start/stop and background request processing."""

    def test_start_stop(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        assert not loop.is_running
        loop.start()
        assert loop.is_running
        loop.stop()
        assert not loop.is_running

    def test_double_start_safe(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        loop.start()
        loop.start()  # should not raise
        assert loop.is_running
        loop.stop()

    def test_background_request(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        loop = InferenceLoop(session_mgr, registry, tel_store)
        loop.start()
        try:
            request = InferenceRequest(model_id="m1", prompt="bg test")
            result = loop.process_request(request)
            assert result["model_id"] == "m1"
        finally:
            loop.stop()

    def test_background_error_propagates(self, components) -> None:
        db, registry, session_mgr, tel_store = components

        def failing_fn(**kw):
            raise RuntimeError("bg boom")

        loop = InferenceLoop(session_mgr, registry, tel_store, inference_fn=failing_fn)
        loop.start()
        try:
            request = InferenceRequest(model_id="m1", prompt="hello")
            with pytest.raises(RuntimeError, match="bg boom"):
                loop.process_request(request)
        finally:
            loop.stop()

    def test_kwargs_forwarded(self, components) -> None:
        db, registry, session_mgr, tel_store = components
        received = {}

        def capture_fn(model_id, version, model_path, prompt, **kw):
            received.update(kw)
            return {"ok": True}

        loop = InferenceLoop(session_mgr, registry, tel_store, inference_fn=capture_fn)
        request = InferenceRequest(model_id="m1", prompt="hi", kwargs={"temperature": 0.7, "max_tokens": 100})
        loop.process_request(request)
        assert received["temperature"] == 0.7
        assert received["max_tokens"] == 100
