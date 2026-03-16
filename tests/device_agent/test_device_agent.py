"""Tests for DeviceAgent — top-level entrypoint wiring and lifecycle."""

from __future__ import annotations

import pytest

from octomil.device_agent.device_agent import DeviceAgent


@pytest.fixture
def agent(tmp_path):
    models_dir = tmp_path / "models"
    agent = DeviceAgent(
        db_path=":memory:",
        models_dir=models_dir,
        server_base_url=None,
        device_id="test-device",
    )
    yield agent
    # Ensure loops are stopped even if test forgets
    agent.stop()


class TestInit:
    def test_components_initialized(self, agent: DeviceAgent) -> None:
        assert agent._db is not None
        assert agent._model_registry is not None
        assert agent._downloader is not None
        assert agent._verifier is not None
        assert agent._activation_manager is not None
        assert agent._session_manager is not None
        assert agent._scheduler is not None
        assert agent._policy_engine is not None
        assert agent._telemetry_store is not None
        assert agent._telemetry_uploader is not None

    def test_loops_initialized(self, agent: DeviceAgent) -> None:
        assert agent._inference_loop is not None
        assert agent._artifact_loop is not None
        assert agent._activation_loop is not None
        assert agent._telemetry_loop is not None

    def test_device_id_auto_generated(self, tmp_path) -> None:
        agent = DeviceAgent(db_path=":memory:", models_dir=tmp_path / "m")
        assert agent._device_id
        assert len(agent._device_id) > 0
        agent.stop()

    def test_custom_device_id(self, agent: DeviceAgent) -> None:
        assert agent._device_id == "test-device"

    def test_policy_config_dict(self, tmp_path) -> None:
        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "m",
            policy_config={"min_battery_for_background": 25},
        )
        state = agent._policy_engine.get_device_state()
        assert state is not None
        agent.stop()

    def test_models_dir_created(self, tmp_path) -> None:
        models_dir = tmp_path / "new_models"
        agent = DeviceAgent(db_path=":memory:", models_dir=models_dir)
        assert models_dir.exists()
        agent.stop()


class TestLifecycle:
    def test_start_all_loops(self, agent: DeviceAgent) -> None:
        agent.start()
        assert agent._inference_loop.is_running
        assert agent._artifact_loop.is_running
        assert agent._activation_loop.is_running
        assert agent._telemetry_loop.is_running

    def test_stop_all_loops(self, agent: DeviceAgent) -> None:
        agent.start()
        agent.stop()
        assert not agent._inference_loop.is_running
        assert not agent._artifact_loop.is_running
        assert not agent._activation_loop.is_running
        assert not agent._telemetry_loop.is_running


class TestInfer:
    def test_infer_stub(self, agent: DeviceAgent) -> None:
        # Set up an active model
        agent._model_registry.set_active_model("m1", "v1")
        result = agent.infer("m1", "hello world")
        assert result["model_id"] == "m1"
        assert result["version"] == "v1"

    def test_infer_with_custom_fn(self, tmp_path) -> None:
        def my_fn(model_id, version, model_path, prompt, **kw):
            return {"answer": f"processed: {prompt}"}

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "m",
            inference_fn=my_fn,
        )
        agent._model_registry.set_active_model("m1", "v1")
        result = agent.infer("m1", "test")
        assert result["answer"] == "processed: test"
        agent.stop()

    def test_infer_no_active_model_raises(self, agent: DeviceAgent) -> None:
        with pytest.raises(ValueError, match="No active model"):
            agent.infer("nonexistent", "hello")

    def test_infer_while_running(self, agent: DeviceAgent) -> None:
        agent._model_registry.set_active_model("m1", "v1")
        agent.start()
        result = agent.infer("m1", "hello from bg")
        assert result["model_id"] == "m1"

    def test_infer_kwargs_forwarded(self, tmp_path) -> None:
        received = {}

        def capture_fn(model_id, version, model_path, prompt, **kw):
            received.update(kw)
            return {"ok": True}

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "m",
            inference_fn=capture_fn,
        )
        agent._model_registry.set_active_model("m1", "v1")
        agent.infer("m1", "test", temperature=0.5)
        assert received["temperature"] == 0.5
        agent.stop()


class TestGetStatus:
    def test_status_structure(self, agent: DeviceAgent) -> None:
        status = agent.get_status()
        assert "device_id" in status
        assert "boot_id" in status
        assert "active_models" in status
        assert "downloads" in status
        assert "loops" in status
        assert "active_sessions" in status
        assert "device_state" in status

    def test_status_device_id(self, agent: DeviceAgent) -> None:
        status = agent.get_status()
        assert status["device_id"] == "test-device"

    def test_status_active_models(self, agent: DeviceAgent) -> None:
        agent._model_registry.set_active_model("m1", "v1")
        status = agent.get_status()
        assert "m1" in status["active_models"]
        assert status["active_models"]["m1"]["active_version"] == "v1"

    def test_status_loop_states_before_start(self, agent: DeviceAgent) -> None:
        status = agent.get_status()
        loops = status["loops"]
        assert loops["inference_loop"] is False
        assert loops["artifact_loop"] is False
        assert loops["activation_loop"] is False
        assert loops["telemetry_loop"] is False

    def test_status_loop_states_after_start(self, agent: DeviceAgent) -> None:
        agent.start()
        status = agent.get_status()
        loops = status["loops"]
        assert loops["inference_loop"] is True
        assert loops["artifact_loop"] is True
        assert loops["activation_loop"] is True
        assert loops["telemetry_loop"] is True


class TestUpdateDeviceState:
    def test_battery_update(self, agent: DeviceAgent) -> None:
        agent.update_device_state(battery_pct=42)
        state = agent._policy_engine.get_device_state()
        assert state["battery_pct"] == 42

    def test_network_update(self, agent: DeviceAgent) -> None:
        agent.update_device_state(network_type="wifi")
        state = agent._policy_engine.get_device_state()
        assert state["network_type"] == "wifi"

    def test_full_state_update(self, agent: DeviceAgent) -> None:
        agent.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="ethernet",
            thermal_state="nominal",
            free_storage_bytes=5_000_000_000,
            is_foreground=True,
        )
        state = agent._policy_engine.get_device_state()
        assert state["battery_pct"] == 80
        assert state["is_charging"] is True
        assert state["network_type"] == "ethernet"
        assert state["thermal_state"] == "nominal"
        assert state["free_storage_bytes"] == 5_000_000_000
        assert state["is_foreground"] is True

    def test_partial_update_preserves_other_fields(self, agent: DeviceAgent) -> None:
        agent.update_device_state(battery_pct=50, network_type="wifi")
        agent.update_device_state(battery_pct=30)
        state = agent._policy_engine.get_device_state()
        assert state["battery_pct"] == 30
        assert state["network_type"] == "wifi"  # unchanged
