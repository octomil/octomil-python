"""Tests for DeviceAgent wiring — OctomilControl integration, CrashDetector
startup, and activation policy handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from octomil.device_agent.activation_manager import ActivationManager
from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.device_agent import DeviceAgent
from octomil.device_agent.inference_session_manager import InferenceSessionManager
from octomil.device_agent.loops.activation_loop import ActivationLoop
from octomil.device_agent.loops.artifact_loop import ArtifactLoop
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.operation_scheduler import OperationScheduler
from octomil.device_agent.policy.policy_engine import PolicyEngine
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def components(tmp_path):
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry = DeviceModelRegistry(db, models_dir=models_dir)
    downloader = MagicMock()
    verifier = MagicMock()
    policy = PolicyEngine()
    scheduler = OperationScheduler(db)
    tel_store = TelemetryStore(db, device_id="dev1", boot_id="boot1")
    activation_mgr = ActivationManager(db, registry)
    session_mgr = InferenceSessionManager()
    yield (
        db,
        registry,
        downloader,
        verifier,
        policy,
        scheduler,
        tel_store,
        activation_mgr,
        session_mgr,
    )
    db.close()


# ---------------------------------------------------------------------------
# OctomilControl → ArtifactLoop wiring
# ---------------------------------------------------------------------------


class TestControlToArtifactLoopWiring:
    """Verify that OctomilControl can serve as the server_client for ArtifactLoop."""

    def test_control_get_desired_state_used_by_artifact_loop(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store, _, _ = components

        mock_control = MagicMock()
        mock_control.get_desired_state.return_value = [
            {
                "model_id": "m1",
                "version": "v2",
                "artifact_id": "art-1",
                "manifest": {"files": [{"path": "model.bin", "size": 1000, "sha256": "abc"}]},
                "total_bytes": 1000,
                "activation_policy": "immediate",
            }
        ]

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
            server_client=mock_control,
        )
        loop._poll_desired_state()

        mock_control.get_desired_state.assert_called_once()
        art = registry.get_artifact("art-1")
        assert art is not None
        assert art["model_id"] == "m1"
        assert art["version"] == "v2"
        assert art["activation_policy"] == "immediate"

    def test_device_agent_uses_control_as_server_client(self, tmp_path) -> None:
        mock_control = MagicMock()
        mock_control.get_desired_state.return_value = []

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
            control=mock_control,
        )

        # The artifact loop should have the control as its server_client
        assert agent._artifact_loop._server_client is mock_control
        agent.stop()

    def test_explicit_server_client_takes_precedence(self, tmp_path) -> None:
        mock_control = MagicMock()
        mock_server_client = MagicMock()

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
            control=mock_control,
            server_client=mock_server_client,
        )

        # Explicit server_client takes precedence over control
        assert agent._artifact_loop._server_client is mock_server_client
        agent.stop()

    def test_activation_policy_passed_through(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store, _, _ = components

        mock_client = MagicMock()
        mock_client.get_desired_state.return_value = [
            {
                "model_id": "m1",
                "version": "v1",
                "artifact_id": "art-manual",
                "manifest": {"files": []},
                "total_bytes": 100,
                "activation_policy": "manual",
            }
        ]

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
            server_client=mock_client,
        )
        loop._poll_desired_state()

        art = registry.get_artifact("art-manual")
        assert art is not None
        assert art["activation_policy"] == "manual"


# ---------------------------------------------------------------------------
# CrashDetector → DeviceAgent startup wiring
# ---------------------------------------------------------------------------


class TestCrashDetectorStartupWiring:
    def test_boot_recorded_on_start(self, tmp_path) -> None:
        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
        )
        agent.start()

        history = agent._crash_detector.get_boot_history()
        assert len(history) == 1
        assert history[0]["boot_id"] == agent._boot_id
        agent.stop()

    def test_clean_shutdown_recorded_on_stop(self, tmp_path) -> None:
        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
        )
        agent.start()
        agent.stop()

        history = agent._crash_detector.get_boot_history()
        assert len(history) == 1
        assert history[0]["clean_shutdown"] == 1

    def test_crash_loop_triggers_rollback(self, tmp_path) -> None:
        db = LocalDB(":memory:")
        for stmt in TELEMETRY_SCHEMA_STATEMENTS:
            db.execute(stmt)

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
        )

        # Set up an active model
        agent._model_registry.set_active_model("m1", "v2")
        # Set previous version so rollback has somewhere to go
        agent._model_registry.set_active_model("m1", "v3")

        # Simulate 3 crashed boots for m1
        for i in range(3):
            agent._crash_detector.record_boot(f"crash-{i}", "m1", "v3")
        # One more to mark previous as crashes
        agent._crash_detector.record_boot("crash-final", "m1", "v3")

        # Now start the agent — crash loop detection should trigger rollback
        agent.start()

        # The active model should have been rolled back from v3 to v2
        active = agent._model_registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v2"

        agent.stop()

    def test_no_crash_loop_no_rollback(self, tmp_path) -> None:
        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=tmp_path / "models",
            device_id="test-dev",
        )

        # Set up active model
        agent._model_registry.set_active_model("m1", "v1")

        agent.start()

        # Should remain on v1
        active = agent._model_registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"

        agent.stop()


# ---------------------------------------------------------------------------
# Activation policy handling
# ---------------------------------------------------------------------------


def _register_staged_artifact(
    db: LocalDB,
    artifact_id: str,
    model_id: str,
    version: str,
    activation_policy: str = "immediate",
) -> None:
    manifest = json.dumps({"files": []})
    db.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id, model_id, version, status, manifest_json, total_bytes, "
        " activation_policy, staged_at, updated_at) "
        "VALUES (?, ?, ?, 'STAGED', ?, 100, ?, 'now', 'now')",
        (artifact_id, model_id, version, manifest, activation_policy),
    )


class TestActivationPolicyImmediate:
    def test_immediate_activates_immediately(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "immediate")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"


class TestActivationPolicyNextLaunch:
    def test_next_launch_skipped_when_not_startup(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "next_launch")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            is_startup=False,
        )
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is None  # Not activated

    def test_next_launch_activates_during_startup(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "next_launch")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            is_startup=True,
        )
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"


class TestActivationPolicyManual:
    def test_manual_never_auto_activates(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "manual")

        # Try both startup and non-startup
        for is_startup in (True, False):
            loop = ActivationLoop(
                model_registry=registry,
                activation_manager=activation_mgr,
                session_manager=session_mgr,
                policy_engine=policy,
                telemetry_store=tel_store,
                is_startup=is_startup,
            )
            loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is None  # Never activated


class TestActivationPolicyWhenIdle:
    def test_when_idle_activates_with_no_sessions(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "when_idle")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"

    def test_when_idle_skipped_with_active_sessions(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components
        _register_staged_artifact(db, "a1", "m1", "v1", "when_idle")

        # Acquire a session to simulate active inference
        handle = session_mgr.acquire("m1", "v0")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is None  # Not activated while session is active

        # Release the session and try again
        session_mgr.release(handle)
        loop._check_staged()

        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"


class TestMarkStartupComplete:
    def test_mark_startup_complete(self, components) -> None:
        db, registry, _, _, policy, _, tel_store, activation_mgr, session_mgr = components

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            is_startup=True,
        )
        assert loop._is_startup is True
        loop.mark_startup_complete()
        assert loop._is_startup is False
