"""Tests for ActivationLoop — staged detection, warmup, drain, rollback."""

from __future__ import annotations

import json

import pytest

from octomil.device_agent.activation_manager import ActivationManager
from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.inference_session_manager import InferenceSessionManager
from octomil.device_agent.loops.activation_loop import ActivationLoop
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.policy.policy_engine import PolicyEngine
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore


@pytest.fixture
def components(tmp_path):
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry = DeviceModelRegistry(db, models_dir=models_dir)
    activation_mgr = ActivationManager(db, registry)
    session_mgr = InferenceSessionManager()
    policy = PolicyEngine()
    tel_store = TelemetryStore(db, device_id="dev1", boot_id="boot1")
    yield db, registry, activation_mgr, session_mgr, policy, tel_store
    db.close()


def _register_staged_artifact(db: LocalDB, artifact_id: str, model_id: str, version: str) -> None:
    """Register an artifact and transition it to STAGED."""
    manifest = json.dumps({"files": []})
    db.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id, model_id, version, status, manifest_json, total_bytes, staged_at, updated_at) "
        "VALUES (?, ?, ?, 'STAGED', ?, 100, 'now', 'now')",
        (artifact_id, model_id, version, manifest),
    )


class TestCheckStaged:
    def test_no_staged_is_noop(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()  # should not raise

    def test_staged_artifact_gets_activated(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        _register_staged_artifact(db, "a1", "m1", "v1")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        # Model should be activated
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"

        # Telemetry event logged
        events = tel_store.get_unsent(batch_size=10)
        event_types = [e["event_type"] for e in events]
        assert "artifact.activated" in event_types

    def test_warmup_failure_triggers_rollback(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components

        # Set up: v1 is active, v2 is staged
        manifest = json.dumps({"files": []})
        db.execute(
            "INSERT INTO model_artifacts "
            "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
            "VALUES (?, ?, ?, 'ACTIVE', ?, 100, 'now')",
            ("a1", "m1", "v1", manifest),
        )
        registry.set_active_model("m1", "v1")

        # v2 staged — but we'll make warmup fail by setting it back to STAGED first
        # The warmup call on ActivationManager expects STAGED->WARMING transition
        # But if the artifact is actually in a bad state, warmup returns False
        db.execute(
            "INSERT INTO model_artifacts "
            "(artifact_id, model_id, version, status, manifest_json, total_bytes, staged_at, updated_at) "
            "VALUES (?, ?, ?, 'STAGED', ?, 100, 'now', 'now')",
            ("a2", "m1", "v2", manifest),
        )

        # Patch ActivationManager.warmup to fail
        original_warmup = activation_mgr.warmup
        activation_mgr.warmup = lambda artifact_id, warmup_fn=None: False

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        # Active model should still be v1 (unchanged)
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"

        # Restore
        activation_mgr.warmup = original_warmup

    def test_policy_blocks_warmup(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        _register_staged_artifact(db, "a1", "m1", "v1")

        # Thermal throttle blocks warmup
        policy.update_device_state(thermal_state="critical")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop._check_staged()

        # Should NOT have activated
        active = registry.get_active_model("m1")
        assert active is None


class TestDrainOldVersion:
    def test_drain_with_zero_refcount(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            drain_timeout=1.0,
        )
        # Should complete immediately since no sessions exist
        loop._drain_old_version("m1", "v1")

    def test_drain_timeout_with_active_session(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        # Acquire a session on v1
        handle = session_mgr.acquire("m1", "v1")

        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            drain_timeout=0.2,
        )
        # Should timeout since session is still held
        loop._drain_old_version("m1", "v1")
        # Still pinned
        assert session_mgr.get_refcount("m1", "v1") == 1
        session_mgr.release(handle)


class TestLifecycle:
    def test_start_stop(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
            check_interval=0.1,
        )
        assert not loop.is_running
        loop.start()
        assert loop.is_running
        loop.stop()
        assert not loop.is_running

    def test_double_start_safe(self, components) -> None:
        db, registry, activation_mgr, session_mgr, policy, tel_store = components
        loop = ActivationLoop(
            model_registry=registry,
            activation_manager=activation_mgr,
            session_manager=session_mgr,
            policy_engine=policy,
            telemetry_store=tel_store,
        )
        loop.start()
        loop.start()
        assert loop.is_running
        loop.stop()
