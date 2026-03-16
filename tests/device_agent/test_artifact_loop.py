"""Tests for ArtifactLoop — server polling, reconciliation, download + verify pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from octomil.device_agent.artifact_downloader import ArtifactDownloader
from octomil.device_agent.artifact_verifier import ArtifactVerifier
from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.loops.artifact_loop import ArtifactLoop
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.operation_scheduler import OperationScheduler
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
    downloader = ArtifactDownloader(db, models_dir=models_dir)
    verifier = ArtifactVerifier(db, models_dir=models_dir)
    policy = PolicyEngine()
    scheduler = OperationScheduler(db)
    tel_store = TelemetryStore(db, device_id="dev1", boot_id="boot1")
    yield db, registry, downloader, verifier, policy, scheduler, tel_store
    db.close()


class TestPollDesiredState:
    def test_no_server_client_is_noop(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
            server_client=None,
        )
        # Should not raise
        loop._poll_desired_state()

    def test_new_artifact_discovered(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        mock_client = MagicMock()
        mock_client.get_desired_state.return_value = [
            {
                "model_id": "m1",
                "version": "v2",
                "artifact_id": "art1",
                "manifest": {"files": [{"path": "model.bin", "size": 1000, "sha256": "abc"}]},
                "total_bytes": 1000,
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

        # Artifact should be registered
        art = registry.get_artifact("art1")
        assert art is not None
        assert art["model_id"] == "m1"
        assert art["version"] == "v2"
        assert art["status"] == "REGISTERED"

        # Telemetry event should be logged
        events = tel_store.get_unsent(batch_size=10)
        event_types = [e["event_type"] for e in events]
        assert "artifact.discovered" in event_types

    def test_duplicate_artifact_not_reregistered(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        manifest = json.dumps({"files": []})
        registry.register_artifact("art1", "m1", "v1", manifest, 100)

        mock_client = MagicMock()
        mock_client.get_desired_state.return_value = [
            {"model_id": "m1", "version": "v1", "artifact_id": "art1", "manifest": {}, "total_bytes": 100}
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
        # No new telemetry events since it was already registered
        events = tel_store.get_unsent(batch_size=10)
        assert len(events) == 0

    def test_server_error_is_graceful(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        mock_client = MagicMock()
        mock_client.get_desired_state.side_effect = RuntimeError("connection refused")
        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
            server_client=mock_client,
        )
        # Should not raise
        loop._poll_desired_state()


class TestReconcile:
    def test_registered_artifact_gets_scheduled(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        manifest = json.dumps({"files": []})
        registry.register_artifact("art1", "m1", "v1", manifest, 100)
        policy.update_device_state(network_type="wifi")

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        loop._reconcile()

        pending = scheduler.get_pending(op_type="artifact_download")
        assert len(pending) == 1
        assert pending[0]["resource_id"] == "art1"

    def test_duplicate_schedule_prevented(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        manifest = json.dumps({"files": []})
        registry.register_artifact("art1", "m1", "v1", manifest, 100)
        policy.update_device_state(network_type="wifi")

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        loop._reconcile()
        loop._reconcile()  # second call

        pending = scheduler.get_pending(op_type="artifact_download")
        assert len(pending) == 1  # not duplicated

    def test_policy_blocks_download(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        manifest = json.dumps({"files": []})
        registry.register_artifact("art1", "m1", "v1", manifest, 100)

        # Set low battery, not charging
        policy.update_device_state(battery_pct=5, is_charging=False)

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        loop._reconcile()

        pending = scheduler.get_pending(op_type="artifact_download")
        assert len(pending) == 0  # blocked by policy


class TestLoopLifecycle:
    def test_start_stop(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
            poll_interval=9999,
            process_interval=0.1,
        )
        assert not loop.is_running
        loop.start()
        assert loop.is_running
        loop.stop()
        assert not loop.is_running

    def test_double_start_safe(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        loop.start()
        loop.start()
        assert loop.is_running
        loop.stop()


class TestProcessArtifact:
    def test_missing_artifact_returns_false(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        assert loop._process_artifact("nonexistent") is False

    def test_policy_blocks_returns_false(self, components) -> None:
        db, registry, downloader, verifier, policy, scheduler, tel_store = components
        manifest = json.dumps({"files": []})
        registry.register_artifact("art1", "m1", "v1", manifest, 100)
        policy.update_device_state(battery_pct=5, is_charging=False)

        loop = ArtifactLoop(
            model_registry=registry,
            downloader=downloader,
            verifier=verifier,
            policy_engine=policy,
            operation_scheduler=scheduler,
            telemetry_store=tel_store,
        )
        assert loop._process_artifact("art1") is False
