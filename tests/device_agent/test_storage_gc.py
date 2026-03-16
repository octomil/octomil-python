"""Tests for StorageGC scan, collect, dry_run, and protection rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.inference_session_manager import InferenceSessionManager
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.storage_gc import StorageGC


def _make_model_dir(models_dir: Path, model_id: str, version: str, size_bytes: int = 1024) -> Path:
    """Create a model version directory with a dummy file of the given size."""
    version_dir = models_dir / model_id / version
    version_dir.mkdir(parents=True, exist_ok=True)
    data_file = version_dir / "model.bin"
    data_file.write_bytes(b"\x00" * size_bytes)
    return version_dir


def _register_artifact(db: LocalDB, artifact_id: str, model_id: str, version: str, status: str = "ACTIVE") -> None:
    """Insert a model_artifacts row at the given status."""
    manifest = json.dumps({"files": []})
    db.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 100, 'now')",
        (artifact_id, model_id, version, status, manifest),
    )


@pytest.fixture
def env(tmp_path):
    db = LocalDB(":memory:")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reg = DeviceModelRegistry(db, models_dir=models_dir)
    sessions = InferenceSessionManager()
    gc = StorageGC(reg, sessions, models_dir=models_dir)
    yield db, reg, sessions, gc, models_dir
    db.close()


class TestScan:
    def test_scan_empty(self, env) -> None:
        _, _, _, gc, _ = env
        assert gc.scan() == []

    def test_scan_finds_eligible(self, env) -> None:
        db, reg, _, gc, models_dir = env
        # v1 = old, v2 = previous, v3 = active
        for v in ["v1", "v2", "v3"]:
            _register_artifact(db, f"a-{v}", "m1", v, status="ACTIVE")
            _make_model_dir(models_dir, "m1", v)
        reg.set_active_model("m1", "v2")
        reg.set_active_model("m1", "v3")

        eligible = gc.scan()
        assert ("m1", "v1") in eligible
        assert ("m1", "v2") not in eligible  # previous
        assert ("m1", "v3") not in eligible  # active

    def test_scan_excludes_staged(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="STAGED")
        db.execute("UPDATE model_artifacts SET staged_at = 'now' WHERE artifact_id = 'a-v2'")
        _make_model_dir(models_dir, "m1", "v1")
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v1")

        eligible = gc.scan()
        versions = [v for _, v in eligible]
        assert "v2" not in versions

    def test_scan_excludes_sessions_with_refcount(self, env) -> None:
        db, reg, sessions, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1")
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v2")

        # Pin v1 with an active session
        handle = sessions.acquire("m1", "v1")
        eligible = gc.scan()
        versions = [v for _, v in eligible]
        assert "v1" not in versions

        # Release and rescan
        sessions.release(handle)
        eligible = gc.scan()
        versions = [v for _, v in eligible]
        assert "v1" in versions


class TestCollect:
    def test_collect_removes_directory(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1", size_bytes=2048)
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v2")

        freed = gc.collect("m1", "v1")
        assert freed == 2048
        assert not (models_dir / "m1" / "v1").exists()

    def test_collect_refuses_active(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1")
        reg.set_active_model("m1", "v1")

        freed = gc.collect("m1", "v1")
        assert freed == 0
        assert (models_dir / "m1" / "v1").exists()

    def test_collect_refuses_previous(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1")
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v1")
        reg.set_active_model("m1", "v2")

        freed = gc.collect("m1", "v1")
        assert freed == 0
        assert (models_dir / "m1" / "v1").exists()

    def test_collect_refuses_active_sessions(self, env) -> None:
        db, reg, sessions, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1")
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v2")

        handle = sessions.acquire("m1", "v1")
        freed = gc.collect("m1", "v1")
        assert freed == 0
        assert (models_dir / "m1" / "v1").exists()
        sessions.release(handle)

    def test_collect_refuses_staged(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="STAGED")
        db.execute("UPDATE model_artifacts SET staged_at = 'now' WHERE artifact_id = 'a-v1'")
        _make_model_dir(models_dir, "m1", "v1")

        freed = gc.collect("m1", "v1")
        assert freed == 0
        assert (models_dir / "m1" / "v1").exists()

    def test_collect_nonexistent_directory(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        reg.set_active_model("m1", "v2")
        # Don't create the directory
        freed = gc.collect("m1", "v1")
        assert freed == 0


class TestDryRun:
    def test_dry_run_does_not_delete(self, env) -> None:
        db, reg, _, gc, models_dir = env
        _register_artifact(db, "a-v1", "m1", "v1", status="ACTIVE")
        _register_artifact(db, "a-v2", "m1", "v2", status="ACTIVE")
        _make_model_dir(models_dir, "m1", "v1", size_bytes=512)
        _make_model_dir(models_dir, "m1", "v2")
        reg.set_active_model("m1", "v2")

        freed = gc.run(dry_run=True)
        assert freed == 0
        assert (models_dir / "m1" / "v1").exists()


class TestRun:
    def test_run_full_gc(self, env) -> None:
        db, reg, _, gc, models_dir = env
        for v in ["v1", "v2", "v3"]:
            _register_artifact(db, f"a-{v}", "m1", v, status="ACTIVE")
            _make_model_dir(models_dir, "m1", v, size_bytes=1024)
        reg.set_active_model("m1", "v2")
        reg.set_active_model("m1", "v3")

        freed = gc.run()
        assert freed == 1024  # Only v1 deleted
        assert not (models_dir / "m1" / "v1").exists()
        assert (models_dir / "m1" / "v2").exists()  # previous — kept
        assert (models_dir / "m1" / "v3").exists()  # active — kept


class TestStorageUsage:
    def test_empty_models_dir(self, env) -> None:
        _, _, _, gc, _ = env
        usage = gc.get_storage_usage()
        assert usage["total_bytes"] == 0
        assert usage["by_model"] == {}

    def test_models_dir_not_created(self, tmp_path) -> None:
        db = LocalDB(":memory:")
        reg = DeviceModelRegistry(db, models_dir=tmp_path / "nonexistent")
        sessions = InferenceSessionManager()
        gc = StorageGC(reg, sessions, models_dir=tmp_path / "nonexistent")
        usage = gc.get_storage_usage()
        assert usage["total_bytes"] == 0
        db.close()

    def test_reports_correct_sizes(self, env) -> None:
        _, _, _, gc, models_dir = env
        _make_model_dir(models_dir, "m1", "v1", size_bytes=1000)
        _make_model_dir(models_dir, "m1", "v2", size_bytes=2000)
        _make_model_dir(models_dir, "m2", "v1", size_bytes=500)

        usage = gc.get_storage_usage()
        assert usage["total_bytes"] == 3500
        assert usage["by_model"]["m1"] == 3000
        assert usage["by_model"]["m2"] == 500


class TestProtectionRules:
    """Verify that active + previous versions are always kept even through run()."""

    def test_active_and_previous_always_protected(self, env) -> None:
        db, reg, _, gc, models_dir = env
        for v in ["v1", "v2", "v3", "v4"]:
            _register_artifact(db, f"a-{v}", "m1", v, status="ACTIVE")
            _make_model_dir(models_dir, "m1", v, size_bytes=100)
        reg.set_active_model("m1", "v3")
        reg.set_active_model("m1", "v4")

        gc.run()
        # v4 = active, v3 = previous, both must survive
        assert (models_dir / "m1" / "v4").exists()
        assert (models_dir / "m1" / "v3").exists()
        # v1 and v2 should be collected
        assert not (models_dir / "m1" / "v1").exists()
        assert not (models_dir / "m1" / "v2").exists()
