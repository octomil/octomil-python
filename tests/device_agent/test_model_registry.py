"""Tests for DeviceModelRegistry CRUD, pointer flip, and rollback."""

from __future__ import annotations

import json

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.model_registry import DeviceModelRegistry


@pytest.fixture
def registry(tmp_path):
    db = LocalDB(":memory:")
    reg = DeviceModelRegistry(db, models_dir=tmp_path / "models")
    yield reg
    db.close()


class TestActivePointer:
    def test_get_active_none(self, registry: DeviceModelRegistry) -> None:
        assert registry.get_active_model("nonexistent") is None

    def test_set_and_get_active(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"
        assert active["previous_version"] is None

    def test_pointer_flip_preserves_previous(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        registry.set_active_model("m1", "v2")
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v2"
        assert active["previous_version"] == "v1"

    def test_triple_flip(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        registry.set_active_model("m1", "v2")
        registry.set_active_model("m1", "v3")
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v3"
        assert active["previous_version"] == "v2"


class TestArtifactCRUD:
    def test_register_and_get(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": [{"path": "model.bin", "size": 100}]})
        registry.register_artifact("a1", "m1", "v1", manifest, 100)
        art = registry.get_artifact("a1")
        assert art is not None
        assert art["model_id"] == "m1"
        assert art["status"] == "REGISTERED"
        assert art["bytes_downloaded"] == 0

    def test_update_status(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": []})
        registry.register_artifact("a1", "m1", "v1", manifest, 100)
        registry.update_artifact_status("a1", "DOWNLOADING", bytes_downloaded=50)
        art = registry.get_artifact("a1")
        assert art is not None
        assert art["status"] == "DOWNLOADING"
        assert art["bytes_downloaded"] == 50

    def test_update_invalid_field_raises(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": []})
        registry.register_artifact("a1", "m1", "v1", manifest, 100)
        with pytest.raises(ValueError, match="Cannot update field"):
            registry.update_artifact_status("a1", "ACTIVE", bad_field="x")

    def test_get_nonexistent(self, registry: DeviceModelRegistry) -> None:
        assert registry.get_artifact("nonexistent") is None

    def test_list_installed(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": []})
        registry.register_artifact("a1", "m1", "v1", manifest, 100)
        registry.update_artifact_status("a1", "DOWNLOADING")
        # DOWNLOADING is not in the list filter
        installed = registry.list_installed_versions("m1")
        assert len(installed) == 0

    def test_staged_versions(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": []})
        registry.register_artifact("a1", "m1", "v1", manifest, 100)
        # Manually set status to STAGED for test
        registry._db.execute("UPDATE model_artifacts SET status = 'STAGED', staged_at = 'now' WHERE artifact_id = 'a1'")
        staged = registry.get_staged_versions("m1")
        assert len(staged) == 1
        assert staged[0]["version"] == "v1"


class TestRollback:
    def test_rollback_no_previous(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        result = registry.rollback("m1", "test")
        assert result is None

    def test_rollback_success(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        registry.set_active_model("m1", "v2")
        result = registry.rollback("m1", "warmup failed")
        assert result == "v1"
        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"
        assert active["previous_version"] == "v2"

    def test_rollback_creates_record(self, registry: DeviceModelRegistry) -> None:
        registry.set_active_model("m1", "v1")
        registry.set_active_model("m1", "v2")
        registry.rollback("m1", "crash loop")
        rows = registry._db.execute("SELECT * FROM rollback_records WHERE model_id = 'm1'")
        assert len(rows) == 1
        assert rows[0]["from_version"] == "v2"
        assert rows[0]["to_version"] == "v1"
        assert rows[0]["reason"] == "crash loop"

    def test_rollback_nonexistent_model(self, registry: DeviceModelRegistry) -> None:
        result = registry.rollback("nonexistent", "test")
        assert result is None


class TestGC:
    def test_gc_eligible(self, registry: DeviceModelRegistry) -> None:
        manifest = json.dumps({"files": []})
        for v in ["v1", "v2", "v3"]:
            registry.register_artifact(f"a-{v}", "m1", v, manifest, 100)
            registry._db.execute(
                "UPDATE model_artifacts SET status = 'ACTIVE' WHERE artifact_id = ?",
                (f"a-{v}",),
            )
        registry.set_active_model("m1", "v2")
        registry.set_active_model("m1", "v3")
        eligible = registry.gc_eligible_versions("m1")
        # v3 is active, v2 is previous, v1 is eligible
        assert "v1" in eligible
        assert "v2" not in eligible
        assert "v3" not in eligible


class TestModelPath:
    def test_path_format(self, registry: DeviceModelRegistry, tmp_path) -> None:
        path = registry.get_model_path("my-model", "v1.0")
        assert str(path) == str(tmp_path / "models" / "my-model" / "v1.0")
