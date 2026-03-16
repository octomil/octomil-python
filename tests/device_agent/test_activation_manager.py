"""Tests for ActivationManager state transitions."""

from __future__ import annotations

import json

import pytest

from octomil.device_agent.activation_manager import ActivationManager
from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.model_registry import DeviceModelRegistry


@pytest.fixture
def setup(tmp_path):
    db = LocalDB(":memory:")
    reg = DeviceModelRegistry(db, models_dir=tmp_path / "models")
    mgr = ActivationManager(db, reg)
    # Register a test artifact in VERIFIED state
    manifest = json.dumps({"files": []})
    db.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
        "VALUES (?, ?, ?, 'VERIFIED', ?, 100, 'now')",
        ("a1", "m1", "v1", manifest),
    )
    return db, reg, mgr


class TestStage:
    def test_stage_from_verified(self, setup) -> None:
        db, reg, mgr = setup
        assert mgr.stage("a1") is True
        assert mgr.get_activation_state("a1") == "STAGED"

    def test_stage_invalid_from_registered(self, setup) -> None:
        db, _, mgr = setup
        db.execute("UPDATE model_artifacts SET status = 'REGISTERED' WHERE artifact_id = 'a1'")
        assert mgr.stage("a1") is False


class TestWarmup:
    def test_warmup_success(self, setup) -> None:
        _, _, mgr = setup
        mgr.stage("a1")
        assert mgr.warmup("a1") is True
        assert mgr.get_activation_state("a1") == "ACTIVE"

    def test_warmup_with_fn_success(self, setup) -> None:
        _, _, mgr = setup
        mgr.stage("a1")
        assert mgr.warmup("a1", warmup_fn=lambda path: True) is True
        assert mgr.get_activation_state("a1") == "ACTIVE"

    def test_warmup_with_fn_failure(self, setup) -> None:
        _, _, mgr = setup
        mgr.stage("a1")
        assert mgr.warmup("a1", warmup_fn=lambda path: False) is False
        assert mgr.get_activation_state("a1") == "FAILED_HEALTHCHECK"

    def test_warmup_with_fn_exception(self, setup) -> None:
        _, _, mgr = setup
        mgr.stage("a1")

        def bad_warmup(path):
            raise RuntimeError("oops")

        assert mgr.warmup("a1", warmup_fn=bad_warmup) is False
        assert mgr.get_activation_state("a1") == "FAILED_HEALTHCHECK"

    def test_warmup_without_stage_fails(self, setup) -> None:
        _, _, mgr = setup
        # Artifact is VERIFIED, not STAGED — WARMING transition invalid
        assert mgr.warmup("a1") is False


class TestActivate:
    def test_activate_flips_pointer(self, setup) -> None:
        _, reg, mgr = setup
        mgr.activate("m1", "v1")
        active = reg.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v1"


class TestAutoRollback:
    def test_auto_rollback(self, setup) -> None:
        _, reg, mgr = setup
        reg.set_active_model("m1", "v1")
        reg.set_active_model("m1", "v2")
        result = mgr.auto_rollback("m1", "crash loop")
        assert result == "v1"
        active = reg.get_active_model("m1")
        assert active["active_version"] == "v1"


class TestGetActivationState:
    def test_state_returns_correct(self, setup) -> None:
        _, _, mgr = setup
        assert mgr.get_activation_state("a1") == "VERIFIED"

    def test_state_nonexistent(self, setup) -> None:
        _, _, mgr = setup
        assert mgr.get_activation_state("nonexistent") is None


class TestDrainOld:
    def test_drain_no_refcount_fn(self, setup) -> None:
        _, _, mgr = setup
        assert mgr.drain_old("m1", "v1") is True

    def test_drain_zero_refcount(self, setup) -> None:
        _, _, mgr = setup
        assert mgr.drain_old("m1", "v1", refcount_fn=lambda m, v: 0) is True

    def test_drain_timeout(self, setup) -> None:
        _, _, mgr = setup
        # Always returns 1, so it times out
        assert mgr.drain_old("m1", "v1", timeout_sec=0.1, refcount_fn=lambda m, v: 1) is False
