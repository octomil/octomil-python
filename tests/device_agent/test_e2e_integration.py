"""End-to-end integration tests for the device agent round-trip.

Validates the full lifecycle of models, inference, activation, telemetry,
policy gating, storage GC, crash detection, training, and runtime updates
using in-process components with temp databases and no real network calls.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from octomil.device_agent.activation_manager import ActivationManager
from octomil.device_agent.artifact_downloader import ArtifactDownloader
from octomil.device_agent.artifact_verifier import ArtifactVerifier
from octomil.device_agent.crash_detector import CrashDetector
from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.device_agent import DeviceAgent
from octomil.device_agent.inference_session_manager import InferenceSessionManager
from octomil.device_agent.loops.inference_loop import InferenceLoop, InferenceRequest
from octomil.device_agent.model_registry import DeviceModelRegistry
from octomil.device_agent.policy.policy_engine import PolicyConfig, PolicyEngine
from octomil.device_agent.runtime_updater import RuntimeUpdater
from octomil.device_agent.storage_gc import StorageGC
from octomil.device_agent.telemetry.events import TelemetryClass
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore
from octomil.device_agent.training.local_trainer import (
    InvalidTransitionError,
    LocalTrainer,
    TrainingLimits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> LocalDB:
    """Create an in-memory LocalDB with base + telemetry + training schemas."""
    db = LocalDB(":memory:")
    from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS

    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS

    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


def _write_model_file(model_dir: Path, file_path: str, content: bytes) -> str:
    """Write a file under model_dir and return its SHA-256 hex digest."""
    full = model_dir / file_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _register_and_complete_artifact(
    db: LocalDB,
    registry: DeviceModelRegistry,
    downloader: ArtifactDownloader,
    models_dir: Path,
    artifact_id: str,
    model_id: str,
    version: str,
    files: list[dict[str, Any]],
) -> None:
    """Register an artifact and simulate a completed download by writing files
    and marking all chunks as COMPLETE."""
    # Strip non-serializable 'content' key from manifest entries
    manifest_files = [{k: v for k, v in f.items() if k != "content"} for f in files]
    manifest = {"files": manifest_files}
    total_bytes = sum(f["size"] for f in files)
    registry.register_artifact(artifact_id, model_id, version, json.dumps(manifest), total_bytes)

    model_path = models_dir / model_id / version
    model_path.mkdir(parents=True, exist_ok=True)

    # Write actual file content
    for f in files:
        content = f.get("content", b"\x00" * f["size"])
        _write_model_file(model_path, f["path"], content)

    # Start download (creates chunk rows) then mark all complete
    downloader.start_download(artifact_id, manifest, "http://fake")
    db.execute(
        "UPDATE download_chunks SET status = 'COMPLETE' WHERE artifact_id = ?",
        (artifact_id,),
    )
    db.execute(
        "UPDATE model_artifacts SET bytes_downloaded = total_bytes, status = 'DOWNLOADING' WHERE artifact_id = ?",
        (artifact_id,),
    )


# ---------------------------------------------------------------------------
# 1. Full model lifecycle
# ---------------------------------------------------------------------------


class TestFullModelLifecycle:
    """In-process lifecycle: register -> download -> verify -> stage -> warmup
    -> activate -> infer -> telemetry -> stop."""

    def test_full_round_trip(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        content = b"fake-model-weights-v1"
        sha = hashlib.sha256(content).hexdigest()

        files = [{"path": "model.bin", "size": len(content), "sha256": sha, "content": content}]

        infer_calls: list[dict[str, Any]] = []

        def stub_infer(model_id: str, version: str, model_path: str, prompt: str, **kw: Any) -> dict[str, Any]:
            infer_calls.append({"model_id": model_id, "version": version, "prompt": prompt})
            return {"output": f"result-{prompt}", "model_id": model_id, "version": version}

        agent = DeviceAgent(
            db_path=":memory:",
            models_dir=models_dir,
            device_id="dev-e2e",
            inference_fn=stub_infer,
        )

        try:
            db = agent._db
            registry = agent._model_registry
            downloader = agent._downloader
            verifier = agent._verifier
            activation = agent._activation_manager

            # Register + simulate download
            _register_and_complete_artifact(db, registry, downloader, models_dir, "art-1", "m1", "v1", files)

            # Verify
            ok = verifier.verify_artifact("art-1")
            assert ok, "Verification should succeed for correct sha256"
            artifact = registry.get_artifact("art-1")
            assert artifact is not None
            assert artifact["status"] == "VERIFIED"

            # Stage
            staged = activation.stage("art-1")
            assert staged, "Staging a VERIFIED artifact should succeed"
            assert activation.get_activation_state("art-1") == "STAGED"

            # Warmup (mock to succeed)
            warmed = activation.warmup("art-1", warmup_fn=lambda path: True)
            assert warmed, "Warmup should succeed when warmup_fn returns True"
            assert activation.get_activation_state("art-1") == "ACTIVE"

            # Activate pointer
            activation.activate("m1", "v1")
            active = registry.get_active_model("m1")
            assert active is not None
            assert active["active_version"] == "v1"

            # Inference (no loops running — direct call)
            result = agent.infer("m1", "hello")
            assert result["output"] == "result-hello"
            assert result["version"] == "v1"
            assert len(infer_calls) == 1
            assert infer_calls[0]["version"] == "v1"

            # Verify telemetry event was logged
            events = agent._telemetry_store.get_unsent(batch_size=100)
            event_types = [e["event_type"] for e in events]
            assert "serving.request.completed" in event_types
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# 2. Model update during active inference
# ---------------------------------------------------------------------------


class TestModelUpdateDuringInference:
    """Activate v2 while v1 inference is in-flight; verify version pinning."""

    def test_version_pinning_during_update(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        db = _make_db()
        registry = DeviceModelRegistry(db, models_dir=models_dir)
        session_mgr = InferenceSessionManager()
        telemetry = TelemetryStore(db, device_id="dev", boot_id="boot1")

        v1_started = threading.Event()
        v1_proceed = threading.Event()
        results: dict[str, Any] = {}

        def slow_infer(model_id: str, version: str, model_path: str, prompt: str, **kw: Any) -> dict[str, Any]:
            if version == "v1":
                v1_started.set()
                v1_proceed.wait(timeout=10)
            return {"version": version}

        loop = InferenceLoop(
            session_manager=session_mgr,
            model_registry=registry,
            telemetry_store=telemetry,
            inference_fn=slow_infer,
        )

        # Setup v1 as active
        registry.set_active_model("m1", "v1")
        (models_dir / "m1" / "v1").mkdir(parents=True, exist_ok=True)
        (models_dir / "m1" / "v2").mkdir(parents=True, exist_ok=True)

        loop.start()
        try:
            # Start v1 inference in another thread
            req_v1 = InferenceRequest(model_id="m1", prompt="p1")

            def run_v1() -> None:
                results["v1"] = loop.process_request(req_v1)

            t1 = threading.Thread(target=run_v1)
            t1.start()

            # Wait for v1 inference to begin
            assert v1_started.wait(timeout=5), "v1 inference did not start"

            # v1 is in-flight — refcount should be 1
            assert session_mgr.get_refcount("m1", "v1") == 1

            # Flip pointer to v2
            registry.set_active_model("m1", "v2")

            # New inference should pin v2
            req_v2 = InferenceRequest(model_id="m1", prompt="p2")
            result_v2 = loop.process_request(req_v2)
            assert result_v2["version"] == "v2"

            # Complete v1
            v1_proceed.set()
            t1.join(timeout=5)
            assert results["v1"]["version"] == "v1"

            # v1 refcount should now be 0
            assert session_mgr.get_refcount("m1", "v1") == 0
        finally:
            loop.stop()


# ---------------------------------------------------------------------------
# 3. Interrupted download and resume
# ---------------------------------------------------------------------------


class TestInterruptedDownloadResume:
    """Register multi-chunk artifact, mark some done, resume only pending."""

    def test_resume_targets_only_pending_chunks(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        db = _make_db()
        registry = DeviceModelRegistry(db, models_dir=models_dir)
        downloader = ArtifactDownloader(db, models_dir=models_dir)

        # File large enough for 3 chunks at 8 MiB each
        file_size = 3 * 8 * 1024 * 1024
        files = [{"path": "big.bin", "size": file_size, "sha256": "abc123"}]
        manifest = {"files": files}
        registry.register_artifact("art-resume", "m1", "v1", json.dumps(manifest), file_size)

        # Start download — creates 3 chunk rows
        downloader.start_download("art-resume", manifest, "http://fake")

        all_chunks = db.execute(
            "SELECT chunk_index FROM download_chunks WHERE artifact_id = 'art-resume' ORDER BY chunk_index"
        )
        assert len(all_chunks) == 3

        # Mark chunk 0 and 1 as complete, leave chunk 2 pending
        db.execute(
            "UPDATE download_chunks SET status = 'COMPLETE' WHERE artifact_id = 'art-resume' AND chunk_index IN (0, 1)"
        )

        # Resume should return only chunk 2
        pending = downloader.resume_download("art-resume")
        assert len(pending) == 1
        assert pending[0]["chunk_index"] == 2

        # Complete chunk 2
        db.execute(
            "UPDATE download_chunks SET status = 'COMPLETE' WHERE artifact_id = 'art-resume' AND chunk_index = 2"
        )

        # No more pending
        pending_after = downloader.resume_download("art-resume")
        assert len(pending_after) == 0

        # Write the actual file so verification can find it
        model_path = models_dir / "m1" / "v1"
        model_path.mkdir(parents=True, exist_ok=True)
        content = b"\x00" * file_size
        sha = hashlib.sha256(content).hexdigest()
        (model_path / "big.bin").write_bytes(content)

        # Update manifest with correct hash and verify
        db.execute(
            "UPDATE model_artifacts SET manifest_json = ? WHERE artifact_id = 'art-resume'",
            (json.dumps({"files": [{"path": "big.bin", "size": file_size, "sha256": sha}]}),),
        )
        verifier = ArtifactVerifier(db, models_dir=models_dir)
        assert verifier.verify_artifact("art-resume") is True


# ---------------------------------------------------------------------------
# 4. Automatic rollback on warmup failure
# ---------------------------------------------------------------------------


class TestAutoRollbackOnWarmupFailure:
    """Stage v2, warmup fails, verify rollback to v1."""

    def test_warmup_failure_triggers_rollback(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        db = _make_db()
        registry = DeviceModelRegistry(db, models_dir=models_dir)
        activation = ActivationManager(db, registry)
        telemetry = TelemetryStore(db, device_id="dev", boot_id="boot1")

        content_v1 = b"model-v1"
        sha_v1 = hashlib.sha256(content_v1).hexdigest()
        content_v2 = b"model-v2"
        sha_v2 = hashlib.sha256(content_v2).hexdigest()

        # Setup v1 as active
        files_v1 = [{"path": "model.bin", "size": len(content_v1), "sha256": sha_v1, "content": content_v1}]
        downloader = ArtifactDownloader(db, models_dir=models_dir)
        _register_and_complete_artifact(db, registry, downloader, models_dir, "art-v1", "m1", "v1", files_v1)
        verifier = ArtifactVerifier(db, models_dir=models_dir)
        verifier.verify_artifact("art-v1")
        activation.stage("art-v1")
        activation.warmup("art-v1", warmup_fn=lambda p: True)
        activation.activate("m1", "v1")

        active_before = registry.get_active_model("m1")
        assert active_before is not None
        assert active_before["active_version"] == "v1"

        # Register v2 and complete its download + verification + staging
        files_v2 = [{"path": "model.bin", "size": len(content_v2), "sha256": sha_v2, "content": content_v2}]
        _register_and_complete_artifact(db, registry, downloader, models_dir, "art-v2", "m1", "v2", files_v2)
        verifier.verify_artifact("art-v2")
        activation.stage("art-v2")

        # Simulate what the activation loop does: attempt warmup on v2.
        # Warmup fails.
        warmed = activation.warmup("art-v2", warmup_fn=lambda p: False)
        assert warmed is False
        assert activation.get_activation_state("art-v2") == "FAILED_HEALTHCHECK"

        # In production, the activation loop would call auto_rollback only if
        # the pointer had already been flipped. To test the rollback path
        # properly, simulate a scenario where v2 was briefly activated
        # (e.g. a premature flip or a crash during warmup):
        activation.activate("m1", "v2")
        active_v2 = registry.get_active_model("m1")
        assert active_v2 is not None
        assert active_v2["active_version"] == "v2"

        # Now auto-rollback should flip back to v1
        rolled_to = activation.auto_rollback("m1", "warmup_failed")
        assert rolled_to == "v1"

        # Log telemetry event
        telemetry.append_auto(
            "artifact.download.failed",
            {"artifact_id": "art-v2", "error": "warmup_failed", "rolled_back_to": rolled_to},
            model_id="m1",
            model_version="v2",
        )

        # Verify rollback record exists
        rollback_rows = db.execute("SELECT * FROM rollback_records WHERE model_id = 'm1'")
        assert len(rollback_rows) >= 1
        assert rollback_rows[0]["from_version"] == "v2"
        assert rollback_rows[0]["to_version"] == "v1"
        assert rollback_rows[0]["reason"] == "warmup_failed"

        # Verify telemetry event
        events = telemetry.get_unsent(batch_size=100)
        types = [e["event_type"] for e in events]
        assert "artifact.download.failed" in types

        # Active pointer should be back on v1
        active_after = registry.get_active_model("m1")
        assert active_after is not None
        assert active_after["active_version"] == "v1"


# ---------------------------------------------------------------------------
# 5. Crash loop detection and rollback
# ---------------------------------------------------------------------------


class TestCrashLoopDetection:
    """Simulate 3 boots with crashes and verify crash loop detection."""

    def test_crash_loop_triggers_rollback_recommendation(self) -> None:
        db = _make_db()
        detector = CrashDetector(db)

        # Simulate 3 boots with same model, no clean shutdown.
        # Each subsequent record_boot marks all prior unclean boots as crashes.
        # Boot 0: inserted (clean_shutdown=0, crash_detected=0)
        detector.record_boot("boot-0", active_model_id="m1", active_model_version="v1")
        # Boot 1: marks boot-0 as crash, inserts boot-1
        detector.record_boot("boot-1", active_model_id="m1", active_model_version="v1")
        # Boot 2: marks boot-1 as crash, inserts boot-2
        detector.record_boot("boot-2", active_model_id="m1", active_model_version="v1")
        # Boot 3: marks boot-2 as crash, inserts boot-3
        crash_count = detector.record_boot("boot-3", active_model_id="m1", active_model_version="v1")
        # Only 1 unclean boot was unmarked at the time of boot-3 (boot-2)
        assert crash_count == 1

        # But the DB now has 3 total crash_detected=1 rows (boot-0, boot-1, boot-2)
        crash_rows = db.execute("SELECT * FROM boot_history WHERE crash_detected = 1")
        assert len(crash_rows) == 3

        # Crash loop should be detected
        assert detector.is_crash_loop("m1", window_sec=600, threshold=3)

        # Should recommend rollback
        should_rollback, reason = detector.should_auto_rollback("m1")
        assert should_rollback is True
        assert "crash_loop" in reason

    def test_clean_shutdowns_prevent_crash_loop(self) -> None:
        db = _make_db()
        detector = CrashDetector(db)

        # 3 boots with clean shutdowns
        for i in range(3):
            boot_id = f"clean-{i}"
            detector.record_boot(boot_id, active_model_id="m1")
            detector.record_clean_shutdown(boot_id)

        crash_count = detector.record_boot("clean-3", active_model_id="m1")
        assert crash_count == 0
        assert not detector.is_crash_loop("m1")

        should_rollback, _ = detector.should_auto_rollback("m1")
        assert should_rollback is False


# ---------------------------------------------------------------------------
# 6. Storage GC
# ---------------------------------------------------------------------------


class TestStorageGC:
    """Active v3, previous v2, gc_eligible v1 — GC should delete v1 only."""

    def test_gc_deletes_only_eligible_versions(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        db = _make_db()
        registry = DeviceModelRegistry(db, models_dir=models_dir)
        session_mgr = InferenceSessionManager()
        gc = StorageGC(registry, session_mgr, models_dir=models_dir)

        # Create directories for v1, v2, v3
        for v in ("v1", "v2", "v3"):
            d = models_dir / "m1" / v
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.bin").write_bytes(b"x" * 1024)

        # Register artifacts so gc_eligible_versions can find them
        for v_num, v in enumerate(("v1", "v2", "v3"), start=1):
            aid = f"art-{v}"
            registry.register_artifact(aid, "m1", v, json.dumps({"files": []}), 1024)
            registry.update_artifact_status(aid, "DOWNLOADING")
            # Use update that doesn't go through activation state machine
            db.execute(
                "UPDATE model_artifacts SET status = 'ACTIVE' WHERE artifact_id = ?",
                (aid,),
            )

        # Set v3 as active, v2 as previous
        registry.set_active_model("m1", "v2")
        registry.set_active_model("m1", "v3")  # This makes v3 active, v2 previous

        active = registry.get_active_model("m1")
        assert active is not None
        assert active["active_version"] == "v3"
        assert active["previous_version"] == "v2"

        # v1 should be gc-eligible
        eligible = registry.gc_eligible_versions("m1")
        assert "v1" in eligible
        assert "v2" not in eligible
        assert "v3" not in eligible

        # Run GC
        freed = gc.run()
        assert freed > 0

        # v1 directory should be gone
        assert not (models_dir / "m1" / "v1").exists()
        # v2 and v3 should remain
        assert (models_dir / "m1" / "v2").exists()
        assert (models_dir / "m1" / "v3").exists()


# ---------------------------------------------------------------------------
# 7. Policy engine gating
# ---------------------------------------------------------------------------


class TestPolicyEngineGating:
    """Verify download/training gates under different device conditions."""

    def test_low_battery_blocks_download_and_training(self) -> None:
        engine = PolicyEngine(PolicyConfig())
        engine.update_device_state(battery_pct=10, is_charging=False, network_type="wifi")

        allowed, reason = engine.should_allow_download(1_000_000)
        assert allowed is False
        assert reason == "battery_low"

        allowed, reason = engine.should_allow_training()
        assert allowed is False
        assert reason == "not_charging"

    def test_good_conditions_allow_all(self) -> None:
        engine = PolicyEngine(PolicyConfig())
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            thermal_state="nominal",
            free_storage_bytes=10_000_000_000,
            is_foreground=False,
        )

        allowed_dl, reason_dl = engine.should_allow_download(1_000_000)
        assert allowed_dl is True
        assert reason_dl == "ok"

        allowed_tr, reason_tr = engine.should_allow_training()
        assert allowed_tr is True
        assert reason_tr == "ok"

        allowed_wu, reason_wu = engine.should_allow_warmup()
        assert allowed_wu is True
        assert reason_wu == "ok"

        allowed_up, reason_up = engine.should_allow_upload(1_000)
        assert allowed_up is True
        assert reason_up == "ok"

    def test_thermal_throttle_blocks_training(self) -> None:
        engine = PolicyEngine(PolicyConfig())
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            thermal_state="serious",
            free_storage_bytes=10_000_000_000,
            is_foreground=False,
        )
        allowed, reason = engine.should_allow_training()
        assert allowed is False
        assert reason == "thermal_throttle"

    def test_no_wifi_blocks_download(self) -> None:
        engine = PolicyEngine(PolicyConfig())
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="none",
            free_storage_bytes=10_000_000_000,
        )
        allowed, reason = engine.should_allow_download(1_000_000)
        assert allowed is False
        assert reason == "network_not_allowed"


# ---------------------------------------------------------------------------
# 8. Telemetry pipeline
# ---------------------------------------------------------------------------


class TestTelemetryPipeline:
    """Priority ordering, mark-as-sent, and storage pressure cleanup."""

    def test_priority_ordering_and_cleanup(self) -> None:
        db = _make_db()
        store = TelemetryStore(db, device_id="dev", boot_id="boot-tel")

        # Append events of all 3 classes
        store.append("evt.must", TelemetryClass.MUST_KEEP, {"k": "must1"})
        store.append("evt.important", TelemetryClass.IMPORTANT, {"k": "imp1"})
        store.append("evt.best", TelemetryClass.BEST_EFFORT, {"k": "be1"})
        store.append("evt.must", TelemetryClass.MUST_KEEP, {"k": "must2"})
        store.append("evt.best", TelemetryClass.BEST_EFFORT, {"k": "be2"})

        # get_unsent should return MUST_KEEP first
        unsent = store.get_unsent(batch_size=100)
        assert len(unsent) == 5
        assert unsent[0]["telemetry_class"] == "MUST_KEEP"
        assert unsent[1]["telemetry_class"] == "MUST_KEEP"
        assert unsent[2]["telemetry_class"] == "IMPORTANT"

        # Mark the first 2 (MUST_KEEP) as sent
        must_ids = [unsent[0]["event_id"], unsent[1]["event_id"]]
        store.mark_sent(must_ids, batch_id="batch-1")

        # After marking, only 3 unsent remain
        remaining = store.get_unsent(batch_size=100)
        assert len(remaining) == 3

        # Storage pressure cleanup: drop BEST_EFFORT first
        dropped = store.storage_pressure_cleanup(target_bytes=0)
        assert dropped >= 2  # at least the 2 BEST_EFFORT events

        # After cleanup, MUST_KEEP events that were sent should still exist in DB
        # and the unsent IMPORTANT event should survive
        final_unsent = store.get_unsent(batch_size=100)
        for evt in final_unsent:
            assert evt["telemetry_class"] != "BEST_EFFORT"

        # Count by class — BEST_EFFORT should be 0
        counts = store.count_by_class()
        assert counts.get("BEST_EFFORT", 0) == 0

    def test_must_keep_preserved_under_pressure(self) -> None:
        db = _make_db()
        store = TelemetryStore(db, device_id="dev", boot_id="boot-mk")

        # Only MUST_KEEP events
        for i in range(5):
            store.append("critical.event", TelemetryClass.MUST_KEEP, {"i": i})

        dropped = store.storage_pressure_cleanup(target_bytes=999_999_999)
        # storage_pressure_cleanup only drops BEST_EFFORT and IMPORTANT
        assert dropped == 0

        counts = store.count_by_class()
        assert counts.get("MUST_KEEP", 0) == 5


# ---------------------------------------------------------------------------
# 9. Local training job lifecycle
# ---------------------------------------------------------------------------


class TestLocalTrainingLifecycle:
    """Drive a training job through the full happy-path state machine."""

    def test_full_training_lifecycle(self) -> None:
        db = _make_db()
        trainer = LocalTrainer(db)

        job_id = trainer.create_job(
            job_type="personalization",
            binding_key="m1:scope1",
            base_model_id="m1",
            base_version="v1",
            limits=TrainingLimits(max_steps=20, checkpoint_every_steps=5),
        )

        job = trainer.get_job(job_id)
        assert job is not None
        assert job["state"] == "NEW"

        # NEW -> ELIGIBLE
        trainer.transition(job_id, "ELIGIBLE")
        job_eligible = trainer.get_job(job_id)
        assert job_eligible is not None
        assert job_eligible["state"] == "ELIGIBLE"

        # ELIGIBLE -> QUEUED
        trainer.transition(job_id, "QUEUED")
        job_queued = trainer.get_job(job_id)
        assert job_queued is not None
        assert job_queued["state"] == "QUEUED"

        # QUEUED -> PREPARING_DATA via prepare_data
        dataset_id = trainer.prepare_data(job_id, {"recency": "7d", "max_examples": 500})
        assert dataset_id
        # prepare_data transitions through PREPARING_DATA -> WAITING_FOR_RESOURCES
        job_waiting = trainer.get_job(job_id)
        assert job_waiting is not None
        assert job_waiting["state"] == "WAITING_FOR_RESOURCES"

        # WAITING_FOR_RESOURCES -> TRAINING -> CHECKPOINTING -> TRAINING -> ...
        step_count = 0
        checkpoint_saved = False

        def train_fn(ctx: dict[str, Any]) -> dict[str, Any]:
            nonlocal step_count, checkpoint_saved
            step_count += 1
            result: dict[str, Any] = {"loss": 1.0 / (step_count + 1), "done": step_count >= 10}
            if step_count % 5 == 0:
                result["checkpoint_data"] = b"checkpoint-bytes"
                checkpoint_saved = True
            return result

        metrics = trainer.train(job_id, train_fn, max_steps=20, checkpoint_every=5)
        assert metrics["step"] >= 10
        job_training = trainer.get_job(job_id)
        assert job_training is not None
        assert job_training["state"] == "TRAINING"
        assert checkpoint_saved

        # Verify checkpoint was saved
        cp = trainer.resume_from_checkpoint(job_id)
        assert cp is not None
        assert cp["step"] >= 5

        # TRAINING -> EVALUATING -> CANDIDATE_READY
        def eval_fn(ctx: dict[str, Any]) -> dict[str, Any]:
            return {"accept": True, "accuracy": 0.92}

        eval_result = trainer.evaluate(job_id, eval_fn)
        assert eval_result["accept"] is True
        job_candidate = trainer.get_job(job_id)
        assert job_candidate is not None
        assert job_candidate["state"] == "CANDIDATE_READY"

        # CANDIDATE_READY -> STAGED -> ACTIVATING -> ACTIVE -> COMPLETED
        trainer.transition(job_id, "STAGED")
        trainer.transition(job_id, "ACTIVATING")
        trainer.transition(job_id, "ACTIVE")
        trainer.transition(job_id, "COMPLETED")

        final = trainer.get_job(job_id)
        assert final is not None
        assert final["state"] == "COMPLETED"

    def test_invalid_transitions_rejected(self) -> None:
        db = _make_db()
        trainer = LocalTrainer(db)
        job_id = trainer.create_job("personalization", "m1:s1", "m1", "v1")

        # NEW -> COMPLETED should fail (not allowed)
        with pytest.raises(InvalidTransitionError):
            trainer.transition(job_id, "COMPLETED")

        # NEW -> ELIGIBLE is allowed
        trainer.transition(job_id, "ELIGIBLE")

        # ELIGIBLE -> TRAINING should fail
        with pytest.raises(InvalidTransitionError):
            trainer.transition(job_id, "TRAINING")


# ---------------------------------------------------------------------------
# 10. Runtime update lifecycle
# ---------------------------------------------------------------------------


class TestRuntimeUpdateLifecycle:
    """Discover -> download -> verify -> pending restart -> activate on boot."""

    def test_full_runtime_update_cycle(self) -> None:
        db = _make_db()
        updater = RuntimeUpdater(db)

        # Discover
        runtime_id = updater.discover("2.0.0", "https://cdn.octomil.com/runtime-2.0.0.tar.gz")
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "DISCOVERED"
        assert rt["version"] == "2.0.0"

        # Download
        assert updater.download(runtime_id)
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "DOWNLOADED"
        assert rt["downloaded_at"] is not None

        # Verify
        assert updater.verify(runtime_id)
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "VERIFIED"
        assert rt["verified_at"] is not None

        # Mark pending restart
        assert updater.mark_pending_restart(runtime_id)
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "PENDING_RESTART"
        assert rt["pending_since"] is not None

        pending = updater.get_pending_runtime()
        assert pending is not None
        assert pending["runtime_id"] == runtime_id

        # Activate on boot
        assert updater.activate_on_boot(runtime_id)
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "ACTIVE_ON_NEXT_BOOT"
        assert rt["activated_at"] is not None

        active = updater.get_active_runtime()
        assert active is not None
        assert active["runtime_id"] == runtime_id
        assert active["version"] == "2.0.0"

    def test_invalid_transitions_rejected(self) -> None:
        db = _make_db()
        updater = RuntimeUpdater(db)
        runtime_id = updater.discover("3.0.0", "https://cdn.octomil.com/runtime-3.0.0")

        # DISCOVERED -> VERIFIED should fail (must go through DOWNLOADED)
        assert updater.verify(runtime_id) is False
        rt = updater.get_runtime(runtime_id)
        assert rt is not None
        assert rt["status"] == "DISCOVERED"

        # DISCOVERED -> PENDING_RESTART should fail
        assert updater.mark_pending_restart(runtime_id) is False

        # DISCOVERED -> ACTIVE_ON_NEXT_BOOT should fail
        assert updater.activate_on_boot(runtime_id) is False
