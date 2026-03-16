"""Tests for ArtifactDownloader chunk tracking and progress."""

from __future__ import annotations

import json

import pytest

from octomil.device_agent.artifact_downloader import CHUNK_SIZE, ArtifactDownloader
from octomil.device_agent.db.local_db import LocalDB


@pytest.fixture
def setup(tmp_path):
    db = LocalDB(":memory:")
    dl = ArtifactDownloader(db, models_dir=tmp_path / "models")
    manifest = {"files": [{"path": "model.bin", "size": CHUNK_SIZE * 3, "sha256": "abc"}]}
    manifest_json = json.dumps(manifest)
    db.execute(
        "INSERT INTO model_artifacts "
        "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
        "VALUES (?, ?, ?, 'REGISTERED', ?, ?, 'now')",
        ("a1", "m1", "v1", manifest_json, CHUNK_SIZE * 3),
    )
    return db, dl, manifest


class TestStartDownload:
    def test_creates_chunks(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        rows = db.execute("SELECT * FROM download_chunks WHERE artifact_id = 'a1' ORDER BY chunk_index")
        assert len(rows) == 3
        assert all(r["status"] == "PENDING" for r in rows)

    def test_sets_downloading_status(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        row = db.execute_one("SELECT status FROM model_artifacts WHERE artifact_id = 'a1'")
        assert row is not None
        assert row["status"] == "DOWNLOADING"

    def test_idempotent_chunk_insert(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        dl.start_download("a1", manifest, "https://example.com")  # should not duplicate
        rows = db.execute("SELECT * FROM download_chunks WHERE artifact_id = 'a1'")
        assert len(rows) == 3


class TestResumeDownload:
    def test_returns_pending_chunks(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        pending = dl.resume_download("a1")
        assert len(pending) == 3

    def test_excludes_complete_chunks(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        db.execute("UPDATE download_chunks SET status = 'COMPLETE' " "WHERE artifact_id = 'a1' AND chunk_index = 0")
        pending = dl.resume_download("a1")
        assert len(pending) == 2
        assert all(p["chunk_index"] != 0 for p in pending)


class TestProgress:
    def test_progress_zero(self, setup) -> None:
        db, dl, manifest = setup
        progress = dl.get_progress("a1")
        assert progress["bytes_downloaded"] == 0
        assert progress["total_bytes"] == CHUNK_SIZE * 3

    def test_progress_nonexistent(self, setup) -> None:
        _, dl, _ = setup
        progress = dl.get_progress("nonexistent")
        assert progress["pct"] == 0.0


class TestPauseCancel:
    def test_pause(self, setup) -> None:
        db, dl, manifest = setup
        dl.pause("a1")
        row = db.execute_one("SELECT status FROM model_artifacts WHERE artifact_id = 'a1'")
        assert row is not None
        assert row["status"] == "PAUSED"

    def test_cancel_removes_chunks(self, setup) -> None:
        db, dl, manifest = setup
        dl.start_download("a1", manifest, "https://example.com")
        dl.cancel("a1")
        rows = db.execute("SELECT * FROM download_chunks WHERE artifact_id = 'a1'")
        assert len(rows) == 0
        row = db.execute_one("SELECT status FROM model_artifacts WHERE artifact_id = 'a1'")
        assert row is not None
        assert row["status"] == "CANCELLED"
