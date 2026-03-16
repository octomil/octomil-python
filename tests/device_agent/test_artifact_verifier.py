"""Tests for ArtifactVerifier hash verification."""

from __future__ import annotations

import hashlib
import json

import pytest

from octomil.device_agent.artifact_verifier import ArtifactVerifier
from octomil.device_agent.db.local_db import LocalDB


@pytest.fixture
def setup(tmp_path):
    db = LocalDB(":memory:")
    verifier = ArtifactVerifier(db, models_dir=tmp_path / "models")
    return db, verifier, tmp_path


class TestVerifyFile:
    def test_valid_hash(self, setup) -> None:
        db, verifier, tmp_path = setup
        content = b"hello world test data"
        expected = hashlib.sha256(content).hexdigest()
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(content)
        assert verifier.verify_file(file_path, expected) is True

    def test_invalid_hash(self, setup) -> None:
        db, verifier, tmp_path = setup
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(b"data")
        assert verifier.verify_file(file_path, "badhash") is False

    def test_missing_file(self, setup) -> None:
        _, verifier, tmp_path = setup
        assert verifier.verify_file(tmp_path / "missing.bin", "abc") is False


class TestVerifyChunk:
    def test_valid_chunk(self, setup) -> None:
        _, verifier, tmp_path = setup
        content = b"A" * 100 + b"B" * 50 + b"C" * 100
        file_path = tmp_path / "chunked.bin"
        file_path.write_bytes(content)
        chunk_data = b"B" * 50
        expected = hashlib.sha256(chunk_data).hexdigest()
        assert verifier.verify_chunk(file_path, offset=100, length=50, expected_sha256=expected)

    def test_invalid_chunk(self, setup) -> None:
        _, verifier, tmp_path = setup
        file_path = tmp_path / "chunked.bin"
        file_path.write_bytes(b"X" * 200)
        assert verifier.verify_chunk(file_path, offset=0, length=100, expected_sha256="bad") is False


class TestVerifyArtifact:
    def test_artifact_verified(self, setup) -> None:
        db, verifier, tmp_path = setup
        # Create model files
        model_dir = tmp_path / "models" / "m1" / "v1"
        model_dir.mkdir(parents=True)
        content = b"model weights here"
        (model_dir / "weights.bin").write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = {"files": [{"path": "weights.bin", "sha256": expected_hash}]}
        db.execute(
            "INSERT INTO model_artifacts "
            "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
            "VALUES (?, ?, ?, 'DOWNLOADED', ?, ?, 'now')",
            ("a1", "m1", "v1", json.dumps(manifest), len(content)),
        )

        assert verifier.verify_artifact("a1") is True
        row = db.execute_one("SELECT status FROM model_artifacts WHERE artifact_id = 'a1'")
        assert row["status"] == "VERIFIED"

    def test_artifact_verification_fails(self, setup) -> None:
        db, verifier, tmp_path = setup
        model_dir = tmp_path / "models" / "m1" / "v1"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_bytes(b"corrupted")

        manifest = {"files": [{"path": "weights.bin", "sha256": "expectedhash"}]}
        db.execute(
            "INSERT INTO model_artifacts "
            "(artifact_id, model_id, version, status, manifest_json, total_bytes, updated_at) "
            "VALUES (?, ?, ?, 'DOWNLOADED', ?, 100, 'now')",
            ("a1", "m1", "v1", json.dumps(manifest)),
        )

        assert verifier.verify_artifact("a1") is False
        row = db.execute_one("SELECT status FROM model_artifacts WHERE artifact_id = 'a1'")
        assert row["status"] == "FAILED_VERIFICATION"

    def test_artifact_nonexistent(self, setup) -> None:
        _, verifier, _ = setup
        assert verifier.verify_artifact("nonexistent") is False
