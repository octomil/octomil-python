"""Artifact integrity verification using SHA-256.

Verifies individual chunks, full files, and complete artifacts against
expected hashes from the download manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)

HASH_BUFFER_SIZE = 128 * 1024  # 128 KiB read buffer


class ArtifactVerifier:
    """SHA-256 verification for downloaded model artifacts."""

    def __init__(self, db: LocalDB, models_dir: str | Path = "/models") -> None:
        self._db = db
        self._models_dir = Path(models_dir)

    def verify_chunk(
        self,
        file_path: str | Path,
        offset: int,
        length: int,
        expected_sha256: str,
    ) -> bool:
        """Verify a byte range within a file against an expected SHA-256 hash."""
        path = Path(file_path)
        if not path.exists():
            return False
        try:
            hasher = hashlib.sha256()
            remaining = length
            with open(path, "rb") as f:
                f.seek(offset)
                while remaining > 0:
                    to_read = min(HASH_BUFFER_SIZE, remaining)
                    data = f.read(to_read)
                    if not data:
                        break
                    hasher.update(data)
                    remaining -= len(data)
            return hasher.hexdigest() == expected_sha256
        except OSError:
            logger.warning("Failed to read %s for chunk verification", file_path)
            return False

    def verify_file(self, file_path: str | Path, expected_sha256: str) -> bool:
        """Verify an entire file against an expected SHA-256 hash."""
        path = Path(file_path)
        if not path.exists():
            return False
        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    data = f.read(HASH_BUFFER_SIZE)
                    if not data:
                        break
                    hasher.update(data)
            return hasher.hexdigest() == expected_sha256
        except OSError:
            logger.warning("Failed to read %s for file verification", file_path)
            return False

    def verify_artifact(self, artifact_id: str) -> bool:
        """Verify all files in an artifact's manifest.

        Updates artifact status to VERIFIED on success or FAILED_VERIFICATION
        on failure.
        """
        row = self._db.execute_one(
            "SELECT model_id, version, manifest_json FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if row is None:
            return False

        model_path = self._models_dir / row["model_id"] / row["version"]
        manifest = json.loads(row["manifest_json"])

        all_ok = True
        for file_info in manifest.get("files", []):
            full_path = model_path / file_info["path"]
            expected = file_info.get("sha256", "")
            if not expected:
                continue
            if not self.verify_file(full_path, expected):
                logger.warning(
                    "Verification failed for %s in artifact %s",
                    file_info["path"],
                    artifact_id,
                )
                all_ok = False

        now_field = "verified_at" if all_ok else "last_error"
        now_value = "datetime('now')" if all_ok else "'verification_failed'"
        new_status = "VERIFIED" if all_ok else "FAILED_VERIFICATION"

        self._db.execute(
            f"UPDATE model_artifacts SET status = ?, {now_field} = {now_value}, "
            "updated_at = datetime('now') WHERE artifact_id = ?",
            (new_status, artifact_id),
        )
        return all_ok
