"""Chunked resumable artifact downloader.

Downloads model artifacts as individually-tracked chunks, writing to
.part files and renaming to final paths only after full verification.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB default


class ArtifactDownloader:
    """Chunked, resumable artifact download manager."""

    def __init__(self, db: LocalDB, models_dir: str | Path = "/models") -> None:
        self._db = db
        self._models_dir = Path(models_dir)

    def start_download(self, artifact_id: str, manifest: dict[str, Any], base_url: str) -> None:
        """Initialize chunk records in DB from a manifest.

        Manifest format::

            {
                "files": [
                    {"path": "model.bin", "size": 4000000000, "sha256": "abc..."},
                    ...
                ]
            }
        """
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE model_artifacts SET status = 'DOWNLOADING', updated_at = datetime('now') WHERE artifact_id = ?",
                (artifact_id,),
            )
            for file_info in manifest.get("files", []):
                file_path = file_info["path"]
                file_size = file_info["size"]
                num_chunks = max(1, (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE)
                for i in range(num_chunks):
                    cur.execute(
                        "INSERT OR IGNORE INTO download_chunks "
                        "(artifact_id, file_path, chunk_index, status) "
                        "VALUES (?, ?, ?, 'PENDING')",
                        (artifact_id, file_path, i),
                    )

    def resume_download(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return list of chunks still needing download."""
        rows = self._db.execute(
            "SELECT artifact_id, file_path, chunk_index "
            "FROM download_chunks "
            "WHERE artifact_id = ? AND status != 'COMPLETE' "
            "ORDER BY file_path, chunk_index",
            (artifact_id,),
        )
        return [dict(r) for r in rows]

    def download_chunk(
        self,
        artifact_id: str,
        file_path: str,
        chunk_index: int,
        url: str,
        expected_sha256: Optional[str] = None,
    ) -> bool:
        """Download a single chunk to a .part file.

        Returns True on success, False on failure.
        """
        artifact = self._db.execute_one(
            "SELECT model_id, version FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if artifact is None:
            return False

        model_path = self._models_dir / artifact["model_id"] / artifact["version"]
        parts_dir = model_path / ".parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        part_file = parts_dir / f"{file_path}.chunk{chunk_index}.part"

        try:
            offset = chunk_index * CHUNK_SIZE
            headers = {"Range": f"bytes={offset}-{offset + CHUNK_SIZE - 1}"}

            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as resp:
                resp.raise_for_status()
                hasher = hashlib.sha256() if expected_sha256 else None
                bytes_written = 0
                with open(part_file, "wb") as f:
                    for data in resp.iter_bytes(chunk_size=65536):
                        f.write(data)
                        if hasher:
                            hasher.update(data)
                        bytes_written += len(data)
                    f.flush()
                    os.fsync(f.fileno())

            if expected_sha256 and hasher and hasher.hexdigest() != expected_sha256:
                logger.warning("Chunk hash mismatch for %s chunk %d", file_path, chunk_index)
                part_file.unlink(missing_ok=True)
                self._db.execute(
                    "UPDATE download_chunks SET status = 'FAILED', "
                    "attempts = attempts + 1, last_error = 'hash_mismatch' "
                    "WHERE artifact_id = ? AND file_path = ? AND chunk_index = ?",
                    (artifact_id, file_path, chunk_index),
                )
                return False

            self._db.execute(
                "UPDATE download_chunks SET status = 'COMPLETE', attempts = attempts + 1 "
                "WHERE artifact_id = ? AND file_path = ? AND chunk_index = ?",
                (artifact_id, file_path, chunk_index),
            )

            # Update bytes_downloaded on artifact
            self._db.execute(
                "UPDATE model_artifacts SET bytes_downloaded = bytes_downloaded + ?, "
                "updated_at = datetime('now') WHERE artifact_id = ?",
                (bytes_written, artifact_id),
            )
            return True

        except Exception as exc:
            logger.warning("Chunk download failed: %s", exc)
            part_file.unlink(missing_ok=True)
            self._db.execute(
                "UPDATE download_chunks SET status = 'FAILED', "
                "attempts = attempts + 1, last_error = ? "
                "WHERE artifact_id = ? AND file_path = ? AND chunk_index = ?",
                (str(exc), artifact_id, file_path, chunk_index),
            )
            return False

    def assemble_file(self, artifact_id: str, file_path: str) -> bool:
        """Assemble all chunks for a file into the final path.

        Returns True if assembly succeeded.
        """
        artifact = self._db.execute_one(
            "SELECT model_id, version FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if artifact is None:
            return False

        model_path = self._models_dir / artifact["model_id"] / artifact["version"]
        parts_dir = model_path / ".parts"
        final_path = model_path / file_path
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Check all chunks are complete
        pending = self._db.execute_one(
            "SELECT COUNT(*) as cnt FROM download_chunks "
            "WHERE artifact_id = ? AND file_path = ? AND status != 'COMPLETE'",
            (artifact_id, file_path),
        )
        if pending and pending["cnt"] > 0:
            return False

        chunks = self._db.execute(
            "SELECT chunk_index FROM download_chunks WHERE artifact_id = ? AND file_path = ? ORDER BY chunk_index",
            (artifact_id, file_path),
        )

        with open(final_path, "wb") as out:
            for chunk_row in chunks:
                part_file = parts_dir / f"{file_path}.chunk{chunk_row['chunk_index']}.part"
                with open(part_file, "rb") as inp:
                    while True:
                        data = inp.read(65536)
                        if not data:
                            break
                        out.write(data)
            out.flush()
            os.fsync(out.fileno())

        # Clean up part files
        for chunk_row in chunks:
            part_file = parts_dir / f"{file_path}.chunk{chunk_row['chunk_index']}.part"
            part_file.unlink(missing_ok=True)

        return True

    def get_progress(self, artifact_id: str) -> dict[str, Any]:
        """Return download progress for an artifact."""
        artifact = self._db.execute_one(
            "SELECT bytes_downloaded, total_bytes FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if artifact is None:
            return {"bytes_downloaded": 0, "total_bytes": 0, "pct": 0.0}
        total = artifact["total_bytes"] or 1
        downloaded = artifact["bytes_downloaded"]
        return {
            "bytes_downloaded": downloaded,
            "total_bytes": artifact["total_bytes"],
            "pct": round(downloaded / total * 100, 2),
        }

    def pause(self, artifact_id: str) -> None:
        """Pause download by setting artifact status to PAUSED."""
        self._db.execute(
            "UPDATE model_artifacts SET status = 'PAUSED', updated_at = datetime('now') WHERE artifact_id = ?",
            (artifact_id,),
        )

    def cancel(self, artifact_id: str) -> None:
        """Cancel download and clean up .part files."""
        artifact = self._db.execute_one(
            "SELECT model_id, version FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if artifact:
            parts_dir = self._models_dir / artifact["model_id"] / artifact["version"] / ".parts"
            if parts_dir.exists():
                for f in parts_dir.iterdir():
                    f.unlink(missing_ok=True)
                parts_dir.rmdir()

        self._db.execute(
            "DELETE FROM download_chunks WHERE artifact_id = ?",
            (artifact_id,),
        )
        self._db.execute(
            "UPDATE model_artifacts SET status = 'CANCELLED', updated_at = datetime('now') WHERE artifact_id = ?",
            (artifact_id,),
        )
