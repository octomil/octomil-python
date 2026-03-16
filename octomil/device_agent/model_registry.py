"""Device-local model registry backed by SQLite.

Manages installed model versions, active pointer, staging, and rollback.
All mutations are atomic via the LocalDB transaction context manager.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeviceModelRegistry:
    """Device-side model registry with atomic version pointer flips."""

    def __init__(self, db: LocalDB, models_dir: str | Path = "/models") -> None:
        self._db = db
        self._models_dir = Path(models_dir)

    # -- Active pointer --

    def get_active_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Return the active version info for a model, or None."""
        row = self._db.execute_one(
            "SELECT active_version, previous_version, updated_at FROM active_model_pointer WHERE model_id = ?",
            (model_id,),
        )
        if row is None:
            return None
        return {
            "model_id": model_id,
            "active_version": row["active_version"],
            "previous_version": row["previous_version"],
            "path": str(self.get_model_path(model_id, row["active_version"])),
            "updated_at": row["updated_at"],
        }

    def set_active_model(self, model_id: str, version: str) -> None:
        """Atomically flip the active pointer for a model."""
        now = _now_iso()
        with self._db.transaction() as cur:
            existing = cur.execute(
                "SELECT active_version FROM active_model_pointer WHERE model_id = ?",
                (model_id,),
            ).fetchone()

            if existing is None:
                cur.execute(
                    "INSERT INTO active_model_pointer "
                    "(model_id, active_version, previous_version, updated_at) "
                    "VALUES (?, ?, NULL, ?)",
                    (model_id, version, now),
                )
            else:
                cur.execute(
                    "UPDATE active_model_pointer "
                    "SET previous_version = active_version, "
                    "    active_version = ?, updated_at = ? "
                    "WHERE model_id = ?",
                    (version, now, model_id),
                )

    # -- Staged versions --

    def get_staged_versions(self, model_id: str) -> list[dict[str, Any]]:
        """Return artifacts in STAGED status for the given model."""
        rows = self._db.execute(
            "SELECT artifact_id, version, staged_at "
            "FROM model_artifacts "
            "WHERE model_id = ? AND status = 'STAGED' "
            "ORDER BY staged_at DESC",
            (model_id,),
        )
        return [dict(r) for r in rows]

    # -- Artifact management --

    def register_artifact(
        self,
        artifact_id: str,
        model_id: str,
        version: str,
        manifest_json: str,
        total_bytes: int,
    ) -> None:
        """Register a new artifact for download tracking."""
        now = _now_iso()
        self._db.execute(
            "INSERT INTO model_artifacts "
            "(artifact_id, model_id, version, status, manifest_json, "
            " bytes_downloaded, total_bytes, updated_at) "
            "VALUES (?, ?, ?, 'REGISTERED', ?, 0, ?, ?)",
            (artifact_id, model_id, version, manifest_json, total_bytes, now),
        )

    def update_artifact_status(self, artifact_id: str, new_status: str, **kwargs: Any) -> None:
        """Transition an artifact to a new status with optional field updates."""
        sets = ["status = ?", "updated_at = ?"]
        params: list[Any] = [new_status, _now_iso()]

        allowed_fields = {
            "bytes_downloaded",
            "verified_at",
            "staged_at",
            "activated_at",
            "last_error",
            "retry_count",
        }
        for key, value in kwargs.items():
            if key not in allowed_fields:
                raise ValueError(f"Cannot update field: {key}")
            sets.append(f"{key} = ?")
            params.append(value)

        params.append(artifact_id)
        self._db.execute(
            f"UPDATE model_artifacts SET {', '.join(sets)} WHERE artifact_id = ?",
            tuple(params),
        )

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        """Return artifact record as dict, or None."""
        row = self._db.execute_one(
            "SELECT * FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        return dict(row) if row else None

    def list_installed_versions(self, model_id: str) -> list[dict[str, Any]]:
        """Return all artifact versions for a model that reached ACTIVE or STAGED."""
        rows = self._db.execute(
            "SELECT artifact_id, version, status, activated_at, staged_at "
            "FROM model_artifacts "
            "WHERE model_id = ? AND status IN ('ACTIVE', 'STAGED', 'VERIFIED') "
            "ORDER BY updated_at DESC",
            (model_id,),
        )
        return [dict(r) for r in rows]

    # -- Paths --

    def get_model_path(self, model_id: str, version: str) -> Path:
        """Return the canonical path for a model version's files."""
        return self._models_dir / model_id / version

    # -- Rollback --

    def rollback(self, model_id: str, reason: str) -> Optional[str]:
        """Roll back to the previous active version.

        Returns the version rolled back to, or None if no previous version.
        """
        now = _now_iso()
        with self._db.transaction() as cur:
            row = cur.execute(
                "SELECT active_version, previous_version FROM active_model_pointer WHERE model_id = ?",
                (model_id,),
            ).fetchone()

            if row is None or row["previous_version"] is None:
                return None

            from_version = row["active_version"]
            to_version = row["previous_version"]

            cur.execute(
                "UPDATE active_model_pointer "
                "SET active_version = ?, previous_version = ?, updated_at = ? "
                "WHERE model_id = ?",
                (to_version, from_version, now, model_id),
            )

            cur.execute(
                "INSERT INTO rollback_records "
                "(model_id, from_version, to_version, reason, rolled_back_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (model_id, from_version, to_version, reason, now),
            )

            return to_version

    # -- Garbage collection --

    def gc_eligible_versions(self, model_id: str) -> list[str]:
        """Return versions that are not active or previous and can be cleaned up."""
        row = self._db.execute_one(
            "SELECT active_version, previous_version FROM active_model_pointer WHERE model_id = ?",
            (model_id,),
        )
        protected: set[str] = set()
        if row:
            protected.add(row["active_version"])
            if row["previous_version"]:
                protected.add(row["previous_version"])

        rows = self._db.execute(
            "SELECT DISTINCT version FROM model_artifacts "
            "WHERE model_id = ? AND status NOT IN ('DOWNLOADING', 'REGISTERED')",
            (model_id,),
        )
        return [r["version"] for r in rows if r["version"] not in protected]
