"""Runtime binary updater with safe state machine transitions.

State machine: DISCOVERED -> DOWNLOADED -> VERIFIED -> PENDING_RESTART -> ACTIVE_ON_NEXT_BOOT

Never hot-swaps the runtime while inference is active. Updates are applied
only after a clean restart cycle.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Valid state transitions for runtime binary updates.
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "DISCOVERED": {"DOWNLOADED"},
    "DOWNLOADED": {"VERIFIED"},
    "VERIFIED": {"PENDING_RESTART"},
    "PENDING_RESTART": {"ACTIVE_ON_NEXT_BOOT"},
}


class RuntimeUpdater:
    """Manages runtime binary updates with a safe state machine.

    Tracks runtime versions in the ``runtime_versions`` table and enforces
    the transition order: DISCOVERED -> DOWNLOADED -> VERIFIED ->
    PENDING_RESTART -> ACTIVE_ON_NEXT_BOOT.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    # -- Discovery --

    def discover(self, version: str, artifact_url: str) -> str:
        """Register a newly discovered runtime version.

        Returns the runtime_id.
        """
        runtime_id = uuid.uuid4().hex
        now = _now_iso()
        self._db.execute(
            "INSERT INTO runtime_versions "
            "(runtime_id, version, status, artifact_path, updated_at) "
            "VALUES (?, ?, 'DISCOVERED', ?, ?)",
            (runtime_id, version, artifact_url, now),
        )
        logger.info("Discovered runtime %s (version=%s)", runtime_id, version)
        return runtime_id

    # -- Download --

    def download(self, runtime_id: str) -> bool:
        """Mark a runtime as downloaded. Returns True if transition succeeded."""
        return self._transition(runtime_id, "DOWNLOADED", downloaded_at=_now_iso())

    # -- Verification --

    def verify(self, runtime_id: str) -> bool:
        """Mark a runtime as verified. Returns True if transition succeeded."""
        return self._transition(runtime_id, "VERIFIED", verified_at=_now_iso())

    # -- Pending restart --

    def mark_pending_restart(self, runtime_id: str) -> bool:
        """Mark a runtime as pending restart. Returns True if transition succeeded."""
        return self._transition(runtime_id, "PENDING_RESTART", pending_since=_now_iso())

    # -- Activation on next boot --

    def activate_on_boot(self, runtime_id: str) -> bool:
        """Activate a runtime on boot. Returns True if transition succeeded.

        This should only be called during boot sequence, after confirming
        inference is not active.
        """
        return self._transition(runtime_id, "ACTIVE_ON_NEXT_BOOT", activated_at=_now_iso())

    # -- Queries --

    def get_active_runtime(self) -> Optional[dict[str, Any]]:
        """Return the currently active runtime, or None."""
        row = self._db.execute_one(
            "SELECT * FROM runtime_versions WHERE status = 'ACTIVE_ON_NEXT_BOOT' ORDER BY activated_at DESC LIMIT 1",
        )
        return dict(row) if row else None

    def get_pending_runtime(self) -> Optional[dict[str, Any]]:
        """Return the runtime pending restart, or None."""
        row = self._db.execute_one(
            "SELECT * FROM runtime_versions WHERE status = 'PENDING_RESTART' ORDER BY pending_since DESC LIMIT 1",
        )
        return dict(row) if row else None

    def get_runtime(self, runtime_id: str) -> Optional[dict[str, Any]]:
        """Return a single runtime record by id, or None."""
        row = self._db.execute_one(
            "SELECT * FROM runtime_versions WHERE runtime_id = ?",
            (runtime_id,),
        )
        return dict(row) if row else None

    # -- Internal helpers --

    def _transition(self, runtime_id: str, new_status: str, **kwargs: Any) -> bool:
        """Attempt a state transition. Returns True if valid and applied."""
        row = self._db.execute_one(
            "SELECT status FROM runtime_versions WHERE runtime_id = ?",
            (runtime_id,),
        )
        if row is None:
            logger.warning("Runtime %s not found", runtime_id)
            return False

        current = row["status"]
        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            logger.warning(
                "Invalid runtime transition %s -> %s for %s",
                current,
                new_status,
                runtime_id,
            )
            return False

        sets = ["status = ?", "updated_at = ?"]
        params: list[Any] = [new_status, _now_iso()]

        allowed_fields = {
            "downloaded_at",
            "verified_at",
            "pending_since",
            "activated_at",
            "artifact_path",
        }
        for key, value in kwargs.items():
            if key not in allowed_fields:
                raise ValueError(f"Cannot update field: {key}")
            sets.append(f"{key} = ?")
            params.append(value)

        params.append(runtime_id)
        self._db.execute(
            f"UPDATE runtime_versions SET {', '.join(sets)} WHERE runtime_id = ?",
            tuple(params),
        )
        logger.info("Runtime %s transitioned to %s", runtime_id, new_status)
        return True
