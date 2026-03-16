"""Two-phase model activation with warmup and rollback.

State machine: VERIFIED -> STAGED -> WARMING -> ACTIVE | FAILED_HEALTHCHECK
Rollback creates a record and flips the active pointer back.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .db.local_db import LocalDB
from .model_registry import DeviceModelRegistry

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Valid state transitions
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "VERIFIED": {"STAGED"},
    "STAGED": {"WARMING"},
    "WARMING": {"ACTIVE", "FAILED_HEALTHCHECK"},
    "ACTIVE": {"STAGED"},  # re-staging for updates
    "FAILED_HEALTHCHECK": {"STAGED", "WARMING"},
}


class ActivationManager:
    """Two-phase activation manager for on-device models."""

    def __init__(self, db: LocalDB, registry: DeviceModelRegistry) -> None:
        self._db = db
        self._registry = registry

    def stage(self, artifact_id: str) -> bool:
        """Move a VERIFIED artifact to STAGED.

        Returns True if the transition succeeded.
        """
        return self._transition(artifact_id, "STAGED", staged_at=_now_iso())

    def warmup(self, artifact_id: str, warmup_fn: Any = None) -> bool:
        """Run warmup checks and transition to ACTIVE or FAILED_HEALTHCHECK.

        If warmup_fn is provided, it is called with the model path and must
        return True for the warmup to succeed. Otherwise, warmup always
        succeeds (useful for testing).
        """
        if not self._transition(artifact_id, "WARMING"):
            return False

        artifact = self._db.execute_one(
            "SELECT model_id, version FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if artifact is None:
            return False

        model_path = self._registry.get_model_path(artifact["model_id"], artifact["version"])

        success = True
        if warmup_fn is not None:
            try:
                success = bool(warmup_fn(model_path))
            except Exception as exc:
                logger.warning("Warmup failed for artifact %s: %s", artifact_id, exc)
                success = False

        if success:
            return self._transition(artifact_id, "ACTIVE", activated_at=_now_iso())
        else:
            self._transition(artifact_id, "FAILED_HEALTHCHECK", last_error="warmup_failed")
            return False

    def activate(self, model_id: str, version: str) -> None:
        """Atomically flip the active pointer. Only affects new sessions."""
        self._registry.set_active_model(model_id, version)

    def drain_old(
        self,
        model_id: str,
        old_version: str,
        timeout_sec: float = 300.0,
        refcount_fn: Any = None,
    ) -> bool:
        """Wait for refcount=0 on old_version or until timeout.

        refcount_fn should be a callable(model_id, version) -> int.
        Returns True if drained, False if timed out.
        """
        if refcount_fn is None:
            return True

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            count = refcount_fn(model_id, old_version)
            if count <= 0:
                return True
            time.sleep(min(1.0, deadline - time.monotonic()))
        return False

    def auto_rollback(self, model_id: str, reason: str) -> Optional[str]:
        """Roll back if warmup fails or crash loop detected.

        Returns the version rolled back to, or None.
        """
        return self._registry.rollback(model_id, reason)

    def get_activation_state(self, artifact_id: str) -> Optional[str]:
        """Return current status of an artifact."""
        row = self._db.execute_one(
            "SELECT status FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        return row["status"] if row else None

    def _transition(self, artifact_id: str, new_status: str, **kwargs: Any) -> bool:
        """Attempt a state transition. Returns True if valid and applied."""
        row = self._db.execute_one(
            "SELECT status FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if row is None:
            return False

        current = row["status"]
        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            logger.warning(
                "Invalid transition %s -> %s for artifact %s",
                current,
                new_status,
                artifact_id,
            )
            return False

        self._registry.update_artifact_status(artifact_id, new_status, **kwargs)
        return True
