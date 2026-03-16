"""Crash loop detection based on boot history analysis.

On startup, checks whether the previous boot completed with a clean shutdown.
If not, marks it as a crash. Detects crash loops by counting crashes within
a sliding time window and recommends auto-rollback when thresholds are hit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_epoch(iso_str: str) -> float:
    """Parse an ISO timestamp to a Unix epoch float."""
    # Handle both with and without timezone info
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%f")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class CrashDetector:
    """Detects crash loops by tracking boot history.

    Records each boot and shutdown in the ``boot_history`` table. On startup,
    any previous boot that did not record a clean shutdown is flagged as a
    crash. Crash loop detection counts recent crashes within a sliding time
    window and recommends rollback when a threshold is exceeded.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    # -- Boot lifecycle --

    def record_boot(
        self,
        boot_id: str,
        active_model_id: Optional[str] = None,
        active_model_version: Optional[str] = None,
        runtime_version: Optional[str] = None,
    ) -> int:
        """Record a new boot event. Returns the number of previous unclean boots detected.

        Scans for any previous boot that has ``clean_shutdown=0`` and
        ``crash_detected=0``, marking those as crashes.
        """
        now = _now_iso()

        # Mark any previous unclean boots as crashes
        unclean_rows = self._db.execute(
            "SELECT boot_id FROM boot_history WHERE clean_shutdown = 0 AND crash_detected = 0",
        )
        crash_count = 0
        for row in unclean_rows:
            self._db.execute(
                "UPDATE boot_history SET crash_detected = 1 WHERE boot_id = ?",
                (row["boot_id"],),
            )
            crash_count += 1
            logger.warning("Previous boot %s did not shut down cleanly — marked as crash", row["boot_id"])

        # Insert new boot record
        self._db.execute(
            "INSERT INTO boot_history "
            "(boot_id, started_at, active_model_id, active_model_version, "
            " runtime_version, clean_shutdown, crash_detected) "
            "VALUES (?, ?, ?, ?, ?, 0, 0)",
            (boot_id, now, active_model_id, active_model_version, runtime_version),
        )
        return crash_count

    def record_clean_shutdown(self, boot_id: str) -> bool:
        """Mark the current boot as cleanly shut down.

        Also computes and stores the duration. Returns True if the boot was
        found and updated.
        """
        row = self._db.execute_one(
            "SELECT started_at FROM boot_history WHERE boot_id = ?",
            (boot_id,),
        )
        if row is None:
            return False

        now = _now_iso()
        start_epoch = _iso_to_epoch(row["started_at"])
        end_epoch = _iso_to_epoch(now)
        duration = max(0.0, end_epoch - start_epoch)

        self._db.execute(
            "UPDATE boot_history SET clean_shutdown = 1, duration_sec = ? WHERE boot_id = ?",
            (duration, boot_id),
        )
        return True

    # -- Crash loop detection --

    def is_crash_loop(
        self,
        model_id: str,
        window_sec: float = 300.0,
        threshold: int = 3,
    ) -> bool:
        """Return True if *model_id* crashed at least *threshold* times within *window_sec*.

        Looks at boot_history rows where ``crash_detected=1`` and
        ``active_model_id`` matches.
        """
        now_epoch = _iso_to_epoch(_now_iso())
        rows = self._db.execute(
            "SELECT started_at FROM boot_history WHERE crash_detected = 1 AND active_model_id = ?",
            (model_id,),
        )
        recent_crashes = 0
        for row in rows:
            boot_epoch = _iso_to_epoch(row["started_at"])
            if now_epoch - boot_epoch <= window_sec:
                recent_crashes += 1
        return recent_crashes >= threshold

    def should_auto_rollback(self, model_id: str) -> tuple[bool, str]:
        """Decide whether *model_id* should be rolled back.

        Returns ``(should_rollback, reason)``.
        """
        if self.is_crash_loop(model_id):
            reason = f"crash_loop: model {model_id} exceeded crash threshold within window"
            return True, reason
        return False, ""

    # -- Query --

    def get_boot_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent boot records, most recent first."""
        rows = self._db.execute(
            "SELECT * FROM boot_history ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]
