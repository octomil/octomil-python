"""Append-only telemetry event store backed by the local SQLite DB."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..db.local_db import LocalDB
from .events import TelemetryClass, default_class_for

logger = logging.getLogger(__name__)

# Priority ordering for class-based sorting (lower = higher priority).
_CLASS_SORT_ORDER: dict[TelemetryClass, int] = {
    TelemetryClass.MUST_KEEP: 0,
    TelemetryClass.IMPORTANT: 1,
    TelemetryClass.BEST_EFFORT: 2,
}


class TelemetryStore:
    """Thread-safe, append-only telemetry event WAL.

    Events are appended with monotonic sequence numbers per boot_id.
    The store supports priority-ordered batch retrieval, cursor-based
    acknowledgement, and storage-pressure cleanup.
    """

    def __init__(self, db: LocalDB, device_id: str, boot_id: str) -> None:
        self._db = db
        self._device_id = device_id
        self._boot_id = boot_id
        self._lock = threading.Lock()
        self._seq = self._recover_sequence()

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def _recover_sequence(self) -> int:
        """Recover the next sequence number from the DB on startup."""
        row = self._db.execute_one(
            "SELECT MAX(sequence_no) AS max_seq FROM telemetry_events WHERE boot_id = ?",
            (self._boot_id,),
        )
        if row and row["max_seq"] is not None:
            return int(row["max_seq"]) + 1
        return 1

    def _next_seq(self) -> int:
        """Return and increment the monotonic sequence counter."""
        seq = self._seq
        self._seq += 1
        return seq

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        telemetry_class: TelemetryClass | str,
        payload: dict[str, Any],
        *,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Append a telemetry event and return the generated event_id."""
        event_id = uuid.uuid4().hex
        occurred_at = datetime.now(timezone.utc).isoformat()
        cls_value = telemetry_class.value if isinstance(telemetry_class, TelemetryClass) else telemetry_class
        payload_json = json.dumps(payload) if payload else None

        with self._lock:
            seq = self._next_seq()
            self._db.execute(
                """
                INSERT INTO telemetry_events
                    (event_id, device_id, boot_id, session_id, sequence_no,
                     event_type, telemetry_class, occurred_at,
                     model_id, model_version, payload_json, uploaded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    event_id,
                    self._device_id,
                    self._boot_id,
                    session_id,
                    seq,
                    event_type,
                    cls_value,
                    occurred_at,
                    model_id,
                    model_version,
                    payload_json,
                ),
            )
        return event_id

    def append_auto(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Append an event, auto-selecting the telemetry class from the event type."""
        cls = default_class_for(event_type)
        return self.append(
            event_type,
            cls,
            payload,
            model_id=model_id,
            model_version=model_version,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_unsent(
        self,
        batch_size: int = 100,
        min_class: Optional[TelemetryClass] = None,
    ) -> list[dict[str, Any]]:
        """Return unsent events ordered by class priority then sequence.

        If *min_class* is set, only events of that class or higher priority
        are included (e.g. min_class=IMPORTANT excludes BEST_EFFORT).
        """
        class_filter = ""
        params: list[Any] = []
        if min_class is not None:
            min_pri = _CLASS_SORT_ORDER.get(min_class, 2)
            allowed = [c.value for c, pri in _CLASS_SORT_ORDER.items() if pri <= min_pri]
            placeholders = ",".join("?" for _ in allowed)
            class_filter = f"AND telemetry_class IN ({placeholders})"
            params.extend(allowed)

        params.append(batch_size)

        rows = self._db.execute(
            f"""
            SELECT event_id, device_id, boot_id, session_id, sequence_no,
                   event_type, telemetry_class, occurred_at,
                   model_id, model_version, payload_json
            FROM telemetry_events
            WHERE uploaded = 0 {class_filter}
            ORDER BY
                CASE telemetry_class
                    WHEN 'MUST_KEEP' THEN 0
                    WHEN 'IMPORTANT' THEN 1
                    WHEN 'BEST_EFFORT' THEN 2
                    ELSE 3
                END,
                sequence_no
            LIMIT ?
            """,
            tuple(params),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------

    def mark_sent(self, event_ids: list[str], batch_id: str) -> None:
        """Mark events as uploaded with the given batch_id."""
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        self._db.execute(
            f"""
            UPDATE telemetry_events
            SET uploaded = 1, batch_id = ?
            WHERE event_id IN ({placeholders})
            """,
            (batch_id, *event_ids),
        )

    def mark_acked(self, boot_id: str, through_seq: int) -> None:
        """Update the cursor to record that the server has acked through *through_seq*."""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """
            INSERT INTO telemetry_cursors (boot_id, last_acked_seq, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(boot_id) DO UPDATE
                SET last_acked_seq = MAX(last_acked_seq, excluded.last_acked_seq),
                    updated_at = excluded.updated_at
            """,
            (boot_id, through_seq, now),
        )

    def get_cursor(self, boot_id: str) -> int:
        """Return the last acked sequence number for a boot_id (0 if none)."""
        row = self._db.execute_one(
            "SELECT last_acked_seq FROM telemetry_cursors WHERE boot_id = ?",
            (boot_id,),
        )
        return int(row["last_acked_seq"]) if row else 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def drop_best_effort(self, older_than_hours: int = 24) -> int:
        """Delete BEST_EFFORT events older than the given threshold. Returns count."""
        cutoff = _hours_ago_iso(older_than_hours)
        return self._delete_count(
            "DELETE FROM telemetry_events WHERE telemetry_class = 'BEST_EFFORT' AND occurred_at < ?",
            (cutoff,),
        )

    def drop_important(self, older_than_hours: int = 168) -> int:
        """Delete IMPORTANT events older than the given threshold (default 7 days)."""
        cutoff = _hours_ago_iso(older_than_hours)
        return self._delete_count(
            "DELETE FROM telemetry_events WHERE telemetry_class = 'IMPORTANT' AND occurred_at < ?",
            (cutoff,),
        )

    def storage_pressure_cleanup(self, target_bytes: int) -> int:
        """Aggressively drop events to free space. BEST_EFFORT first, then IMPORTANT.

        Returns total number of events dropped.
        """
        total_dropped = 0
        # Phase 1: drop all BEST_EFFORT
        total_dropped += self._delete_count(
            "DELETE FROM telemetry_events WHERE telemetry_class = 'BEST_EFFORT'",
        )

        # Check if we've freed enough (heuristic: ~512 bytes per event)
        if total_dropped * 512 >= target_bytes:
            return total_dropped

        # Phase 2: drop IMPORTANT
        total_dropped += self._delete_count(
            "DELETE FROM telemetry_events WHERE telemetry_class = 'IMPORTANT'",
        )
        return total_dropped

    def _delete_count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """Execute a DELETE and return the number of rows affected."""
        with self._db.transaction() as cursor:
            cursor.execute(sql, params)
            return cursor.rowcount

    def count_by_class(self) -> dict[str, int]:
        """Return a dict of event counts keyed by telemetry class."""
        rows = self._db.execute(
            """
            SELECT telemetry_class, COUNT(*) as cnt
            FROM telemetry_events
            GROUP BY telemetry_class
            """,
        )
        return {row["telemetry_class"]: int(row["cnt"]) for row in rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hours_ago_iso(hours: int) -> str:
    """Return an ISO timestamp for *hours* ago."""
    from datetime import timedelta

    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()
