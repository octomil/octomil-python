"""SQLite-based local gradient cache for resilient training."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional


class GradientCache:
    """Persist gradient updates locally so they survive upload failures.

    Uses a single SQLite file that is safe for concurrent reads from the
    same process.  Each entry stores the serialized weights for a
    (round_id, device_id) pair along with metadata.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        round_id: str,
        device_id: str,
        weights_data: bytes,
        sample_count: int = 0,
    ) -> None:
        """Store a gradient update in the cache."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO gradient_cache "
                "(round_id, device_id, weights_data, sample_count, submitted, created_at) "
                "VALUES (?, ?, ?, ?, 0, ?)",
                (round_id, device_id, weights_data, sample_count, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, round_id: str, device_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single cached entry, or None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT round_id, device_id, weights_data, sample_count, submitted, created_at "
                "FROM gradient_cache WHERE round_id = ? AND device_id = ?",
                (round_id, device_id),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    def list_pending(self, device_id: str) -> List[Dict[str, Any]]:
        """Return all unsubmitted entries for a device."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT round_id, device_id, weights_data, sample_count, submitted, created_at "
                "FROM gradient_cache WHERE device_id = ? AND submitted = 0 "
                "ORDER BY created_at ASC",
                (device_id,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def mark_submitted(self, round_id: str, device_id: str) -> None:
        """Mark an entry as successfully submitted."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE gradient_cache SET submitted = 1 "
                "WHERE round_id = ? AND device_id = ?",
                (round_id, device_id),
            )
            conn.commit()
        finally:
            conn.close()

    def purge_older_than(self, max_age_seconds: float) -> int:
        """Delete entries older than *max_age_seconds*. Returns count deleted."""
        cutoff = time.time() - max_age_seconds
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM gradient_cache WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS gradient_cache ("
                "  round_id TEXT NOT NULL,"
                "  device_id TEXT NOT NULL,"
                "  weights_data BLOB NOT NULL,"
                "  sample_count INTEGER NOT NULL DEFAULT 0,"
                "  submitted INTEGER NOT NULL DEFAULT 0,"
                "  created_at REAL NOT NULL,"
                "  PRIMARY KEY (round_id, device_id)"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        return {
            "round_id": row[0],
            "device_id": row[1],
            "weights_data": row[2],
            "sample_count": row[3],
            "submitted": bool(row[4]),
            "created_at": row[5],
        }
