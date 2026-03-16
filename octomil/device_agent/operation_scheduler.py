"""Lease-based operation scheduler with retry and priority queuing.

Operations are persisted in SQLite so incomplete work survives restarts.
Leases prevent duplicate execution; expired leases are reclaimed automatically.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)


# Priority levels (lower number = higher priority)
PRIORITY_CRITICAL_FOREGROUND = 0
PRIORITY_BACKGROUND_IMPORTANT = 50
PRIORITY_BACKGROUND_BEST_EFFORT = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_plus(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _retry_delay(attempt: int, base: float = 1.0, max_delay: float = 300.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base * (2**attempt), max_delay)
    return delay + random.uniform(0, delay * 0.1)


class OperationScheduler:
    """Lease-based operation scheduling with retry and priority."""

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def schedule(
        self,
        op_type: str,
        resource_id: str,
        payload: Optional[dict[str, Any]] = None,
        priority: int = PRIORITY_BACKGROUND_BEST_EFFORT,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Schedule a new operation. Returns the op_id."""
        if idempotency_key:
            existing = self._db.execute_one(
                "SELECT op_id FROM operations WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            if existing:
                return existing["op_id"]

        op_id = uuid.uuid4().hex
        import json

        self._db.execute(
            "INSERT INTO operations "
            "(op_id, op_type, resource_id, state, idempotency_key, "
            " priority, payload_json, updated_at) "
            "VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?)",
            (
                op_id,
                op_type,
                resource_id,
                idempotency_key,
                priority,
                json.dumps(payload) if payload else None,
                _now_iso(),
            ),
        )
        return op_id

    def lease(self, op_id: str, owner: str, duration_sec: float = 60.0) -> bool:
        """Attempt to acquire a lease on an operation.

        Returns True if the lease was acquired.
        """
        now = _now_iso()
        expires = _now_plus(duration_sec)

        with self._db.transaction() as cur:
            row = cur.execute(
                "SELECT state, lease_owner, lease_expires_at FROM operations WHERE op_id = ?",
                (op_id,),
            ).fetchone()

            if row is None:
                return False

            # Can lease if PENDING, or if existing lease is expired
            if row["state"] not in ("PENDING", "LEASED"):
                return False

            if row["state"] == "LEASED" and row["lease_expires_at"]:
                if row["lease_expires_at"] > now:
                    return False  # Lease still held

            cur.execute(
                "UPDATE operations SET state = 'LEASED', lease_owner = ?, "
                "lease_expires_at = ?, attempt_count = attempt_count + 1, "
                "updated_at = ? WHERE op_id = ?",
                (owner, expires, now, op_id),
            )
            return True

    def renew_lease(self, op_id: str, owner: str, duration_sec: float = 60.0) -> bool:
        """Renew an existing lease. Returns True if renewed."""
        expires = _now_plus(duration_sec)
        now = _now_iso()

        with self._db.transaction() as cur:
            row = cur.execute(
                "SELECT lease_owner FROM operations WHERE op_id = ? AND state = 'LEASED'",
                (op_id,),
            ).fetchone()

            if row is None or row["lease_owner"] != owner:
                return False

            cur.execute(
                "UPDATE operations SET lease_expires_at = ?, updated_at = ? WHERE op_id = ?",
                (expires, now, op_id),
            )
            return True

    def complete(self, op_id: str) -> None:
        """Mark an operation as successfully completed."""
        self._db.execute(
            "UPDATE operations SET state = 'SUCCESS', lease_owner = NULL, "
            "lease_expires_at = NULL, updated_at = ? WHERE op_id = ?",
            (_now_iso(), op_id),
        )

    def fail(self, op_id: str, error: str) -> None:
        """Mark an operation as failed and schedule retry with backoff."""
        row = self._db.execute_one(
            "SELECT attempt_count FROM operations WHERE op_id = ?",
            (op_id,),
        )
        attempt = row["attempt_count"] if row else 0
        delay = _retry_delay(attempt)
        next_retry = _now_plus(delay)

        self._db.execute(
            "UPDATE operations SET state = 'FAILED', lease_owner = NULL, "
            "lease_expires_at = NULL, next_retry_at = ?, "
            "updated_at = ? WHERE op_id = ?",
            (next_retry, _now_iso(), op_id),
        )

    def recover_expired_leases(self) -> list[str]:
        """Reclaim operations with expired leases. Returns reclaimed op_ids."""
        now = _now_iso()
        rows = self._db.execute(
            "SELECT op_id FROM operations WHERE state = 'LEASED' AND lease_expires_at < ?",
            (now,),
        )
        reclaimed = []
        for row in rows:
            self._db.execute(
                "UPDATE operations SET state = 'PENDING', lease_owner = NULL, "
                "lease_expires_at = NULL, updated_at = ? WHERE op_id = ?",
                (now, row["op_id"]),
            )
            reclaimed.append(row["op_id"])
        return reclaimed

    def get_pending(
        self,
        op_type: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return pending operations, ordered by priority then creation."""
        sql = "SELECT * FROM operations WHERE state = 'PENDING'"
        params: list[Any] = []

        if op_type is not None:
            sql += " AND op_type = ?"
            params.append(op_type)
        if priority is not None:
            sql += " AND priority <= ?"
            params.append(priority)

        sql += " ORDER BY priority ASC, updated_at ASC"
        rows = self._db.execute(sql, tuple(params))
        return [dict(r) for r in rows]

    def get_operation(self, op_id: str) -> Optional[dict[str, Any]]:
        """Return a single operation by ID."""
        row = self._db.execute_one(
            "SELECT * FROM operations WHERE op_id = ?",
            (op_id,),
        )
        return dict(row) if row else None
