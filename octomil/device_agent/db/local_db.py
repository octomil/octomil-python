"""SQLite WAL database manager for device agent local state."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .schema import SCHEMA_STATEMENTS

logger = logging.getLogger(__name__)


class LocalDB:
    """Thread-safe SQLite database with WAL mode for device agent state.

    Uses a single reusable connection with serialized access via a lock.
    Schema is auto-created on first use.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._apply_schema()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def _apply_schema(self) -> None:
        with self._lock:
            assert self._conn is not None
            for stmt in SCHEMA_STATEMENTS:
                self._conn.execute(stmt)
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Yield a cursor inside a serialized transaction.

        Commits on success, rolls back on exception.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a single statement and return all rows."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor.fetchall()

    def execute_one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        """Execute a single statement and return the first row, or None."""
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
