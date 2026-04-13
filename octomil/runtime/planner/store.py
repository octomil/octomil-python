"""Local SQLite cache for runtime plans and benchmarks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".cache" / "octomil" / "runtime_planner.sqlite3"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS plan_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    capability  TEXT NOT NULL,
    policy      TEXT NOT NULL,
    plan_json   TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    ttl_seconds INTEGER NOT NULL DEFAULT 604800,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_cache (
    cache_key          TEXT PRIMARY KEY,
    model              TEXT NOT NULL,
    policy             TEXT NOT NULL DEFAULT '',
    capability         TEXT NOT NULL,
    engine             TEXT NOT NULL,
    engine_version     TEXT,
    platform           TEXT NOT NULL DEFAULT '',
    arch               TEXT NOT NULL DEFAULT '',
    chip               TEXT,
    sdk_version        TEXT NOT NULL DEFAULT '',
    installed_hash     TEXT NOT NULL DEFAULT '',
    tokens_per_second  REAL NOT NULL DEFAULT 0.0,
    ttft_ms            REAL NOT NULL DEFAULT 0.0,
    memory_mb          REAL NOT NULL DEFAULT 0.0,
    metadata_json      TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL,
    expires_at         REAL NOT NULL
);
"""


class RuntimePlannerStore:
    """Local SQLite cache for runtime plans and benchmark results.

    The store is created lazily on first access and is safe for single-process
    use.  Concurrent access from multiple processes is not guaranteed but will
    not corrupt the database (SQLite WAL mode).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(os.environ.get("OCTOMIL_RUNTIME_PLANNER_DB", "") or str(db_path or _DEFAULT_DB_PATH))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist. Recreates DB on corruption."""
        try:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            self._ensure_benchmark_columns(conn)
        except sqlite3.DatabaseError:
            logger.warning("Corrupt planner DB at %s — recreating", self._db_path, exc_info=True)
            self.close()
            try:
                self._db_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                self._conn = None
                conn = self._get_conn()
                conn.executescript(_SCHEMA_SQL)
                self._ensure_benchmark_columns(conn)
            except Exception:
                logger.error("Failed to recreate planner DB", exc_info=True)

    def _ensure_benchmark_columns(self, conn: sqlite3.Connection) -> None:
        """Add cache columns for users who tried earlier pre-merge builds."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(benchmark_cache)").fetchall()}
        columns = {
            "policy": "TEXT NOT NULL DEFAULT ''",
            "engine_version": "TEXT",
            "platform": "TEXT NOT NULL DEFAULT ''",
            "arch": "TEXT NOT NULL DEFAULT ''",
            "chip": "TEXT",
            "sdk_version": "TEXT NOT NULL DEFAULT ''",
            "installed_hash": "TEXT NOT NULL DEFAULT ''",
            "expires_at": "REAL NOT NULL DEFAULT 0",
        }
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE benchmark_cache ADD COLUMN {name} {ddl}")
        conn.commit()

    @staticmethod
    def _make_cache_key(**kwargs: str | None) -> str:
        """Build a deterministic cache key from components."""
        parts = "|".join(f"{k}={v or ''}" for k, v in sorted(kwargs.items()))
        return hashlib.sha256(parts.encode()).hexdigest()[:32]

    # ------------------------------------------------------------------
    # Plan cache
    # ------------------------------------------------------------------

    def get_plan(self, cache_key: str) -> dict | None:
        """Return cached plan if fresh, else None."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT plan_json, expires_at FROM plan_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if time.time() > row["expires_at"]:
                # Expired — clean it up
                conn.execute("DELETE FROM plan_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None
            return json.loads(row["plan_json"])
        except (sqlite3.Error, json.JSONDecodeError):
            logger.debug("Failed to read plan cache", exc_info=True)
            return None

    def put_plan(
        self,
        cache_key: str,
        *,
        model: str,
        capability: str,
        policy: str,
        plan_json: str,
        source: str,
        ttl_seconds: int,
    ) -> None:
        """Insert or replace a plan in the cache."""
        now = time.time()
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO plan_cache
                   (cache_key, model, capability, policy, plan_json, source, ttl_seconds, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cache_key, model, capability, policy, plan_json, source, ttl_seconds, now, now + ttl_seconds),
            )
            conn.commit()
        except sqlite3.Error:
            logger.debug("Failed to write plan cache", exc_info=True)

    # ------------------------------------------------------------------
    # Benchmark cache
    # ------------------------------------------------------------------

    def get_benchmark(self, cache_key: str) -> dict | None:
        """Return cached benchmark result, or None."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM benchmark_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if time.time() > row["expires_at"]:
                conn.execute("DELETE FROM benchmark_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None
            return dict(row)
        except sqlite3.Error:
            logger.debug("Failed to read benchmark cache", exc_info=True)
            return None

    def put_benchmark(
        self,
        cache_key: str,
        *,
        model: str,
        capability: str,
        engine: str,
        policy: str = "",
        engine_version: str | None = None,
        platform: str = "",
        arch: str = "",
        chip: str | None = None,
        sdk_version: str = "",
        installed_hash: str = "",
        tokens_per_second: float = 0.0,
        ttft_ms: float = 0.0,
        memory_mb: float = 0.0,
        metadata_json: str = "{}",
        ttl_seconds: int = 1_209_600,
    ) -> None:
        """Insert or replace a benchmark result in the cache."""
        now = time.time()
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_cache
                   (cache_key, model, policy, capability, engine, engine_version, platform, arch, chip, sdk_version,
                    installed_hash, tokens_per_second, ttft_ms, memory_mb, metadata_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_key,
                    model,
                    policy,
                    capability,
                    engine,
                    engine_version,
                    platform,
                    arch,
                    chip,
                    sdk_version,
                    installed_hash,
                    tokens_per_second,
                    ttft_ms,
                    memory_mb,
                    metadata_json,
                    now,
                    now + ttl_seconds,
                ),
            )
            conn.commit()
        except sqlite3.Error:
            logger.debug("Failed to write benchmark cache", exc_info=True)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
