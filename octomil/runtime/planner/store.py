"""Local cache for runtime plans and benchmarks.

Three backends:

  - :class:`SQLiteRuntimePlannerStore` — durable, on-disk; the
    production default.
  - :class:`MemoryRuntimePlannerStore` — bounded TTL-based in-memory
    cache used as the automatic fallback when ``_sqlite3`` is not
    available (Ren'Py, some PyInstaller builds, sandboxed embeds).
    Process-local; thread-safe via ``RLock``; expiry by
    ``time.monotonic()`` for monotonicity but absolute ``time.time()``
    for cache-key TTL accounting consistent with the SQLite backend.
  - :class:`NullRuntimePlannerStore` — explicit no-op for tests and
    `OCTOMIL_RUNTIME_PLANNER_CACHE=0`. Never selected automatically.

The factory :func:`build_runtime_planner_store` is the only thing
callers should use; it picks SQLite when available, Memory when
``_sqlite3`` is missing, and emits a one-time WARNING log so the
operator can see why the on-disk cache isn't being used. The
WARNING is "*sqlite cache unavailable; using in-memory*", **not**
"planner disabled" — the planner still attempts network resolution
and benchmark uploads.

The legacy public name :class:`RuntimePlannerStore` aliases the
chosen backend so existing call sites keep working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".cache" / "octomil" / "runtime_planner.sqlite3"

# Memory-cache bounds. The plan cache holds compact JSON blobs; the
# benchmark cache holds dicts of telemetry. 256 entries each is well
# under any embedded memory budget and far more than a single user
# session typically touches.
_MEMORY_CACHE_MAX_ENTRIES = 256

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


# ---------------------------------------------------------------------------
# Lazy sqlite3 import
# ---------------------------------------------------------------------------


def _try_import_sqlite3() -> Any:
    """Return the ``sqlite3`` module, or ``None`` if ``_sqlite3`` is missing.

    Some Python distributions (Ren'Py's bundled CPython, certain
    PyInstaller builds, sandboxed embeds) ship without the
    ``_sqlite3`` C extension. ``import sqlite3`` then raises
    :class:`ImportError` at import time. Doing it lazily inside this
    helper means top-level ``import octomil`` still succeeds in
    those environments — the planner just falls back to the memory
    backend.
    """
    try:
        import sqlite3 as _sqlite3  # noqa: F401  (probe import)

        return _sqlite3
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RuntimePlannerStoreProtocol(Protocol):
    """Cache contract shared by SQLite/Memory/Null backends."""

    def get_plan(self, cache_key: str) -> Optional[dict]: ...

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
    ) -> None: ...

    def get_benchmark(self, cache_key: str) -> Optional[dict]: ...

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
    ) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Cache-key helpers (shared)
# ---------------------------------------------------------------------------


def _make_cache_key(**kwargs: str | None) -> str:
    """Build a deterministic cache key from components.

    Callers typically include capability, model/app ref, policy
    preset, org id hash, key type, API base, SDK/schema version, and
    device/runtime profile so that the cache doesn't collide across
    user contexts. The hash collapses to 32 hex chars for short
    primary keys; the SHA-256 input is the joined sorted parts.
    """
    parts = "|".join(f"{k}={v or ''}" for k, v in sorted(kwargs.items()))
    return hashlib.sha256(parts.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteRuntimePlannerStore:
    """Durable on-disk cache backed by SQLite.

    Created lazily on first access; safe for single-process use.
    Concurrent multi-process access is not guaranteed but won't
    corrupt the database (WAL mode).

    Falls back transparently to :class:`MemoryRuntimePlannerStore`
    when sqlite3 is somehow loadable but a runtime ``DatabaseError``
    fires the connect loop on a corrupt DB the SDK can't unlink.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._sqlite3 = _try_import_sqlite3()
        if self._sqlite3 is None:
            raise ImportError(
                "SQLiteRuntimePlannerStore requires the _sqlite3 extension. "
                "Use build_runtime_planner_store() to get an automatic fallback."
            )
        self._db_path = Path(os.environ.get("OCTOMIL_RUNTIME_PLANNER_DB", "") or str(db_path or _DEFAULT_DB_PATH))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Any = None
        self._ensure_schema()

    def _get_conn(self) -> Any:
        if self._conn is None:
            self._conn = self._sqlite3.connect(str(self._db_path), timeout=5.0)
            self._conn.row_factory = self._sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist. Recreates DB on corruption."""
        try:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            self._ensure_benchmark_columns(conn)
        except self._sqlite3.DatabaseError:
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

    def _ensure_benchmark_columns(self, conn: Any) -> None:
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
        return _make_cache_key(**kwargs)

    # --- Plan cache ----------------------------------------------------

    def get_plan(self, cache_key: str) -> dict | None:
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT plan_json, expires_at FROM plan_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if time.time() > row["expires_at"]:
                conn.execute("DELETE FROM plan_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None
            return json.loads(row["plan_json"])
        except (self._sqlite3.Error, json.JSONDecodeError):
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
        except self._sqlite3.Error:
            logger.debug("Failed to write plan cache", exc_info=True)

    # --- Benchmark cache -----------------------------------------------

    def get_benchmark(self, cache_key: str) -> dict | None:
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
        except self._sqlite3.Error:
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
        except self._sqlite3.Error:
            logger.debug("Failed to write benchmark cache", exc_info=True)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except self._sqlite3.Error:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Memory backend (bounded, TTL, thread-safe)
# ---------------------------------------------------------------------------


class MemoryRuntimePlannerStore:
    """Process-local in-memory cache used when sqlite3 is unavailable.

    LRU-evicting (capacity ``_MEMORY_CACHE_MAX_ENTRIES`` per cache),
    expiry by absolute ``time.time()`` so TTL semantics match the
    SQLite backend, thread-safe via :class:`threading.RLock`. No new
    dependency.

    Caveats vs SQLite:
      - per-process; not shared across processes or restarts;
      - unbounded cross-process call patterns (multiple short-lived
        Python processes hitting the planner) lose cache locality
        and hit the network more often. Acceptable trade-off for
        environments where SQLite simply isn't available; the
        planner still works and the SDK doesn't crash on import.
    """

    def __init__(self, *, max_entries: int = _MEMORY_CACHE_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._plans: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
        self._benchmarks: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_cache_key(**kwargs: str | None) -> str:
        return _make_cache_key(**kwargs)

    def _trim(self, store: "OrderedDict[str, tuple[float, dict]]") -> None:
        while len(store) > self._max_entries:
            store.popitem(last=False)

    def get_plan(self, cache_key: str) -> dict | None:
        now = time.time()
        with self._lock:
            entry = self._plans.get(cache_key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, plan = entry
            if now > expires_at:
                self._plans.pop(cache_key, None)
                self._misses += 1
                return None
            self._plans.move_to_end(cache_key)
            self._hits += 1
            return plan

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
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError:
            logger.debug("Refusing to cache non-JSON planner payload", exc_info=True)
            return
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._plans[cache_key] = (expires_at, plan)
            self._plans.move_to_end(cache_key)
            self._trim(self._plans)

    def get_benchmark(self, cache_key: str) -> dict | None:
        now = time.time()
        with self._lock:
            entry = self._benchmarks.get(cache_key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, payload = entry
            if now > expires_at:
                self._benchmarks.pop(cache_key, None)
                self._misses += 1
                return None
            self._benchmarks.move_to_end(cache_key)
            self._hits += 1
            return dict(payload)

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
        now = time.time()
        payload = {
            "cache_key": cache_key,
            "model": model,
            "policy": policy,
            "capability": capability,
            "engine": engine,
            "engine_version": engine_version,
            "platform": platform,
            "arch": arch,
            "chip": chip,
            "sdk_version": sdk_version,
            "installed_hash": installed_hash,
            "tokens_per_second": tokens_per_second,
            "ttft_ms": ttft_ms,
            "memory_mb": memory_mb,
            "metadata_json": metadata_json,
            "created_at": now,
            "expires_at": now + ttl_seconds,
        }
        with self._lock:
            self._benchmarks[cache_key] = (now + ttl_seconds, payload)
            self._benchmarks.move_to_end(cache_key)
            self._trim(self._benchmarks)

    def stats(self) -> dict[str, int]:
        """Return basic counters; intended for diagnostic logging."""
        with self._lock:
            return {
                "plan_entries": len(self._plans),
                "benchmark_entries": len(self._benchmarks),
                "hits": self._hits,
                "misses": self._misses,
            }

    def close(self) -> None:
        with self._lock:
            self._plans.clear()
            self._benchmarks.clear()


# ---------------------------------------------------------------------------
# Null backend (explicit no-op)
# ---------------------------------------------------------------------------


class NullRuntimePlannerStore:
    """Explicit no-op cache.

    Selected when ``OCTOMIL_RUNTIME_PLANNER_CACHE=0`` is set or by
    tests that want to disable caching entirely. **Never** chosen
    automatically — the auto fallback for missing sqlite3 is the
    bounded :class:`MemoryRuntimePlannerStore`.
    """

    @staticmethod
    def _make_cache_key(**kwargs: str | None) -> str:
        return _make_cache_key(**kwargs)

    def get_plan(self, cache_key: str) -> dict | None:
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
        return None

    def get_benchmark(self, cache_key: str) -> dict | None:
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
        return None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_warned_about_missing_sqlite = False


def build_runtime_planner_store(
    db_path: str | Path | None = None,
    *,
    force_memory: bool = False,
    force_null: bool = False,
) -> RuntimePlannerStoreProtocol:
    """Pick the right backend for this process.

    Resolution order:

      1. ``force_null=True`` (or env ``OCTOMIL_RUNTIME_PLANNER_CACHE=0``)
         → :class:`NullRuntimePlannerStore`.
      2. ``force_memory=True`` → :class:`MemoryRuntimePlannerStore`.
         Test hook; production code never sets this directly.
      3. ``_sqlite3`` importable → :class:`SQLiteRuntimePlannerStore`.
      4. ``_sqlite3`` missing → :class:`MemoryRuntimePlannerStore`,
         with a one-time WARNING log so the operator can see why
         the on-disk cache isn't being used.

    The planner itself doesn't get disabled in any of these branches.
    Network resolution and benchmark uploads continue normally; only
    the caching layer differs.
    """
    if force_null or os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE", "").strip() == "0":
        return NullRuntimePlannerStore()
    if force_memory:
        return MemoryRuntimePlannerStore()

    if _try_import_sqlite3() is None:
        global _warned_about_missing_sqlite
        if not _warned_about_missing_sqlite:
            logger.warning(
                "runtime planner sqlite cache unavailable; using in-memory planner cache "
                "(this is expected on Ren'Py / some PyInstaller / sandboxed Python builds "
                "that ship without the _sqlite3 extension; planner will still attempt "
                "network resolution)"
            )
            _warned_about_missing_sqlite = True
        return MemoryRuntimePlannerStore()

    try:
        return SQLiteRuntimePlannerStore(db_path=db_path)
    except ImportError:
        # Race: ``_try_import_sqlite3`` succeeded but the constructor
        # disagreed (very unusual; defensive only).
        logger.warning("Falling back to in-memory planner cache after SQLite construction failed", exc_info=True)
        return MemoryRuntimePlannerStore()


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


# Existing callers do ``from .store import RuntimePlannerStore`` and
# ``RuntimePlannerStore._make_cache_key(...)``. Keep those working by
# aliasing to the production backend factory and exposing the
# ``_make_cache_key`` static method on a thin shim that delegates to
# :func:`build_runtime_planner_store`.
class RuntimePlannerStore:
    """Legacy entry point.

    Constructing this class returns the chosen backend transparently
    (SQLite when available, Memory otherwise). The class also exposes
    ``_make_cache_key`` as a staticmethod so existing callers that
    use it as a class-level utility keep working.
    """

    def __new__(cls, db_path: str | Path | None = None) -> Any:  # type: ignore[misc]
        return build_runtime_planner_store(db_path=db_path)

    @staticmethod
    def _make_cache_key(**kwargs: str | None) -> str:
        return _make_cache_key(**kwargs)
