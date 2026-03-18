"""Benchmark result storage keyed on model identity + device context.

Persists per-engine benchmark results (latency, throughput, memory) in
the device agent SQLite database. Provides lookup for best engine selection
and full result retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .db.local_db import LocalDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkResult:
    """A single benchmark measurement for one engine."""

    model_id: str
    model_version: str
    device_class: str
    sdk_version: str
    engine: str
    latency_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    memory_bytes: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkStore:
    """CRUD store for per-engine benchmark results.

    Key structure: (model_id, model_version, device_class, sdk_version, engine).
    Does NOT use file path or file size as identity.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def record(self, result: BenchmarkResult) -> None:
        """Insert or update a benchmark result (upsert)."""
        now = _now_iso()
        metadata_json = json.dumps(result.metadata) if result.metadata else None
        self._db.execute(
            "INSERT INTO benchmark_results "
            "(model_id, model_version, device_class, sdk_version, engine, "
            " latency_ms, throughput_tps, memory_bytes, metadata_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id, model_version, device_class, sdk_version, engine) "
            "DO UPDATE SET latency_ms=excluded.latency_ms, "
            "  throughput_tps=excluded.throughput_tps, "
            "  memory_bytes=excluded.memory_bytes, "
            "  metadata_json=excluded.metadata_json, "
            "  recorded_at=excluded.recorded_at",
            (
                result.model_id,
                result.model_version,
                result.device_class,
                result.sdk_version,
                result.engine,
                result.latency_ms,
                result.throughput_tps,
                result.memory_bytes,
                metadata_json,
                now,
            ),
        )

    def get_results(
        self,
        model_id: str,
        model_version: str,
        device_class: Optional[str] = None,
        sdk_version: Optional[str] = None,
    ) -> list[BenchmarkResult]:
        """Return all benchmark results for a model, optionally filtered."""
        sql = "SELECT * FROM benchmark_results WHERE model_id = ? AND model_version = ?"
        params: list[Any] = [model_id, model_version]

        if device_class is not None:
            sql += " AND device_class = ?"
            params.append(device_class)
        if sdk_version is not None:
            sql += " AND sdk_version = ?"
            params.append(sdk_version)

        sql += " ORDER BY latency_ms ASC NULLS LAST"
        rows = self._db.execute(sql, tuple(params))
        return [self._row_to_result(row) for row in rows]

    def get_best_engine(
        self,
        model_id: str,
        model_version: str,
        device_class: Optional[str] = None,
        sdk_version: Optional[str] = None,
    ) -> Optional[str]:
        """Return the engine with the lowest latency for the given model.

        Returns None if no benchmark results exist.
        """
        results = self.get_results(model_id, model_version, device_class, sdk_version)
        if not results:
            return None
        # Filter to results that have latency data
        with_latency = [r for r in results if r.latency_ms is not None]
        if not with_latency:
            return results[0].engine
        return min(with_latency, key=lambda r: r.latency_ms).engine  # type: ignore[arg-type,return-value]

    def delete(
        self,
        model_id: str,
        model_version: str,
        engine: Optional[str] = None,
    ) -> int:
        """Delete benchmark results. Returns the number of rows deleted."""
        if engine is not None:
            self._db.execute(
                "DELETE FROM benchmark_results WHERE model_id = ? AND model_version = ? AND engine = ?",
                (model_id, model_version, engine),
            )
        else:
            self._db.execute(
                "DELETE FROM benchmark_results WHERE model_id = ? AND model_version = ?",
                (model_id, model_version),
            )
        # SQLite DELETE doesn't return rows; use changes count via a separate query
        count_row = self._db.execute_one("SELECT changes() AS cnt")
        return count_row["cnt"] if count_row else 0

    @staticmethod
    def _row_to_result(row: Any) -> BenchmarkResult:
        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return BenchmarkResult(
            model_id=row["model_id"],
            model_version=row["model_version"],
            device_class=row["device_class"],
            sdk_version=row["sdk_version"],
            engine=row["engine"],
            latency_ms=row["latency_ms"],
            throughput_tps=row["throughput_tps"],
            memory_bytes=row["memory_bytes"],
            metadata=metadata,
        )
