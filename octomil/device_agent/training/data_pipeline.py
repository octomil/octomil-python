"""Data pipeline for selecting, snapshotting, and loading training examples.

Provides bounded, deterministic dataset snapshots for local training
with configurable selection policies and automatic expiry.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..db.local_db import LocalDB

logger = logging.getLogger(__name__)


class DataPipeline:
    """Manages training data selection, snapshotting, and lifecycle.

    Datasets are created as immutable snapshots with stable IDs and
    bounded sizes. Old snapshots are expired automatically.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def select_examples(
        self,
        source_scope: str,
        max_count: int,
        selection_policy: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Select examples from a source scope for training.

        This is a placeholder that returns synthetic examples. In production,
        this would query the device's local interaction store filtered by
        scope and apply the selection policy.

        ``selection_policy`` supports:
          - strategy: "recent" | "random" | "diverse" (default: "recent")
          - min_quality: float (default: 0.0)
          - exclude_pii: bool (default: True)
        """
        policy = selection_policy or {}
        strategy = policy.get("strategy", "recent")
        examples: list[dict[str, Any]] = []

        # Placeholder: in production this reads from a local interaction store
        for i in range(min(max_count, 100)):
            examples.append(
                {
                    "example_id": str(uuid.uuid4()),
                    "source_scope": source_scope,
                    "index": i,
                    "strategy": strategy,
                }
            )

        logger.debug(
            "Selected %d examples from scope=%s (strategy=%s)",
            len(examples),
            source_scope,
            strategy,
        )
        return examples

    def create_snapshot(
        self,
        dataset_id: str,
        examples: list[dict[str, Any]],
        source_scope: str,
        expires_hours: int = 24,
    ) -> dict[str, Any]:
        """Persist a dataset snapshot and return its record.

        The snapshot is stored in the training_datasets table with a
        deterministic ID. An optional expiry is set for automatic cleanup.
        """
        now = self._now()
        payload = json.dumps(examples)
        byte_size = len(payload.encode("utf-8"))

        # Calculate expiry
        expires_at: Optional[str] = None
        if expires_hours > 0:
            from datetime import timedelta

            expires_dt = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
            expires_at = expires_dt.isoformat()

        self._db.execute(
            """
            INSERT OR REPLACE INTO training_datasets
                (dataset_id, source_scope, example_count, byte_size,
                 selection_policy_json, snapshot_created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                source_scope,
                len(examples),
                byte_size,
                payload,
                now,
                expires_at,
            ),
        )

        logger.info(
            "Created dataset snapshot %s (%d examples, %d bytes)",
            dataset_id,
            len(examples),
            byte_size,
        )
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def load_snapshot(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load a dataset snapshot for training.

        Returns the list of examples from the snapshot.
        Raises ValueError if the dataset does not exist.
        """
        row = self._db.execute_one(
            "SELECT selection_policy_json FROM training_datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        if row is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        return json.loads(dict(row)["selection_policy_json"])

    def expire_old_snapshots(self) -> int:
        """Remove expired dataset snapshots.

        Returns the number of snapshots removed.
        """
        now = self._now()
        rows = self._db.execute(
            """
            SELECT dataset_id FROM training_datasets
            WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
            (now,),
        )
        removed = 0
        for row in rows:
            self._db.execute(
                "DELETE FROM training_datasets WHERE dataset_id = ?",
                (dict(row)["dataset_id"],),
            )
            removed += 1

        if removed:
            logger.info("Expired %d old dataset snapshots", removed)
        return removed

    def get_dataset(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a dataset record by ID."""
        row = self._db.execute_one(
            "SELECT * FROM training_datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        return dict(row) if row else None
