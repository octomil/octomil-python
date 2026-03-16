"""Personalization adapter lifecycle management.

Manages versioned adapter artifacts and their bindings to base models.
Supports atomic activation flips, shadow comparison, and rollback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..db.local_db import LocalDB

logger = logging.getLogger(__name__)


@dataclass
class ActiveBinding:
    """Snapshot of the active model+adapter binding for a given key."""

    binding_key: str
    base_model_id: str
    base_version: str
    adapter_id: Optional[str]
    adapter_version: Optional[str]


class AdapterManager:
    """Manages personalization adapter lifecycle on the device.

    Adapters are versioned and immutable once registered. Activation is
    done via binding keys that map a base model slot to an adapter version.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_adapter(
        self,
        adapter_id: str,
        base_model_id: str,
        base_version: str,
        scope: str,
        version: str,
        artifact_path: str,
        parent_version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Register a new adapter artifact. Returns the adapter record."""
        now = self._now()
        self._db.execute(
            """
            INSERT INTO personalization_adapters
                (adapter_id, base_model_id, base_version, adapter_scope, version,
                 parent_adapter_version, status, artifact_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'REGISTERED', ?, ?)
            """,
            (
                adapter_id,
                base_model_id,
                base_version,
                scope,
                version,
                parent_version,
                artifact_path,
                now,
            ),
        )
        logger.info(
            "Registered adapter %s (model=%s, scope=%s, v=%s)",
            adapter_id,
            base_model_id,
            scope,
            version,
        )
        return self._get_adapter_row(adapter_id)  # type: ignore[return-value]

    def get_active_adapter(self, binding_key: str) -> Optional[dict[str, Any]]:
        """Get the active adapter for a binding key, or None."""
        binding = self.get_binding(binding_key)
        if binding is None or binding.adapter_id is None:
            return None
        return self._get_adapter_row(binding.adapter_id)

    def get_binding(self, binding_key: str) -> Optional[ActiveBinding]:
        """Get the active binding for a key."""
        row = self._db.execute_one(
            "SELECT * FROM active_bindings WHERE binding_key = ?",
            (binding_key,),
        )
        if row is None:
            return None
        d = dict(row)
        return ActiveBinding(
            binding_key=d["binding_key"],
            base_model_id=d["base_model_id"],
            base_version=d["base_version"],
            adapter_id=d.get("adapter_id"),
            adapter_version=d.get("adapter_version"),
        )

    def activate_adapter(
        self,
        binding_key: str,
        adapter_id: str,
        adapter_version: str,
    ) -> None:
        """Atomically flip a binding to point to a new adapter version.

        If no binding exists for the key, looks up the adapter record to
        populate the base model info.
        """
        adapter = self._get_adapter_row(adapter_id)
        if adapter is None:
            raise ValueError(f"Adapter not found: {adapter_id}")

        now = self._now()
        existing = self.get_binding(binding_key)
        if existing is None:
            self._db.execute(
                """
                INSERT INTO active_bindings
                    (binding_key, base_model_id, base_version,
                     adapter_id, adapter_version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    binding_key,
                    adapter["base_model_id"],
                    adapter["base_version"],
                    adapter_id,
                    adapter_version,
                    now,
                ),
            )
        else:
            self._db.execute(
                """
                UPDATE active_bindings
                SET adapter_id = ?, adapter_version = ?, updated_at = ?
                WHERE binding_key = ?
                """,
                (adapter_id, adapter_version, now, binding_key),
            )

        # Update adapter status to ACTIVE
        self._db.execute(
            "UPDATE personalization_adapters SET status = 'ACTIVE' WHERE adapter_id = ?",
            (adapter_id,),
        )
        logger.info(
            "Activated adapter %s (v=%s) for binding %s",
            adapter_id,
            adapter_version,
            binding_key,
        )

    def shadow_compare(
        self,
        binding_key: str,
        candidate_adapter_id: str,
        eval_prompts: list[str],
    ) -> dict[str, Any]:
        """Compare current adapter vs candidate silently.

        Returns comparison results without changing the active binding.
        This is a placeholder that returns the comparison metadata;
        actual inference comparison is driven by the caller.
        """
        current = self.get_active_adapter(binding_key)
        candidate = self._get_adapter_row(candidate_adapter_id)
        return {
            "binding_key": binding_key,
            "current_adapter_id": current["adapter_id"] if current else None,
            "candidate_adapter_id": candidate_adapter_id,
            "candidate_version": candidate["version"] if candidate else None,
            "prompts_count": len(eval_prompts),
            "status": "comparison_ready",
        }

    def rollback_adapter(self, binding_key: str, reason: str) -> Optional[str]:
        """Revert binding to previous adapter version.

        Records the rollback in the rollback_records table.
        Returns the previous adapter_id, or None if no previous version.
        """
        binding = self.get_binding(binding_key)
        if binding is None:
            return None

        current_adapter_id = binding.adapter_id
        if current_adapter_id is None:
            return None

        # Find the previous version of this adapter (by parent)
        current = self._get_adapter_row(current_adapter_id)
        if current is None:
            return None

        if current.get("parent_adapter_version") is None:
            # No parent version to roll back to; clear the binding
            now = self._now()
            self._db.execute(
                """
                UPDATE active_bindings
                SET adapter_id = NULL, adapter_version = NULL, updated_at = ?
                WHERE binding_key = ?
                """,
                (now, binding_key),
            )
            self._db.execute(
                """
                INSERT INTO rollback_records
                    (model_id, from_version, to_version, reason, rolled_back_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    binding.base_model_id,
                    current["version"],
                    "none",
                    reason,
                    now,
                ),
            )
            return None

        # Find the parent adapter
        parent_row = self._db.execute_one(
            """
            SELECT * FROM personalization_adapters
            WHERE base_model_id = ? AND version = ? AND adapter_scope = ?
            """,
            (
                current["base_model_id"],
                current["parent_adapter_version"],
                current["adapter_scope"],
            ),
        )
        if parent_row is None:
            return None

        parent = dict(parent_row)
        now = self._now()
        self._db.execute(
            """
            UPDATE active_bindings
            SET adapter_id = ?, adapter_version = ?, updated_at = ?
            WHERE binding_key = ?
            """,
            (parent["adapter_id"], parent["version"], now, binding_key),
        )
        self._db.execute(
            """
            INSERT INTO rollback_records
                (model_id, from_version, to_version, reason, rolled_back_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                binding.base_model_id,
                current["version"],
                parent["version"],
                reason,
                now,
            ),
        )
        logger.info(
            "Rolled back binding %s: %s -> %s (reason: %s)",
            binding_key,
            current["version"],
            parent["version"],
            reason,
        )
        return parent["adapter_id"]

    def list_adapters(
        self,
        base_model_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List registered adapters, optionally filtered."""
        conditions: list[str] = []
        params: list[Any] = []
        if base_model_id:
            conditions.append("base_model_id = ?")
            params.append(base_model_id)
        if scope:
            conditions.append("adapter_scope = ?")
            params.append(scope)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._db.execute(
            f"SELECT * FROM personalization_adapters {where} ORDER BY created_at DESC",
            tuple(params),
        )
        return [dict(r) for r in rows]

    def gc_old_adapters(self, keep_versions: int = 3) -> int:
        """Garbage collect old adapter versions, keeping the latest N per scope.

        Returns the number of adapters removed.
        """
        # Get unique (base_model_id, adapter_scope) pairs
        groups = self._db.execute("SELECT DISTINCT base_model_id, adapter_scope FROM personalization_adapters")
        removed = 0
        for group in groups:
            g = dict(group)
            # Get all versions for this group, newest first
            rows = self._db.execute(
                """
                SELECT adapter_id FROM personalization_adapters
                WHERE base_model_id = ? AND adapter_scope = ? AND status != 'ACTIVE'
                ORDER BY created_at DESC
                """,
                (g["base_model_id"], g["adapter_scope"]),
            )
            # Skip the first keep_versions, delete the rest
            to_delete = [dict(r)["adapter_id"] for r in rows[keep_versions:]]
            for aid in to_delete:
                self._db.execute(
                    "DELETE FROM personalization_adapters WHERE adapter_id = ?",
                    (aid,),
                )
                removed += 1
        if removed:
            logger.info("GC removed %d old adapters", removed)
        return removed

    def _get_adapter_row(self, adapter_id: str) -> Optional[dict[str, Any]]:
        row = self._db.execute_one(
            "SELECT * FROM personalization_adapters WHERE adapter_id = ?",
            (adapter_id,),
        )
        return dict(row) if row else None
