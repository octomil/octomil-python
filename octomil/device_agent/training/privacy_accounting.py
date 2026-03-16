"""Privacy budget tracking for differential privacy on device.

Tracks cumulative epsilon and delta spent per scope, enforces daily
budgets, and refuses participation when budgets are exhausted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..db.local_db import LocalDB

logger = logging.getLogger(__name__)


class PrivacyAccountant:
    """Track epsilon/delta spent per scope for differential privacy.

    Each scope (e.g. a model or federation) has its own cumulative budget.
    The accountant supports daily resets for rolling privacy budgets.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_budget(self, scope_key: str) -> dict[str, Any]:
        """Get current privacy budget for a scope.

        Returns {"epsilon_spent": float, "delta_spent": float, "last_updated_at": str}.
        If no record exists, returns zeros.
        """
        row = self._db.execute_one(
            "SELECT * FROM privacy_accounting WHERE scope_key = ?",
            (scope_key,),
        )
        if row is None:
            return {
                "scope_key": scope_key,
                "epsilon_spent": 0.0,
                "delta_spent": 0.0,
                "last_updated_at": self._now(),
            }
        return dict(row)

    def spend(self, scope_key: str, epsilon: float, delta: float) -> dict[str, Any]:
        """Record epsilon/delta expenditure for a scope.

        Creates the record if it does not exist, otherwise accumulates.
        Returns the updated budget record.
        """
        existing = self._db.execute_one(
            "SELECT * FROM privacy_accounting WHERE scope_key = ?",
            (scope_key,),
        )
        now = self._now()

        if existing is None:
            self._db.execute(
                """
                INSERT INTO privacy_accounting
                    (scope_key, epsilon_spent, delta_spent, last_updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (scope_key, epsilon, delta, now),
            )
        else:
            self._db.execute(
                """
                UPDATE privacy_accounting
                SET epsilon_spent = epsilon_spent + ?,
                    delta_spent = delta_spent + ?,
                    last_updated_at = ?
                WHERE scope_key = ?
                """,
                (epsilon, delta, now, scope_key),
            )

        logger.debug(
            "Privacy spend scope=%s: eps=+%.4f, delta=+%.6f",
            scope_key,
            epsilon,
            delta,
        )
        return self.get_budget(scope_key)

    def has_budget(
        self,
        scope_key: str,
        epsilon_needed: float,
        daily_budget_eps: float,
    ) -> bool:
        """Check whether spending epsilon_needed would stay within daily budget.

        Returns True if the scope has sufficient remaining budget.
        """
        budget = self.get_budget(scope_key)
        remaining = daily_budget_eps - budget["epsilon_spent"]
        return remaining >= epsilon_needed

    def reset_daily(self, scope_key: str) -> None:
        """Reset the budget for a scope if the last update was on a previous day.

        This implements a rolling daily budget: if the last update timestamp
        falls on a different UTC date than today, the counters are zeroed.
        """
        budget = self.get_budget(scope_key)
        last_date = budget["last_updated_at"][:10]  # YYYY-MM-DD
        today = self._today()

        if last_date < today:
            now = self._now()
            self._db.execute(
                """
                UPDATE privacy_accounting
                SET epsilon_spent = 0.0, delta_spent = 0.0, last_updated_at = ?
                WHERE scope_key = ?
                """,
                (now, scope_key),
            )
            logger.info("Daily privacy reset for scope %s", scope_key)

    def get_all_scopes(self) -> list[dict[str, Any]]:
        """Return privacy budget records for all tracked scopes."""
        rows = self._db.execute(
            "SELECT * FROM privacy_accounting ORDER BY scope_key"
        )
        return [dict(r) for r in rows]
