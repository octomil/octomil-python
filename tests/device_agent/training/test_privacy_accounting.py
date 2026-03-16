"""Tests for PrivacyAccountant budget tracking."""

from __future__ import annotations

import unittest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS
from octomil.device_agent.training.privacy_accounting import PrivacyAccountant


def _make_db() -> LocalDB:
    db = LocalDB(":memory:")
    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


class TestPrivacyAccountant(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.accountant = PrivacyAccountant(self.db)

    def test_get_budget_empty(self) -> None:
        budget = self.accountant.get_budget("scope_1")
        self.assertEqual(budget["epsilon_spent"], 0.0)
        self.assertEqual(budget["delta_spent"], 0.0)
        self.assertEqual(budget["scope_key"], "scope_1")

    def test_spend_creates_record(self) -> None:
        result = self.accountant.spend("scope_1", 0.5, 1e-5)
        self.assertAlmostEqual(result["epsilon_spent"], 0.5)
        self.assertAlmostEqual(result["delta_spent"], 1e-5)

    def test_spend_accumulates(self) -> None:
        self.accountant.spend("scope_1", 0.5, 1e-5)
        self.accountant.spend("scope_1", 0.3, 2e-5)
        budget = self.accountant.get_budget("scope_1")
        self.assertAlmostEqual(budget["epsilon_spent"], 0.8)
        self.assertAlmostEqual(budget["delta_spent"], 3e-5)

    def test_spend_multiple_scopes_independent(self) -> None:
        self.accountant.spend("scope_1", 1.0, 1e-5)
        self.accountant.spend("scope_2", 0.5, 2e-5)

        b1 = self.accountant.get_budget("scope_1")
        b2 = self.accountant.get_budget("scope_2")
        self.assertAlmostEqual(b1["epsilon_spent"], 1.0)
        self.assertAlmostEqual(b2["epsilon_spent"], 0.5)

    def test_has_budget_sufficient(self) -> None:
        self.accountant.spend("scope_1", 0.5, 1e-5)
        self.assertTrue(self.accountant.has_budget("scope_1", epsilon_needed=0.3, daily_budget_eps=1.0))

    def test_has_budget_insufficient(self) -> None:
        self.accountant.spend("scope_1", 0.9, 1e-5)
        self.assertFalse(self.accountant.has_budget("scope_1", epsilon_needed=0.3, daily_budget_eps=1.0))

    def test_has_budget_exact(self) -> None:
        self.accountant.spend("scope_1", 0.7, 1e-5)
        self.assertTrue(self.accountant.has_budget("scope_1", epsilon_needed=0.3, daily_budget_eps=1.0))

    def test_has_budget_no_prior_spend(self) -> None:
        self.assertTrue(self.accountant.has_budget("new_scope", epsilon_needed=0.5, daily_budget_eps=1.0))

    def test_reset_daily(self) -> None:
        self.accountant.spend("scope_1", 1.0, 1e-5)

        # Manually set last_updated_at to yesterday
        self.db.execute(
            "UPDATE privacy_accounting SET last_updated_at = '2020-01-01T00:00:00+00:00' WHERE scope_key = ?",
            ("scope_1",),
        )

        self.accountant.reset_daily("scope_1")
        budget = self.accountant.get_budget("scope_1")
        self.assertAlmostEqual(budget["epsilon_spent"], 0.0)
        self.assertAlmostEqual(budget["delta_spent"], 0.0)

    def test_reset_daily_same_day_no_reset(self) -> None:
        self.accountant.spend("scope_1", 1.0, 1e-5)
        self.accountant.reset_daily("scope_1")
        # Should NOT reset since last_updated_at is today
        budget = self.accountant.get_budget("scope_1")
        self.assertAlmostEqual(budget["epsilon_spent"], 1.0)

    def test_get_all_scopes(self) -> None:
        self.accountant.spend("alpha", 0.1, 1e-6)
        self.accountant.spend("beta", 0.2, 2e-6)
        self.accountant.spend("gamma", 0.3, 3e-6)

        scopes = self.accountant.get_all_scopes()
        self.assertEqual(len(scopes), 3)
        keys = [s["scope_key"] for s in scopes]
        self.assertEqual(keys, ["alpha", "beta", "gamma"])

    def test_get_all_scopes_empty(self) -> None:
        scopes = self.accountant.get_all_scopes()
        self.assertEqual(scopes, [])


if __name__ == "__main__":
    unittest.main()
