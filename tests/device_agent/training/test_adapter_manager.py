"""Tests for AdapterManager CRUD and binding activation."""

from __future__ import annotations

import unittest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.training.adapter_manager import (
    AdapterManager,
)
from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS


def _make_db() -> LocalDB:
    db = LocalDB(":memory:")
    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


class TestAdapterManagerCRUD(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.mgr = AdapterManager(self.db)

    def test_register_adapter(self) -> None:
        record = self.mgr.register_adapter(
            adapter_id="a1",
            base_model_id="m1",
            base_version="1.0",
            scope="user_1",
            version="v1",
            artifact_path="/adapters/a1",
        )
        self.assertEqual(record["adapter_id"], "a1")
        self.assertEqual(record["status"], "REGISTERED")
        self.assertEqual(record["base_model_id"], "m1")
        self.assertEqual(record["adapter_scope"], "user_1")

    def test_register_adapter_with_parent(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        record = self.mgr.register_adapter(
            adapter_id="a2",
            base_model_id="m1",
            base_version="1.0",
            scope="user_1",
            version="v2",
            artifact_path="/adapters/a2",
            parent_version="v1",
        )
        self.assertEqual(record["parent_adapter_version"], "v1")

    def test_list_adapters_all(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m2", "1.0", "user_2", "v1", "/a2")
        adapters = self.mgr.list_adapters()
        self.assertEqual(len(adapters), 2)

    def test_list_adapters_by_model(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m2", "1.0", "user_2", "v1", "/a2")
        adapters = self.mgr.list_adapters(base_model_id="m1")
        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0]["adapter_id"], "a1")

    def test_list_adapters_by_scope(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m1", "1.0", "user_2", "v1", "/a2")
        adapters = self.mgr.list_adapters(scope="user_1")
        self.assertEqual(len(adapters), 1)


class TestAdapterManagerBindings(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.mgr = AdapterManager(self.db)

    def test_get_binding_nonexistent_returns_none(self) -> None:
        self.assertIsNone(self.mgr.get_binding("nonexistent"))

    def test_activate_adapter_creates_binding(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.activate_adapter("m1::user_1", "a1", "v1")

        binding = self.mgr.get_binding("m1::user_1")
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding.adapter_id, "a1")
        self.assertEqual(binding.adapter_version, "v1")
        self.assertEqual(binding.base_model_id, "m1")

    def test_activate_adapter_updates_existing_binding(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m1", "1.0", "user_1", "v2", "/a2")
        self.mgr.activate_adapter("m1::user_1", "a1", "v1")
        self.mgr.activate_adapter("m1::user_1", "a2", "v2")

        binding = self.mgr.get_binding("m1::user_1")
        assert binding is not None
        self.assertEqual(binding.adapter_id, "a2")
        self.assertEqual(binding.adapter_version, "v2")

    def test_activate_nonexistent_adapter_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.mgr.activate_adapter("b1", "nonexistent", "v1")

    def test_get_active_adapter(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.activate_adapter("m1::user_1", "a1", "v1")
        adapter = self.mgr.get_active_adapter("m1::user_1")
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertEqual(adapter["adapter_id"], "a1")
        self.assertEqual(adapter["status"], "ACTIVE")

    def test_get_active_adapter_no_binding(self) -> None:
        self.assertIsNone(self.mgr.get_active_adapter("nonexistent"))


class TestAdapterManagerShadowAndRollback(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.mgr = AdapterManager(self.db)

    def test_shadow_compare(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m1", "1.0", "user_1", "v2", "/a2")
        self.mgr.activate_adapter("m1::user_1", "a1", "v1")

        result = self.mgr.shadow_compare("m1::user_1", "a2", ["prompt1", "prompt2"])
        self.assertEqual(result["current_adapter_id"], "a1")
        self.assertEqual(result["candidate_adapter_id"], "a2")
        self.assertEqual(result["prompts_count"], 2)

    def test_rollback_no_parent_clears_binding(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.activate_adapter("m1::user_1", "a1", "v1")

        result = self.mgr.rollback_adapter("m1::user_1", "quality degraded")
        self.assertIsNone(result)

        binding = self.mgr.get_binding("m1::user_1")
        assert binding is not None
        self.assertIsNone(binding.adapter_id)

    def test_rollback_with_parent(self) -> None:
        self.mgr.register_adapter("a1", "m1", "1.0", "user_1", "v1", "/a1")
        self.mgr.register_adapter("a2", "m1", "1.0", "user_1", "v2", "/a2", parent_version="v1")
        self.mgr.activate_adapter("m1::user_1", "a2", "v2")

        result = self.mgr.rollback_adapter("m1::user_1", "bad quality")
        self.assertEqual(result, "a1")

        binding = self.mgr.get_binding("m1::user_1")
        assert binding is not None
        self.assertEqual(binding.adapter_id, "a1")
        self.assertEqual(binding.adapter_version, "v1")

    def test_rollback_nonexistent_binding(self) -> None:
        result = self.mgr.rollback_adapter("nonexistent", "test")
        self.assertIsNone(result)


class TestAdapterManagerGC(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.mgr = AdapterManager(self.db)

    def test_gc_removes_old_versions(self) -> None:
        for i in range(5):
            self.mgr.register_adapter(
                f"a{i}",
                "m1",
                "1.0",
                "user_1",
                f"v{i}",
                f"/a{i}",
            )

        removed = self.mgr.gc_old_adapters(keep_versions=2)
        self.assertEqual(removed, 3)
        remaining = self.mgr.list_adapters()
        self.assertEqual(len(remaining), 2)

    def test_gc_preserves_active(self) -> None:
        for i in range(5):
            self.mgr.register_adapter(
                f"a{i}",
                "m1",
                "1.0",
                "user_1",
                f"v{i}",
                f"/a{i}",
            )
        self.mgr.activate_adapter("m1::user_1", "a0", "v0")

        removed = self.mgr.gc_old_adapters(keep_versions=2)
        # a0 is ACTIVE so not counted in the non-active pool
        # Non-active: a1, a2, a3, a4 -> keep 2, remove 2
        self.assertEqual(removed, 2)
        # Should still have a0 (active) + 2 kept = 3
        remaining = self.mgr.list_adapters()
        self.assertEqual(len(remaining), 3)


if __name__ == "__main__":
    unittest.main()
