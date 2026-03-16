"""Tests for DataPipeline snapshot creation and lifecycle."""

from __future__ import annotations

import unittest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.training.data_pipeline import DataPipeline
from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS


def _make_db() -> LocalDB:
    db = LocalDB(":memory:")
    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


class TestDataPipelineSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.pipeline = DataPipeline(self.db)

    def test_select_examples_default(self) -> None:
        examples = self.pipeline.select_examples("user_1", max_count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertEqual(ex["source_scope"], "user_1")
            self.assertEqual(ex["strategy"], "recent")

    def test_select_examples_with_policy(self) -> None:
        policy = {"strategy": "diverse", "min_quality": 0.5}
        examples = self.pipeline.select_examples("user_1", max_count=5, selection_policy=policy)
        self.assertEqual(len(examples), 5)
        for ex in examples:
            self.assertEqual(ex["strategy"], "diverse")

    def test_select_examples_caps_at_100(self) -> None:
        examples = self.pipeline.select_examples("user_1", max_count=200)
        self.assertEqual(len(examples), 100)


class TestDataPipelineSnapshots(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.pipeline = DataPipeline(self.db)

    def test_create_snapshot(self) -> None:
        examples = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        record = self.pipeline.create_snapshot(
            dataset_id="ds_1",
            examples=examples,
            source_scope="user_1",
        )
        self.assertEqual(record["dataset_id"], "ds_1")
        self.assertEqual(record["source_scope"], "user_1")
        self.assertEqual(record["example_count"], 2)
        self.assertGreater(record["byte_size"], 0)
        self.assertIsNotNone(record["expires_at"])

    def test_create_snapshot_no_expiry(self) -> None:
        examples = [{"id": "1"}]
        record = self.pipeline.create_snapshot(
            dataset_id="ds_2",
            examples=examples,
            source_scope="user_1",
            expires_hours=0,
        )
        self.assertIsNone(record["expires_at"])

    def test_load_snapshot(self) -> None:
        examples = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        self.pipeline.create_snapshot("ds_1", examples, "user_1")

        loaded = self.pipeline.load_snapshot("ds_1")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["id"], "1")
        self.assertEqual(loaded[1]["text"], "world")

    def test_load_nonexistent_snapshot_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.load_snapshot("nonexistent")

    def test_get_dataset(self) -> None:
        examples = [{"id": "1"}]
        self.pipeline.create_snapshot("ds_1", examples, "user_1")
        ds = self.pipeline.get_dataset("ds_1")
        self.assertIsNotNone(ds)
        assert ds is not None
        self.assertEqual(ds["dataset_id"], "ds_1")

    def test_get_dataset_nonexistent(self) -> None:
        self.assertIsNone(self.pipeline.get_dataset("nonexistent"))


class TestDataPipelineExpiry(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.pipeline = DataPipeline(self.db)

    def test_expire_old_snapshots(self) -> None:
        examples = [{"id": "1"}]
        self.pipeline.create_snapshot("ds_1", examples, "user_1", expires_hours=24)
        self.pipeline.create_snapshot("ds_2", examples, "user_2", expires_hours=24)

        # Manually backdate the expiry
        self.db.execute(
            "UPDATE training_datasets SET expires_at = '2020-01-01T00:00:00+00:00' WHERE dataset_id = ?",
            ("ds_1",),
        )

        removed = self.pipeline.expire_old_snapshots()
        self.assertEqual(removed, 1)

        # ds_1 should be gone, ds_2 should remain
        self.assertIsNone(self.pipeline.get_dataset("ds_1"))
        self.assertIsNotNone(self.pipeline.get_dataset("ds_2"))

    def test_expire_no_expired(self) -> None:
        examples = [{"id": "1"}]
        self.pipeline.create_snapshot("ds_1", examples, "user_1", expires_hours=24)
        removed = self.pipeline.expire_old_snapshots()
        self.assertEqual(removed, 0)

    def test_expire_no_datasets(self) -> None:
        removed = self.pipeline.expire_old_snapshots()
        self.assertEqual(removed, 0)

    def test_snapshot_replace(self) -> None:
        examples_v1 = [{"id": "1"}]
        examples_v2 = [{"id": "2"}, {"id": "3"}]

        self.pipeline.create_snapshot("ds_1", examples_v1, "user_1")
        self.pipeline.create_snapshot("ds_1", examples_v2, "user_1")

        ds = self.pipeline.get_dataset("ds_1")
        assert ds is not None
        self.assertEqual(ds["example_count"], 2)

        loaded = self.pipeline.load_snapshot("ds_1")
        self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
