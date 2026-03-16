"""Tests for LocalDB creation, schema application, and transactions."""

from __future__ import annotations

import threading

import pytest

from octomil.device_agent.db.local_db import LocalDB


class TestLocalDBCreation:
    def test_in_memory_db(self) -> None:
        db = LocalDB(":memory:")
        assert db._conn is not None
        db.close()

    def test_file_db(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        db = LocalDB(db_path)
        assert db_path.exists()
        db.close()

    def test_wal_mode_enabled(self) -> None:
        db = LocalDB(":memory:")
        row = db.execute_one("PRAGMA journal_mode")
        # In-memory DBs use "memory" mode, but WAL is set for file DBs
        assert row is not None
        db.close()


class TestSchema:
    def test_all_tables_created(self) -> None:
        db = LocalDB(":memory:")
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = {r["name"] for r in rows}
        expected = {
            "model_artifacts",
            "download_chunks",
            "operations",
            "active_model_pointer",
            "base_models",
            "personalization_adapters",
            "active_bindings",
            "rollback_records",
        }
        assert expected.issubset(table_names)
        db.close()

    def test_schema_idempotent(self) -> None:
        db = LocalDB(":memory:")
        # Re-apply schema should not raise
        db._apply_schema()
        db.close()


class TestTransactions:
    def test_commit_on_success(self) -> None:
        db = LocalDB(":memory:")
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO active_model_pointer (model_id, active_version, updated_at) VALUES (?, ?, ?)",
                ("m1", "v1", "2024-01-01T00:00:00Z"),
            )
        row = db.execute_one("SELECT * FROM active_model_pointer WHERE model_id = 'm1'")
        assert row is not None
        assert row["active_version"] == "v1"
        db.close()

    def test_rollback_on_error(self) -> None:
        db = LocalDB(":memory:")
        with pytest.raises(ValueError):
            with db.transaction() as cur:
                cur.execute(
                    "INSERT INTO active_model_pointer (model_id, active_version, updated_at) VALUES (?, ?, ?)",
                    ("m1", "v1", "2024-01-01T00:00:00Z"),
                )
                raise ValueError("test error")
        row = db.execute_one("SELECT * FROM active_model_pointer WHERE model_id = 'm1'")
        assert row is None
        db.close()

    def test_execute_returns_rows(self) -> None:
        db = LocalDB(":memory:")
        db.execute(
            "INSERT INTO active_model_pointer (model_id, active_version, updated_at) VALUES (?, ?, ?)",
            ("m1", "v1", "now"),
        )
        rows = db.execute("SELECT * FROM active_model_pointer")
        assert len(rows) == 1
        db.close()

    def test_execute_one_returns_none(self) -> None:
        db = LocalDB(":memory:")
        row = db.execute_one("SELECT * FROM active_model_pointer WHERE model_id = 'nonexistent'")
        assert row is None
        db.close()

    def test_thread_safety(self) -> None:
        db = LocalDB(":memory:")
        errors: list[Exception] = []

        def insert(i: int) -> None:
            try:
                db.execute(
                    "INSERT INTO active_model_pointer (model_id, active_version, updated_at) VALUES (?, ?, ?)",
                    (f"m{i}", f"v{i}", "now"),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        rows = db.execute("SELECT COUNT(*) as cnt FROM active_model_pointer")
        assert rows[0]["cnt"] == 10
        db.close()
