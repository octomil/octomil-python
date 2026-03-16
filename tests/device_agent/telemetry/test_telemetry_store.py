"""Tests for TelemetryStore — append, ordering, cursors, cleanup."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.events import TelemetryClass
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore


def _make_db() -> LocalDB:
    """Create an in-memory LocalDB with telemetry schema applied."""
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


def _make_store(db: LocalDB | None = None) -> TelemetryStore:
    db = db or _make_db()
    return TelemetryStore(db, device_id="dev-1", boot_id="boot-1")


class TestAppend:
    def test_append_returns_event_id(self) -> None:
        store = _make_store()
        eid = store.append("test.event", TelemetryClass.IMPORTANT, {"k": "v"})
        assert isinstance(eid, str)
        assert len(eid) == 32  # uuid4 hex

    def test_append_increments_sequence(self) -> None:
        store = _make_store()
        store.append("e1", TelemetryClass.IMPORTANT, {})
        store.append("e2", TelemetryClass.IMPORTANT, {})
        events = store.get_unsent(batch_size=10)
        seqs = [e["sequence_no"] for e in events]
        assert seqs == [1, 2]

    def test_append_auto_selects_class(self) -> None:
        store = _make_store()
        store.append_auto("training.job.created", {"job_id": "j1"})
        events = store.get_unsent()
        assert events[0]["telemetry_class"] == "MUST_KEEP"

    def test_append_with_model_info(self) -> None:
        store = _make_store()
        store.append(
            "serving.request.completed",
            TelemetryClass.IMPORTANT,
            {"latency_ms": 42},
            model_id="model-a",
            model_version="v1",
            session_id="sess-1",
        )
        events = store.get_unsent()
        assert events[0]["model_id"] == "model-a"
        assert events[0]["model_version"] == "v1"
        assert events[0]["session_id"] == "sess-1"


class TestGetUnsent:
    def test_returns_empty_when_no_events(self) -> None:
        store = _make_store()
        assert store.get_unsent() == []

    def test_orders_by_class_priority_then_sequence(self) -> None:
        store = _make_store()
        # Insert in reverse priority order
        store.append("e1", TelemetryClass.BEST_EFFORT, {"n": 1})
        store.append("e2", TelemetryClass.IMPORTANT, {"n": 2})
        store.append("e3", TelemetryClass.MUST_KEEP, {"n": 3})
        store.append("e4", TelemetryClass.MUST_KEEP, {"n": 4})

        events = store.get_unsent(batch_size=10)
        classes = [e["telemetry_class"] for e in events]
        assert classes == ["MUST_KEEP", "MUST_KEEP", "IMPORTANT", "BEST_EFFORT"]

    def test_respects_batch_size(self) -> None:
        store = _make_store()
        for i in range(10):
            store.append(f"e{i}", TelemetryClass.IMPORTANT, {})
        events = store.get_unsent(batch_size=3)
        assert len(events) == 3

    def test_min_class_filters_lower_priority(self) -> None:
        store = _make_store()
        store.append("e1", TelemetryClass.BEST_EFFORT, {})
        store.append("e2", TelemetryClass.IMPORTANT, {})
        store.append("e3", TelemetryClass.MUST_KEEP, {})

        events = store.get_unsent(min_class=TelemetryClass.IMPORTANT)
        classes = {e["telemetry_class"] for e in events}
        assert "BEST_EFFORT" not in classes
        assert "MUST_KEEP" in classes
        assert "IMPORTANT" in classes

    def test_excludes_already_sent(self) -> None:
        store = _make_store()
        eid = store.append("e1", TelemetryClass.IMPORTANT, {})
        store.mark_sent([eid], "batch-1")
        assert store.get_unsent() == []


class TestMarkSent:
    def test_marks_events_uploaded(self) -> None:
        store = _make_store()
        eid1 = store.append("e1", TelemetryClass.IMPORTANT, {})
        eid2 = store.append("e2", TelemetryClass.IMPORTANT, {})
        store.mark_sent([eid1, eid2], "batch-x")
        assert store.get_unsent() == []

    def test_no_op_on_empty_list(self) -> None:
        store = _make_store()
        store.mark_sent([], "batch-x")  # should not raise


class TestCursors:
    def test_get_cursor_returns_zero_initially(self) -> None:
        store = _make_store()
        assert store.get_cursor("boot-1") == 0

    def test_mark_acked_updates_cursor(self) -> None:
        store = _make_store()
        store.mark_acked("boot-1", 5)
        assert store.get_cursor("boot-1") == 5

    def test_mark_acked_never_goes_backwards(self) -> None:
        store = _make_store()
        store.mark_acked("boot-1", 10)
        store.mark_acked("boot-1", 5)  # lower — should be ignored
        assert store.get_cursor("boot-1") == 10


class TestCleanup:
    def test_drop_best_effort_removes_old_events(self) -> None:
        db = _make_db()
        store = TelemetryStore(db, device_id="dev-1", boot_id="boot-1")

        # Insert a BEST_EFFORT event with an old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        db.execute(
            """
            INSERT INTO telemetry_events
                (event_id, device_id, boot_id, sequence_no, event_type,
                 telemetry_class, occurred_at, uploaded)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            ("old-1", "dev-1", "boot-1", 999, "test.old", "BEST_EFFORT", old_ts),
        )
        # Insert a recent one
        store.append("test.new", TelemetryClass.BEST_EFFORT, {})

        dropped = store.drop_best_effort(older_than_hours=24)
        assert dropped == 1

        remaining = store.get_unsent()
        assert len(remaining) == 1
        assert remaining[0]["event_type"] == "test.new"

    def test_drop_important_removes_old_events(self) -> None:
        db = _make_db()
        store = TelemetryStore(db, device_id="dev-1", boot_id="boot-1")

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
        db.execute(
            """
            INSERT INTO telemetry_events
                (event_id, device_id, boot_id, sequence_no, event_type,
                 telemetry_class, occurred_at, uploaded)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            ("old-imp", "dev-1", "boot-1", 999, "test.old", "IMPORTANT", old_ts),
        )
        dropped = store.drop_important(older_than_hours=168)
        assert dropped == 1

    def test_storage_pressure_cleanup_drops_best_effort_first(self) -> None:
        store = _make_store()
        for i in range(5):
            store.append(f"be{i}", TelemetryClass.BEST_EFFORT, {})
        for i in range(3):
            store.append(f"imp{i}", TelemetryClass.IMPORTANT, {})
        store.append("mk1", TelemetryClass.MUST_KEEP, {})

        store.storage_pressure_cleanup(target_bytes=1024)
        remaining = store.get_unsent()

        # MUST_KEEP should survive
        assert any(e["telemetry_class"] == "MUST_KEEP" for e in remaining)

    def test_storage_pressure_drops_important_if_needed(self) -> None:
        store = _make_store()
        store.append("be1", TelemetryClass.BEST_EFFORT, {})
        store.append("imp1", TelemetryClass.IMPORTANT, {})
        store.append("mk1", TelemetryClass.MUST_KEEP, {})

        # Request more bytes than BEST_EFFORT alone frees
        dropped = store.storage_pressure_cleanup(target_bytes=999_999_999)
        assert dropped >= 2

        remaining = store.get_unsent()
        classes = {e["telemetry_class"] for e in remaining}
        assert classes == {"MUST_KEEP"}


class TestCountByClass:
    def test_counts_all_classes(self) -> None:
        store = _make_store()
        store.append("e1", TelemetryClass.MUST_KEEP, {})
        store.append("e2", TelemetryClass.MUST_KEEP, {})
        store.append("e3", TelemetryClass.IMPORTANT, {})
        store.append("e4", TelemetryClass.BEST_EFFORT, {})

        counts = store.count_by_class()
        assert counts["MUST_KEEP"] == 2
        assert counts["IMPORTANT"] == 1
        assert counts["BEST_EFFORT"] == 1


class TestSequenceRecovery:
    def test_recovers_sequence_on_new_store(self) -> None:
        db = _make_db()
        store1 = TelemetryStore(db, device_id="dev-1", boot_id="boot-1")
        store1.append("e1", TelemetryClass.IMPORTANT, {})
        store1.append("e2", TelemetryClass.IMPORTANT, {})

        # Simulate restart
        store2 = TelemetryStore(db, device_id="dev-1", boot_id="boot-1")
        store2.append("e3", TelemetryClass.IMPORTANT, {})

        events = store2.get_unsent()
        seqs = sorted(e["sequence_no"] for e in events)
        assert seqs == [1, 2, 3]


class TestThreadSafety:
    def test_concurrent_appends(self) -> None:
        store = _make_store()
        errors: list[Exception] = []

        def _append(n: int) -> None:
            try:
                for i in range(20):
                    store.append(f"t{n}.e{i}", TelemetryClass.IMPORTANT, {"n": n, "i": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_append, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        events = store.get_unsent(batch_size=200)
        assert len(events) == 80
        # All sequence numbers should be unique
        seqs = [e["sequence_no"] for e in events]
        assert len(set(seqs)) == 80
