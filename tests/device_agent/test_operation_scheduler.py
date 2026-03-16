"""Tests for OperationScheduler lease management."""

from __future__ import annotations

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.operation_scheduler import (
    PRIORITY_BACKGROUND_BEST_EFFORT,
    PRIORITY_BACKGROUND_IMPORTANT,
    PRIORITY_CRITICAL_FOREGROUND,
    OperationScheduler,
)


@pytest.fixture
def scheduler():
    db = LocalDB(":memory:")
    sched = OperationScheduler(db)
    yield sched
    db.close()


class TestSchedule:
    def test_schedule_returns_op_id(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "model-1")
        assert op_id
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["op_type"] == "download"
        assert op["state"] == "PENDING"

    def test_schedule_with_payload(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "model-1", payload={"url": "https://example.com"})
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["payload_json"] is not None

    def test_idempotency_key(self, scheduler: OperationScheduler) -> None:
        op1 = scheduler.schedule("download", "m1", idempotency_key="key1")
        op2 = scheduler.schedule("download", "m1", idempotency_key="key1")
        assert op1 == op2


class TestLease:
    def test_lease_success(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        assert scheduler.lease(op_id, "worker-1") is True
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["state"] == "LEASED"
        assert op["lease_owner"] == "worker-1"

    def test_lease_already_held(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1", duration_sec=3600)
        assert scheduler.lease(op_id, "worker-2") is False

    def test_lease_nonexistent(self, scheduler: OperationScheduler) -> None:
        assert scheduler.lease("nonexistent", "worker-1") is False

    def test_lease_completed_op(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1")
        scheduler.complete(op_id)
        assert scheduler.lease(op_id, "worker-2") is False


class TestRenewLease:
    def test_renew_success(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1")
        assert scheduler.renew_lease(op_id, "worker-1") is True

    def test_renew_wrong_owner(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1")
        assert scheduler.renew_lease(op_id, "worker-2") is False


class TestComplete:
    def test_complete(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1")
        scheduler.complete(op_id)
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["state"] == "SUCCESS"
        assert op["lease_owner"] is None


class TestFail:
    def test_fail_schedules_retry(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        scheduler.lease(op_id, "worker-1")
        scheduler.fail(op_id, "network error")
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["state"] == "FAILED"
        assert op["next_retry_at"] is not None


class TestRecoverExpiredLeases:
    def test_recover_expired(self, scheduler: OperationScheduler) -> None:
        op_id = scheduler.schedule("download", "m1")
        # Lease with very short duration so it expires
        scheduler.lease(op_id, "worker-1", duration_sec=0)
        # Force the lease_expires_at to the past
        scheduler._db.execute(
            "UPDATE operations SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE op_id = ?",
            (op_id,),
        )
        reclaimed = scheduler.recover_expired_leases()
        assert op_id in reclaimed
        op = scheduler.get_operation(op_id)
        assert op is not None
        assert op["state"] == "PENDING"
        assert op["lease_owner"] is None


class TestGetPending:
    def test_get_pending(self, scheduler: OperationScheduler) -> None:
        scheduler.schedule("download", "m1", priority=PRIORITY_BACKGROUND_BEST_EFFORT)
        scheduler.schedule("verify", "m2", priority=PRIORITY_CRITICAL_FOREGROUND)
        pending = scheduler.get_pending()
        assert len(pending) == 2
        # Should be ordered by priority
        assert pending[0]["priority"] <= pending[1]["priority"]

    def test_filter_by_type(self, scheduler: OperationScheduler) -> None:
        scheduler.schedule("download", "m1")
        scheduler.schedule("verify", "m2")
        pending = scheduler.get_pending(op_type="download")
        assert len(pending) == 1
        assert pending[0]["op_type"] == "download"

    def test_filter_by_priority(self, scheduler: OperationScheduler) -> None:
        scheduler.schedule("a", "m1", priority=PRIORITY_CRITICAL_FOREGROUND)
        scheduler.schedule("b", "m2", priority=PRIORITY_BACKGROUND_BEST_EFFORT)
        pending = scheduler.get_pending(priority=PRIORITY_BACKGROUND_IMPORTANT)
        assert len(pending) == 1
        assert pending[0]["op_type"] == "a"
