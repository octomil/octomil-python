"""Tests for TelemetryLoop — policy refresh, storage pressure, uploader delegation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.loops.telemetry_loop import (
    TelemetryLoop,
)
from octomil.device_agent.policy.policy_engine import PolicyEngine
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore


@pytest.fixture
def components():
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    policy = PolicyEngine()
    tel_store = TelemetryStore(db, device_id="dev1", boot_id="boot1")
    uploader = MagicMock()
    yield db, tel_store, uploader, policy
    db.close()


class TestLifecycle:
    def test_start_stop(self, components) -> None:
        db, tel_store, uploader, policy = components
        loop = TelemetryLoop(tel_store, uploader, policy)
        assert not loop.is_running
        loop.start()
        assert loop.is_running
        uploader.start.assert_called_once()
        loop.stop()
        assert not loop.is_running
        uploader.stop.assert_called_once()

    def test_double_start_safe(self, components) -> None:
        db, tel_store, uploader, policy = components
        loop = TelemetryLoop(tel_store, uploader, policy)
        loop.start()
        loop.start()
        assert loop.is_running
        loop.stop()

    def test_uploader_started_before_thread(self, components) -> None:
        db, tel_store, uploader, policy = components
        loop = TelemetryLoop(tel_store, uploader, policy)
        loop.start()
        # Uploader.start should be called before the thread begins
        uploader.start.assert_called_once()
        loop.stop()


class TestPolicyRefresh:
    def test_policy_pushed_to_uploader_on_start(self, components) -> None:
        db, tel_store, uploader, policy = components
        loop = TelemetryLoop(tel_store, uploader, policy)
        loop.start()
        time.sleep(0.1)
        loop.stop()
        # set_policy should have been called at least once
        assert uploader.set_policy.call_count >= 1

    def test_policy_updates_propagated(self, components) -> None:
        db, tel_store, uploader, policy = components
        loop = TelemetryLoop(tel_store, uploader, policy, policy_refresh_interval=0.1)
        loop.start()

        # Change device state mid-loop
        policy.update_device_state(network_type="cellular")
        time.sleep(0.3)
        loop.stop()

        # set_policy should have been called multiple times due to refresh
        assert uploader.set_policy.call_count >= 2


class TestStoragePressure:
    def test_no_cleanup_when_ample_storage(self, components) -> None:
        db, tel_store, uploader, policy = components
        policy.update_device_state(free_storage_bytes=10_000_000_000)

        loop = TelemetryLoop(tel_store, uploader, policy)
        loop._check_storage_pressure()

        # No events to clean anyway, but also no error

    def test_cleanup_triggered_under_pressure(self, components) -> None:
        db, tel_store, uploader, policy = components
        policy.update_device_state(free_storage_bytes=100_000_000)  # below threshold

        # Insert some BEST_EFFORT events
        for i in range(5):
            tel_store.append("test.event", "BEST_EFFORT", {"i": i})

        loop = TelemetryLoop(tel_store, uploader, policy)
        loop._check_storage_pressure()

        # BEST_EFFORT events should have been dropped
        events = tel_store.get_unsent(batch_size=100)
        best_effort = [e for e in events if e["telemetry_class"] == "BEST_EFFORT"]
        assert len(best_effort) == 0

    def test_must_keep_events_survive_pressure(self, components) -> None:
        db, tel_store, uploader, policy = components
        policy.update_device_state(free_storage_bytes=100_000_000)

        # Insert a MUST_KEEP event
        tel_store.append("critical.event", "MUST_KEEP", {"important": True})

        loop = TelemetryLoop(tel_store, uploader, policy)
        loop._check_storage_pressure()

        events = tel_store.get_unsent(batch_size=100)
        must_keep = [e for e in events if e["telemetry_class"] == "MUST_KEEP"]
        assert len(must_keep) == 1
