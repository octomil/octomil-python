"""Tests for RuntimeUpdater state machine transitions."""

from __future__ import annotations

import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.runtime_updater import RuntimeUpdater


@pytest.fixture
def setup():
    db = LocalDB(":memory:")
    updater = RuntimeUpdater(db)
    yield db, updater
    db.close()


class TestDiscover:
    def test_discover_creates_record(self, setup) -> None:
        db, updater = setup
        rid = updater.discover("1.0.0", "https://cdn.example.com/runtime-1.0.0.tar.gz")
        assert rid is not None
        rt = updater.get_runtime(rid)
        assert rt is not None
        assert rt["version"] == "1.0.0"
        assert rt["status"] == "DISCOVERED"
        assert rt["artifact_path"] == "https://cdn.example.com/runtime-1.0.0.tar.gz"

    def test_discover_multiple_versions(self, setup) -> None:
        _, updater = setup
        r1 = updater.discover("1.0.0", "url1")
        r2 = updater.discover("2.0.0", "url2")
        assert r1 != r2
        assert updater.get_runtime(r1)["version"] == "1.0.0"
        assert updater.get_runtime(r2)["version"] == "2.0.0"


class TestStateMachineTransitions:
    def test_full_happy_path(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")

        assert updater.download(rid) is True
        assert updater.get_runtime(rid)["status"] == "DOWNLOADED"
        assert updater.get_runtime(rid)["downloaded_at"] is not None

        assert updater.verify(rid) is True
        assert updater.get_runtime(rid)["status"] == "VERIFIED"
        assert updater.get_runtime(rid)["verified_at"] is not None

        assert updater.mark_pending_restart(rid) is True
        assert updater.get_runtime(rid)["status"] == "PENDING_RESTART"
        assert updater.get_runtime(rid)["pending_since"] is not None

        assert updater.activate_on_boot(rid) is True
        assert updater.get_runtime(rid)["status"] == "ACTIVE_ON_NEXT_BOOT"
        assert updater.get_runtime(rid)["activated_at"] is not None

    def test_skip_download_fails(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")
        # Try to verify without downloading
        assert updater.verify(rid) is False
        assert updater.get_runtime(rid)["status"] == "DISCOVERED"

    def test_skip_verify_fails(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")
        updater.download(rid)
        # Try to mark pending restart without verifying
        assert updater.mark_pending_restart(rid) is False
        assert updater.get_runtime(rid)["status"] == "DOWNLOADED"

    def test_skip_pending_restart_fails(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")
        updater.download(rid)
        updater.verify(rid)
        # Try to activate without marking pending restart
        assert updater.activate_on_boot(rid) is False
        assert updater.get_runtime(rid)["status"] == "VERIFIED"

    def test_backward_transition_fails(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")
        updater.download(rid)
        updater.verify(rid)
        # Try to go back to DOWNLOADED
        assert updater.download(rid) is False
        assert updater.get_runtime(rid)["status"] == "VERIFIED"

    def test_nonexistent_runtime_fails(self, setup) -> None:
        _, updater = setup
        assert updater.download("nonexistent") is False
        assert updater.verify("nonexistent") is False
        assert updater.mark_pending_restart("nonexistent") is False
        assert updater.activate_on_boot("nonexistent") is False


class TestPendingRestartFlow:
    def test_get_pending_runtime(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("2.0.0", "url")
        updater.download(rid)
        updater.verify(rid)
        updater.mark_pending_restart(rid)

        pending = updater.get_pending_runtime()
        assert pending is not None
        assert pending["runtime_id"] == rid
        assert pending["status"] == "PENDING_RESTART"
        assert pending["version"] == "2.0.0"

    def test_no_pending_runtime(self, setup) -> None:
        _, updater = setup
        assert updater.get_pending_runtime() is None

    def test_pending_to_active(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("2.0.0", "url")
        updater.download(rid)
        updater.verify(rid)
        updater.mark_pending_restart(rid)

        # On next boot, activate
        updater.activate_on_boot(rid)

        assert updater.get_pending_runtime() is None
        active = updater.get_active_runtime()
        assert active is not None
        assert active["runtime_id"] == rid
        assert active["version"] == "2.0.0"


class TestGetActiveRuntime:
    def test_no_active_runtime(self, setup) -> None:
        _, updater = setup
        assert updater.get_active_runtime() is None

    def test_active_runtime_after_full_cycle(self, setup) -> None:
        _, updater = setup
        rid = updater.discover("1.0.0", "url")
        updater.download(rid)
        updater.verify(rid)
        updater.mark_pending_restart(rid)
        updater.activate_on_boot(rid)

        active = updater.get_active_runtime()
        assert active is not None
        assert active["runtime_id"] == rid
        assert active["status"] == "ACTIVE_ON_NEXT_BOOT"

    def test_get_runtime_nonexistent(self, setup) -> None:
        _, updater = setup
        assert updater.get_runtime("nonexistent") is None
