"""Tests for CrashDetector boot tracking and crash loop detection."""

from __future__ import annotations

import pytest

from octomil.device_agent.crash_detector import CrashDetector
from octomil.device_agent.db.local_db import LocalDB


@pytest.fixture
def setup():
    db = LocalDB(":memory:")
    detector = CrashDetector(db)
    yield db, detector
    db.close()


class TestBootTracking:
    def test_record_boot(self, setup) -> None:
        db, detector = setup
        crash_count = detector.record_boot("boot-1", "m1", "v1", "rt-1.0")
        assert crash_count == 0

        history = detector.get_boot_history()
        assert len(history) == 1
        assert history[0]["boot_id"] == "boot-1"
        assert history[0]["active_model_id"] == "m1"
        assert history[0]["active_model_version"] == "v1"
        assert history[0]["runtime_version"] == "rt-1.0"
        assert history[0]["clean_shutdown"] == 0
        assert history[0]["crash_detected"] == 0

    def test_record_boot_optional_fields(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1")
        history = detector.get_boot_history()
        assert history[0]["active_model_id"] is None
        assert history[0]["active_model_version"] is None
        assert history[0]["runtime_version"] is None

    def test_clean_shutdown(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1", "m1", "v1")
        assert detector.record_clean_shutdown("boot-1") is True

        history = detector.get_boot_history()
        assert history[0]["clean_shutdown"] == 1
        assert history[0]["duration_sec"] is not None
        assert history[0]["duration_sec"] >= 0.0

    def test_clean_shutdown_nonexistent(self, setup) -> None:
        _, detector = setup
        assert detector.record_clean_shutdown("nonexistent") is False


class TestCrashDetection:
    def test_unclean_boot_marked_as_crash(self, setup) -> None:
        _, detector = setup
        # Boot 1 starts, does not shut down cleanly
        detector.record_boot("boot-1", "m1", "v1")

        # Boot 2 starts — should detect boot-1 as a crash
        crash_count = detector.record_boot("boot-2", "m1", "v1")
        assert crash_count == 1

        history = detector.get_boot_history(limit=10)
        boot1 = next(h for h in history if h["boot_id"] == "boot-1")
        assert boot1["crash_detected"] == 1

    def test_clean_shutdown_prevents_crash_flag(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1", "m1", "v1")
        detector.record_clean_shutdown("boot-1")

        # Boot 2 should not detect any crash
        crash_count = detector.record_boot("boot-2", "m1", "v1")
        assert crash_count == 0

    def test_multiple_unclean_boots(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1", "m1", "v1")
        detector.record_boot("boot-2", "m1", "v1")  # marks boot-1
        detector.record_boot("boot-3", "m1", "v1")  # marks boot-2

        history = detector.get_boot_history(limit=10)
        boot1 = next(h for h in history if h["boot_id"] == "boot-1")
        boot2 = next(h for h in history if h["boot_id"] == "boot-2")
        assert boot1["crash_detected"] == 1
        assert boot2["crash_detected"] == 1


class TestCrashLoopDetection:
    def _simulate_crashes(self, detector: CrashDetector, model_id: str, count: int) -> None:
        """Simulate *count* crashed boots for the given model."""
        for i in range(count):
            boot_id = f"crash-boot-{i}"
            detector.record_boot(boot_id, model_id, "v1")
        # One more boot to mark all previous as crashed
        detector.record_boot("final-boot", model_id, "v1")

    def test_below_threshold_no_crash_loop(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 2)
        # Default threshold is 3, we only have 2 crashes
        assert detector.is_crash_loop("m1") is False

    def test_at_threshold_is_crash_loop(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 3)
        assert detector.is_crash_loop("m1") is True

    def test_above_threshold_is_crash_loop(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 5)
        assert detector.is_crash_loop("m1") is True

    def test_custom_threshold(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 1)
        assert detector.is_crash_loop("m1", threshold=1) is True

    def test_different_model_not_affected(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 5)
        assert detector.is_crash_loop("m2") is False

    def test_crash_loop_with_large_window(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 3)
        # All crashes happened just now, so even a 1-second window should catch them
        assert detector.is_crash_loop("m1", window_sec=86400) is True


class TestAutoRollback:
    def _simulate_crashes(self, detector: CrashDetector, model_id: str, count: int) -> None:
        for i in range(count):
            detector.record_boot(f"crash-{i}", model_id, "v1")
        detector.record_boot("final", model_id, "v1")

    def test_should_rollback_on_crash_loop(self, setup) -> None:
        _, detector = setup
        self._simulate_crashes(detector, "m1", 3)
        should, reason = detector.should_auto_rollback("m1")
        assert should is True
        assert "crash_loop" in reason

    def test_should_not_rollback_no_crashes(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1", "m1", "v1")
        detector.record_clean_shutdown("boot-1")
        should, reason = detector.should_auto_rollback("m1")
        assert should is False
        assert reason == ""


class TestBootHistory:
    def test_history_limit(self, setup) -> None:
        _, detector = setup
        for i in range(20):
            detector.record_boot(f"boot-{i}", "m1", "v1")
            detector.record_clean_shutdown(f"boot-{i}")
        history = detector.get_boot_history(limit=5)
        assert len(history) == 5

    def test_history_ordering(self, setup) -> None:
        _, detector = setup
        detector.record_boot("boot-1", "m1", "v1")
        detector.record_clean_shutdown("boot-1")
        detector.record_boot("boot-2", "m1", "v2")
        detector.record_clean_shutdown("boot-2")

        history = detector.get_boot_history()
        # Most recent first
        assert history[0]["boot_id"] == "boot-2"
        assert history[1]["boot_id"] == "boot-1"
