"""Tests for BandwidthBudget — token bucket consumption and refill."""

from __future__ import annotations

from unittest.mock import patch

from octomil.device_agent.policy.bandwidth_budget import (
    BandwidthBudget,
    foreground_download_budget,
    foreground_upload_budget,
    idle_download_budget,
    idle_upload_budget,
)


class TestConsume:
    def test_consume_succeeds_when_budget_available(self) -> None:
        bucket = BandwidthBudget(max_rate=1000, refill_rate=100, name="test")
        assert bucket.consume(500) is True
        assert bucket.available() == 500

    def test_consume_fails_when_over_budget(self) -> None:
        bucket = BandwidthBudget(max_rate=1000, refill_rate=100, name="test")
        assert bucket.consume(1500) is False

    def test_consume_drains_to_zero(self) -> None:
        bucket = BandwidthBudget(max_rate=100, refill_rate=100, name="test")
        bucket.consume(100)
        assert bucket.available() == 0
        assert bucket.consume(1) is False

    def test_multiple_consumes(self) -> None:
        bucket = BandwidthBudget(max_rate=100, refill_rate=100, name="test")
        assert bucket.consume(30) is True
        assert bucket.consume(30) is True
        assert bucket.consume(30) is True
        assert bucket.consume(30) is False  # only 10 left


class TestRefill:
    def test_refill_adds_tokens_over_time(self) -> None:
        t = [0.0]

        def _mock_monotonic() -> float:
            return t[0]

        with patch("octomil.device_agent.policy.bandwidth_budget.time") as mock_time:
            mock_time.monotonic = _mock_monotonic
            bucket = BandwidthBudget(max_rate=1000, refill_rate=500, name="test")
            bucket.consume(1000)  # drain fully

            # Advance 1 second
            t[0] = 1.0
            available = bucket.available()

        assert available == 500

    def test_refill_caps_at_max(self) -> None:
        bucket = BandwidthBudget(max_rate=100, refill_rate=1000, name="test")
        # Even with high refill rate, tokens cap at max_rate
        assert bucket.available() <= 100


class TestReset:
    def test_reset_refills_to_max(self) -> None:
        bucket = BandwidthBudget(max_rate=1000, refill_rate=100, name="test")
        bucket.consume(999)
        bucket.reset()
        assert bucket.available() == 1000


class TestProperties:
    def test_name_property(self) -> None:
        bucket = BandwidthBudget(max_rate=100, refill_rate=100, name="my-bucket")
        assert bucket.name == "my-bucket"

    def test_max_rate_property(self) -> None:
        bucket = BandwidthBudget(max_rate=42, refill_rate=10, name="test")
        assert bucket.max_rate == 42


class TestFactoryFunctions:
    def test_foreground_download_budget(self) -> None:
        budget = foreground_download_budget()
        assert budget.name == "fg-download"
        assert budget.max_rate == 1_048_576  # 1 MB

    def test_foreground_upload_budget(self) -> None:
        budget = foreground_upload_budget()
        assert budget.name == "fg-upload"
        assert budget.max_rate == 65_536  # 64 KB

    def test_idle_download_budget(self) -> None:
        budget = idle_download_budget()
        assert budget.name == "idle-download"
        assert budget.max_rate == 1_073_741_824  # 1 GB (unlimited)

    def test_idle_upload_budget(self) -> None:
        budget = idle_upload_budget()
        assert budget.name == "idle-upload"
        assert budget.max_rate == 524_288  # 512 KB
