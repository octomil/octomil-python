"""Tests for client-side training resilience: battery, network, gradient cache, train_if_eligible."""

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from octomil.device_info import get_battery_level, is_charging


class TestBatteryEligibility(unittest.TestCase):
    """Battery-level and charging checks for training eligibility."""

    def test_battery_above_threshold_eligible(self):
        """Device with battery > threshold should be eligible."""
        from octomil.resilience import check_training_eligibility

        result = check_training_eligibility(battery_level=80, min_battery=15, charging=False)
        self.assertTrue(result.eligible)
        self.assertIsNone(result.reason)

    def test_battery_below_threshold_ineligible(self):
        """Device with battery < threshold should be ineligible."""
        from octomil.resilience import check_training_eligibility

        result = check_training_eligibility(battery_level=10, min_battery=15, charging=False)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "low_battery")

    def test_charging_overrides_battery(self):
        """Charging device should be eligible even with low battery."""
        from octomil.resilience import check_training_eligibility

        result = check_training_eligibility(battery_level=5, min_battery=15, charging=True)
        self.assertTrue(result.eligible)
        self.assertIsNone(result.reason)

    def test_no_battery_sensor_eligible(self):
        """Desktop without battery sensor (battery_level=None) should be eligible."""
        from octomil.resilience import check_training_eligibility

        result = check_training_eligibility(battery_level=None, min_battery=15, charging=False)
        self.assertTrue(result.eligible)
        self.assertIsNone(result.reason)


class TestNetworkCheck(unittest.TestCase):
    """Network reachability checks."""

    @patch("octomil.resilience.httpx")
    def test_network_check_reachable(self, mock_httpx):
        """When the API is reachable, network quality should be suitable."""
        from octomil.resilience import check_network_quality

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.get.return_value = mock_response

        quality = check_network_quality("https://api.octomil.com")
        self.assertTrue(quality.reachable)

    @patch("octomil.resilience.httpx")
    def test_network_check_unreachable(self, mock_httpx):
        """When the API is unreachable, network quality should not be suitable."""
        from octomil.resilience import check_network_quality

        mock_httpx.get.side_effect = Exception("Connection refused")

        quality = check_network_quality("https://api.octomil.com")
        self.assertFalse(quality.reachable)

    @patch("octomil.resilience.httpx")
    def test_network_check_timeout(self, mock_httpx):
        """Timeout should result in unreachable."""
        from octomil.resilience import check_network_quality

        import httpx
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.get.side_effect = httpx.TimeoutException("timed out")

        quality = check_network_quality("https://api.octomil.com")
        self.assertFalse(quality.reachable)


class TestGradientCache(unittest.TestCase):
    """SQLite-based gradient caching."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "gradients.db")

    def tearDown(self):
        try:
            os.remove(self._db_path)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def test_gradient_cache_store_retrieve(self):
        """Store and retrieve a gradient entry."""
        from octomil.gradient_cache import GradientCache

        cache = GradientCache(self._db_path)
        cache.store("round-1", "device-1", b"\x01\x02\x03", sample_count=100)

        entry = cache.get("round-1", "device-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["round_id"], "round-1")
        self.assertEqual(entry["device_id"], "device-1")
        self.assertEqual(entry["weights_data"], b"\x01\x02\x03")
        self.assertEqual(entry["sample_count"], 100)
        self.assertFalse(entry["submitted"])

    def test_gradient_cache_list_pending(self):
        """List pending (unsubmitted) gradient entries."""
        from octomil.gradient_cache import GradientCache

        cache = GradientCache(self._db_path)
        cache.store("round-1", "device-1", b"\x01", sample_count=10)
        cache.store("round-2", "device-1", b"\x02", sample_count=20)
        cache.store("round-3", "device-1", b"\x03", sample_count=30)

        # Mark one as submitted
        cache.mark_submitted("round-1", "device-1")

        pending = cache.list_pending("device-1")
        self.assertEqual(len(pending), 2)
        round_ids = [p["round_id"] for p in pending]
        self.assertIn("round-2", round_ids)
        self.assertIn("round-3", round_ids)

    def test_gradient_cache_mark_submitted(self):
        """Marking a gradient as submitted should update its state."""
        from octomil.gradient_cache import GradientCache

        cache = GradientCache(self._db_path)
        cache.store("round-1", "device-1", b"\x01", sample_count=10)

        cache.mark_submitted("round-1", "device-1")
        entry = cache.get("round-1", "device-1")
        self.assertTrue(entry["submitted"])

    def test_gradient_cache_purge_old(self):
        """Purge old entries beyond a retention limit."""
        from octomil.gradient_cache import GradientCache

        cache = GradientCache(self._db_path)
        # Insert entries with artificially old timestamps
        conn = sqlite3.connect(self._db_path)
        old_time = time.time() - 7200  # 2 hours ago
        conn.execute(
            "INSERT INTO gradient_cache (round_id, device_id, weights_data, sample_count, submitted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("old-round", "device-1", b"\x00", 5, 0, old_time),
        )
        conn.commit()
        conn.close()

        # Add a recent entry through the normal API
        cache.store("new-round", "device-1", b"\x01", sample_count=10)

        # Purge entries older than 1 hour
        purged = cache.purge_older_than(3600)
        self.assertEqual(purged, 1)

        # Old entry gone, new entry still there
        self.assertIsNone(cache.get("old-round", "device-1"))
        self.assertIsNotNone(cache.get("new-round", "device-1"))


class TestTrainIfEligible(unittest.TestCase):
    """FederatedClient.train_if_eligible integration."""

    def _make_client(self, **kwargs):
        """Create a FederatedClient with a stubbed API."""
        from octomil.federated_client import FederatedClient

        client = FederatedClient(
            auth_token_provider=lambda: "test-token",
            org_id="test-org",
            api_base="https://api.octomil.com/api/v1",
        )
        client.api = MagicMock()
        client.device_id = "device-123"
        return client

    @patch("octomil.federated_client.get_battery_level", return_value=5)
    @patch("octomil.federated_client.is_charging", return_value=False)
    def test_train_if_eligible_skips_low_battery(self, mock_charging, mock_battery):
        """train_if_eligible should skip training when battery is too low."""
        client = self._make_client()

        train_fn = MagicMock()
        result = client.train_if_eligible(
            round_id="round-1",
            local_train_fn=train_fn,
            min_battery=15,
        )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "low_battery")
        train_fn.assert_not_called()

    @patch("octomil.federated_client.get_battery_level", return_value=50)
    @patch("octomil.federated_client.is_charging", return_value=False)
    @patch("octomil.federated_client.check_network_quality")
    def test_train_if_eligible_caches_on_upload_failure(
        self, mock_network, mock_charging, mock_battery
    ):
        """When upload fails, train_if_eligible should cache the gradient locally."""
        from octomil.gradient_cache import GradientCache

        client = self._make_client()

        # Make join_round raise to simulate upload failure
        client.join_round = MagicMock(
            side_effect=Exception("upload failed")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GradientCache(os.path.join(tmpdir, "cache.db"))
            mock_network.return_value = MagicMock(reachable=True)

            result = client.train_if_eligible(
                round_id="round-1",
                local_train_fn=lambda state: (state, 10, {"loss": 0.5}),
                gradient_cache=cache,
            )

            self.assertTrue(result["skipped"])
            self.assertEqual(result["reason"], "upload_failed")

            # Gradient should be cached
            pending = cache.list_pending("device-123")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["round_id"], "round-1")


class TestIsCharging(unittest.TestCase):
    """Tests for the is_charging device_info function."""

    def test_is_charging_true(self):
        fake_battery = MagicMock()
        fake_battery.power_plugged = True
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.return_value = fake_battery

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = is_charging()
        self.assertTrue(result)

    def test_is_charging_false(self):
        fake_battery = MagicMock()
        fake_battery.power_plugged = False
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.return_value = fake_battery

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = is_charging()
        self.assertFalse(result)

    def test_is_charging_no_battery(self):
        """Desktop without battery should return False."""
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.return_value = None

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = is_charging()
        self.assertFalse(result)

    def test_is_charging_no_psutil(self):
        """No psutil should return False."""
        with patch.dict(sys.modules, {"psutil": None}):
            result = is_charging()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
