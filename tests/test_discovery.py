"""Tests for octomil.discovery — mDNS device discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from octomil.discovery import (
    DiscoveredDevice,
    detect_platform_on_network,
    scan_for_devices,
    wait_for_device,
)


# ---------------------------------------------------------------------------
# scan_for_devices
# ---------------------------------------------------------------------------


class TestScanForDevices:
    def test_returns_empty_without_zeroconf(self):
        with patch.dict("sys.modules", {"zeroconf": None}):
            result = scan_for_devices(timeout=0.1)
            assert result == []

    def test_returns_discovered_devices(self):
        """When zeroconf finds a service, it should be returned as DiscoveredDevice."""
        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_info.properties = {
            b"device_name": b"Sean's iPhone",
            b"platform": b"ios",
            b"device_id": b"abc-123",
        }
        mock_info.parsed_addresses.return_value = ["192.168.1.42"]
        mock_info.port = 8080
        mock_zc.get_service_info.return_value = mock_info

        mock_zeroconf_cls = MagicMock(return_value=mock_zc)
        mock_browser_cls = MagicMock()

        # Simulate the ServiceBrowser calling add_service immediately
        def fake_browser(zc, type_, listener):
            listener.add_service(zc, type_, "test._octomil._tcp.local.")
            return MagicMock()

        mock_browser_cls.side_effect = fake_browser

        mock_module = MagicMock()
        mock_module.Zeroconf = mock_zeroconf_cls
        mock_module.ServiceBrowser = mock_browser_cls

        with patch.dict("sys.modules", {"zeroconf": mock_module}):
            # Re-import to pick up mocked module
            import importlib

            import octomil.discovery

            importlib.reload(octomil.discovery)

            result = octomil.discovery.scan_for_devices(timeout=0.01)

        assert len(result) == 1
        assert result[0].name == "Sean's iPhone"
        assert result[0].platform == "ios"
        assert result[0].ip == "192.168.1.42"
        assert result[0].port == 8080
        assert result[0].device_id == "abc-123"

    def test_returns_empty_on_no_services(self):
        mock_zc = MagicMock()
        mock_zeroconf_cls = MagicMock(return_value=mock_zc)
        mock_browser_cls = MagicMock()

        mock_module = MagicMock()
        mock_module.Zeroconf = mock_zeroconf_cls
        mock_module.ServiceBrowser = mock_browser_cls

        with patch.dict("sys.modules", {"zeroconf": mock_module}):
            import importlib

            import octomil.discovery

            importlib.reload(octomil.discovery)

            result = octomil.discovery.scan_for_devices(timeout=0.01)

        assert result == []


# ---------------------------------------------------------------------------
# detect_platform_on_network
# ---------------------------------------------------------------------------


class TestDetectPlatformOnNetwork:
    def test_returns_none_without_zeroconf(self):
        with patch.dict("sys.modules", {"zeroconf": None}):
            result = detect_platform_on_network(timeout=0.1)
            assert result is None

    def test_returns_ios_when_apple_service_found(self):
        mock_zc = MagicMock()
        mock_zeroconf_cls = MagicMock(return_value=mock_zc)

        def fake_browser(zc, type_, listener):
            if "_airplay._tcp" in type_:
                listener.add_service(zc, type_, "test._airplay._tcp.local.")
            return MagicMock()

        mock_browser_cls = MagicMock(side_effect=fake_browser)

        mock_module = MagicMock()
        mock_module.Zeroconf = mock_zeroconf_cls
        mock_module.ServiceBrowser = mock_browser_cls

        with patch.dict("sys.modules", {"zeroconf": mock_module}):
            import importlib

            import octomil.discovery

            importlib.reload(octomil.discovery)

            result = octomil.discovery.detect_platform_on_network(timeout=0.01)

        assert result == "ios"

    def test_returns_none_when_no_apple_services(self):
        mock_zc = MagicMock()
        mock_zeroconf_cls = MagicMock(return_value=mock_zc)
        mock_browser_cls = MagicMock()

        mock_module = MagicMock()
        mock_module.Zeroconf = mock_zeroconf_cls
        mock_module.ServiceBrowser = mock_browser_cls

        with patch.dict("sys.modules", {"zeroconf": mock_module}):
            import importlib

            import octomil.discovery

            importlib.reload(octomil.discovery)

            result = octomil.discovery.detect_platform_on_network(timeout=0.01)

        assert result is None


# ---------------------------------------------------------------------------
# wait_for_device
# ---------------------------------------------------------------------------


class TestWaitForDevice:
    @patch("octomil.discovery.scan_for_devices")
    def test_returns_device_when_found(self, mock_scan):
        device = DiscoveredDevice(
            name="Pixel 8",
            platform="android",
            ip="192.168.1.50",
            port=9090,
            device_id="pixel-123",
        )
        mock_scan.return_value = [device]

        result = wait_for_device(timeout=1.0, poll_interval=0.01)
        assert result is not None
        assert result.name == "Pixel 8"

    @patch("octomil.discovery.scan_for_devices")
    def test_returns_none_on_timeout(self, mock_scan):
        mock_scan.return_value = []

        result = wait_for_device(timeout=0.05, poll_interval=0.01)
        assert result is None

    @patch("octomil.discovery.scan_for_devices")
    def test_calls_on_found_callback(self, mock_scan):
        device = DiscoveredDevice(
            name="iPhone 16",
            platform="ios",
            ip="192.168.1.10",
            port=8080,
            device_id="iphone-456",
        )
        mock_scan.return_value = [device]
        callback = MagicMock()

        result = wait_for_device(timeout=1.0, poll_interval=0.01, on_found=callback)
        assert result is not None
        callback.assert_called_once_with(device)

    @patch("octomil.discovery.scan_for_devices")
    def test_finds_device_after_retries(self, mock_scan):
        device = DiscoveredDevice(
            name="Pixel 8",
            platform="android",
            ip="192.168.1.50",
            port=9090,
            device_id="pixel-123",
        )
        # First two scans find nothing, third finds device
        mock_scan.side_effect = [[], [], [device]]

        result = wait_for_device(timeout=5.0, poll_interval=0.01)
        assert result is not None
        assert result.name == "Pixel 8"
        assert mock_scan.call_count == 3
