import subprocess
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

try:
    from edgeml.device_info import (
        DeviceInfo,
        detect_gpu,
        get_battery_level,
        get_manufacturer,
        get_memory_info,
        get_model,
        get_network_type,
        get_stable_device_id,
        get_storage_info,
        get_timezone,
    )
    HAS_DEVICE_INFO = True
except ImportError:
    HAS_DEVICE_INFO = False


@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetStableDeviceId(unittest.TestCase):
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_returns_hardware_uuid(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = (
            "Hardware Overview:\n"
            "  Hardware UUID: ABCD1234-5678-EFGH-IJKL-MNOPQRSTUVWX\n"
        )
        result = get_stable_device_id()
        self.assertEqual(result, "MacBook-ABCD1234")
        mock_subprocess.assert_called_once_with(
            ["system_profiler", "SPHardwareDataType"], text=True
        )

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_uuid_line_generic(self, mock_subprocess, mock_system):
        """A line containing 'UUID' but not 'Hardware UUID' should still match."""
        mock_subprocess.return_value = (
            "Hardware Overview:\n"
            "  Provisioning UUID: 11112222-3333-4444-5555-666677778888\n"
        )
        result = get_stable_device_id()
        self.assertEqual(result, "MacBook-11112222")

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    @patch("edgeml.device_info.socket.gethostname", return_value="my-mac")
    def test_macos_no_uuid_line_falls_back_to_hostname(
        self, mock_hostname, mock_subprocess, mock_system
    ):
        mock_subprocess.return_value = "Hardware Overview:\n  Model: MacBookPro\n"
        result = get_stable_device_id()
        self.assertEqual(result, "my-mac")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("builtins.open", mock_open(read_data="abcdef1234567890\n"))
    def test_linux_reads_machine_id(self, mock_system):
        result = get_stable_device_id()
        self.assertEqual(result, "Linux-abcdef12")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("edgeml.device_info.socket.gethostname", return_value="linux-box")
    def test_linux_machine_id_missing_falls_back_to_hostname(
        self, mock_hostname, mock_open_fn, mock_system
    ):
        result = get_stable_device_id()
        self.assertEqual(result, "linux-box")

    @patch("edgeml.device_info.platform.system", return_value="Windows")
    @patch("edgeml.device_info.socket.gethostname", return_value="windows-pc")
    def test_fallback_to_hostname(self, mock_hostname, mock_system):
        result = get_stable_device_id()
        self.assertEqual(result, "windows-pc")

    @patch("edgeml.device_info.platform.system", side_effect=Exception("boom"))
    @patch("edgeml.device_info.socket.gethostname", return_value="fallback-host")
    def test_exception_falls_back_to_hostname(self, mock_hostname, mock_system):
        result = get_stable_device_id()
        self.assertEqual(result, "fallback-host")

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch(
        "edgeml.device_info.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "system_profiler"),
    )
    @patch("edgeml.device_info.socket.gethostname", return_value="err-host")
    def test_macos_subprocess_error_falls_back(
        self, mock_hostname, mock_subprocess, mock_system
    ):
        result = get_stable_device_id()
        self.assertEqual(result, "err-host")


# ---------------------------------------------------------------------------
# get_battery_level
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetBatteryLevel(unittest.TestCase):
    def test_with_psutil_available(self):
        fake_battery = MagicMock()
        fake_battery.percent = 73.5
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.return_value = fake_battery

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_battery_level()

        self.assertEqual(result, 73)

    def test_psutil_returns_none_battery(self):
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.return_value = None

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_battery_level()

        self.assertIsNone(result)

    def test_psutil_not_installed(self):
        """When psutil is not importable, get_battery_level returns None."""
        with patch.dict(sys.modules, {"psutil": None}):
            result = get_battery_level()
        self.assertIsNone(result)

    def test_psutil_raises_runtime_error(self):
        fake_psutil = MagicMock()
        fake_psutil.sensors_battery.side_effect = RuntimeError("sensor failure")

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_battery_level()

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# get_network_type
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetNetworkType(unittest.TestCase):
    def test_always_returns_wifi(self):
        self.assertEqual(get_network_type(), "wifi")


# ---------------------------------------------------------------------------
# get_timezone
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetTimezone(unittest.TestCase):
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_timezone(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = (
            "/var/db/timezone/zoneinfo/America/New_York\n"
        )
        result = get_timezone()
        self.assertEqual(result, "America/New_York")
        mock_subprocess.assert_called_once_with(
            ["readlink", "/etc/localtime"], text=True
        )

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_no_zoneinfo_in_path_returns_utc(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "/some/random/path\n"
        result = get_timezone()
        self.assertEqual(result, "UTC")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_linux_timezone(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "Europe/London\n"
        result = get_timezone()
        self.assertEqual(result, "Europe/London")
        mock_subprocess.assert_called_once_with(
            ["timedatectl", "show", "--value", "-p", "Timezone"], text=True
        )

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch(
        "edgeml.device_info.subprocess.check_output",
        side_effect=FileNotFoundError,
    )
    def test_exception_returns_utc(self, mock_subprocess, mock_system):
        result = get_timezone()
        self.assertEqual(result, "UTC")

    @patch("edgeml.device_info.platform.system", return_value="Windows")
    def test_unsupported_platform_returns_utc(self, mock_system):
        """Neither Darwin nor Linux branch is entered, so the fallback fires."""
        self.assertEqual(get_timezone(), "UTC")
        mock_system.assert_called()


# ---------------------------------------------------------------------------
# get_manufacturer
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetManufacturer(unittest.TestCase):
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    def test_macos_returns_apple(self, mock_system):
        self.assertEqual(get_manufacturer(), "Apple")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("builtins.open", mock_open(read_data="Lenovo\n"))
    def test_linux_reads_sys_vendor(self, mock_system):
        self.assertEqual(get_manufacturer(), "Lenovo")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_linux_file_missing_returns_platform_system(
        self, mock_open_fn, mock_system
    ):
        self.assertEqual(get_manufacturer(), "Linux")

    @patch("edgeml.device_info.platform.system", return_value="Windows")
    def test_other_platform_returns_system_name(self, mock_system):
        self.assertEqual(get_manufacturer(), "Windows")


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetModel(unittest.TestCase):
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_sysctl(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "MacBookPro18,3\n"
        result = get_model()
        self.assertEqual(result, "MacBookPro18,3")
        mock_subprocess.assert_called_once_with(
            ["sysctl", "-n", "hw.model"], text=True
        )

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch(
        "edgeml.device_info.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "sysctl"),
    )
    @patch("edgeml.device_info.platform.node", return_value="my-node")
    def test_macos_sysctl_failure_falls_back_to_node(
        self, mock_node, mock_subprocess, mock_system
    ):
        result = get_model()
        self.assertEqual(result, "my-node")

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    @patch("edgeml.device_info.platform.node", return_value="linux-node")
    def test_non_macos_returns_platform_node(self, mock_node, mock_system):
        result = get_model()
        self.assertEqual(result, "linux-node")


# ---------------------------------------------------------------------------
# get_memory_info
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetMemoryInfo(unittest.TestCase):
    def test_with_psutil(self):
        fake_memory = MagicMock()
        fake_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
        fake_psutil = MagicMock()
        fake_psutil.virtual_memory.return_value = fake_memory

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_memory_info()

        self.assertEqual(result, 16384)

    def test_psutil_not_installed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            result = get_memory_info()
        self.assertIsNone(result)

    def test_psutil_raises_runtime_error(self):
        fake_psutil = MagicMock()
        fake_psutil.virtual_memory.side_effect = RuntimeError("fail")

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_memory_info()

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# get_storage_info
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestGetStorageInfo(unittest.TestCase):
    def test_with_psutil(self):
        fake_disk = MagicMock()
        fake_disk.free = 100 * 1024 * 1024 * 1024  # 100 GB
        fake_psutil = MagicMock()
        fake_psutil.disk_usage.return_value = fake_disk

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_storage_info()

        self.assertEqual(result, 102400)
        fake_psutil.disk_usage.assert_called_once_with("/")

    def test_psutil_not_installed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            result = get_storage_info()
        self.assertIsNone(result)

    def test_psutil_raises_runtime_error(self):
        fake_psutil = MagicMock()
        fake_psutil.disk_usage.side_effect = RuntimeError("fail")

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = get_storage_info()

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# detect_gpu
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestDetectGpu(unittest.TestCase):
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_with_chipset_model(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = (
            "Graphics/Displays:\n  Chipset Model: Apple M1 Pro\n"
        )
        self.assertTrue(detect_gpu())

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_with_gpu_keyword(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "GPU: Apple M2 Max\n"
        self.assertTrue(detect_gpu())

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch("edgeml.device_info.subprocess.check_output")
    def test_macos_no_gpu_info(self, mock_subprocess, mock_system):
        mock_subprocess.return_value = "Graphics/Displays:\n  Nothing useful\n"
        self.assertFalse(detect_gpu())

    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    @patch(
        "edgeml.device_info.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "system_profiler"),
    )
    def test_macos_subprocess_error(self, mock_subprocess, mock_system):
        self.assertFalse(detect_gpu())

    @patch("edgeml.device_info.platform.system", return_value="Linux")
    def test_non_macos_returns_false(self, mock_system):
        self.assertFalse(detect_gpu())

    @patch("edgeml.device_info.platform.system", return_value="Windows")
    def test_windows_returns_false(self, mock_system):
        self.assertFalse(detect_gpu())


# ---------------------------------------------------------------------------
# DeviceInfo class
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_DEVICE_INFO, "edgeml.device_info not available in this install")
class TestDeviceInfo(unittest.TestCase):
    def test_init_state(self):
        info = DeviceInfo()
        self.assertIsNone(info._device_id)
        self.assertIsNone(info._cached_info)

    @patch("edgeml.device_info.get_stable_device_id", return_value="test-id-123")
    def test_device_id_lazy_initialization(self, mock_get_id):
        info = DeviceInfo()
        self.assertIsNone(info._device_id)

        device_id = info.device_id
        self.assertEqual(device_id, "test-id-123")
        mock_get_id.assert_called_once()

    @patch("edgeml.device_info.get_stable_device_id", return_value="cached-id")
    def test_device_id_caching(self, mock_get_id):
        info = DeviceInfo()

        first_call = info.device_id
        second_call = info.device_id
        self.assertEqual(first_call, "cached-id")
        self.assertEqual(second_call, "cached-id")
        mock_get_id.assert_called_once()

    @patch("edgeml.device_info.get_storage_info", return_value=51200)
    @patch("edgeml.device_info.get_memory_info", return_value=8192)
    @patch("edgeml.device_info.detect_gpu", return_value=True)
    @patch("edgeml.device_info.platform.machine", return_value="arm64")
    @patch("edgeml.device_info.get_model", return_value="MacBookPro18,3")
    @patch("edgeml.device_info.get_manufacturer", return_value="Apple")
    def test_collect_device_info_keys_and_values(
        self,
        mock_manufacturer,
        mock_model,
        mock_machine,
        mock_gpu,
        mock_memory,
        mock_storage,
    ):
        info = DeviceInfo()
        result = info.collect_device_info()

        expected_keys = {
            "manufacturer",
            "model",
            "cpu_architecture",
            "gpu_available",
            "total_memory_mb",
            "available_storage_mb",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["manufacturer"], "Apple")
        self.assertEqual(result["model"], "MacBookPro18,3")
        self.assertEqual(result["cpu_architecture"], "arm64")
        self.assertTrue(result["gpu_available"])
        self.assertEqual(result["total_memory_mb"], 8192)
        self.assertEqual(result["available_storage_mb"], 51200)

    @patch("edgeml.device_info.get_network_type", return_value="wifi")
    @patch("edgeml.device_info.get_battery_level", return_value=85)
    def test_collect_metadata_keys_and_values(self, mock_battery, mock_network):
        info = DeviceInfo()
        result = info.collect_metadata()

        expected_keys = {"battery_level", "network_type"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["battery_level"], 85)
        self.assertEqual(result["network_type"], "wifi")

    @patch("edgeml.device_info.platform.python_version", return_value="3.11.5")
    @patch("edgeml.device_info.detect_gpu", return_value=False)
    @patch("edgeml.device_info.platform.machine", return_value="x86_64")
    def test_collect_capabilities_keys_and_values(
        self, mock_machine, mock_gpu, mock_python
    ):
        info = DeviceInfo()
        result = info.collect_capabilities()

        expected_keys = {"cpu_architecture", "gpu_available", "python_version"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["cpu_architecture"], "x86_64")
        self.assertFalse(result["gpu_available"])
        self.assertEqual(result["python_version"], "3.11.5")

    @patch("edgeml.device_info.get_timezone", return_value="US/Pacific")
    @patch("edgeml.device_info.get_network_type", return_value="wifi")
    @patch("edgeml.device_info.get_battery_level", return_value=50)
    @patch("edgeml.device_info.platform.python_version", return_value="3.11.5")
    @patch("edgeml.device_info.detect_gpu", return_value=True)
    @patch("edgeml.device_info.platform.machine", return_value="arm64")
    @patch("edgeml.device_info.get_storage_info", return_value=51200)
    @patch("edgeml.device_info.get_memory_info", return_value=16384)
    @patch("edgeml.device_info.get_model", return_value="MacBookPro18,3")
    @patch("edgeml.device_info.get_manufacturer", return_value="Apple")
    @patch("edgeml.device_info.get_stable_device_id", return_value="test-device-id")
    @patch("edgeml.device_info.platform.release", return_value="23.1.0")
    @patch("edgeml.device_info.platform.system", return_value="Darwin")
    def test_to_registration_dict_all_keys(
        self,
        mock_system,
        mock_release,
        mock_device_id,
        mock_manufacturer,
        mock_model,
        mock_memory,
        mock_storage,
        mock_machine,
        mock_gpu,
        mock_python,
        mock_battery,
        mock_network,
        mock_timezone,
    ):
        info = DeviceInfo()
        result = info.to_registration_dict()

        expected_top_keys = {
            "device_identifier",
            "platform",
            "os_version",
            "device_info",
            "locale",
            "region",
            "timezone",
            "metadata",
            "capabilities",
        }
        self.assertEqual(set(result.keys()), expected_top_keys)
        self.assertEqual(result["device_identifier"], "test-device-id")
        self.assertEqual(result["platform"], "darwin")
        self.assertEqual(result["os_version"], "Darwin 23.1.0")
        self.assertEqual(result["locale"], "en_US")
        self.assertEqual(result["region"], "US")
        self.assertEqual(result["timezone"], "US/Pacific")
        self.assertIsInstance(result["device_info"], dict)
        self.assertIsInstance(result["metadata"], dict)
        self.assertIsInstance(result["capabilities"], dict)

    @patch("edgeml.device_info.get_network_type", return_value="wifi")
    @patch("edgeml.device_info.get_battery_level", return_value=42)
    def test_update_metadata_returns_same_as_collect_metadata(
        self, mock_battery, mock_network
    ):
        info = DeviceInfo()
        metadata = info.update_metadata()
        collected = info.collect_metadata()
        self.assertEqual(metadata, collected)

    @patch("edgeml.device_info.get_network_type", return_value="wifi")
    @patch("edgeml.device_info.get_battery_level", return_value=None)
    def test_collect_metadata_with_no_battery(self, mock_battery, mock_network):
        info = DeviceInfo()
        result = info.collect_metadata()
        self.assertIsNone(result["battery_level"])
        self.assertEqual(result["network_type"], "wifi")


if __name__ == "__main__":
    unittest.main()
