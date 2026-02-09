"""
Device information collection for EdgeML Python SDK.

Automatically collects hardware metadata and system constraints
for device monitoring and training eligibility.
"""

import platform
import socket
import subprocess
from typing import Dict, Any, Optional


def get_stable_device_id() -> str:
    """
    Get a stable device identifier that persists across SDK runs.

    Returns:
        Stable device identifier based on hardware UUID or machine ID.
    """
    try:
        if platform.system() == "Darwin":  # macOS
            hardware_uuid = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"], text=True  # noqa: S603,S607
            )
            for line in hardware_uuid.split("\n"):
                if "Hardware UUID" in line or "UUID" in line:
                    uuid = line.split(":")[-1].strip()
                    return f"MacBook-{uuid[:8]}"
        elif platform.system() == "Linux":
            try:
                with open("/etc/machine-id", "r") as f:
                    machine_id = f.read().strip()
                    return f"Linux-{machine_id[:8]}"
            except (OSError, IOError):
                pass
    except (subprocess.SubprocessError, OSError):
        pass

    # Fallback to hostname
    return socket.gethostname()


def get_battery_level() -> Optional[int]:
    """
    Get current battery level as percentage.

    Returns:
        Battery percentage (0-100) or None if unavailable.
    """
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery:
            return int(battery.percent)
    except ImportError:
        pass
    except (AttributeError, OSError):
        pass
    return None


def get_network_type() -> str:
    """
    Get current network connection type.

    Returns:
        Network type: 'wifi', 'cellular', 'ethernet', or 'unknown'.
    """
    # Basic implementation - can be enhanced with platform-specific checks
    return "wifi"  # Default assumption for desktop/laptop


def get_timezone() -> str:
    """Get system timezone."""
    try:
        if platform.system() == "Darwin":
            tz = subprocess.check_output(
                ["readlink", "/etc/localtime"], text=True  # noqa: S603,S607
            ).strip()
            return tz.split("/zoneinfo/")[-1] if "/zoneinfo/" in tz else "UTC"
        elif platform.system() == "Linux":
            tz = subprocess.check_output(
                ["timedatectl", "show", "--value", "-p", "Timezone"], text=True  # noqa: S603,S607
            ).strip()
            return tz
    except (subprocess.SubprocessError, OSError):
        pass
    return "UTC"


def get_manufacturer() -> Optional[str]:
    """Get device manufacturer."""
    system = platform.system()
    if system == "Darwin":
        return "Apple"
    elif system == "Linux":
        try:
            with open("/sys/devices/virtual/dmi/id/sys_vendor", "r") as f:
                return f.read().strip()
        except (OSError, IOError):
            pass
    return system


def get_model() -> Optional[str]:
    """Get device model name."""
    try:
        if platform.system() == "Darwin":
            model = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], text=True  # noqa: S603,S607
            ).strip()
            return model
    except (subprocess.SubprocessError, OSError):
        pass
    return platform.node()


def get_memory_info() -> Optional[int]:
    """
    Get total system memory in MB.

    Returns:
        Total memory in MB or None if unavailable.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return int(memory.total / (1024 * 1024))
    except ImportError:
        pass
    except (AttributeError, OSError):
        pass
    return None


def get_storage_info() -> Optional[int]:
    """
    Get available storage in MB.

    Returns:
        Available storage in MB or None if unavailable.
    """
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return int(disk.free / (1024 * 1024))
    except ImportError:
        pass
    except (AttributeError, OSError):
        pass
    return None


def detect_gpu() -> bool:
    """
    Detect if GPU is available.

    Returns:
        True if GPU is detected, False otherwise.
    """
    try:
        if platform.system() == "Darwin":
            gpu_info = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True  # noqa: S603,S607
            )
            return "Chipset Model" in gpu_info or "GPU" in gpu_info
    except (subprocess.SubprocessError, OSError):
        pass
    return False


class DeviceInfo:
    """
    Collects and manages device information for EdgeML platform.

    Automatically gathers:
    - Stable device identifier
    - Hardware specs (CPU, memory, storage, GPU)
    - System info (OS, manufacturer, model)
    - Runtime constraints (battery, network)
    - Locale and timezone

    Example:
        >>> info = DeviceInfo()
        >>> registration_data = info.to_registration_dict()
    """

    def __init__(self):
        """Initialize device information collector."""
        self._device_id: Optional[str] = None
        self._cached_info: Optional[Dict[str, Any]] = None

    @property
    def device_id(self) -> str:
        """Get stable device identifier."""
        if self._device_id is None:
            self._device_id = get_stable_device_id()
        return self._device_id

    def collect_device_info(self) -> Dict[str, Any]:
        """
        Collect complete device information.

        Returns:
            Dictionary with all device hardware and capability info.
        """
        return {
            "manufacturer": get_manufacturer(),
            "model": get_model(),
            "cpu_architecture": platform.machine(),
            "gpu_available": detect_gpu(),
            "total_memory_mb": get_memory_info(),
            "available_storage_mb": get_storage_info(),
        }

    def collect_metadata(self) -> Dict[str, Any]:
        """
        Collect runtime metadata (battery, network).

        Returns:
            Dictionary with current runtime constraints.
        """
        return {
            "battery_level": get_battery_level(),
            "network_type": get_network_type(),
        }

    def collect_capabilities(self) -> Dict[str, Any]:
        """
        Collect ML capabilities.

        Returns:
            Dictionary with ML framework availability.
        """
        return {
            "cpu_architecture": platform.machine(),
            "gpu_available": detect_gpu(),
            "python_version": platform.python_version(),
        }

    def to_registration_dict(self) -> Dict[str, Any]:
        """
        Create registration payload for EdgeML API.

        Returns:
            Complete registration dictionary with all device information.
        """
        return {
            "device_identifier": self.device_id,
            "platform": platform.system().lower(),
            "os_version": f"{platform.system()} {platform.release()}",
            "device_info": self.collect_device_info(),
            "locale": "en_US",  # Can be enhanced with locale detection
            "region": "US",     # Can be enhanced with region detection
            "timezone": get_timezone(),
            "metadata": self.collect_metadata(),
            "capabilities": self.collect_capabilities(),
        }

    def update_metadata(self) -> Dict[str, Any]:
        """
        Get updated metadata for heartbeat updates.

        Call this periodically to send updated battery/network status.

        Returns:
            Current runtime metadata.
        """
        return self.collect_metadata()
