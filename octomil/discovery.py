"""mDNS/Zeroconf device discovery for Octomil devices on the local network."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class DiscoveredDevice:
    """An Octomil device found via mDNS service advertisement."""

    name: str
    platform: str  # "ios", "android", "python"
    ip: str
    port: int
    device_id: str


def scan_for_devices(timeout: float = 5.0) -> List[DiscoveredDevice]:
    """Scan the local network for Octomil devices via mDNS.

    Looks for services advertising ``_octomil._tcp.local.``.
    Returns an empty list if zeroconf is not installed or no devices are found.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore[import-untyped]
    except ImportError:
        return []

    devices: List[DiscoveredDevice] = []

    class _Listener:
        """Collect discovered Octomil services."""

        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info:
                txt = {k.decode(): v.decode() for k, v in info.properties.items()}
                addresses = info.parsed_addresses()
                devices.append(
                    DiscoveredDevice(
                        name=txt.get("device_name", name),
                        platform=txt.get("platform", "unknown"),
                        ip=str(addresses[0]) if addresses else "",
                        port=info.port,
                        device_id=txt.get("device_id", ""),
                    )
                )

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

    zc = Zeroconf()
    listener = _Listener()
    _browser = ServiceBrowser(zc, "_octomil._tcp.local.", listener)

    time.sleep(timeout)
    zc.close()
    return devices


def detect_platform_on_network(timeout: float = 3.0) -> Optional[str]:
    """Detect if iOS or Android devices are on the local network.

    Scans for Apple-specific Bonjour services (_apple-mobdev2._tcp,
    _airplay._tcp, _companion-link._tcp). If found, returns "ios".
    Returns None if no platform can be detected.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore[import-untyped]
    except ImportError:
        return None

    apple_services = [
        "_apple-mobdev2._tcp.local.",
        "_airplay._tcp.local.",
        "_companion-link._tcp.local.",
    ]

    found_apple = False

    class _AppleListener:
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            nonlocal found_apple
            found_apple = True

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

    zc = Zeroconf()
    listener = _AppleListener()
    browsers = [ServiceBrowser(zc, svc, listener) for svc in apple_services]

    time.sleep(timeout)
    zc.close()

    if found_apple:
        return "ios"
    return None


def wait_for_device(
    timeout: float = 300.0,
    poll_interval: float = 2.0,
    on_found: Optional[Callable[[DiscoveredDevice], None]] = None,
) -> Optional[DiscoveredDevice]:
    """Continuously scan for Octomil devices until one appears or timeout.

    Used after showing the app-download QR code — keeps scanning for new
    ``_octomil._tcp`` services instead of requiring the user to press Enter.

    Returns the first device found, or None on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        devices = scan_for_devices(timeout=poll_interval)
        if devices:
            device = devices[0]
            if on_found:
                on_found(device)
            return device
    return None
