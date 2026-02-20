"""mDNS/Zeroconf device discovery for EdgeML devices on the local network."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List


@dataclass
class DiscoveredDevice:
    """An EdgeML device found via mDNS service advertisement."""

    name: str
    platform: str  # "ios", "android", "python"
    ip: str
    port: int
    device_id: str


def scan_for_devices(timeout: float = 5.0) -> List[DiscoveredDevice]:
    """Scan the local network for EdgeML devices via mDNS.

    Looks for services advertising ``_edgeml._tcp.local.``.
    Returns an empty list if zeroconf is not installed or no devices are found.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore[import-untyped]
    except ImportError:
        return []

    devices: List[DiscoveredDevice] = []

    class _Listener:
        """Collect discovered EdgeML services."""

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
    _browser = ServiceBrowser(zc, "_edgeml._tcp.local.", listener)

    time.sleep(timeout)
    zc.close()
    return devices
