"""Training eligibility checks: battery, network quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy httpx — defer ~55ms import cost. Exposed as module attribute for test mocking.
def __getattr__(name: str) -> object:
    if name == "httpx":
        import httpx as _httpx
        globals()["httpx"] = _httpx
        return _httpx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class EligibilityResult:
    """Result of a training eligibility check."""
    eligible: bool
    reason: Optional[str] = None


@dataclass
class NetworkQuality:
    """Result of a network quality probe."""
    reachable: bool
    latency_ms: Optional[float] = None


def check_training_eligibility(
    battery_level: Optional[int],
    min_battery: int = 15,
    charging: bool = False,
) -> EligibilityResult:
    """Check whether the device should participate in training.

    Args:
        battery_level: Current battery percentage (0-100), or None if unavailable.
        min_battery: Minimum battery level required (default 15%).
        charging: Whether the device is currently charging.

    Returns:
        EligibilityResult with eligible=True if training is allowed.
    """
    # No battery sensor (desktop) -> eligible
    if battery_level is None:
        return EligibilityResult(eligible=True)

    # Charging overrides low battery
    if charging:
        return EligibilityResult(eligible=True)

    if battery_level < min_battery:
        return EligibilityResult(eligible=False, reason="low_battery")

    return EligibilityResult(eligible=True)


def check_network_quality(
    api_base: str,
    timeout: float = 5.0,
) -> NetworkQuality:
    """Probe the API endpoint to check network reachability.

    Args:
        api_base: Base URL to probe (e.g. "https://api.octomil.com").
        timeout: Request timeout in seconds.

    Returns:
        NetworkQuality with reachable=True if the server responds.
    """
    try:
        import time

        start = time.monotonic()
        response = httpx.get(f"{api_base.rstrip('/')}/health", timeout=timeout)
        latency = (time.monotonic() - start) * 1000
        return NetworkQuality(reachable=response.status_code < 500, latency_ms=latency)
    except Exception:
        return NetworkQuality(reachable=False)
