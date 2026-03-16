"""Monitoring configuration for device heartbeat and telemetry.

Provides a simple dataclass to control background monitoring behaviour
when using the SDK device registration flow.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MonitoringConfig"]


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for device monitoring (heartbeat, telemetry).

    Attributes:
        enabled: Whether background monitoring is active.
        heartbeat_interval_seconds: Interval in seconds between heartbeat pings.
    """

    enabled: bool = False
    heartbeat_interval_seconds: int = 300
