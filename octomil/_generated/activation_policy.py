"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ActivationPolicy(str, Enum):
    IMMEDIATE = "immediate"
    """Activate as soon as artifact is verified and staged."""
    NEXT_LAUNCH = "next_launch"
    """Activate on next app launch."""
    MANUAL = "manual"
    """Hold in staged state; activate only on explicit SDK call."""
    WHEN_IDLE = "when_idle"
    """Activate when no inference is in flight and device is idle."""
