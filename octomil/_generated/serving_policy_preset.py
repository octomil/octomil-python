"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ServingPolicyPreset(str, Enum):
    PRIVATE = "private"
    """Local-only inference. Never use cloud."""
    LOCAL_FIRST = "local_first"
    """Prefer on-device; allow cloud fallback."""
    PERFORMANCE_FIRST = "performance_first"
    """Minimize latency; allow cloud fallback."""
    QUALITY_FIRST = "quality_first"
    """Maximize output quality; allow cloud fallback."""
    CLOUD_FIRST = "cloud_first"
    """Prefer cloud inference. Routing engine uses local engines when cloud unavailable."""
    CLOUD_ONLY = "cloud_only"
    """Cloud-only inference. Never use local."""
