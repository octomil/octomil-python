"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class RoutingPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    """Always use on-device inference. Fail if no local model is available."""
    LOCAL_FIRST = "local_first"
    """Prefer on-device inference; fall back to cloud if local is unavailable or insufficient."""
    CLOUD_ONLY = "cloud_only"
    """Always use cloud inference. Never attempt local model execution."""
