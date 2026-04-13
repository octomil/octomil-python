"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class RoutingPolicy(str, Enum):
    PRIVATE = "private"
    """Never send inference inputs to cloud. On-device only with no fallback."""
    LOCAL_ONLY = "local_only"
    """Always use on-device inference. Fail if no local model is available."""
    LOCAL_FIRST = "local_first"
    """Prefer on-device inference; fall back to cloud if local is unavailable or insufficient."""
    CLOUD_FIRST = "cloud_first"
    """Prefer cloud inference; fall back to on-device execution when cloud is unavailable."""
    CLOUD_ONLY = "cloud_only"
    """Always use cloud inference. Never attempt local model execution."""
    PERFORMANCE_FIRST = "performance_first"
    """Choose the lowest-latency viable route across local and cloud candidates."""
    AUTO = "auto"
    """SDK automatically selects the best inference path based on model availability, hardware capability, and network conditions."""
