"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class QueryRoutingTier(str, Enum):
    FAST = "fast"
    """Prefer speed over quality. Select smallest available model."""
    BALANCED = "balanced"
    """Balance quality and speed. Default tier."""
    QUALITY = "quality"
    """Prefer output quality. Select largest available model."""
