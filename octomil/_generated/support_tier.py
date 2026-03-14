"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class SupportTier(str, Enum):
    BLESSED = "blessed"
    """Primary recommended option. Fully tested, optimized, documented."""
    SUPPORTED = "supported"
    """Production-ready with active maintenance and test coverage."""
    EXPERIMENTAL = "experimental"
    """Functional but not production-hardened. API may change. Limited test coverage."""
    RESEARCH = "research"
    """Reference implementation for evaluation. Not shipped in release builds."""
