"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class CompatibilityLevel(str, Enum):
    STABLE = "stable"
    """Breaking changes require major version bump. Full backward compatibility guaranteed within major."""
    BETA = "beta"
    """Shape is mostly final. Minor breaking changes possible in minor versions with migration guide."""
    EXPERIMENTAL = "experimental"
    """Shape may change significantly. Not required for SDK conformance."""
    COMPATIBILITY = "compatibility"
    """Stable but frozen. No new features. Exists for migration from older API patterns. Has a support window."""
