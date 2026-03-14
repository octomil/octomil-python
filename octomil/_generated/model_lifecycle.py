"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelLifecycle(str, Enum):
    ACTIVE = "active"
    """Available for deployment. Default state for new versions."""
    DEPRECATED = "deprecated"
    """Still functional but scheduled for removal. SDKs should warn."""
    RETIRED = "retired"
    """No longer available. SDKs must not resolve to this version."""
    PREVIEW = "preview"
    """Pre-release version. Available for testing, not production."""
