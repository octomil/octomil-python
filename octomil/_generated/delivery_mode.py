"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class DeliveryMode(str, Enum):
    BUNDLED = "bundled"
    """Model is included in the app binary. No download required."""
    MANAGED = "managed"
    """SDK downloads and manages the model artifact at runtime."""
    CLOUD = "cloud"
    """Inference runs remotely. No model artifact stored on device."""
