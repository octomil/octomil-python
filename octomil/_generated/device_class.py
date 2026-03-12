"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class DeviceClass(str, Enum):
    FLAGSHIP = "flagship"
    """High-end device (large RAM, GPU/NPU, fast CPU)"""
    HIGH = "high"
    """Upper-mid device (good RAM, some acceleration)"""
    MID = "mid"
    """Mid-range device (moderate RAM, CPU-only inference likely)"""
    LOW = "low"
    """Low-end device (limited RAM, may not support on-device inference)"""
