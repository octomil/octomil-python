"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class GateClass(str, Enum):
    READINESS = "readiness"
    """Hard checks that must pass before local execution can start"""
    PERFORMANCE = "performance"
    """Latency, throughput, resource, or benchmark checks"""
    OUTPUT_QUALITY = "output_quality"
    """Post-inference checks against the generated result"""
