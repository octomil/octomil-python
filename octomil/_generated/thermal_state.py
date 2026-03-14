"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ThermalState(str, Enum):
    NOMINAL = "nominal"
    """Normal operating temperature"""
    FAIR = "fair"
    """Slightly elevated temperature, no throttling"""
    SERIOUS = "serious"
    """High temperature, performance may be throttled"""
    CRITICAL = "critical"
    """Critical temperature, heavy throttling or shutdown imminent"""
