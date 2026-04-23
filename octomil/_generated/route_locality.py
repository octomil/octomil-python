"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class RouteLocality(str, Enum):
    LOCAL = "local"
    """Inference runs in a local SDK/device/browser runtime or explicit local endpoint"""
    CLOUD = "cloud"
    """Inference runs through the hosted cloud gateway"""
