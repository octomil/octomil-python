"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class RouteMode(str, Enum):
    SDK_RUNTIME = "sdk_runtime"
    """Direct local inference via SDK-managed runtime engine"""
    EXTERNAL_ENDPOINT = "external_endpoint"
    """User-configured external inference endpoint"""
    HOSTED_GATEWAY = "hosted_gateway"
    """Octomil hosted cloud inference gateway"""
