"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ExecutionMode(str, Enum):
    SDK_RUNTIME = "sdk_runtime"
    """Local inference via SDK-managed runtime engine"""
    HOSTED_GATEWAY = "hosted_gateway"
    """Cloud inference via server gateway"""
    EXTERNAL_ENDPOINT = "external_endpoint"
    """Inference via user-configured external server"""
