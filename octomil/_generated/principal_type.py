"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class PrincipalType(str, Enum):
    USER = "user"
    """Human user authenticated via session/JWT"""
    ORG_API_CLIENT = "org_api_client"
    """Organization-scoped API key client"""
    DEVICE = "device"
    """Registered edge device with device token"""
    SERVICE_WORKER = "service_worker"
    """Internal service-to-service caller"""
