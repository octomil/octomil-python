"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class AuthType(str, Enum):
    ORG_API_KEY = "org_api_key"
    """Organization-scoped API key (edg_ prefix)"""
    DEVICE_TOKEN = "device_token"
    """Short-lived device access token (JWT with typ=device_access)"""
    SERVICE_TOKEN = "service_token"
    """Internal service-to-service token"""
    PUBLISHABLE_KEY = "publishable_key"
    """Client-safe API key with restricted scopes, safe to embed in mobile apps and browser code."""
