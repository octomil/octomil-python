"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class DeviceConnectivityStatus(str, Enum):
    ONLINE = "online"
    """Heartbeat received within 2x expected interval"""
    STALE = "stale"
    """Heartbeat received within 5x interval but beyond 2x"""
    OFFLINE = "offline"
    """No heartbeat beyond 5x interval or device deregistered/suspended"""
    REVOKED = "revoked"
    """Device trust has been revoked by an admin"""
    ERROR = "error"
    """Device is in an error state"""
