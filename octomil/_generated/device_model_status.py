"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class DeviceModelStatus(str, Enum):
    NOT_ASSIGNED = "not_assigned"
    """No model assigned to this device"""
    ASSIGNED = "assigned"
    """Model assigned but not yet downloaded"""
    DOWNLOADING = "downloading"
    """Model download in progress"""
    DOWNLOAD_FAILED = "download_failed"
    """Model download failed"""
    VERIFYING = "verifying"
    """Model integrity verification in progress"""
    READY = "ready"
    """Model downloaded and verified, not yet loaded into runtime"""
    LOADING = "loading"
    """Model being loaded into runtime memory"""
    LOAD_FAILED = "load_failed"
    """Model failed to load into runtime"""
    ACTIVE = "active"
    """Model loaded and serving inference requests"""
    FALLBACK_ACTIVE = "fallback_active"
    """Fallback model active instead of assigned model"""
    DEPRECATED_ASSIGNED = "deprecated_assigned"
    """Assigned model version has been deprecated"""
