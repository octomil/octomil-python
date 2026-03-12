"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelStatus(str, Enum):
    NOT_CACHED = "not_cached"
    """Model not present on device"""
    DOWNLOADING = "downloading"
    """Model download in progress"""
    READY = "ready"
    """Model cached and ready for inference"""
    ERROR = "error"
    """Model in error state (download failed, corrupt, load failed)"""
