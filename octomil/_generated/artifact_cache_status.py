"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ArtifactCacheStatus(str, Enum):
    HIT = "hit"
    """Artifact found in local cache with valid digest"""
    MISS = "miss"
    """Artifact not in cache, download required"""
    DOWNLOADED = "downloaded"
    """Artifact was downloaded during this request"""
    NOT_APPLICABLE = "not_applicable"
    """No artifact involved in this routing decision"""
    UNAVAILABLE = "unavailable"
    """Artifact required but cannot be obtained"""
