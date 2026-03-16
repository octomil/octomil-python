"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class TelemetryClass(str, Enum):
    MUST_KEEP = "must_keep"
    """Critical events that must never be dropped; uploaded immediately and persisted with highest durability."""
    IMPORTANT = "important"
    """High-value events uploaded on a best-effort basis with moderate persistence; dropped only under severe storage pressure."""
    BEST_EFFORT = "best_effort"
    """Low-priority diagnostic events; may be dropped if the upload queue is full or device is under resource pressure."""
