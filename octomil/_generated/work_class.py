"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class WorkClass(str, Enum):
    CRITICAL_FOREGROUND = "critical_foreground"
    """Must run immediately and may interact with the user; scheduled at foreground priority with no deferral."""
    BACKGROUND_IMPORTANT = "background_important"
    """Important background work (e.g., model download, pre-warm); scheduled promptly but yields to foreground."""
    BACKGROUND_BEST_EFFORT = "background_best_effort"
    """Low-priority background work (e.g., telemetry flush, GC); deferred until the device is idle and charging."""
