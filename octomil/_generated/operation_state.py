"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class OperationState(str, Enum):
    QUEUED = "queued"
    """Operation is enqueued and waiting to be picked up by a worker."""
    LEASED = "leased"
    """Operation has been claimed by a worker but execution has not yet started."""
    RUNNING = "running"
    """Operation is actively executing."""
    SUCCESS = "success"
    """Operation completed successfully."""
    FAILED = "failed"
    """Operation failed; may be retried depending on retry policy."""
    PAUSED = "paused"
    """Operation execution suspended (e.g., waiting for connectivity or charging)."""
