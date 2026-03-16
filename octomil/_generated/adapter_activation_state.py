"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class AdapterActivationState(str, Enum):
    NONE = "none"
    """No adapter tracked for this slot."""
    STAGED = "staged"
    """Adapter is on-disk and ready to be warmed."""
    WARMING = "warming"
    """Adapter is being loaded into the runtime alongside the base model."""
    SHADOW = "shadow"
    """Adapter loaded in shadow mode; traffic not yet switched to it."""
    ACTIVE = "active"
    """Adapter is active and serving inference requests."""
    DRAINING_OLD = "draining_old"
    """Previous adapter is draining in-flight requests before eviction."""
    FINALIZED = "finalized"
    """Adapter has been replaced by a newer version and is no longer active."""
    FAILED_HEALTHCHECK = "failed_healthcheck"
    """Adapter loaded but failed post-load inference health check."""
    REJECTED = "rejected"
    """Adapter rejected by policy or compatibility check (e.g., base model mismatch)."""
    ROLLBACK_PENDING = "rollback_pending"
    """Activation failed; system is reverting to the previous adapter version."""
