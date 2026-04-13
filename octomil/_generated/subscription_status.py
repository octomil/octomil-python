"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    """Subscription is current and paid."""
    PAST_DUE = "past_due"
    """Payment failed, grace period active."""
    CANCELED = "canceled"
    """Subscription canceled, no longer active."""
    TRIALING = "trialing"
    """In trial period before first charge."""
    INCOMPLETE = "incomplete"
    """Initial payment attempt failed, awaiting customer action."""
    INCOMPLETE_EXPIRED = "incomplete_expired"
    """Initial payment window expired without successful payment."""
    UNPAID = "unpaid"
    """All retry attempts exhausted, subscription still exists but not active."""
    PAUSED = "paused"
    """Subscription paused by provider or admin."""
