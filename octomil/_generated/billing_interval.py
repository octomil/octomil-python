"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class BillingInterval(str, Enum):
    MONTHLY = "monthly"
    """Billed every calendar month."""
    ANNUAL = "annual"
    """Billed once per year (20% discount)."""
