"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class KvCacheStrategy(str, Enum):
    DISABLED = "disabled"
    """KV cache management not applied."""
    BUDGET_ONLY = "budget_only"
    """Memory-budget KV allocation without compression."""
    COMPRESSED = "compressed"
    """Quantized KV cache for extended context windows."""
