"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelRefKind(str, Enum):
    MODEL = "model"
    """Plain model identifier or slug (e.g. 'gemma-2b', 'phi-4'). No prefix, no special grammar."""
    APP = "app"
    """Application reference (@app/{slug}/{capability}). Requires server resolution."""
    CAPABILITY = "capability"
    """Capability reference (@capability/{capability}). Resolves to org/global default model for the capability."""
    DEPLOYMENT = "deployment"
    """Deployment reference (deploy_{id_or_key}). Resolves to pinned model and config."""
    EXPERIMENT = "experiment"
    """Experiment reference (exp_{experiment_id}/{variant_id}). Resolves to experiment variant."""
    ALIAS = "alias"
    """Named alias (alias:{name}). Resolves to whatever the alias points to."""
    DEFAULT = "default"
    """Empty or unspecified model reference. Uses org/config default."""
    UNKNOWN = "unknown"
    """Unrecognized format. Passed to server as-is for best-effort resolution."""
