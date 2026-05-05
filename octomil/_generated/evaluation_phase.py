"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class EvaluationPhase(str, Enum):
    PRE_INFERENCE = "pre_inference"
    """Evaluated before model execution begins"""
    DURING_INFERENCE = "during_inference"
    """Measured while generation starts, before visible output in buffered modes"""
    POST_INFERENCE = "post_inference"
    """Evaluated after a complete result exists, before it is returned"""
