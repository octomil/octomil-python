"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelFeature(str, Enum):
    STREAMING = "streaming"
    """Real-time incremental output (token-by-token or chunk-by-chunk)"""
    BATCH = "batch"
    """Full input processed, full output returned"""
    FUNCTION_CALLING = "function_calling"
    """Tool use / function calling support"""
    STRUCTURED_OUTPUT = "structured_output"
    """JSON schema constrained output"""
