"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class FinishReason(str, Enum):
    STOP = "stop"
    """Natural end of generation (EOS token or stop sequence)"""
    TOOL_CALLS = "tool_calls"
    """Model emitted one or more tool calls"""
    LENGTH = "length"
    """Hit max_output_tokens limit"""
    CONTENT_FILTER = "content_filter"
    """Content was filtered by safety system"""
