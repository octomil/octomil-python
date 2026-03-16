"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ToolCallTier(str, Enum):
    NONE = "NONE"
    """No tool-calling capability. Tool definitions are ignored."""
    TEXT_JSON = "TEXT_JSON"
    """Prompt-injected. SDK instructs model to emit structured JSON, extracts tool calls from raw text."""
    GRAMMAR = "GRAMMAR"
    """Constrained decoding guarantees valid JSON tool-call output."""
    NATIVE = "NATIVE"
    """Engine natively supports function calling at the API level."""
