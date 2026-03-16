"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    """System prompt or instructions"""
    USER = "user"
    """User-provided input"""
    ASSISTANT = "assistant"
    """Model-generated response"""
    TOOL = "tool"
    """Tool execution result"""
