"""Octomil Responses API — Layers 1-3 for structured on-device inference."""

from __future__ import annotations

# Re-export runtime and tools subpackages
from . import runtime, tools
from .prompt_formatter import PromptFormatter
from .responses import OctomilResponses
from .types import (
    ContentPart,
    InputItem,
    OutputItem,
    Response,
    ResponseFormat,
    ResponseRequest,
    ResponseStreamEvent,
    ResponseToolCall,
    ResponseUsage,
    ToolChoice,
)

__all__ = [
    "ContentPart",
    "InputItem",
    "OctomilResponses",
    "OutputItem",
    "PromptFormatter",
    "Response",
    "ResponseFormat",
    "ResponseRequest",
    "ResponseStreamEvent",
    "ResponseToolCall",
    "ResponseUsage",
    "ToolChoice",
    "runtime",
    "tools",
]
