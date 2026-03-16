"""Runtime layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ToolCallTier(Enum):
    """How an inference engine handles tool/function calling.

    Matches the ToolCallTier contract enum (since 1.3.0).
    """

    NONE = "NONE"
    TEXT_JSON = "TEXT_JSON"
    GRAMMAR = "GRAMMAR"
    NATIVE = "NATIVE"


@dataclass
class RuntimeCapabilities:
    tool_call_tier: ToolCallTier = ToolCallTier.NONE
    supports_structured_output: bool = False
    supports_multimodal_input: bool = False
    supports_streaming: bool = True
    max_context_length: Optional[int] = None
    supported_families: frozenset[str] = field(default_factory=frozenset)

    @property
    def supports_tool_calls(self) -> bool:
        """True if tool calling is available at any tier (backward compat)."""
        return self.tool_call_tier != ToolCallTier.NONE

    @property
    def supports_reliable_tool_calls(self) -> bool:
        """True if tool calling is guaranteed structurally valid (GRAMMAR/NATIVE)."""
        return self.tool_call_tier in (ToolCallTier.GRAMMAR, ToolCallTier.NATIVE)

    @property
    def supports_text_tool_calls(self) -> bool:
        """True if tool calls are extracted from model text output."""
        return self.tool_call_tier == ToolCallTier.TEXT_JSON


@dataclass
class RuntimeToolDef:
    name: str
    description: str
    parameters_schema: Optional[str] = None


@dataclass
class RuntimeRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    tool_definitions: Optional[list[RuntimeToolDef]] = None
    json_schema: Optional[str] = None


@dataclass
class RuntimeToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class RuntimeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class RuntimeResponse:
    text: str
    tool_calls: Optional[list[RuntimeToolCall]] = None
    finish_reason: str = "stop"
    usage: Optional[RuntimeUsage] = None
    raw_text: Optional[str] = None


@dataclass
class RuntimeToolCallDelta:
    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_delta: Optional[str] = None


@dataclass
class RuntimeChunk:
    text: Optional[str] = None
    tool_call_delta: Optional[RuntimeToolCallDelta] = None
    finish_reason: Optional[str] = None
    usage: Optional[RuntimeUsage] = None
