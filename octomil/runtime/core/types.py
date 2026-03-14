"""Runtime layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeCapabilities:
    supports_tool_calls: bool = False
    supports_text_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_multimodal_input: bool = False
    supports_streaming: bool = True
    max_context_length: Optional[int] = None
    supported_families: frozenset[str] = field(default_factory=frozenset)


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
