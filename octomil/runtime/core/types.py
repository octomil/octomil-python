"""Runtime layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from octomil._generated.message_role import MessageRole
from octomil._generated.modality import Modality


class ToolCallTier(Enum):
    """How an inference engine handles tool/function calling.

    Matches the ToolCallTier contract enum (since 1.3.0).
    """

    NONE = "NONE"
    TEXT_JSON = "TEXT_JSON"
    GRAMMAR = "GRAMMAR"
    NATIVE = "NATIVE"


@dataclass
class RuntimeContentPart:
    """A single content part in a runtime message.

    Uses Modality enum values as type discriminator.
    Media parts hold raw decoded bytes (not base64).
    """

    type: Modality
    text: Optional[str] = None
    data: Optional[bytes] = None
    media_type: Optional[str] = None

    @staticmethod
    def text_part(text: str) -> RuntimeContentPart:
        return RuntimeContentPart(type=Modality.TEXT, text=text)

    @staticmethod
    def image_part(data: bytes, media_type: str) -> RuntimeContentPart:
        return RuntimeContentPart(type=Modality.IMAGE, data=data, media_type=media_type)

    @staticmethod
    def audio_part(data: bytes, media_type: str) -> RuntimeContentPart:
        return RuntimeContentPart(type=Modality.AUDIO, data=data, media_type=media_type)

    @staticmethod
    def video_part(data: bytes, media_type: str) -> RuntimeContentPart:
        return RuntimeContentPart(type=Modality.VIDEO, data=data, media_type=media_type)


@dataclass
class RuntimeMessage:
    """A single message in a runtime conversation.

    Parts are ordered — order is significant and MUST be preserved.
    """

    role: MessageRole
    parts: list[RuntimeContentPart]
    tool_calls: Optional[list[RuntimeToolCall]] = None
    tool_call_id: Optional[str] = None


@dataclass
class GenerationConfig:
    """Generation parameters for inference."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[list[str]] = None


@dataclass
class RuntimeCapabilities:
    tool_call_tier: ToolCallTier = ToolCallTier.NONE
    supports_structured_output: bool = False
    supports_multimodal_input: bool = False
    supports_streaming: bool = True
    max_context_length: Optional[int] = None
    supported_families: frozenset[str] = field(default_factory=frozenset)
    input_modalities: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))
    max_media_parts_per_message: Optional[int] = None
    supports_historical_media: bool = False
    supports_interleaved_content: bool = False

    @property
    def supports_tool_calls(self) -> bool:
        """True if tool calling is available at any tier."""
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
    """Runtime inference request.

    Contains structured multimodal messages with resolved binary data.
    """

    messages: list[RuntimeMessage]
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
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
