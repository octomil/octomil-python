"""Response API data types (Layer 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# -- Content Parts (multimodal input) --


@dataclass
class TextContent:
    text: str


@dataclass
class ImageContent:
    data: Optional[str] = None
    url: Optional[str] = None
    media_type: Optional[str] = None
    detail: str = "auto"


@dataclass
class AudioContent:
    data: str
    media_type: str


@dataclass
class FileContent:
    data: str
    media_type: str
    filename: Optional[str] = None


# Union type for content parts
ContentPart = TextContent | ImageContent | AudioContent | FileContent


# -- Input Items --


@dataclass
class SystemInput:
    content: str


@dataclass
class UserInput:
    content: list[ContentPart]


@dataclass
class AssistantInput:
    content: Optional[list[ContentPart]] = None
    tool_calls: Optional[list[ResponseToolCall]] = None


@dataclass
class ToolResultInput:
    tool_call_id: str
    content: str


InputItem = SystemInput | UserInput | AssistantInput | ToolResultInput


# -- Tool Call --


@dataclass
class ResponseToolCall:
    id: str
    name: str
    arguments: str


# -- Tool Choice --


class ToolChoice(Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass
class SpecificToolChoice:
    name: str


# -- Response Format --


class ResponseFormat(Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


@dataclass
class JsonSchemaFormat:
    schema: str


# -- Request --


@dataclass
class ResponseRequest:
    model: str
    input: list[InputItem] | str
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: ToolChoice | SpecificToolChoice = ToolChoice.AUTO
    response_format: ResponseFormat | JsonSchemaFormat = ResponseFormat.TEXT
    stream: bool = False
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[list[str]] = None
    metadata: Optional[dict[str, str]] = None
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None

    @staticmethod
    def text(model: str, text: str, **kwargs: Any) -> ResponseRequest:
        """Create a ResponseRequest with a single text input."""
        return ResponseRequest(model=model, input=[text_input(text)], **kwargs)


# -- Output Items --


@dataclass
class TextOutput:
    text: str


@dataclass
class ToolCallOutput:
    tool_call: ResponseToolCall


@dataclass
class JsonOutput:
    json: str


OutputItem = TextOutput | ToolCallOutput | JsonOutput


# -- Usage --


@dataclass
class ResponseUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# -- Response --


@dataclass
class Response:
    id: str
    model: str
    output: list[OutputItem]
    finish_reason: str
    usage: Optional[ResponseUsage] = None


# -- Stream Events --


@dataclass
class TextDeltaEvent:
    delta: str


@dataclass
class ToolCallDeltaEvent:
    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_delta: Optional[str] = None


@dataclass
class DoneEvent:
    response: Response


@dataclass
class ErrorEvent:
    error: Exception


ResponseStreamEvent = TextDeltaEvent | ToolCallDeltaEvent | DoneEvent | ErrorEvent


# -- Convenience constructors --


def text_input(value: str) -> UserInput:
    """Create a user message with a single text part."""
    return UserInput(content=[TextContent(text=value)])


def system_input(value: str) -> SystemInput:
    """Create a system message."""
    return SystemInput(content=value)
