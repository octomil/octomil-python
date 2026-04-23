"""Request normalization — input validation, message conversion, request preprocessing.

Converts Layer 2 ResponseRequest / InputItems into Layer 1 RuntimeMessages
and RuntimeRequests.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional, Union

from octomil._generated.message_role import MessageRole
from octomil.model_ref import ModelRef, _ModelRefCapability, _ModelRefId
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeToolCall,
    RuntimeToolDef,
)

from .types import (
    AssistantInput,
    AudioContent,
    FileContent,
    ImageContent,
    InputItem,
    JsonSchemaFormat,
    ResponseFormat,
    ResponseRequest,
    SystemInput,
    TextContent,
    TextOutput,
    ToolResultInput,
    UserInput,
    system_input,
    text_input,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model ID normalization
# ---------------------------------------------------------------------------


def _model_id_str(model: Union[str, ModelRef]) -> str:
    """Normalize a model ref to a plain string ID."""
    if isinstance(model, _ModelRefId):
        return model.model_id
    if isinstance(model, _ModelRefCapability):
        return model.capability.value
    return str(model)


# ---------------------------------------------------------------------------
# Previous response merging
# ---------------------------------------------------------------------------


def apply_previous_response(
    request: ResponseRequest,
    response_cache: dict[str, Any],
) -> ResponseRequest:
    """Prepend previous response output as assistant context."""
    if not request.previous_response_id:
        return request
    prev = response_cache.get(request.previous_response_id)
    if prev is None:
        return request

    assistant_text = "".join(item.text for item in prev.output if isinstance(item, TextOutput))
    assistant_item = AssistantInput(content=[TextContent(text=assistant_text)] if assistant_text else None)

    input_items: list[InputItem]
    if isinstance(request.input, str):
        input_items = [text_input(request.input)]
    else:
        input_items = list(request.input)

    return ResponseRequest(
        model=request.model,
        input=[assistant_item] + input_items,
        tools=request.tools,
        tool_choice=request.tool_choice,
        response_format=request.response_format,
        stream=request.stream,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        metadata=request.metadata,
        instructions=request.instructions,
    )


# ---------------------------------------------------------------------------
# RuntimeRequest building
# ---------------------------------------------------------------------------


def build_runtime_request(request: ResponseRequest) -> RuntimeRequest:
    """Build a RuntimeRequest from a ResponseRequest."""
    input_items: list[InputItem]
    if isinstance(request.input, str):
        input_items = [text_input(request.input)]
    else:
        input_items = list(request.input)

    if request.instructions:
        input_items = [system_input(request.instructions)] + input_items

    messages = _input_items_to_messages(input_items)

    tool_defs: Optional[list[RuntimeToolDef]] = None
    if request.tools:
        tool_defs = []
        for t in request.tools:
            fn = t.get("function", t)
            schema = fn.get("input_schema") or fn.get("parameters")
            tool_defs.append(
                RuntimeToolDef(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters_schema=json.dumps(schema) if schema else None,
                )
            )

    json_schema: Optional[str] = None
    if isinstance(request.response_format, JsonSchemaFormat):
        json_schema = request.response_format.schema
    elif request.response_format == ResponseFormat.JSON_OBJECT:
        json_schema = "{}"

    return RuntimeRequest(
        messages=messages,
        generation_config=GenerationConfig(
            max_tokens=request.max_output_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 1.0,
            stop=request.stop,
        ),
        tool_definitions=tool_defs,
        json_schema=json_schema,
        model=request.model,
    )


# ---------------------------------------------------------------------------
# Layer 2 -> Layer 1 message conversion
# ---------------------------------------------------------------------------


def _input_items_to_messages(input_items: list[InputItem]) -> list[RuntimeMessage]:
    """Convert Layer 2 InputItems to Layer 1 RuntimeMessages."""
    messages: list[RuntimeMessage] = []
    for item in input_items:
        if isinstance(item, SystemInput):
            messages.append(
                RuntimeMessage(
                    role=MessageRole.SYSTEM,
                    parts=[RuntimeContentPart.text_part(item.content)],
                )
            )
        elif isinstance(item, UserInput):
            parts = [_resolve_content_part(p) for p in item.content]
            messages.append(RuntimeMessage(role=MessageRole.USER, parts=parts))
        elif isinstance(item, AssistantInput):
            asst_parts: list[RuntimeContentPart] = []
            if item.content:
                for p in item.content:
                    if isinstance(p, TextContent):
                        asst_parts.append(RuntimeContentPart.text_part(p.text))
            rt_tool_calls: Optional[list[RuntimeToolCall]] = None
            if item.tool_calls:
                rt_tool_calls = [
                    RuntimeToolCall(id=call.id, name=call.name, arguments=call.arguments) for call in item.tool_calls
                ]
            if not asst_parts:
                asst_parts = [RuntimeContentPart.text_part("")]
            messages.append(
                RuntimeMessage(
                    role=MessageRole.ASSISTANT,
                    parts=asst_parts,
                    tool_calls=rt_tool_calls,
                )
            )
        elif isinstance(item, ToolResultInput):
            messages.append(
                RuntimeMessage(
                    role=MessageRole.TOOL,
                    parts=[RuntimeContentPart.text_part(item.content)],
                    tool_call_id=item.tool_call_id,
                )
            )
    return messages


def _resolve_content_part(part: object) -> RuntimeContentPart:
    """Convert a Layer 2 ContentPart to a Layer 1 RuntimeContentPart."""
    if isinstance(part, TextContent):
        return RuntimeContentPart.text_part(part.text)
    if isinstance(part, ImageContent):
        if part.data:
            raw = base64.b64decode(part.data)
            return RuntimeContentPart.image_part(raw, part.media_type or "image/png")
        logger.warning("ImageContent without data")
        return RuntimeContentPart.text_part("[image: unresolved]")
    if isinstance(part, AudioContent):
        raw = base64.b64decode(part.data)
        return RuntimeContentPart.audio_part(raw, part.media_type)
    if isinstance(part, FileContent):
        mt = part.media_type.lower()
        raw = base64.b64decode(part.data)
        if mt.startswith("image/"):
            return RuntimeContentPart.image_part(raw, part.media_type)
        if mt.startswith("audio/"):
            return RuntimeContentPart.audio_part(raw, part.media_type)
        if mt.startswith("video/"):
            return RuntimeContentPart.video_part(raw, part.media_type)
        raise ValueError(
            f"Cannot resolve FileContent with mediaType '{part.media_type}' "
            f"to a runtime content part. Supported prefixes: image/*, audio/*, video/*"
        )
    raise TypeError(f"Unknown content part type: {type(part)}")
