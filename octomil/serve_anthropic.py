"""Anthropic Messages API translation layer for ``octomil serve``.

Translates Anthropic ``/v1/messages`` requests to the OpenAI-compatible
backend and returns Anthropic-format responses.  This enables
``octomil launch claude --model <local-model>`` to work with local models.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .serve import GenerationRequest


# ---------------------------------------------------------------------------
# Pydantic models for Anthropic Messages API
# ---------------------------------------------------------------------------


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str = ""


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, list[AnthropicContentBlock]]


class AnthropicMessagesBody(BaseModel):
    model: str = ""
    messages: list[AnthropicMessage] = Field(default_factory=list)
    max_tokens: int = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    system: Optional[str] = None
    stop_sequences: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def _anthropic_to_openai_messages(
    body: AnthropicMessagesBody,
) -> list[dict[str, str]]:
    """Convert Anthropic message format to OpenAI chat messages."""
    messages: list[dict[str, str]] = []

    if body.system:
        messages.append({"role": "system", "content": body.system})

    for msg in body.messages:
        if isinstance(msg.content, str):
            text = msg.content
        else:
            text = "".join(block.text for block in msg.content if block.type == "text")
        messages.append({"role": msg.role, "content": text})

    return messages


# ---------------------------------------------------------------------------
# Streaming adapter (Anthropic SSE format)
# ---------------------------------------------------------------------------


async def _stream_anthropic(
    state: Any,
    gen_req: "GenerationRequest",
    msg_id: str,
) -> AsyncIterator[str]:
    """Yield Anthropic-format SSE events from the backend stream."""
    assert state.backend is not None

    # message_start
    start_event = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": gen_req.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

    # content_block_start
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

    # Stream content deltas
    output_tokens = 0
    async for chunk in state.backend.generate_stream(gen_req):
        if chunk.text:
            output_tokens += 1
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk.text},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

    # content_block_stop
    block_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

    # message_delta
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_anthropic_routes(app: Any, state: Any) -> None:
    """Register ``/v1/messages`` endpoint on the given FastAPI app."""
    from fastapi.responses import StreamingResponse

    from .errors import OctomilError, OctomilErrorCode
    from .serve import GenerationRequest

    @app.post("/v1/messages")
    async def anthropic_messages(body: AnthropicMessagesBody) -> Any:
        if state.backend is None:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="Model not loaded")

        openai_messages = _anthropic_to_openai_messages(body)

        gen_req = GenerationRequest(
            model=body.model or state.model_name,
            messages=openai_messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature if body.temperature is not None else 0.7,
            top_p=body.top_p if body.top_p is not None else 1.0,
            stream=body.stream,
        )

        state.request_count += 1
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        if body.stream:
            return StreamingResponse(
                _stream_anthropic(state, gen_req, msg_id),
                media_type="text/event-stream",
            )

        text, metrics = state.backend.generate(gen_req)

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": gen_req.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.total_tokens,
            },
        }
