"""Response builder — constructing public Response objects from runtime results.

Converts Layer 1 RuntimeResponse into Layer 2 Response, including output
item construction, usage mapping, and ID generation.
"""

from __future__ import annotations

import uuid
from typing import Optional

from octomil.execution.route_metadata_mapper import RouteMetadata
from octomil.runtime.core.types import (
    RuntimeResponse as _RuntimeResponse,
)

from .types import (
    OutputItem,
    Response,
    ResponseToolCall,
    ResponseUsage,
    TextOutput,
    ToolCallOutput,
)

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Generate a unique response ID."""
    return f"resp_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------


def build_response(
    model: str,
    runtime_response: _RuntimeResponse,
    locality: Optional[str] = None,
    route: Optional[RouteMetadata] = None,
) -> Response:
    """Build a Response from a RuntimeResponse."""
    output: list[OutputItem] = []
    if runtime_response.text:
        output.append(TextOutput(text=runtime_response.text))
    if runtime_response.tool_calls:
        for call in runtime_response.tool_calls:
            output.append(
                ToolCallOutput(
                    tool_call=ResponseToolCall(
                        id=call.id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )
            )

    finish_reason = "tool_calls" if runtime_response.tool_calls else runtime_response.finish_reason
    usage = (
        ResponseUsage(
            prompt_tokens=runtime_response.usage.prompt_tokens,
            completion_tokens=runtime_response.usage.completion_tokens,
            total_tokens=runtime_response.usage.total_tokens,
        )
        if runtime_response.usage
        else None
    )

    return Response(
        id=generate_id(),
        model=model,
        output=output,
        finish_reason=finish_reason,
        usage=usage,
        locality=locality,
        route=route,
    )


# ---------------------------------------------------------------------------
# Streaming accumulation helpers
# ---------------------------------------------------------------------------


class ToolCallBuffer:
    """Accumulates streaming tool call deltas into a complete tool call."""

    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""
