"""Canonical ChatML renderer for text-only engines.

Implements the rendering spec from runtime_chatml_rendering.md.
Used ONLY by engine adapters wrapping text-only backends.
"""

from __future__ import annotations

from typing import Optional

from octomil._generated.modality import Modality

from .types import (
    RuntimeMessage,
    RuntimeRequest,
    RuntimeToolDef,
)

# Placeholder tokens for media parts
_MEDIA_PLACEHOLDERS: dict[Modality, str] = {
    Modality.IMAGE: "[image]",
    Modality.AUDIO: "[audio]",
    Modality.VIDEO: "[video]",
}


def render_chatml(
    request: RuntimeRequest,
    *,
    tool_choice: str = "auto",
    specific_tool_name: Optional[str] = None,
) -> str:
    """Render a RuntimeRequest to a ChatML prompt string.

    Args:
        request: The runtime request to render.
        tool_choice: "auto", "none", "required", or "specific".
        specific_tool_name: Required when tool_choice is "specific".

    Returns:
        Deterministic ChatML string matching the canonical rendering spec.
    """
    sb: list[str] = []

    # Tool definitions block
    if request.tool_definitions and tool_choice != "none":
        sb.append(_render_tool_block(request.tool_definitions, tool_choice, specific_tool_name))

    # Messages
    for msg in request.messages:
        sb.append(_render_message(msg))

    # Generation prompt
    sb.append("<|assistant|>\n")
    return "".join(sb)


def _render_tool_block(
    tool_defs: list[RuntimeToolDef],
    tool_choice: str,
    specific_tool_name: Optional[str],
) -> str:
    sb: list[str] = []
    sb.append("<|system|>\nYou have access to the following tools:\n\n")
    for td in tool_defs:
        sb.append(f"Function: {td.name}\n")
        sb.append(f"Description: {td.description}\n")
        if td.parameters_schema:
            sb.append(f"Parameters: {td.parameters_schema}\n")
        sb.append("\n")
    sb.append(
        "To use a tool, respond with ONLY this JSON and nothing else:\n"
        '{"type": "tool_call", "name": "function_name", "arguments": {...}}\n'
        "If you do not need a tool, respond with normal text.\n"
    )
    if tool_choice == "required":
        sb.append("You MUST use one of the available tools.\n")
    elif tool_choice == "specific" and specific_tool_name:
        sb.append(f"You MUST use the tool: {specific_tool_name}\n")
    sb.append("\n")
    return "".join(sb)


def _render_message(msg: RuntimeMessage) -> str:
    sb: list[str] = []
    sb.append(f"<|{msg.role.value}|>\n")

    prev_was_text = False
    for part in msg.parts:
        if part.type == Modality.TEXT:
            if prev_was_text:
                sb.append("\n")
            sb.append(part.text or "")
            prev_was_text = True
        else:
            placeholder = _MEDIA_PLACEHOLDERS.get(part.type, "")
            sb.append(placeholder)
            prev_was_text = False

    sb.append("\n")
    return "".join(sb)
