"""Transitional tool-call extraction from plain text.

Parses LLM text output that contains tool-call JSON in the format
emitted by PromptFormatter's tool instruction:

    {"tool_call": {"name": "function_name", "arguments": {...}}}

This is a tactical bridge — long-term, tool-call parsing moves to a
response postprocessor layer above the adapter.
"""

from __future__ import annotations

import logging
import uuid

from octomil.grammar import extract_json
from octomil.runtime.core.types import RuntimeToolCall

logger = logging.getLogger(__name__)


def extract_tool_call_from_text(
    text: str,
    declared_tools: list[str] | None = None,
) -> RuntimeToolCall | None:
    """Extract a tool call from model text output.

    Returns a RuntimeToolCall if the text contains valid tool-call JSON,
    or None on any failure (malformed JSON, unknown tool, wrong shape).
    Never raises.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # Fast path: skip parsing if text doesn't look like it could contain tool JSON
    if not stripped.startswith("{") and '"tool_call"' not in stripped:
        return None

    try:
        obj = extract_json(stripped)
        if obj is None:
            logger.debug("tool_parser: no JSON extracted from text")
            return None

        tool_call = obj.get("tool_call")
        if not isinstance(tool_call, dict):
            logger.debug("tool_parser: missing or invalid 'tool_call' key")
            return None

        name = tool_call.get("name")
        if not isinstance(name, str) or not name:
            logger.debug("tool_parser: missing or invalid tool name")
            return None

        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            logger.debug("tool_parser: arguments is not a dict")
            return None

        # Validate against declared tools if provided
        if declared_tools is not None and name not in declared_tools:
            logger.debug("tool_parser: unknown tool '%s' (declared: %s)", name, declared_tools)
            return None

        return RuntimeToolCall(
            id=f"call_{uuid.uuid4().hex[:16]}",
            name=name,
            arguments=_serialize_arguments(arguments),
        )
    except Exception:
        logger.debug("tool_parser: unexpected error during extraction", exc_info=True)
        return None


def _serialize_arguments(arguments: dict) -> str:
    """Serialize arguments dict to JSON string."""
    import json

    return json.dumps(arguments)
