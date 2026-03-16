"""Tool-call extraction from model text output.

Expected format:
  {"type": "tool_call", "name": "fn", "arguments": {...}}

The parser requires the ENTIRE response to be a single JSON object
(after whitespace trimming and code-fence stripping). This prevents
false positives from models that mention JSON in prose.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from octomil.runtime.core.types import RuntimeToolCall

logger = logging.getLogger(__name__)


@dataclass
class ToolCallParseResult:
    """Result of tool-call extraction with validation metadata."""

    tool_call: Optional[RuntimeToolCall] = None
    schema_valid: Optional[bool] = None
    schema_errors: list[str] = field(default_factory=list)


def extract_tool_call_from_text(
    text: str,
    declared_tools: list[str] | None = None,
    tool_schemas: dict[str, dict] | None = None,
) -> RuntimeToolCall | None:
    """Extract a tool call from model text output.

    Returns a RuntimeToolCall if the full response is valid tool-call JSON,
    or None on any failure. Never raises.
    """
    result = extract_tool_call_with_validation(text, declared_tools, tool_schemas)
    return result.tool_call


def extract_tool_call_with_validation(
    text: str,
    declared_tools: list[str] | None = None,
    tool_schemas: dict[str, dict] | None = None,
) -> ToolCallParseResult:
    """Extract a tool call with schema validation metadata.

    Returns ToolCallParseResult with tool_call set if extraction succeeded,
    plus schema_valid/schema_errors if schemas were provided.
    """
    if not text or not text.strip():
        return ToolCallParseResult()

    stripped = text.strip()

    # Strip code fences if present
    stripped = _strip_code_fence(stripped)

    # Full-response-only: entire output must be a JSON object
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return ToolCallParseResult()

    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        logger.debug("tool_parser: invalid JSON in full response")
        return ToolCallParseResult()

    if not isinstance(obj, dict):
        return ToolCallParseResult()

    # Required format: {"type": "tool_call", "name": "...", "arguments": {...}}
    if obj.get("type") != "tool_call":
        logger.debug("tool_parser: missing or wrong 'type' field (expected 'tool_call')")
        return ToolCallParseResult()

    name = obj.get("name")
    arguments = obj.get("arguments")

    if not isinstance(name, str) or not name:
        logger.debug("tool_parser: missing or invalid tool name")
        return ToolCallParseResult()

    if not isinstance(arguments, dict):
        logger.debug("tool_parser: arguments is not a dict")
        return ToolCallParseResult()

    # Validate against declared tools if provided
    if declared_tools is not None and name not in declared_tools:
        logger.debug("tool_parser: unknown tool '%s' (declared: %s)", name, declared_tools)
        return ToolCallParseResult()

    tool_call = RuntimeToolCall(
        id=f"call_{uuid.uuid4().hex[:16]}",
        name=name,
        arguments=json.dumps(arguments),
    )

    # Schema validation if schemas provided
    schema_valid = None
    schema_errors: list[str] = []
    if tool_schemas and name in tool_schemas:
        schema_valid, schema_errors = _validate_arguments(arguments, tool_schemas[name])

    return ToolCallParseResult(
        tool_call=tool_call,
        schema_valid=schema_valid,
        schema_errors=schema_errors,
    )


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fences if the entire text is fenced."""
    lines = text.split("\n")
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _validate_arguments(arguments: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate arguments against a JSON schema. Returns (valid, errors)."""
    try:
        import jsonschema

        jsonschema.validate(instance=arguments, schema=schema)
        return True, []
    except ImportError:
        # jsonschema not installed — skip validation
        return True, []
    except Exception as e:
        return False, [str(e)]
