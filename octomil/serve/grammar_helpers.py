"""Grammar resolution and JSON system prompt injection helpers."""

from __future__ import annotations

from typing import Any, Optional

from ..errors import OctomilError, OctomilErrorCode
from .models import ChatCompletionBody


def _reject_explicit_grammar_on_non_grammar_backend(
    backend_name: str,
    grammar_str: Optional[str],
    is_json: bool,
    uses_grammar_natively: bool,
) -> None:
    """Cutover follow-up #71 (R1 Codex): raise UNSUPPORTED_MODALITY when the
    caller supplied an explicit GBNF grammar (``body.grammar``) and the
    backend's ``capabilities.grammar_supported`` is False.

    Pre-fix the serve layer silently dropped the grammar string before
    forwarding to the backend, so the request returned 200 OK with text
    that ignored the requested constraint. JSON-mode (``is_json=True``,
    grammar derived from ``response_format``) is unaffected — system-
    prompt fallback handles those requests.
    """
    if grammar_str and not is_json and not uses_grammar_natively:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                f"Backend '{backend_name}' does not support GBNF grammar "
                f"(capabilities.grammar_supported=False). Use "
                f"response_format=json_object/json_schema for structured "
                f"output, or route to a grammar-capable backend."
            ),
        )


def _resolve_grammar(body: ChatCompletionBody, default_json_mode: bool = False) -> tuple[Optional[str], bool]:
    """Determine the GBNF grammar string and json_mode flag from a request.

    Returns (grammar_string_or_None, is_json_mode).
    """
    from ..grammar import json_mode_grammar, json_schema_to_gbnf

    # Explicit grammar takes precedence
    if body.grammar:
        return body.grammar, False

    rf = body.response_format
    if rf is None and default_json_mode:
        rf = {"type": "json_object"}

    if rf is None:
        return None, False

    fmt_type = rf.get("type")
    if fmt_type == "json_object":
        return json_mode_grammar(), True
    if fmt_type == "json_schema":
        schema = rf.get("json_schema") or rf.get("schema")
        if schema:
            # The OpenAI API wraps the actual schema under a "schema" key
            # inside json_schema. Handle both nesting levels.
            actual_schema = schema.get("schema", schema)
            return json_schema_to_gbnf(actual_schema), True
        return json_mode_grammar(), True

    return None, False


def _inject_json_system_prompt(
    messages: list[dict[str, Any]],
    schema: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Prepend a JSON-mode system prompt if one isn't already present."""
    from ..grammar import json_system_prompt

    # Don't double-inject
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        if "JSON" in existing or "json" in existing:
            return messages

    prompt = json_system_prompt(schema)
    return [{"role": "system", "content": prompt}] + list(messages)
