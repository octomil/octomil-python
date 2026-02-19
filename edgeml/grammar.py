"""
Structured / constrained decoding support.

Provides JSON Schema to GBNF grammar conversion for use with llama.cpp,
plus JSON validation and extraction utilities.

GBNF (GGML BNF) is the grammar format used by llama.cpp to constrain
model output to match specific patterns. This module converts JSON schemas
into GBNF grammars so the model can only produce valid, schema-conformant JSON.

Usage::

    from edgeml.grammar import json_schema_to_gbnf, json_mode_grammar

    # Any valid JSON
    grammar = json_mode_grammar()

    # Schema-constrained JSON
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    grammar = json_schema_to_gbnf(schema)
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Generic JSON GBNF grammar
# ---------------------------------------------------------------------------

_JSON_GRAMMAR = r"""root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"

object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" .)* "\"" ws
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)? ws
ws     ::= [ \t\n]*
"""


def json_mode_grammar() -> str:
    """Return a GBNF grammar that accepts any valid JSON object.

    This is the grammar used when ``response_format={"type": "json_object"}``
    is specified without a schema.
    """
    return _JSON_GRAMMAR.strip() + "\n"


# ---------------------------------------------------------------------------
# JSON Schema -> GBNF conversion
# ---------------------------------------------------------------------------


class _GBNFBuilder:
    """Incrementally builds a GBNF grammar from a JSON schema.

    Each unique sub-schema gets its own named rule.  Primitive types share
    global rules (``string``, ``number``, etc.).
    """

    def __init__(self) -> None:
        self._rules: dict[str, str] = {}
        self._counter: int = 0
        # Add shared primitives
        self._rules["ws"] = "[ \\t\\n]*"
        self._rules["string"] = '"\\""  ([^"\\\\] | "\\\\" .)* "\\""  ws'
        self._rules["number"] = '"-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)? ws'
        self._rules["integer"] = '"-"? [0-9]+ ws'
        self._rules["boolean"] = '("true" | "false") ws'
        self._rules["null"] = '"null" ws'
        self._rules["value"] = "object | array | string | number | boolean | null"
        self._rules["object"] = (
            '"{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws'
        )
        self._rules["array"] = '"[" ws (value ("," ws value)*)? "]" ws'

    def _fresh_name(self, hint: str = "rule") -> str:
        self._counter += 1
        # Sanitise hint to valid GBNF rule name chars
        safe = re.sub(r"[^a-zA-Z0-9_]", "-", hint)
        return f"{safe}-{self._counter}"

    def build(self, schema: dict[str, Any]) -> str:
        """Convert *schema* to a full GBNF grammar string."""
        root_rule = self._schema_to_rule(schema, "root")
        # Ensure root is first
        lines = [f"root ::= {root_rule}"]
        for name, body in self._rules.items():
            if name == "root":
                continue
            lines.append(f"{name} ::= {body}")
        return "\n".join(lines) + "\n"

    # -- recursive schema handling ------------------------------------------

    def _schema_to_rule(self, schema: dict[str, Any], hint: str = "val") -> str:
        """Return a GBNF expression (or rule name) for *schema*."""

        # enum — fixed set of literal values
        if "enum" in schema:
            return self._enum_rule(schema["enum"], hint)

        schema_type = schema.get("type")

        if schema_type == "string":
            return "string"
        if schema_type == "integer":
            return "integer"
        if schema_type == "number":
            return "number"
        if schema_type == "boolean":
            return "boolean"
        if schema_type == "null":
            return "null"
        if schema_type == "array":
            return self._array_rule(schema, hint)
        if schema_type == "object":
            return self._object_rule(schema, hint)

        # No type specified — accept any JSON value
        return "value"

    def _enum_rule(self, values: list[Any], hint: str) -> str:
        """Create a rule that matches one of the literal enum values."""
        parts: list[str] = []
        for v in values:
            if isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                parts.append(f'"\\"{escaped}\\""')
            elif isinstance(v, bool):
                parts.append(f'"{str(v).lower()}"')
            elif isinstance(v, int):
                parts.append(f'"{v}"')
            elif isinstance(v, float):
                parts.append(f'"{v}"')
            elif v is None:
                parts.append('"null"')
        name = self._fresh_name(hint)
        self._rules[name] = " | ".join(parts) + " ws" if parts else '"null" ws'
        return name

    def _array_rule(self, schema: dict[str, Any], hint: str) -> str:
        """Create a rule for a JSON array, optionally with typed items."""
        items_schema = schema.get("items")
        if items_schema:
            item_ref = self._schema_to_rule(items_schema, f"{hint}-item")
            name = self._fresh_name(hint)
            self._rules[name] = f'"[" ws ({item_ref} ("," ws {item_ref})*)? "]" ws'
            return name
        return "array"

    def _object_rule(self, schema: dict[str, Any], hint: str) -> str:
        """Create a rule for a JSON object with known properties."""
        properties = schema.get("properties")
        if not properties:
            return "object"

        required = set(schema.get("required", []))
        all_keys = list(properties.keys())

        # Build a field rule for each property
        field_parts: list[str] = []
        optional_parts: list[str] = []

        for key in all_keys:
            prop_schema = properties[key]
            val_ref = self._schema_to_rule(prop_schema, f"{hint}-{key}")
            escaped_key = key.replace("\\", "\\\\").replace('"', '\\"')
            field_expr = f'"\\"{escaped_key}\\"" ":" ws {val_ref}'

            if key in required:
                field_parts.append(field_expr)
            else:
                optional_parts.append(field_expr)

        name = self._fresh_name(hint)

        if not field_parts and not optional_parts:
            # No properties defined at all — generic object
            return "object"

        # Strategy: required fields in order, then optional fields
        # This is a simplification — full permutation support would
        # blow up the grammar exponentially. For structured decoding
        # the model is expected to produce fields in the declared order.
        parts_expr: list[str] = []
        for i, fp in enumerate(field_parts):
            if i > 0:
                parts_expr.append('"," ws')
            parts_expr.append(fp)

        for ofp in optional_parts:
            opt_name = self._fresh_name(f"{hint}-opt")
            self._rules[opt_name] = f'"," ws {ofp}'
            if parts_expr:
                parts_expr.append(f"({opt_name})?")
            else:
                # All fields are optional — first one gets special treatment
                parts_expr.append(f"({ofp})?")

        body = " ".join(parts_expr)
        self._rules[name] = f'"{{"  ws {body} "}}" ws'
        return name


def json_schema_to_gbnf(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema dict to a GBNF grammar string.

    Supports the following JSON Schema features:

    * Primitive types: ``string``, ``number``, ``integer``, ``boolean``, ``null``
    * ``object`` with ``properties`` and ``required``
    * ``array`` with ``items``
    * ``enum`` with literal values
    * Nested objects and arrays

    Parameters
    ----------
    schema:
        A JSON Schema dict (e.g. ``{"type": "object", "properties": {...}}``).

    Returns
    -------
    str
        A GBNF grammar string suitable for passing to llama.cpp.
    """
    builder = _GBNFBuilder()
    return builder.build(schema)


# ---------------------------------------------------------------------------
# JSON validation and extraction
# ---------------------------------------------------------------------------


def validate_json_output(text: str) -> bool:
    """Check whether *text* is valid JSON.

    Returns ``True`` if *text* can be parsed by ``json.loads``.
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract a JSON object from *text*, tolerating wrapping.

    Handles common LLM output patterns:
    * Bare JSON: ``{"key": "value"}``
    * Fenced code blocks: ````json\\n{...}\\n` `` ``
    * Leading/trailing prose around JSON

    Returns the parsed ``dict`` or ``None`` if no valid JSON is found.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # 1. Try direct parse
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 2. Try extracting from fenced code blocks: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    for match in fence_pattern.finditer(stripped):
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # 3. Try finding the outermost { ... } pair
    start = stripped.find("{")
    if start == -1:
        return None

    # Find matching closing brace (handle nesting)
    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(stripped)):
        ch = stripped[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    candidate = stripped[start : end + 1]
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# System prompt helpers for MLX JSON mode
# ---------------------------------------------------------------------------

_JSON_SYSTEM_PROMPT = (
    "You must respond with valid JSON only. "
    "Do not include any text, explanation, or markdown outside the JSON object. "
    "Output a single JSON object."
)

_JSON_SCHEMA_SYSTEM_PROMPT_TEMPLATE = (
    "You must respond with valid JSON that conforms to this schema:\n"
    "{schema}\n"
    "Do not include any text, explanation, or markdown outside the JSON object. "
    "Output a single JSON object."
)


def json_system_prompt(schema: Optional[dict[str, Any]] = None) -> str:
    """Return a system prompt instructing the model to output JSON.

    If *schema* is provided, the prompt includes the schema definition.
    Used as a fallback for backends that don't support grammar-based
    constrained decoding (e.g. MLX).
    """
    if schema:
        return _JSON_SCHEMA_SYSTEM_PROMPT_TEMPLATE.format(
            schema=json.dumps(schema, indent=2)
        )
    return _JSON_SYSTEM_PROMPT
