"""JSON-mode validation and repair helpers for OpenAI-compatible serve."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..errors import OctomilError, OctomilErrorCode


@dataclass(frozen=True)
class JsonModeConfig:
    """Resolved JSON response-format constraints."""

    schema: Optional[dict[str, Any]] = None
    strict: bool = False
    require_object: bool = True


@dataclass(frozen=True)
class JsonValidationResult:
    """Validated JSON text and the status surfaced to clients."""

    text: str
    status: str


_FULL_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z", re.DOTALL | re.IGNORECASE)


def resolve_json_mode_config(response_format: Optional[dict[str, Any]]) -> JsonModeConfig:
    """Resolve schema and strict flags from an OpenAI-style response_format."""
    if not response_format:
        return JsonModeConfig()

    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        wrapper = response_format.get("json_schema") or response_format.get("schema") or {}
        if not isinstance(wrapper, dict):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="response_format.json_schema must be an object.",
            )
        schema = wrapper.get("schema", wrapper)
        if not isinstance(schema, dict):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="response_format.json_schema.schema must be an object.",
            )
        _validate_schema_definition(schema)
        return JsonModeConfig(
            schema=schema,
            strict=bool(wrapper.get("strict", response_format.get("strict", False))),
        )

    return JsonModeConfig(strict=bool(response_format.get("strict", False)))


def coerce_json_mode_output(text: str, config: JsonModeConfig) -> JsonValidationResult:
    """Validate JSON-mode output, repairing only contract-safe wrappers.

    Non-strict mode repairs common local-model wrappers:
    - a full fenced JSON code block
    - prose with one extractable JSON object

    Strict mode accepts bare JSON or a full fenced JSON block only. It does
    not extract an object from surrounding prose because that can silently drop
    content in structured extraction workloads.
    """
    parsed = _loads_json_object(text, require_object=config.require_object)
    if parsed is not None:
        _validate_against_schema(parsed, config.schema)
        return JsonValidationResult(text=json.dumps(parsed), status="valid")

    fenced = _extract_full_fence(text)
    if fenced is not None:
        parsed = _loads_json_object(fenced, require_object=config.require_object)
        if parsed is not None:
            _validate_against_schema(parsed, config.schema)
            return JsonValidationResult(text=json.dumps(parsed), status="repaired_fenced")

    if not config.strict:
        from ..grammar import extract_json

        extracted = extract_json(text)
        if extracted is not None:
            _validate_against_schema(extracted, config.schema)
            return JsonValidationResult(text=json.dumps(extracted), status="repaired_extracted")

    raise OctomilError(
        code=OctomilErrorCode.INFERENCE_FAILED,
        message=(
            "JSON mode validation failed: backend returned content that could not be parsed "
            "into a valid JSON object"
            + (" matching the requested schema" if config.schema else "")
            + ". Try a larger model, lower temperature, or use a grammar/schema-capable backend."
        ),
    )


def _loads_json_object(text: str, *, require_object: bool) -> Any | None:
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return None
    if require_object and not isinstance(parsed, dict):
        return None
    return parsed


def _extract_full_fence(text: str) -> str | None:
    if not text:
        return None
    match = _FULL_FENCE_RE.match(text)
    if not match:
        return None
    return match.group(1).strip()


def _validate_schema_definition(schema: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError:
        return

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"Invalid JSON schema in response_format: {exc.message}",
        ) from exc


def _validate_against_schema(parsed: Any, schema: Optional[dict[str, Any]]) -> None:
    if schema is None:
        return
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError:
        return

    try:
        Draft202012Validator(schema).validate(parsed)
    except ValidationError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INFERENCE_FAILED,
            message=f"JSON schema validation failed: {exc.message}",
        ) from exc
