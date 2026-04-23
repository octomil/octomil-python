"""Built-in output quality gate evaluators.

Each evaluator implements the ``OutputQualityEvaluator`` protocol and
returns an ``EvaluatorResult``. Evaluators run **in the SDK process** —
prompt/output content never leaves the caller's machine.

Built-in evaluators:

- ``JsonParseableEvaluator``   — checks that output parses as JSON.
- ``JsonSchemaEvaluator``      — validates output against a JSON Schema.
- ``ToolCallValidEvaluator``   — validates tool-call structure.
- ``RegexPredicateEvaluator``  — matches output against a regex pattern.
- ``SafetyPassedEvaluator``    — adapter stub for app-provided safety check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# EvaluatorResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluatorResult:
    """Privacy-safe result from an output quality evaluator.

    ``safe_metadata`` is sanitized by the forbidden-key filter before
    inclusion in telemetry — no prompt, output, or content fields may
    survive.
    """

    passed: bool
    score: float | None = None
    reason_code: str | None = None
    safe_metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# OutputQualityEvaluator protocol
# ---------------------------------------------------------------------------


class OutputQualityEvaluator(Protocol):
    """Protocol for post-inference output quality evaluation.

    Implementations receive the gate definition and the inference response.
    They MUST NOT upload or log prompt/output content.
    """

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        *,
        gate: dict[str, Any],
        response: Any,
    ) -> EvaluatorResult: ...


# ---------------------------------------------------------------------------
# JsonParseableEvaluator
# ---------------------------------------------------------------------------


class JsonParseableEvaluator:
    """Checks that the response text is valid JSON.

    Maps to gate code ``json_parseable``.
    """

    name = "json_parseable"

    def evaluate(self, *, gate: dict[str, Any], response: Any) -> EvaluatorResult:
        text = _extract_text(response)
        if text is None:
            return EvaluatorResult(
                passed=False,
                reason_code="no_text_content",
                safe_metadata={"evaluator_name": self.name},
            )
        try:
            json.loads(text)
            return EvaluatorResult(
                passed=True,
                safe_metadata={"evaluator_name": self.name},
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return EvaluatorResult(
                passed=False,
                reason_code="json_parse_error",
                safe_metadata={
                    "evaluator_name": self.name,
                    "error_type": type(exc).__name__,
                },
            )


# ---------------------------------------------------------------------------
# JsonSchemaEvaluator
# ---------------------------------------------------------------------------


class JsonSchemaEvaluator:
    """Validates the response text against a JSON Schema.

    The schema is taken from ``gate["config"]["schema"]`` (a dict) or
    provided at construction time.

    Maps to gate code ``schema_valid``.
    """

    name = "json_schema"

    def __init__(self, *, default_schema: dict[str, Any] | None = None) -> None:
        self._default_schema = default_schema

    def evaluate(self, *, gate: dict[str, Any], response: Any) -> EvaluatorResult:
        text = _extract_text(response)
        if text is None:
            return EvaluatorResult(
                passed=False,
                reason_code="no_text_content",
                safe_metadata={"evaluator_name": self.name},
            )

        schema = (gate.get("config") or {}).get("schema") or self._default_schema
        if schema is None:
            return EvaluatorResult(
                passed=False,
                reason_code="no_schema_configured",
                safe_metadata={"evaluator_name": self.name},
            )

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return EvaluatorResult(
                passed=False,
                reason_code="json_parse_error",
                safe_metadata={"evaluator_name": self.name},
            )

        try:
            import jsonschema  # type: ignore[import-untyped]

            jsonschema.validate(data, schema)
            return EvaluatorResult(
                passed=True,
                safe_metadata={"evaluator_name": self.name},
            )
        except ImportError:
            return EvaluatorResult(
                passed=False,
                reason_code="jsonschema_not_installed",
                safe_metadata={"evaluator_name": self.name},
            )
        except jsonschema.ValidationError as exc:
            return EvaluatorResult(
                passed=False,
                reason_code="schema_validation_error",
                safe_metadata={
                    "evaluator_name": self.name,
                    "validation_path": ".".join(str(p) for p in exc.absolute_path),
                },
            )


# ---------------------------------------------------------------------------
# ToolCallValidEvaluator
# ---------------------------------------------------------------------------


class ToolCallValidEvaluator:
    """Validates that tool calls in the response have the required structure.

    Checks that each tool call has ``name`` and ``arguments`` fields and
    that ``arguments`` is valid JSON (if it is a string).

    Maps to gate code ``tool_call_valid``.
    """

    name = "tool_call_valid"

    def evaluate(self, *, gate: dict[str, Any], response: Any) -> EvaluatorResult:
        tool_calls = _extract_tool_calls(response)
        if tool_calls is None:
            # No tool calls in response — pass (gate only applies when tools present)
            return EvaluatorResult(
                passed=True,
                safe_metadata={"evaluator_name": self.name, "tool_call_count": "0"},
            )

        errors: list[str] = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                errors.append(f"tool_call[{i}]:not_dict")
                continue
            if "name" not in tc:
                errors.append(f"tool_call[{i}]:missing_name")
            args = tc.get("arguments")
            if args is not None and isinstance(args, str):
                try:
                    json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    errors.append(f"tool_call[{i}]:invalid_arguments_json")

        if errors:
            return EvaluatorResult(
                passed=False,
                reason_code="tool_call_validation_error",
                safe_metadata={
                    "evaluator_name": self.name,
                    "error_count": str(len(errors)),
                    "first_error": errors[0],
                },
            )
        return EvaluatorResult(
            passed=True,
            safe_metadata={
                "evaluator_name": self.name,
                "tool_call_count": str(len(tool_calls)),
            },
        )


# ---------------------------------------------------------------------------
# RegexPredicateEvaluator
# ---------------------------------------------------------------------------


class RegexPredicateEvaluator:
    """Matches the response text against a regex pattern.

    The pattern is taken from ``gate["config"]["pattern"]`` (a string) or
    provided at construction time. A match anywhere in the text passes.

    Maps to gate code ``evaluator_score_min`` or custom codes.
    """

    name = "regex_predicate"

    def __init__(self, *, default_pattern: str | None = None) -> None:
        self._default_pattern = default_pattern

    def evaluate(self, *, gate: dict[str, Any], response: Any) -> EvaluatorResult:
        text = _extract_text(response)
        if text is None:
            return EvaluatorResult(
                passed=False,
                reason_code="no_text_content",
                safe_metadata={"evaluator_name": self.name},
            )

        pattern = (gate.get("config") or {}).get("pattern") or self._default_pattern
        if pattern is None:
            return EvaluatorResult(
                passed=False,
                reason_code="no_pattern_configured",
                safe_metadata={"evaluator_name": self.name},
            )

        try:
            match = re.search(pattern, text)
        except re.error:
            return EvaluatorResult(
                passed=False,
                reason_code="invalid_regex_pattern",
                safe_metadata={"evaluator_name": self.name},
            )

        return EvaluatorResult(
            passed=match is not None,
            score=1.0 if match else 0.0,
            reason_code=None if match else "pattern_not_matched",
            safe_metadata={"evaluator_name": self.name},
        )


# ---------------------------------------------------------------------------
# SafetyPassedEvaluator (adapter stub)
# ---------------------------------------------------------------------------


class SafetyPassedEvaluator:
    """Adapter stub for app-provided safety evaluation.

    This evaluator does NOT implement a classifier itself. It delegates to
    an app-provided ``check`` callback. If no callback is provided, it
    passes by default (fail-open for advisory, fail-closed handled by the
    runner when no evaluator is registered).

    Maps to gate code ``safety_passed``.
    """

    name = "safety_passed"

    def __init__(
        self,
        *,
        check: Any | None = None,
    ) -> None:
        self._check = check

    def evaluate(self, *, gate: dict[str, Any], response: Any) -> EvaluatorResult:
        if self._check is None:
            return EvaluatorResult(
                passed=True,
                reason_code="no_safety_checker_configured",
                safe_metadata={"evaluator_name": self.name},
            )
        try:
            result = self._check(response)
            if isinstance(result, bool):
                return EvaluatorResult(
                    passed=result,
                    reason_code=None if result else "safety_check_failed",
                    safe_metadata={"evaluator_name": self.name},
                )
            # Assume result is an EvaluatorResult-like object
            return EvaluatorResult(
                passed=bool(getattr(result, "passed", True)),
                score=getattr(result, "score", None),
                reason_code=getattr(result, "reason_code", None),
                safe_metadata=getattr(result, "safe_metadata", {"evaluator_name": self.name}),
            )
        except Exception:
            return EvaluatorResult(
                passed=False,
                reason_code="safety_checker_error",
                safe_metadata={"evaluator_name": self.name},
            )


# ---------------------------------------------------------------------------
# EvaluatorRegistry
# ---------------------------------------------------------------------------


@dataclass
class EvaluatorRegistry:
    """Maps gate codes to evaluator instances.

    Default built-in evaluators are registered automatically. Apps can
    override or extend by passing custom evaluators.
    """

    _evaluators: dict[str, OutputQualityEvaluator] = field(default_factory=dict)

    def register(self, gate_code: str, evaluator: OutputQualityEvaluator) -> None:
        """Register an evaluator for a gate code."""
        self._evaluators[gate_code] = evaluator

    def get(self, gate_code: str) -> OutputQualityEvaluator | None:
        """Get the evaluator for a gate code, or None."""
        return self._evaluators.get(gate_code)

    @classmethod
    def with_defaults(
        cls,
        *,
        json_schema: dict[str, Any] | None = None,
        safety_check: Any | None = None,
        extra: dict[str, OutputQualityEvaluator] | None = None,
    ) -> EvaluatorRegistry:
        """Create a registry with built-in evaluators pre-registered.

        Args:
            json_schema: Default JSON Schema for schema_valid gate.
            safety_check: Callback for safety_passed gate.
            extra: Additional evaluators to register.
        """
        reg = cls()
        reg.register("json_parseable", JsonParseableEvaluator())
        reg.register("schema_valid", JsonSchemaEvaluator(default_schema=json_schema))
        reg.register("tool_call_valid", ToolCallValidEvaluator())
        reg.register("safety_passed", SafetyPassedEvaluator(check=safety_check))
        if extra:
            for code, evaluator in extra.items():
                reg.register(code, evaluator)
        return reg


# ---------------------------------------------------------------------------
# RegistryBackedEvaluator
# ---------------------------------------------------------------------------


class RegistryBackedEvaluator:
    """OutputQualityGateEvaluator that delegates to an EvaluatorRegistry.

    This bridges the per-gate evaluator registry into the single-evaluator
    interface expected by ``CandidateAttemptRunner``.
    """

    name = "registry"

    def __init__(self, registry: EvaluatorRegistry) -> None:
        self._registry = registry

    def evaluate(self, gate: dict[str, Any], response: Any) -> Any:
        """Evaluate a gate using the registered evaluator.

        Returns a GateResult-compatible object for the attempt runner.
        """
        from octomil.runtime.routing.attempt_runner import GateResult, GateStatus

        code = gate.get("code", "")
        evaluator = self._registry.get(code)
        if evaluator is None:
            return GateResult(
                code=code,
                status=GateStatus.FAILED,
                reason_code="evaluator_missing",
            )
        result = evaluator.evaluate(gate=gate, response=response)
        return GateResult(
            code=code,
            status=GateStatus.PASSED if result.passed else GateStatus.FAILED,
            observed_number=result.score,
            reason_code=result.reason_code,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_text(response: Any) -> str | None:
    """Extract text content from a response object.

    Supports: str, dict with "text"/"content"/"output" key, objects with
    .text/.content/.output attributes.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("text", "content", "output"):
            if key in response and isinstance(response[key], str):
                return response[key]
        return None
    for attr in ("text", "content", "output"):
        val = getattr(response, attr, None)
        if isinstance(val, str):
            return val
    return None


def _extract_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from a response object.

    Supports: dict with "tool_calls" key, objects with .tool_calls attribute.
    Returns None if no tool calls are present.
    """
    if isinstance(response, dict):
        tc = response.get("tool_calls")
        if isinstance(tc, list) and len(tc) > 0:
            return tc
        return None
    tc = getattr(response, "tool_calls", None)
    if isinstance(tc, list) and len(tc) > 0:
        return tc
    return None
