"""Tests for built-in output quality gate evaluators."""

from __future__ import annotations

import json

import pytest

from octomil.runtime.routing.evaluators import (
    EvaluatorRegistry,
    EvaluatorResult,
    JsonParseableEvaluator,
    JsonSchemaEvaluator,
    RegexPredicateEvaluator,
    RegistryBackedEvaluator,
    SafetyPassedEvaluator,
    ToolCallValidEvaluator,
)

# ---------------------------------------------------------------------------
# JsonParseableEvaluator
# ---------------------------------------------------------------------------


class TestJsonParseableEvaluator:
    def test_valid_json_string(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response='{"key": "value"}')
        assert result.passed is True

    def test_valid_json_array(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response="[1, 2, 3]")
        assert result.passed is True

    def test_invalid_json(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response="not json at all")
        assert result.passed is False
        assert result.reason_code == "json_parse_error"

    def test_empty_string(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response="")
        assert result.passed is False

    def test_dict_response_with_text(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(
            gate={"code": "json_parseable"},
            response={"text": '{"valid": true}'},
        )
        assert result.passed is True

    def test_dict_response_with_content(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(
            gate={"code": "json_parseable"},
            response={"content": '{"valid": true}'},
        )
        assert result.passed is True

    def test_no_text_content(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(
            gate={"code": "json_parseable"},
            response={"some_other_key": 123},
        )
        assert result.passed is False
        assert result.reason_code == "no_text_content"

    def test_none_response(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response=None)
        assert result.passed is False
        assert result.reason_code == "no_text_content"

    def test_safe_metadata_included(self):
        ev = JsonParseableEvaluator()
        result = ev.evaluate(gate={"code": "json_parseable"}, response='{"a": 1}')
        assert result.safe_metadata is not None
        assert result.safe_metadata["evaluator_name"] == "json_parseable"


# ---------------------------------------------------------------------------
# JsonSchemaEvaluator
# ---------------------------------------------------------------------------


class TestJsonSchemaEvaluator:
    @pytest.fixture()
    def simple_schema(self):
        return {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

    def test_valid_against_schema(self, simple_schema):
        ev = JsonSchemaEvaluator(default_schema=simple_schema)
        result = ev.evaluate(
            gate={"code": "schema_valid"},
            response=json.dumps({"name": "Alice", "age": 30}),
        )
        assert result.passed is True

    def test_invalid_against_schema(self, simple_schema):
        ev = JsonSchemaEvaluator(default_schema=simple_schema)
        result = ev.evaluate(
            gate={"code": "schema_valid"},
            response=json.dumps({"age": 30}),  # missing required "name"
        )
        assert result.passed is False
        assert result.reason_code == "schema_validation_error"

    def test_schema_from_gate_config(self):
        ev = JsonSchemaEvaluator()
        gate = {
            "code": "schema_valid",
            "config": {"schema": {"type": "array", "items": {"type": "number"}}},
        }
        result = ev.evaluate(gate=gate, response="[1, 2, 3]")
        assert result.passed is True

    def test_gate_config_overrides_default(self, simple_schema):
        ev = JsonSchemaEvaluator(default_schema=simple_schema)
        gate = {
            "code": "schema_valid",
            "config": {"schema": {"type": "array"}},
        }
        # "[1,2]" matches array schema, not object schema
        result = ev.evaluate(gate=gate, response="[1, 2]")
        assert result.passed is True

    def test_no_schema_configured(self):
        ev = JsonSchemaEvaluator()
        result = ev.evaluate(gate={"code": "schema_valid"}, response='{"a": 1}')
        assert result.passed is False
        assert result.reason_code == "no_schema_configured"

    def test_invalid_json_input(self, simple_schema):
        ev = JsonSchemaEvaluator(default_schema=simple_schema)
        result = ev.evaluate(gate={"code": "schema_valid"}, response="not json")
        assert result.passed is False
        assert result.reason_code == "json_parse_error"

    def test_no_text(self, simple_schema):
        ev = JsonSchemaEvaluator(default_schema=simple_schema)
        result = ev.evaluate(gate={"code": "schema_valid"}, response=None)
        assert result.passed is False
        assert result.reason_code == "no_text_content"


# ---------------------------------------------------------------------------
# ToolCallValidEvaluator
# ---------------------------------------------------------------------------


class TestToolCallValidEvaluator:
    def test_no_tool_calls_passes(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={"text": "just text, no tools"},
        )
        assert result.passed is True

    def test_valid_tool_calls(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={
                "tool_calls": [
                    {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    {"name": "search", "arguments": '{"q": "test"}'},
                ]
            },
        )
        assert result.passed is True

    def test_tool_call_missing_name(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={"tool_calls": [{"arguments": "{}"}]},
        )
        assert result.passed is False
        assert result.reason_code == "tool_call_validation_error"

    def test_tool_call_invalid_arguments_json(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={"tool_calls": [{"name": "fn", "arguments": "not json"}]},
        )
        assert result.passed is False
        assert result.reason_code == "tool_call_validation_error"

    def test_tool_call_dict_arguments_ok(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={"tool_calls": [{"name": "fn", "arguments": {"k": "v"}}]},
        )
        assert result.passed is True

    def test_tool_call_not_dict(self):
        ev = ToolCallValidEvaluator()
        result = ev.evaluate(
            gate={"code": "tool_call_valid"},
            response={"tool_calls": ["not_a_dict"]},
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# RegexPredicateEvaluator
# ---------------------------------------------------------------------------


class TestRegexPredicateEvaluator:
    def test_match_found(self):
        ev = RegexPredicateEvaluator(default_pattern=r"\d{3}-\d{4}")
        result = ev.evaluate(gate={"code": "evaluator_score_min"}, response="Call 555-1234")
        assert result.passed is True
        assert result.score == 1.0

    def test_no_match(self):
        ev = RegexPredicateEvaluator(default_pattern=r"\d{3}-\d{4}")
        result = ev.evaluate(gate={"code": "evaluator_score_min"}, response="no numbers here")
        assert result.passed is False
        assert result.score == 0.0
        assert result.reason_code == "pattern_not_matched"

    def test_pattern_from_gate_config(self):
        ev = RegexPredicateEvaluator()
        gate = {"code": "evaluator_score_min", "config": {"pattern": r"^OK$"}}
        result = ev.evaluate(gate=gate, response="OK")
        assert result.passed is True

    def test_no_pattern_configured(self):
        ev = RegexPredicateEvaluator()
        result = ev.evaluate(gate={"code": "evaluator_score_min"}, response="anything")
        assert result.passed is False
        assert result.reason_code == "no_pattern_configured"

    def test_invalid_regex(self):
        ev = RegexPredicateEvaluator(default_pattern="[invalid")
        result = ev.evaluate(gate={"code": "evaluator_score_min"}, response="test")
        assert result.passed is False
        assert result.reason_code == "invalid_regex_pattern"


# ---------------------------------------------------------------------------
# SafetyPassedEvaluator
# ---------------------------------------------------------------------------


class TestSafetyPassedEvaluator:
    def test_no_checker_passes(self):
        ev = SafetyPassedEvaluator()
        result = ev.evaluate(gate={"code": "safety_passed"}, response="anything")
        assert result.passed is True
        assert result.reason_code == "no_safety_checker_configured"

    def test_checker_returns_true(self):
        ev = SafetyPassedEvaluator(check=lambda _: True)
        result = ev.evaluate(gate={"code": "safety_passed"}, response="safe content")
        assert result.passed is True

    def test_checker_returns_false(self):
        ev = SafetyPassedEvaluator(check=lambda _: False)
        result = ev.evaluate(gate={"code": "safety_passed"}, response="unsafe content")
        assert result.passed is False
        assert result.reason_code == "safety_check_failed"

    def test_checker_raises_exception(self):
        def bad_checker(_):
            raise RuntimeError("checker broke")

        ev = SafetyPassedEvaluator(check=bad_checker)
        result = ev.evaluate(gate={"code": "safety_passed"}, response="anything")
        assert result.passed is False
        assert result.reason_code == "safety_checker_error"

    def test_checker_returns_result_object(self):
        class CheckResult:
            passed = False
            score = 0.2
            reason_code = "toxic_content"
            safe_metadata = {"evaluator_name": "my_safety"}

        ev = SafetyPassedEvaluator(check=lambda _: CheckResult())
        result = ev.evaluate(gate={"code": "safety_passed"}, response="test")
        assert result.passed is False
        assert result.score == 0.2
        assert result.reason_code == "toxic_content"


# ---------------------------------------------------------------------------
# EvaluatorRegistry
# ---------------------------------------------------------------------------


class TestEvaluatorRegistry:
    def test_register_and_get(self):
        reg = EvaluatorRegistry()
        ev = JsonParseableEvaluator()
        reg.register("json_parseable", ev)
        assert reg.get("json_parseable") is ev

    def test_get_unknown_returns_none(self):
        reg = EvaluatorRegistry()
        assert reg.get("nonexistent") is None

    def test_with_defaults(self):
        reg = EvaluatorRegistry.with_defaults()
        assert reg.get("json_parseable") is not None
        assert reg.get("schema_valid") is not None
        assert reg.get("tool_call_valid") is not None
        assert reg.get("safety_passed") is not None

    def test_with_defaults_and_extras(self):
        custom = RegexPredicateEvaluator(default_pattern="test")
        reg = EvaluatorRegistry.with_defaults(extra={"custom_gate": custom})
        assert reg.get("custom_gate") is custom
        assert reg.get("json_parseable") is not None


# ---------------------------------------------------------------------------
# RegistryBackedEvaluator
# ---------------------------------------------------------------------------


class TestRegistryBackedEvaluator:
    def test_delegates_to_registered_evaluator(self):
        reg = EvaluatorRegistry.with_defaults()
        evaluator = RegistryBackedEvaluator(reg)
        result = evaluator.evaluate(
            {"code": "json_parseable"},
            '{"valid": true}',
        )
        assert result.status.value == "passed"

    def test_missing_evaluator_fails(self):
        reg = EvaluatorRegistry()
        evaluator = RegistryBackedEvaluator(reg)
        result = evaluator.evaluate(
            {"code": "unknown_gate"},
            "response",
        )
        assert result.status.value == "failed"
        assert result.reason_code == "evaluator_missing"

    def test_failed_evaluation_returns_failed_status(self):
        reg = EvaluatorRegistry.with_defaults()
        evaluator = RegistryBackedEvaluator(reg)
        result = evaluator.evaluate(
            {"code": "json_parseable"},
            "not valid json",
        )
        assert result.status.value == "failed"
        assert result.reason_code == "json_parse_error"


# ---------------------------------------------------------------------------
# EvaluatorResult
# ---------------------------------------------------------------------------


class TestEvaluatorResult:
    def test_defaults(self):
        r = EvaluatorResult(passed=True)
        assert r.passed is True
        assert r.score is None
        assert r.reason_code is None
        assert r.safe_metadata is None

    def test_with_all_fields(self):
        r = EvaluatorResult(
            passed=False,
            score=0.5,
            reason_code="low_score",
            safe_metadata={"evaluator_name": "test"},
        )
        assert r.passed is False
        assert r.score == 0.5
        assert r.reason_code == "low_score"
        assert r.safe_metadata == {"evaluator_name": "test"}
