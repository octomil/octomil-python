"""Tests for octomil.runtime.core.tool_parser — tool call extraction."""

import json

from octomil.runtime.core.tool_parser import (
    ToolCallParseResult,
    extract_tool_call_from_text,
    extract_tool_call_with_validation,
)


class TestExtractToolCallFromText:
    """Unit tests for extract_tool_call_from_text (new format only)."""

    def test_valid_tool_call_new_format(self):
        text = '{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "get_weather"
        assert json.loads(result.arguments) == {"city": "NYC"}
        assert result.id.startswith("call_")

    def test_valid_with_declared_tools(self):
        text = '{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}'
        result = extract_tool_call_from_text(text, declared_tools=["get_weather", "search"])
        assert result is not None
        assert result.name == "get_weather"

    def test_old_format_rejected(self):
        """Old format is no longer supported — hard cutover."""
        text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "NYC"}}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_malformed_json_returns_none(self):
        text = '{"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_wrong_type_returns_none(self):
        text = '{"type": "message", "content": "hello"}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_missing_type_returns_none(self):
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_unknown_tool_name_rejected(self):
        text = '{"type": "tool_call", "name": "hack_system", "arguments": {}}'
        result = extract_tool_call_from_text(text, declared_tools=["get_weather"])
        assert result is None

    def test_unknown_tool_allowed_without_declared_tools(self):
        text = '{"type": "tool_call", "name": "hack_system", "arguments": {}}'
        result = extract_tool_call_from_text(text, declared_tools=None)
        assert result is not None

    def test_arguments_not_dict_returns_none(self):
        text = '{"type": "tool_call", "name": "get_weather", "arguments": "invalid"}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_arguments_list_returns_none(self):
        text = '{"type": "tool_call", "name": "get_weather", "arguments": [1, 2, 3]}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_plain_text_returns_none(self):
        text = "I don't know the weather right now."
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_json_in_prose_returns_none(self):
        """Full-response-only: JSON embedded in prose is rejected."""
        text = 'Here is the result: {"type": "tool_call", "name": "search", "arguments": {"q": "test"}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_nested_argument_values_preserved(self):
        text = json.dumps(
            {
                "type": "tool_call",
                "name": "deploy",
                "arguments": {
                    "config": {"replicas": 3, "env": {"DEBUG": "true"}},
                    "tags": ["prod", "v2"],
                },
            }
        )
        result = extract_tool_call_from_text(text)
        assert result is not None
        parsed = json.loads(result.arguments)
        assert parsed["config"]["replicas"] == 3
        assert parsed["tags"] == ["prod", "v2"]

    def test_empty_string_returns_none(self):
        assert extract_tool_call_from_text("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_tool_call_from_text("   \n  ") is None

    def test_none_name_returns_none(self):
        text = '{"type": "tool_call", "name": null, "arguments": {}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_empty_name_returns_none(self):
        text = '{"type": "tool_call", "name": "", "arguments": {}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_empty_arguments_dict_valid(self):
        text = '{"type": "tool_call", "name": "list_all", "arguments": {}}'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert json.loads(result.arguments) == {}

    def test_tool_call_in_code_fence(self):
        text = '```json\n{"type": "tool_call", "name": "search", "arguments": {"q": "test"}}\n```'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "search"

    def test_whitespace_padded_json(self):
        text = '  \n  {"type": "tool_call", "name": "search", "arguments": {"q": "test"}}  \n  '
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "search"


class TestExtractToolCallWithValidation:
    """Tests for extract_tool_call_with_validation with schema checking."""

    def test_returns_parse_result_type(self):
        text = '{"type": "tool_call", "name": "fn", "arguments": {}}'
        result = extract_tool_call_with_validation(text)
        assert isinstance(result, ToolCallParseResult)
        assert result.tool_call is not None

    def test_no_schemas_means_no_validation(self):
        text = '{"type": "tool_call", "name": "fn", "arguments": {"x": 1}}'
        result = extract_tool_call_with_validation(text)
        assert result.tool_call is not None
        assert result.schema_valid is None
        assert result.schema_errors == []

    def test_valid_schema_returns_true(self):
        text = '{"type": "tool_call", "name": "fn", "arguments": {"city": "NYC"}}'
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        result = extract_tool_call_with_validation(text, tool_schemas={"fn": schema})
        assert result.tool_call is not None
        assert result.schema_valid is True
        assert result.schema_errors == []

    def test_invalid_schema_returns_false_with_errors(self):
        text = '{"type": "tool_call", "name": "fn", "arguments": {"city": 42}}'
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        result = extract_tool_call_with_validation(text, tool_schemas={"fn": schema})
        assert result.tool_call is not None  # tool call still returned
        assert result.schema_valid is False
        assert len(result.schema_errors) > 0

    def test_schema_for_different_tool_skipped(self):
        text = '{"type": "tool_call", "name": "fn_a", "arguments": {"x": 1}}'
        result = extract_tool_call_with_validation(text, tool_schemas={"fn_b": {"type": "object"}})
        assert result.tool_call is not None
        assert result.schema_valid is None

    def test_failed_parse_returns_empty_result(self):
        result = extract_tool_call_with_validation("not json")
        assert result.tool_call is None
        assert result.schema_valid is None
