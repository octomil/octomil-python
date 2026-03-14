"""Tests for octomil.runtime.core.tool_parser — text-based tool call extraction."""

import json

from octomil.runtime.core.tool_parser import extract_tool_call_from_text


class TestExtractToolCallFromText:
    """Unit tests for extract_tool_call_from_text."""

    def test_valid_tool_call(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "NYC"}}}'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "get_weather"
        assert json.loads(result.arguments) == {"city": "NYC"}
        assert result.id.startswith("call_")

    def test_valid_tool_call_with_declared_tools(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "NYC"}}}'
        result = extract_tool_call_from_text(text, declared_tools=["get_weather", "search"])
        assert result is not None
        assert result.name == "get_weather"

    def test_malformed_json_returns_none(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "NYC"'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_missing_tool_call_key_returns_none(self):
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_unknown_tool_name_rejected(self):
        text = '{"tool_call": {"name": "hack_system", "arguments": {}}}'
        result = extract_tool_call_from_text(text, declared_tools=["get_weather"])
        assert result is None

    def test_unknown_tool_allowed_without_declared_tools(self):
        text = '{"tool_call": {"name": "hack_system", "arguments": {}}}'
        result = extract_tool_call_from_text(text, declared_tools=None)
        assert result is not None

    def test_arguments_not_dict_returns_none(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": "invalid"}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_arguments_list_returns_none(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": [1, 2, 3]}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_text_without_brace_skips_parsing(self):
        text = "I don't know the weather right now."
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_text_with_tool_call_marker_in_prose(self):
        text = 'The model mentioned "tool_call" but no actual JSON here.'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_nested_argument_values_preserved(self):
        text = json.dumps(
            {
                "tool_call": {
                    "name": "deploy",
                    "arguments": {
                        "config": {"replicas": 3, "env": {"DEBUG": "true"}},
                        "tags": ["prod", "v2"],
                    },
                }
            }
        )
        result = extract_tool_call_from_text(text)
        assert result is not None
        parsed = json.loads(result.arguments)
        assert parsed["config"]["replicas"] == 3
        assert parsed["config"]["env"]["DEBUG"] == "true"
        assert parsed["tags"] == ["prod", "v2"]

    def test_empty_string_returns_none(self):
        assert extract_tool_call_from_text("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_tool_call_from_text("   \n  ") is None

    def test_none_name_returns_none(self):
        text = '{"tool_call": {"name": null, "arguments": {}}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_empty_name_returns_none(self):
        text = '{"tool_call": {"name": "", "arguments": {}}}'
        result = extract_tool_call_from_text(text)
        assert result is None

    def test_empty_arguments_dict_valid(self):
        text = '{"tool_call": {"name": "list_all", "arguments": {}}}'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert json.loads(result.arguments) == {}

    def test_tool_call_in_code_fence(self):
        text = '```json\n{"tool_call": {"name": "search", "arguments": {"q": "test"}}}\n```'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "search"

    def test_tool_call_with_surrounding_text(self):
        text = 'Let me use a tool: {"tool_call": {"name": "search", "arguments": {"q": "test"}}}'
        result = extract_tool_call_from_text(text)
        assert result is not None
        assert result.name == "search"
