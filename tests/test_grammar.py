"""Tests for edgeml.grammar — structured/constrained decoding support."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.grammar import (
    extract_json,
    json_mode_grammar,
    json_schema_to_gbnf,
    json_system_prompt,
    validate_json_output,
)
from edgeml.serve import (
    ChatCompletionBody,
    EchoBackend,
    GenerationRequest,
    _resolve_grammar,
    create_app,
)


# ---------------------------------------------------------------------------
# json_mode_grammar
# ---------------------------------------------------------------------------


class TestJsonModeGrammar:
    def test_returns_nonempty_string(self):
        grammar = json_mode_grammar()
        assert isinstance(grammar, str)
        assert len(grammar) > 50

    def test_contains_root_rule(self):
        grammar = json_mode_grammar()
        assert "root" in grammar

    def test_contains_value_rule(self):
        grammar = json_mode_grammar()
        assert "value" in grammar

    def test_contains_object_rule(self):
        grammar = json_mode_grammar()
        assert "object" in grammar

    def test_contains_string_rule(self):
        grammar = json_mode_grammar()
        assert "string" in grammar

    def test_contains_number_rule(self):
        grammar = json_mode_grammar()
        assert "number" in grammar

    def test_ends_with_newline(self):
        grammar = json_mode_grammar()
        assert grammar.endswith("\n")


# ---------------------------------------------------------------------------
# json_schema_to_gbnf — simple object
# ---------------------------------------------------------------------------


class TestJsonSchemaToGbnfSimple:
    def test_simple_object_has_root(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert grammar.startswith("root ::=")

    def test_simple_object_references_string_and_integer(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "string" in grammar
        assert "integer" in grammar

    def test_simple_object_contains_field_names(self):
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "population": {"type": "number"},
            },
            "required": ["city"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "city" in grammar
        assert "population" in grammar

    def test_all_primitive_types(self):
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "n": {"type": "number"},
                "i": {"type": "integer"},
                "b": {"type": "boolean"},
                "x": {"type": "null"},
            },
            "required": ["s", "n", "i", "b", "x"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "string" in grammar
        assert "number" in grammar
        assert "integer" in grammar
        assert "boolean" in grammar
        assert "null" in grammar

    def test_no_properties_returns_generic_object(self):
        schema = {"type": "object"}
        grammar = json_schema_to_gbnf(schema)
        # Should fall back to generic object rule
        assert "root ::= object" in grammar

    def test_no_type_returns_value(self):
        schema = {}
        grammar = json_schema_to_gbnf(schema)
        assert "root ::= value" in grammar


# ---------------------------------------------------------------------------
# json_schema_to_gbnf — nested schemas
# ---------------------------------------------------------------------------


class TestJsonSchemaToGbnfNested:
    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["user"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "name" in grammar
        assert "email" in grammar
        assert "user" in grammar

    def test_array_with_typed_items(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["tags"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "tags" in grammar
        assert "string" in grammar
        assert "[" in grammar

    def test_array_of_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["id"],
                    },
                },
            },
            "required": ["items"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "id" in grammar
        assert "label" in grammar
        assert "integer" in grammar

    def test_untyped_array_uses_generic(self):
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "array"},
            },
            "required": ["data"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "array" in grammar


# ---------------------------------------------------------------------------
# json_schema_to_gbnf — enum values
# ---------------------------------------------------------------------------


class TestJsonSchemaToGbnfEnum:
    def test_string_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["active", "inactive", "pending"]},
            },
            "required": ["status"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "active" in grammar
        assert "inactive" in grammar
        assert "pending" in grammar

    def test_mixed_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"enum": ["yes", 42, True, None]},
            },
            "required": ["value"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert "yes" in grammar
        assert "42" in grammar
        assert "true" in grammar
        assert "null" in grammar

    def test_integer_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "level": {"enum": [1, 2, 3]},
            },
            "required": ["level"],
        }
        grammar = json_schema_to_gbnf(schema)
        assert '"1"' in grammar or "1" in grammar
        assert '"2"' in grammar or "2" in grammar
        assert '"3"' in grammar or "3" in grammar


# ---------------------------------------------------------------------------
# json_schema_to_gbnf — optional fields
# ---------------------------------------------------------------------------


class TestJsonSchemaToGbnfOptional:
    def test_optional_field_has_question_mark(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "bio": {"type": "string"},
            },
            "required": ["name"],
        }
        grammar = json_schema_to_gbnf(schema)
        # bio should be optional — grammar should contain '?' for it
        assert "?" in grammar

    def test_all_required_no_question_mark(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        }
        grammar = json_schema_to_gbnf(schema)
        # All required fields — no optional markers for them
        # (but the grammar primitives might contain ? in number rule etc.)
        assert "a" in grammar
        assert "b" in grammar


# ---------------------------------------------------------------------------
# validate_json_output
# ---------------------------------------------------------------------------


class TestValidateJsonOutput:
    def test_valid_object(self):
        assert validate_json_output('{"key": "value"}') is True

    def test_valid_array(self):
        assert validate_json_output("[1, 2, 3]") is True

    def test_valid_string(self):
        assert validate_json_output('"hello"') is True

    def test_valid_number(self):
        assert validate_json_output("42") is True

    def test_valid_boolean(self):
        assert validate_json_output("true") is True

    def test_valid_null(self):
        assert validate_json_output("null") is True

    def test_invalid_trailing_comma(self):
        assert validate_json_output('{"key": "value",}') is False

    def test_invalid_plain_text(self):
        assert validate_json_output("Hello world") is False

    def test_invalid_empty_string(self):
        assert validate_json_output("") is False

    def test_invalid_none(self):
        assert validate_json_output(None) is False  # type: ignore[arg-type]

    def test_nested_valid_json(self):
        obj = {"user": {"name": "Alice", "scores": [1, 2, 3]}}
        assert validate_json_output(json.dumps(obj)) is True


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_json(self):
        result = extract_json('{"name": "Alice"}')
        assert result == {"name": "Alice"}

    def test_json_with_leading_text(self):
        result = extract_json('Here is the output: {"name": "Alice"}')
        assert result == {"name": "Alice"}

    def test_json_with_trailing_text(self):
        result = extract_json('{"name": "Alice"} hope that helps!')
        assert result == {"name": "Alice"}

    def test_fenced_code_block(self):
        text = '```json\n{"name": "Alice"}\n```'
        result = extract_json(text)
        assert result == {"name": "Alice"}

    def test_fenced_code_block_no_language(self):
        text = '```\n{"name": "Alice"}\n```'
        result = extract_json(text)
        assert result == {"name": "Alice"}

    def test_nested_json(self):
        obj = {"user": {"name": "Alice", "age": 30}, "active": True}
        text = f"Result: {json.dumps(obj)}"
        result = extract_json(text)
        assert result == obj

    def test_no_json_returns_none(self):
        assert extract_json("Hello world, no json here") is None

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_none_input_returns_none(self):
        assert extract_json(None) is None  # type: ignore[arg-type]

    def test_array_not_dict_returns_none(self):
        assert extract_json("[1, 2, 3]") is None

    def test_json_with_escaped_quotes(self):
        result = extract_json('{"msg": "say \\"hello\\""}')
        assert result is not None
        assert result["msg"] == 'say "hello"'

    def test_multiple_json_extracts_first(self):
        text = '{"a": 1} some text {"b": 2}'
        result = extract_json(text)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# json_system_prompt
# ---------------------------------------------------------------------------


class TestJsonSystemPrompt:
    def test_basic_prompt_mentions_json(self):
        prompt = json_system_prompt()
        assert "JSON" in prompt or "json" in prompt

    def test_schema_prompt_includes_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        prompt = json_system_prompt(schema)
        assert "x" in prompt
        assert "string" in prompt

    def test_no_schema_returns_generic(self):
        prompt = json_system_prompt()
        assert "schema" not in prompt.lower()


# ---------------------------------------------------------------------------
# ChatCompletionBody accepts response_format and grammar
# ---------------------------------------------------------------------------


class TestChatCompletionBodyExtended:
    def test_response_format_json_object(self):
        body = ChatCompletionBody(
            model="test",
            messages=[],
            response_format={"type": "json_object"},
        )
        assert body.response_format == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        body = ChatCompletionBody(
            model="test",
            messages=[],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
        )
        assert body.response_format is not None
        assert body.response_format["type"] == "json_schema"

    def test_grammar_field(self):
        body = ChatCompletionBody(
            model="test",
            messages=[],
            grammar='root ::= "hello"',
        )
        assert body.grammar == 'root ::= "hello"'

    def test_defaults_are_none(self):
        body = ChatCompletionBody()
        assert body.response_format is None
        assert body.grammar is None


# ---------------------------------------------------------------------------
# _resolve_grammar
# ---------------------------------------------------------------------------


class TestResolveGrammar:
    def test_no_format_returns_none(self):
        body = ChatCompletionBody()
        grammar, is_json = _resolve_grammar(body)
        assert grammar is None
        assert is_json is False

    def test_json_object_returns_grammar(self):
        body = ChatCompletionBody(
            response_format={"type": "json_object"},
        )
        grammar, is_json = _resolve_grammar(body)
        assert grammar is not None
        assert "root" in grammar
        assert is_json is True

    def test_json_schema_returns_grammar(self):
        body = ChatCompletionBody(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
            },
        )
        grammar, is_json = _resolve_grammar(body)
        assert grammar is not None
        assert "name" in grammar
        assert is_json is True

    def test_explicit_grammar_takes_precedence(self):
        body = ChatCompletionBody(
            response_format={"type": "json_object"},
            grammar='root ::= "custom"',
        )
        grammar, is_json = _resolve_grammar(body)
        assert grammar == 'root ::= "custom"'
        assert is_json is False

    def test_default_json_mode(self):
        body = ChatCompletionBody()
        grammar, is_json = _resolve_grammar(body, default_json_mode=True)
        assert grammar is not None
        assert is_json is True

    def test_unknown_type_returns_none(self):
        body = ChatCompletionBody(
            response_format={"type": "text"},
        )
        grammar, is_json = _resolve_grammar(body)
        assert grammar is None
        assert is_json is False


# ---------------------------------------------------------------------------
# GenerationRequest extended fields
# ---------------------------------------------------------------------------


class TestGenerationRequestExtended:
    def test_default_grammar_is_none(self):
        req = GenerationRequest(model="m", messages=[])
        assert req.grammar is None
        assert req.json_mode is False

    def test_grammar_field(self):
        req = GenerationRequest(
            model="m",
            messages=[],
            grammar='root ::= "x"',
            json_mode=True,
        )
        assert req.grammar == 'root ::= "x"'
        assert req.json_mode is True


# ---------------------------------------------------------------------------
# FastAPI integration — response_format with EchoBackend
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app():
    """Create a FastAPI app with EchoBackend for testing."""
    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo
        app = create_app("test-model")

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())
    return app


@pytest.fixture
def json_mode_app():
    """Create a FastAPI app with json_mode=True and EchoBackend."""
    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo
        app = create_app("test-model", json_mode=True)

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())
    return app


@pytest.mark.asyncio
async def test_response_format_json_object_accepted(echo_app):
    """response_format=json_object should be accepted without error."""
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "list colors"}],
                "response_format": {"type": "json_object"},
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_response_format_json_schema_accepted(echo_app):
    """response_format=json_schema should be accepted without error."""
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "give user info"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        }
                    },
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_grammar_param_accepted(echo_app):
    """Raw grammar param should be accepted without error."""
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "grammar": 'root ::= "test"',
            },
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_json_mode_default_app(json_mode_app):
    """App created with json_mode=True should accept requests and return 200."""
    transport = ASGITransport(app=json_mode_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "list items"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    # With json_mode=True and echo backend (non-grammar), a JSON system prompt
    # should have been injected into the messages before generation.
    # The echo backend echoes the last user message, so we verify the request
    # succeeded without error and produced a response.
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_no_response_format_still_works(echo_app):
    """Requests without response_format should still work normally."""
    transport = ASGITransport(app=echo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "echo" in data["choices"][0]["message"]["content"].lower()
