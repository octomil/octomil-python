"""Tests for octomil.tool_schemas — coding agent tool-use presets."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

try:
    import fastapi  # noqa: F401
    import pytest_asyncio  # noqa: F401

    _has_serve_deps = True
except ImportError:
    _has_serve_deps = False

from httpx import ASGITransport, AsyncClient

from octomil.serve import EchoBackend, create_app
from octomil.tool_schemas import CODING_TOOL_SCHEMAS, get_tool_use_tools


class TestCodingToolSchemas:
    """Validate the raw schema definitions."""

    def test_expected_tool_count(self):
        assert len(CODING_TOOL_SCHEMAS) == 5

    def test_expected_tool_names(self):
        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "run_command",
            "search_files",
        }
        assert set(CODING_TOOL_SCHEMAS.keys()) == expected

    @pytest.mark.parametrize("name", list(CODING_TOOL_SCHEMAS.keys()))
    def test_schema_has_type_object(self, name: str):
        schema = CODING_TOOL_SCHEMAS[name]
        assert schema["type"] == "object"

    @pytest.mark.parametrize("name", list(CODING_TOOL_SCHEMAS.keys()))
    def test_schema_has_properties(self, name: str):
        schema = CODING_TOOL_SCHEMAS[name]
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)
        assert len(schema["properties"]) > 0

    @pytest.mark.parametrize("name", list(CODING_TOOL_SCHEMAS.keys()))
    def test_schema_has_required(self, name: str):
        schema = CODING_TOOL_SCHEMAS[name]
        assert "required" in schema
        assert isinstance(schema["required"], list)
        assert len(schema["required"]) > 0

    @pytest.mark.parametrize("name", list(CODING_TOOL_SCHEMAS.keys()))
    def test_required_fields_exist_in_properties(self, name: str):
        schema = CODING_TOOL_SCHEMAS[name]
        for req_field in schema["required"]:
            assert (
                req_field in schema["properties"]
            ), f"Required field '{req_field}' not in properties for '{name}'"

    def test_read_file_schema(self):
        schema = CODING_TOOL_SCHEMAS["read_file"]
        assert schema["required"] == ["path"]
        assert "path" in schema["properties"]

    def test_write_file_schema(self):
        schema = CODING_TOOL_SCHEMAS["write_file"]
        assert set(schema["required"]) == {"path", "content"}

    def test_edit_file_schema(self):
        schema = CODING_TOOL_SCHEMAS["edit_file"]
        assert set(schema["required"]) == {"path", "old_text", "new_text"}

    def test_run_command_schema(self):
        schema = CODING_TOOL_SCHEMAS["run_command"]
        assert schema["required"] == ["command"]
        # working_dir is optional
        assert "working_dir" in schema["properties"]

    def test_search_files_schema(self):
        schema = CODING_TOOL_SCHEMAS["search_files"]
        assert schema["required"] == ["pattern"]
        # path is optional
        assert "path" in schema["properties"]


class TestGetToolUseTools:
    """Validate the OpenAI-compatible tool format."""

    def test_returns_list(self):
        tools = get_tool_use_tools()
        assert isinstance(tools, list)

    def test_tool_count_matches_schemas(self):
        tools = get_tool_use_tools()
        assert len(tools) == len(CODING_TOOL_SCHEMAS)

    def test_each_tool_has_type_function(self):
        for tool in get_tool_use_tools():
            assert tool["type"] == "function"

    def test_each_tool_has_function_block(self):
        for tool in get_tool_use_tools():
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_tool_names_match_schemas(self):
        tool_names = {t["function"]["name"] for t in get_tool_use_tools()}
        assert tool_names == set(CODING_TOOL_SCHEMAS.keys())

    def test_parameters_are_valid_json_schema(self):
        for tool in get_tool_use_tools():
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_descriptions_are_non_empty(self):
        for tool in get_tool_use_tools():
            assert len(tool["function"]["description"]) > 0


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------


def _make_echo_app(tool_use: bool):
    """Create a FastAPI app with EchoBackend and lifespan triggered."""
    if not _has_serve_deps:
        pytest.skip("fastapi and pytest-asyncio required")
    with patch("octomil.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", tool_use=tool_use)

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.fixture
def tool_use_app():
    """Create an app with tool_use=True (echo backend)."""
    return _make_echo_app(tool_use=True)


@pytest.fixture
def no_tool_use_app():
    """Create an app with tool_use=False (echo backend)."""
    return _make_echo_app(tool_use=False)


@pytest.mark.asyncio
async def test_tool_schemas_endpoint_enabled(tool_use_app):
    transport = ASGITransport(app=tool_use_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/tool-schemas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert len(data["tools"]) == 5
    # Verify OpenAI tool format
    for tool in data["tools"]:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]


@pytest.mark.asyncio
async def test_tool_schemas_endpoint_disabled(no_tool_use_app):
    transport = ASGITransport(app=no_tool_use_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/tool-schemas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["tools"] == []


@pytest.mark.asyncio
async def test_agent_context_header_accepted(tool_use_app):
    """Requests with X-Octomil-Agent-Context header should succeed normally."""
    transport = ASGITransport(app=tool_use_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"X-Octomil-Agent-Context": "aider/0.50.0"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_agent_context_header_absent(tool_use_app):
    """Requests without X-Octomil-Agent-Context should also succeed."""
    transport = ASGITransport(app=tool_use_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
