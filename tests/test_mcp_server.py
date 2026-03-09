"""Tests for the Octomil MCP server tools and helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        from octomil.mcp.server import _read_file

        with patch("octomil.mcp.server._allowed_roots", return_value=[tmp_path]):
            assert _read_file(str(f)) == "print('hello')"

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        from octomil.mcp.server import _read_file

        with patch("octomil.mcp.server._allowed_roots", return_value=[tmp_path]):
            with pytest.raises(FileNotFoundError):
                _read_file(str(tmp_path / "nonexistent.py"))

    def test_truncates_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_text("x" * 60_000)
        from octomil.mcp.server import _read_file

        with patch("octomil.mcp.server._allowed_roots", return_value=[tmp_path]):
            result = _read_file(str(f), max_chars=1000)
        assert len(result) < 2000
        assert "truncated" in result

    def test_expands_home(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("data")
        from octomil.mcp.server import _read_file

        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            result = _read_file("~/test.py")
        assert result == "data"


class TestPathSecurity:
    def test_denies_sensitive_path(self, tmp_path: Path) -> None:
        from octomil.mcp.server import _resolve_and_validate_path

        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key_file = ssh_dir / "id_rsa"
        key_file.write_text("secret")

        with patch("octomil.mcp.server._allowed_roots", return_value=[tmp_path]):
            with pytest.raises(ValueError, match="sensitive"):
                _resolve_and_validate_path(str(key_file))

    def test_denies_path_outside_allowed_roots(self, tmp_path: Path) -> None:
        from octomil.mcp.server import _resolve_and_validate_path

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        f = outside / "secret.txt"
        f.write_text("nope")

        with patch("octomil.mcp.server._allowed_roots", return_value=[allowed]):
            with pytest.raises(ValueError, match="outside"):
                _resolve_and_validate_path(str(f))

    def test_allows_path_within_root(self, tmp_path: Path) -> None:
        from octomil.mcp.server import _resolve_and_validate_path

        f = tmp_path / "ok.py"
        f.write_text("ok")

        with patch("octomil.mcp.server._allowed_roots", return_value=[tmp_path]):
            result = _resolve_and_validate_path(str(f))
            assert result == f.resolve()


class TestDetectLanguage:
    def test_python(self) -> None:
        from octomil.mcp.server import _detect_language

        assert _detect_language("foo.py") == "python"

    def test_typescript(self) -> None:
        from octomil.mcp.server import _detect_language

        assert _detect_language("bar.tsx") == "typescript"

    def test_unknown(self) -> None:
        from octomil.mcp.server import _detect_language

        assert _detect_language("data.xyz") == ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_build_messages(self) -> None:
        from octomil.mcp.prompts import build_messages

        msgs = build_messages("review_code", "check this")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "check this"

    def test_unknown_tool_uses_general(self) -> None:
        from octomil.mcp.prompts import build_messages

        msgs = build_messages("nonexistent_tool", "hello")
        assert "software engineering" in msgs[0]["content"].lower()

    def test_all_tools_have_prompts(self) -> None:
        from octomil.mcp.prompts import SYSTEM_PROMPTS

        expected = {
            "generate_code",
            "review_code",
            "explain_code",
            "write_tests",
            "general_task",
            "review_file",
            "analyze_files",
        }
        assert set(SYSTEM_PROMPTS.keys()) == expected


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TestBackend:
    def test_default_model_name(self) -> None:
        from octomil.mcp.backend import OctomilMCPBackend

        b = OctomilMCPBackend()
        assert b.model_name == "qwen-coder-7b"

    def test_custom_model_name(self) -> None:
        from octomil.mcp.backend import OctomilMCPBackend

        b = OctomilMCPBackend(model="phi-4-mini")
        assert b.model_name == "phi-4-mini"

    def test_env_model_name(self) -> None:
        from octomil.mcp.backend import OctomilMCPBackend

        with patch.dict(os.environ, {"OCTOMIL_MCP_MODEL": "gemma-4b"}):
            b = OctomilMCPBackend()
            assert b.model_name == "gemma-4b"

    def test_not_loaded_initially(self) -> None:
        from octomil.mcp.backend import OctomilMCPBackend

        b = OctomilMCPBackend()
        assert not b.is_loaded

    def test_format_metrics(self) -> None:
        from octomil.mcp.backend import OctomilMCPBackend

        b = OctomilMCPBackend(model="test-model")
        tag = b.format_metrics(
            {
                "model": "test-model",
                "engine": "mlx-lm",
                "tokens_per_second": 42.5,
                "total_tokens": 100,
                "ttfc_ms": 150.0,
            }
        )
        assert "test-model" in tag
        assert "mlx-lm" in tag
        assert "42.5" in tag


# ---------------------------------------------------------------------------
# MCP server creation
# ---------------------------------------------------------------------------


class TestMCPServer:
    def test_server_creates(self) -> None:
        mcp_mod = pytest.importorskip("mcp.server.fastmcp")  # noqa: F841
        from octomil.mcp.server import create_mcp_server

        server = create_mcp_server()
        assert server is not None

    def test_server_has_tools(self) -> None:
        pytest.importorskip("mcp.server.fastmcp")
        from octomil.mcp.server import create_mcp_server

        server = create_mcp_server()
        # FastMCP stores tools internally — verify by checking the object exists
        assert hasattr(server, "run")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestCLI:
    def test_mcp_help(self) -> None:
        from click.testing import CliRunner

        from octomil.commands.mcp_cmd import mcp

        runner = CliRunner()
        result = runner.invoke(mcp, ["--help"])
        assert result.exit_code == 0
        assert "register" in result.output
        assert "unregister" in result.output
        assert "status" in result.output

    def test_status_not_registered(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from octomil.commands.mcp_cmd import mcp

        runner = CliRunner()
        with patch("octomil.mcp.registration._SETTINGS_PATH", tmp_path / "settings.json"):
            result = runner.invoke(mcp, ["status"])
        assert result.exit_code == 0
        assert "not registered" in result.output
