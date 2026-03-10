"""Tests for MCP server registration across AI coding tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from octomil.mcp.registration import (
    MCPTarget,
    RegistrationError,
    get_all_status,
    is_registered,
    register_mcp_server,
    unregister_mcp_server,
)


def _make_targets(tmp_path: Path) -> list[MCPTarget]:
    """Create test targets pointing at tmp_path."""
    return [
        MCPTarget("claude", "Claude Code", tmp_path / "claude" / "settings.json", "json", "mcpServers"),
        MCPTarget("cursor", "Cursor", tmp_path / "cursor" / "mcp.json", "json", "mcpServers"),
        MCPTarget("vscode", "VS Code", tmp_path / "vscode" / "mcp.json", "json", "servers"),
        MCPTarget("codex", "Codex CLI", tmp_path / "codex" / "config.toml", "toml", "mcp.servers"),
    ]


@pytest.fixture()
def targets(tmp_path: Path):
    """Patch TARGETS to use temp dirs."""
    test_targets = _make_targets(tmp_path)
    with patch("octomil.mcp.registration.TARGETS", test_targets):
        yield test_targets


class TestRegisterAll:
    def test_creates_all_config_files(self, targets: list[MCPTarget]) -> None:
        results = register_mcp_server()
        assert all(r.success for r in results)
        for t in targets:
            assert t.path.is_file(), f"{t.display} config not created"

    def test_json_targets_correct_structure(self, targets: list[MCPTarget]) -> None:
        register_mcp_server(model="test-model")
        for t in targets:
            if t.format != "json":
                continue
            data = json.loads(t.path.read_text())
            entry = data[t.server_key]["octomil"]
            assert entry["command"] == sys.executable
            assert entry["args"] == ["-m", "octomil.mcp"]
            assert entry["env"]["OCTOMIL_MCP_MODEL"] == "test-model"

    def test_toml_target_correct_structure(self, targets: list[MCPTarget]) -> None:
        register_mcp_server(model="phi-mini")
        codex = next(t for t in targets if t.name == "codex")
        content = codex.path.read_text()
        assert "[mcp.servers.octomil]" in content
        assert sys.executable in content
        assert "phi-mini" in content

    def test_no_model_omits_env(self, targets: list[MCPTarget]) -> None:
        register_mcp_server()
        claude = next(t for t in targets if t.name == "claude")
        data = json.loads(claude.path.read_text())
        assert data["mcpServers"]["octomil"]["env"] == {}


class TestRegisterTarget:
    def test_register_single_target(self, targets: list[MCPTarget]) -> None:
        results = register_mcp_server(target="claude")
        assert len(results) == 1
        assert results[0].target == "claude"
        assert results[0].success

    def test_invalid_target_raises(self, targets: list[MCPTarget]) -> None:
        with pytest.raises(RegistrationError, match="Unknown target"):
            register_mcp_server(target="invalid")


class TestPreserveExisting:
    def test_preserves_other_settings(self, targets: list[MCPTarget]) -> None:
        claude = next(t for t in targets if t.name == "claude")
        claude.path.parent.mkdir(parents=True, exist_ok=True)
        claude.path.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "foo"}}}))
        register_mcp_server()
        data = json.loads(claude.path.read_text())
        assert data["theme"] == "dark"
        assert "other" in data["mcpServers"]
        assert "octomil" in data["mcpServers"]

    def test_overwrites_existing_octomil(self, targets: list[MCPTarget]) -> None:
        register_mcp_server(model="old")
        register_mcp_server(model="new")
        claude = next(t for t in targets if t.name == "claude")
        data = json.loads(claude.path.read_text())
        assert data["mcpServers"]["octomil"]["env"]["OCTOMIL_MCP_MODEL"] == "new"


class TestUnregister:
    def test_removes_from_all(self, targets: list[MCPTarget]) -> None:
        register_mcp_server()
        results = unregister_mcp_server()
        assert all(r.success for r in results)
        for t in targets:
            if t.format == "json":
                data = json.loads(t.path.read_text())
                assert "octomil" not in data.get(t.server_key, {})

    def test_preserves_other_servers(self, targets: list[MCPTarget]) -> None:
        claude = next(t for t in targets if t.name == "claude")
        claude.path.parent.mkdir(parents=True, exist_ok=True)
        claude.path.write_text(json.dumps({"mcpServers": {"octomil": {}, "other": {"command": "bar"}}}))
        unregister_mcp_server(target="claude")
        data = json.loads(claude.path.read_text())
        assert "other" in data["mcpServers"]
        assert "octomil" not in data["mcpServers"]

    def test_not_registered_returns_false(self, targets: list[MCPTarget]) -> None:
        results = unregister_mcp_server()
        assert not any(r.success for r in results)


class TestStatus:
    def test_all_registered(self, targets: list[MCPTarget]) -> None:
        register_mcp_server()
        statuses = get_all_status()
        assert all(statuses.values())

    def test_none_registered(self, targets: list[MCPTarget]) -> None:
        statuses = get_all_status()
        assert not any(statuses.values())

    def test_partial_registration(self, targets: list[MCPTarget]) -> None:
        register_mcp_server(target="claude")
        statuses = get_all_status()
        assert statuses["claude"] is True
        assert statuses["cursor"] is False


class TestBackwardCompat:
    def test_is_registered_checks_claude(self, targets: list[MCPTarget]) -> None:
        assert is_registered() is False
        register_mcp_server(target="claude")
        assert is_registered() is True


class TestMalformedConfig:
    def test_invalid_json_fails_gracefully(self, targets: list[MCPTarget]) -> None:
        claude = next(t for t in targets if t.name == "claude")
        claude.path.parent.mkdir(parents=True, exist_ok=True)
        claude.path.write_text("{not valid json")
        results = register_mcp_server(target="claude")
        assert not results[0].success
        assert "Malformed JSON" in (results[0].error or "")
