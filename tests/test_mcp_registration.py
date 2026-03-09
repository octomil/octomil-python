"""Tests for MCP server registration in Claude Code settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def settings_dir(tmp_path: Path):
    """Provide a temp settings path and patch the module constant."""
    settings_file = tmp_path / "settings.json"
    with patch("octomil.mcp.registration._SETTINGS_PATH", settings_file):
        yield settings_file


class TestRegister:
    def test_creates_settings_file(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        register_mcp_server()
        assert settings_dir.is_file()
        data = json.loads(settings_dir.read_text())
        assert "octomil" in data["mcpServers"]

    def test_uses_sys_executable(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        register_mcp_server()
        data = json.loads(settings_dir.read_text())
        assert data["mcpServers"]["octomil"]["command"] == sys.executable

    def test_preserves_existing_settings(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        settings_dir.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "foo"}}}))
        register_mcp_server()
        data = json.loads(settings_dir.read_text())
        assert data["theme"] == "dark"
        assert "other" in data["mcpServers"]
        assert "octomil" in data["mcpServers"]

    def test_overwrites_existing_octomil_entry(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        register_mcp_server(model="old-model")
        register_mcp_server(model="new-model")
        data = json.loads(settings_dir.read_text())
        assert data["mcpServers"]["octomil"]["env"]["OCTOMIL_MCP_MODEL"] == "new-model"

    def test_returns_path(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        result = register_mcp_server()
        assert result == str(settings_dir)

    def test_sets_model_env(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server

        register_mcp_server(model="phi-4-mini")
        data = json.loads(settings_dir.read_text())
        assert data["mcpServers"]["octomil"]["env"]["OCTOMIL_MCP_MODEL"] == "phi-4-mini"


class TestUnregister:
    def test_removes_entry(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import register_mcp_server, unregister_mcp_server

        register_mcp_server()
        assert unregister_mcp_server() is True
        data = json.loads(settings_dir.read_text())
        assert "mcpServers" not in data

    def test_preserves_other_servers(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import unregister_mcp_server

        settings_dir.write_text(json.dumps({"mcpServers": {"octomil": {}, "other": {"command": "bar"}}}))
        assert unregister_mcp_server() is True
        data = json.loads(settings_dir.read_text())
        assert "other" in data["mcpServers"]
        assert "octomil" not in data["mcpServers"]

    def test_returns_false_when_not_registered(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import unregister_mcp_server

        assert unregister_mcp_server() is False


class TestIsRegistered:
    def test_true_when_registered(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import is_registered, register_mcp_server

        register_mcp_server()
        assert is_registered() is True

    def test_false_when_not_registered(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import is_registered

        assert is_registered() is False


class TestGetRegistrationInfo:
    def test_returns_config(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import get_registration_info, register_mcp_server

        register_mcp_server(model="test-model")
        info = get_registration_info()
        assert info is not None
        assert info["command"] == sys.executable
        assert info["env"]["OCTOMIL_MCP_MODEL"] == "test-model"

    def test_returns_none_when_not_registered(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import get_registration_info

        assert get_registration_info() is None


class TestMalformedSettings:
    def test_invalid_json(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import RegistrationError, register_mcp_server

        settings_dir.write_text("{not valid json")
        with pytest.raises(RegistrationError, match="Malformed JSON"):
            register_mcp_server()

    def test_invalid_mcp_servers_type(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import RegistrationError, register_mcp_server

        settings_dir.write_text(json.dumps({"mcpServers": "not a dict"}))
        with pytest.raises(RegistrationError, match="Expected mcpServers"):
            register_mcp_server()

    def test_non_dict_root(self, settings_dir: Path) -> None:
        from octomil.mcp.registration import RegistrationError, register_mcp_server

        settings_dir.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(RegistrationError, match="Expected dict"):
            register_mcp_server()
