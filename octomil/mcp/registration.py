"""Register/unregister the Octomil MCP server in Claude Code settings."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_SERVER_NAME = "octomil"


class RegistrationError(RuntimeError):
    """Raised when settings.json is malformed or unreadable."""


def _read_settings() -> dict[str, Any]:
    """Read ~/.claude/settings.json, returning {} if missing."""
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        raw = _SETTINGS_PATH.read_text()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistrationError(f"Malformed JSON in {_SETTINGS_PATH}: {exc}") from exc

    if not isinstance(data, dict):
        raise RegistrationError(f"Expected dict in {_SETTINGS_PATH}, got {type(data).__name__}")
    return data


def _write_settings(data: dict[str, Any]) -> None:
    """Write settings.json, creating ~/.claude/ if needed."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def register_mcp_server(model: str | None = None) -> str:
    """Register the Octomil MCP server in Claude Code settings.

    Uses the absolute path to ``sys.executable`` to avoid venv/PATH issues.

    Returns the path to settings.json.
    """
    settings = _read_settings()
    servers = settings.setdefault("mcpServers", {})

    if not isinstance(servers, dict):
        raise RegistrationError(f"Expected mcpServers to be dict, got {type(servers).__name__}")

    env: dict[str, str] = {}
    if model:
        env["OCTOMIL_MCP_MODEL"] = model

    servers[_SERVER_NAME] = {
        "command": sys.executable,
        "args": ["-m", "octomil.mcp"],
        "env": env,
    }

    _write_settings(settings)
    logger.info("Registered Octomil MCP server in %s", _SETTINGS_PATH)
    return str(_SETTINGS_PATH)


def unregister_mcp_server() -> bool:
    """Remove the Octomil MCP server entry from settings.json.

    Returns True if an entry was removed, False if none existed.
    """
    settings = _read_settings()
    servers = settings.get("mcpServers", {})

    if not isinstance(servers, dict) or _SERVER_NAME not in servers:
        return False

    del servers[_SERVER_NAME]

    if not servers:
        del settings["mcpServers"]

    _write_settings(settings)
    logger.info("Unregistered Octomil MCP server from %s", _SETTINGS_PATH)
    return True


def is_registered() -> bool:
    """Check if the Octomil MCP server is registered."""
    try:
        settings = _read_settings()
    except RegistrationError:
        return False
    servers = settings.get("mcpServers", {})
    return isinstance(servers, dict) and _SERVER_NAME in servers


def get_registration_info() -> dict[str, Any] | None:
    """Return the Octomil MCP server config, or None if not registered."""
    try:
        settings = _read_settings()
    except RegistrationError:
        return None
    servers = settings.get("mcpServers", {})
    if isinstance(servers, dict):
        return servers.get(_SERVER_NAME)  # type: ignore[return-value]
    return None


def get_settings_path() -> str:
    """Return the path to the Claude Code settings file."""
    return str(_SETTINGS_PATH)
