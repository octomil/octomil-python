"""Register/unregister the Octomil MCP server across AI coding tools.

Supports Claude Code, Cursor, VS Code, and OpenAI Codex CLI.
Each tool has its own config file location and format but the server
spec (command, args, env) is the same across all of them.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SERVER_NAME = "octomil"


class RegistrationError(RuntimeError):
    """Raised when a config file is malformed or unreadable."""


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPTarget:
    """An AI tool that supports MCP server configuration."""

    name: str  # short key: "claude", "cursor", "vscode", "codex"
    display: str  # human-readable: "Claude Code", "Cursor", etc.
    path: Path  # config file path
    format: str  # "json" or "toml"
    server_key: str  # top-level key: "mcpServers" or "servers"


TARGETS: list[MCPTarget] = [
    MCPTarget("claude", "Claude Code", Path.home() / ".claude" / "settings.json", "json", "mcpServers"),
    MCPTarget("cursor", "Cursor", Path.home() / ".cursor" / "mcp.json", "json", "mcpServers"),
    MCPTarget("vscode", "VS Code", Path.home() / ".vscode" / "mcp.json", "json", "servers"),
    MCPTarget("codex", "Codex CLI", Path.home() / ".codex" / "config.toml", "toml", "mcp.servers"),
]


def _get_targets(target_filter: Optional[str] = None) -> list[MCPTarget]:
    """Return targets matching the filter, or all targets."""
    if target_filter is None:
        return list(TARGETS)
    for t in TARGETS:
        if t.name == target_filter:
            return [t]
    valid = ", ".join(t.name for t in TARGETS)
    raise RegistrationError(f"Unknown target '{target_filter}'. Valid: {valid}")


# ---------------------------------------------------------------------------
# JSON config helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON config file, returning {} if missing."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RegistrationError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistrationError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON config, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _register_json(target: MCPTarget, server_entry: dict[str, Any]) -> None:
    """Add the octomil server to a JSON config file."""
    data = _read_json(target.path)
    servers = data.setdefault(target.server_key, {})
    if not isinstance(servers, dict):
        raise RegistrationError(f"Expected {target.server_key} to be dict in {target.path}")
    servers[_SERVER_NAME] = server_entry
    _write_json(target.path, data)


def _unregister_json(target: MCPTarget) -> bool:
    """Remove octomil from a JSON config file. Returns True if removed."""
    data = _read_json(target.path)
    servers = data.get(target.server_key, {})
    if not isinstance(servers, dict) or _SERVER_NAME not in servers:
        return False
    del servers[_SERVER_NAME]
    if not servers:
        del data[target.server_key]
    _write_json(target.path, data)
    return True


def _is_registered_json(target: MCPTarget) -> bool:
    """Check if octomil is registered in a JSON config."""
    try:
        data = _read_json(target.path)
    except RegistrationError:
        return False
    servers = data.get(target.server_key, {})
    return isinstance(servers, dict) and _SERVER_NAME in servers


# ---------------------------------------------------------------------------
# TOML config helpers (Codex CLI)
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML config file, returning {} if missing."""
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            raise RegistrationError("tomllib/tomli not available for TOML parsing")
    try:
        data = tomllib.loads(path.read_text())
    except Exception as exc:
        raise RegistrationError(f"Malformed TOML in {path}: {exc}") from exc
    return data


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    """Write TOML config. Uses a simple serializer for the flat MCP structure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in data.items():
        if key == "mcp":
            # MCP section with nested servers
            servers = value.get("servers", {})
            for srv_name, srv_cfg in servers.items():
                lines.append(f"[mcp.servers.{srv_name}]")
                for k, v in srv_cfg.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    elif isinstance(v, list):
                        items = ", ".join(f'"{i}"' for i in v)
                        lines.append(f"args = [{items}]")
                    elif isinstance(v, dict):
                        # Inline table for env
                        pairs = ", ".join(f'{ek} = "{ev}"' for ek, ev in v.items())
                        lines.append(f"env = {{ {pairs} }}")
                lines.append("")
        else:
            # Preserve other top-level keys as simple key = value
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _register_toml(target: MCPTarget, server_entry: dict[str, Any]) -> None:
    """Add the octomil server to a TOML config file."""
    data = _read_toml(target.path)
    mcp = data.setdefault("mcp", {})
    servers = mcp.setdefault("servers", {})
    # Convert server entry for TOML: command + args as flat keys
    toml_entry: dict[str, Any] = {
        "type": "stdio",
        "command": server_entry["command"],
        "args": server_entry.get("args", []),
    }
    env = server_entry.get("env", {})
    if env:
        toml_entry["env"] = env
    servers[_SERVER_NAME] = toml_entry
    _write_toml(target.path, data)


def _unregister_toml(target: MCPTarget) -> bool:
    """Remove octomil from a TOML config file. Returns True if removed."""
    data = _read_toml(target.path)
    servers = data.get("mcp", {}).get("servers", {})
    if _SERVER_NAME not in servers:
        return False
    del servers[_SERVER_NAME]
    if not servers:
        data.get("mcp", {}).pop("servers", None)
    if not data.get("mcp"):
        data.pop("mcp", None)
    _write_toml(target.path, data)
    return True


def _is_registered_toml(target: MCPTarget) -> bool:
    """Check if octomil is registered in a TOML config."""
    try:
        data = _read_toml(target.path)
    except RegistrationError:
        return False
    return _SERVER_NAME in data.get("mcp", {}).get("servers", {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Result of registering/unregistering with a single target."""

    target: str
    display: str
    path: str
    success: bool
    error: Optional[str] = None


def _build_server_entry(model: Optional[str] = None) -> dict[str, Any]:
    """Build the MCP server entry dict."""
    env: dict[str, str] = {}
    if model:
        env["OCTOMIL_MCP_MODEL"] = model
    return {
        "command": sys.executable,
        "args": ["-m", "octomil.mcp"],
        "env": env,
    }


def register_mcp_server(
    model: Optional[str] = None,
    target: Optional[str] = None,
) -> list[RegistrationResult]:
    """Register the Octomil MCP server across AI tools.

    Returns a list of results, one per target attempted.
    """
    targets = _get_targets(target)
    server_entry = _build_server_entry(model)
    results: list[RegistrationResult] = []

    for t in targets:
        try:
            if t.format == "json":
                _register_json(t, server_entry)
            elif t.format == "toml":
                _register_toml(t, server_entry)
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=True))
            logger.info("Registered in %s (%s)", t.display, t.path)
        except RegistrationError as exc:
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=False, error=str(exc)))
            logger.debug("Failed to register in %s: %s", t.display, exc)
        except Exception as exc:
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=False, error=str(exc)))
            logger.debug("Failed to register in %s: %s", t.display, exc)

    return results


def unregister_mcp_server(target: Optional[str] = None) -> list[RegistrationResult]:
    """Remove the Octomil MCP server from AI tools.

    Returns a list of results, one per target attempted.
    """
    targets = _get_targets(target)
    results: list[RegistrationResult] = []

    for t in targets:
        try:
            if t.format == "json":
                removed = _unregister_json(t)
            elif t.format == "toml":
                removed = _unregister_toml(t)
            else:
                removed = False
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=removed))
        except RegistrationError as exc:
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=False, error=str(exc)))
        except Exception as exc:
            results.append(RegistrationResult(t.name, t.display, str(t.path), success=False, error=str(exc)))

    return results


def get_all_status() -> dict[str, bool]:
    """Return registration status for each target."""
    status: dict[str, bool] = {}
    for t in TARGETS:
        if t.format == "json":
            status[t.name] = _is_registered_json(t)
        elif t.format == "toml":
            status[t.name] = _is_registered_toml(t)
    return status


# ---------------------------------------------------------------------------
# Backward-compatible API (used by existing tests and mcp_cmd.py)
# ---------------------------------------------------------------------------


def is_registered() -> bool:
    """Check if registered in Claude Code (backward compat)."""
    return get_all_status().get("claude", False)


def get_registration_info() -> Optional[dict[str, Any]]:
    """Return Claude Code registration info (backward compat)."""
    claude = next(t for t in TARGETS if t.name == "claude")
    try:
        data = _read_json(claude.path)
    except RegistrationError:
        return None
    servers = data.get(claude.server_key, {})
    if isinstance(servers, dict):
        return servers.get(_SERVER_NAME)  # type: ignore[return-value]
    return None


def get_settings_path() -> str:
    """Return Claude Code settings path (backward compat)."""
    claude = next(t for t in TARGETS if t.name == "claude")
    return str(claude.path)
