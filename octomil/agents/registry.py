"""Agent registry for ``octomil launch`` integrations."""

from __future__ import annotations

import dataclasses
import shutil
from typing import Any, Callable, Optional


@dataclasses.dataclass
class AgentDef:
    """Definition for a coding agent that can be launched via ``octomil launch``."""

    name: str  # short key: "claude", "codex", etc.
    display_name: str  # "Claude Code", "Codex CLI", etc.
    description: str  # one-line description for picker
    env_key: str  # env var for API base URL
    install_check: str  # command to check if installed (e.g. "claude")
    install_cmd: str  # command to install the agent
    exec_cmd: str  # command to launch the agent
    needs_local_model: bool = True  # whether agent requires a local model server
    # Format string for --model flag when using a local model.
    # Use {model} as placeholder.  e.g. "--model openai/{model}"
    model_flag: Optional[str] = None
    # Optional hook to configure the agent for a local model server.
    # Called with (base_url, model_name) before launch.  Returns env overrides.
    configure_local: Optional[Callable[[str, str], dict[str, str]]] = dataclasses.field(default=None, repr=False)


def _configure_openclaw(base_url: str, model: str) -> dict[str, str]:
    """Configure OpenClaw to use a local octomil serve endpoint.

    OpenClaw reads model providers from its config file, not env vars.
    We set the provider as a single JSON object (openclaw validates all
    required fields together), then set the default model.
    """
    import json
    import subprocess

    provider = json.dumps(
        {
            "baseUrl": base_url,
            "apiKey": "octomil-local",
            "api": "openai-completions",
            "models": [{"id": model, "name": model}],
        }
    )
    cmds = [
        ["openclaw", "config", "set", "models.providers.octomil", provider],
        ["openclaw", "config", "set", "agents.defaults.model.primary", f"octomil/{model}"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"Failed to configure OpenClaw: {' '.join(cmd[2:])}\n  {detail}")

    return {}  # no env overrides needed


def _configure_opencode(base_url: str, model: str) -> dict[str, str]:
    """Configure OpenCode to use a local octomil serve endpoint.

    OpenCode uses a JSON config file with provider definitions.
    We write an ``opencode.json`` in the current directory that defines
    an ``octomil`` provider pointing at the local server.
    """
    import json
    import os

    provider_def: dict[str, Any] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Octomil Local",
        "options": {
            "baseURL": base_url,
            "apiKey": "octomil-local",
        },
        "models": {
            model: {"name": model},
        },
    }

    config_dir = os.path.expanduser("~/.config/opencode")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "opencode.json")

    # Merge with existing config if present
    existing: dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
    existing.setdefault("$schema", "https://opencode.ai/config.json")
    providers: dict[str, Any] = existing.setdefault("provider", {})
    providers["octomil"] = provider_def

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    return {}  # no env overrides needed


AGENTS: dict[str, AgentDef] = {
    "claude": AgentDef(
        name="claude",
        display_name="Claude Code",
        description="Anthropic's agentic coding tool",
        env_key="ANTHROPIC_BASE_URL",
        install_check="claude",
        install_cmd="npm install -g @anthropic-ai/claude-code",
        exec_cmd="claude",
        needs_local_model=False,
        model_flag="--model {model}",
    ),
    "codex": AgentDef(
        name="codex",
        display_name="Codex CLI",
        description="OpenAI's lightweight coding agent",
        env_key="OPENAI_BASE_URL",
        install_check="codex",
        install_cmd="npm install -g @openai/codex",
        exec_cmd="codex",
        model_flag="-c model_provider=openai --model {model}",
    ),
    "droid": AgentDef(
        name="droid",
        display_name="Droid",
        description="Factory's software development agent",
        env_key="OPENAI_BASE_URL",
        install_check="droid",
        install_cmd="npm install -g droid",
        exec_cmd="droid",
        needs_local_model=False,
    ),
    "opencode": AgentDef(
        name="opencode",
        display_name="OpenCode",
        description="Open source coding agent, 75+ providers",
        env_key="OPENAI_BASE_URL",
        install_check="opencode",
        install_cmd="npm install -g opencode-ai@latest",
        exec_cmd="opencode",
        model_flag="--model octomil/{model}",
        configure_local=_configure_opencode,
    ),
    "openclaw": AgentDef(
        name="openclaw",
        display_name="OpenClaw",
        description="Conversational agent with chat channels",
        env_key="OPENAI_BASE_URL",
        install_check="openclaw",
        install_cmd="npm install -g openclaw",
        exec_cmd="openclaw",
        configure_local=_configure_openclaw,
    ),
    "aider": AgentDef(
        name="aider",
        display_name="Aider",
        description="AI pair programming in your terminal",
        env_key="OPENAI_API_BASE",
        install_check="aider",
        install_cmd="pip install aider-chat",
        exec_cmd="aider",
        model_flag="--model openai/{model} --no-show-model-warnings",
    ),
}


def get_agent(name: str) -> Optional[AgentDef]:
    """Return the agent definition for *name*, or ``None`` if unknown."""
    return AGENTS.get(name)


def list_agents() -> list[AgentDef]:
    """Return all registered agent definitions."""
    return list(AGENTS.values())


def is_agent_installed(agent: AgentDef) -> bool:
    """Return ``True`` if the agent binary is on ``$PATH``."""
    return shutil.which(agent.install_check) is not None
