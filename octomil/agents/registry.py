"""Agent registry for ``octomil launch`` integrations."""

from __future__ import annotations

import dataclasses
import shutil
from typing import Callable, Optional


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
    # Optional hook to configure the agent for a local model server.
    # Called with (base_url, model_name) before launch.  Returns env overrides.
    configure_local: Optional[Callable[[str, str], dict[str, str]]] = dataclasses.field(default=None, repr=False)


def _configure_openclaw(base_url: str, model: str) -> dict[str, str]:
    """Configure OpenClaw to use a local octomil serve endpoint.

    OpenClaw reads model providers from its config file, not env vars.
    We run ``openclaw config set`` to add an ``octomil`` provider pointing
    at the local server, then set it as the default model.
    """
    import subprocess

    cmds = [
        ["openclaw", "config", "set", "models.providers.octomil.baseUrl", base_url],
        ["openclaw", "config", "set", "models.providers.octomil.apiKey", "octomil-local"],
        ["openclaw", "config", "set", "models.providers.octomil.api", "openai-completions"],
        ["openclaw", "config", "set", "agents.defaults.model.primary", f"octomil/{model}"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True, capture_output=True)

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
    ),
    "codex": AgentDef(
        name="codex",
        display_name="Codex CLI",
        description="OpenAI's lightweight coding agent",
        env_key="OPENAI_BASE_URL",
        install_check="codex",
        install_cmd="npm install -g @openai/codex",
        exec_cmd="codex",
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
