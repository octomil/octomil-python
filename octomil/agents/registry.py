"""Agent registry for ``octomil launch`` integrations."""

from __future__ import annotations

import dataclasses
import shutil
from typing import Optional


@dataclasses.dataclass
class AgentDef:
    """Definition for a coding agent that can be launched via ``octomil launch``."""

    name: str  # short key: "claude", "codex", etc.
    display_name: str  # "Claude Code", "Codex CLI", etc.
    env_key: str  # env var for API base URL
    install_check: str  # command to check if installed (e.g. "claude")
    install_cmd: str  # command to install the agent
    exec_cmd: str  # command to launch the agent
    needs_local_model: bool = True  # whether agent requires a local model server


AGENTS: dict[str, AgentDef] = {
    "claude": AgentDef(
        name="claude",
        display_name="Claude Code",
        env_key="ANTHROPIC_BASE_URL",
        install_check="claude",
        install_cmd="npm install -g @anthropic-ai/claude-code",
        exec_cmd="claude",
    ),
    "codex": AgentDef(
        name="codex",
        display_name="Codex CLI",
        env_key="OPENAI_BASE_URL",
        install_check="codex",
        install_cmd="npm install -g @openai/codex",
        exec_cmd="codex",
    ),
    "openclaw": AgentDef(
        name="openclaw",
        display_name="OpenClaw",
        env_key="OPENAI_BASE_URL",
        install_check="openclaw",
        install_cmd="npm install -g openclaw",
        exec_cmd="openclaw",
    ),
    "aider": AgentDef(
        name="aider",
        display_name="Aider",
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
