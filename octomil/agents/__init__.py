"""Agent registry and launcher for ``octomil launch``."""

from __future__ import annotations

from .registry import AGENTS, AgentDef, get_agent, is_agent_installed, list_agents

__all__ = [
    "AGENTS",
    "AgentDef",
    "get_agent",
    "is_agent_installed",
    "list_agents",
]
