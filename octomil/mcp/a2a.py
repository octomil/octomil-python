"""A2A agent card generation from registered MCP tool definitions.

The agent card is generated at startup from the actual tool definitions,
so it's always in sync with the server's capabilities. No static file
to maintain.

See: https://google.github.io/A2A/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentCardConfig:
    """Configuration for agent card generation."""

    name: str = "Octomil Agent"
    description: str = "On-device ML inference, model resolution, and deployment"
    url: str = "http://localhost:8402"
    version: str = "1.0.0"
    provider: str = "Octomil"
    documentation_url: str = "https://docs.octomil.com"
    extra_capabilities: dict[str, bool] = field(default_factory=dict)


def _tool_to_skill(
    name: str,
    description: str,
    schema: dict[str, Any] | None = None,
    ready: bool | None = None,
) -> dict[str, Any]:
    """Convert a tool definition to an A2A skill entry."""
    skill: dict[str, Any] = {
        "id": name,
        "name": name,
        "description": description or f"Octomil {name} tool",
    }
    if schema:
        skill["inputSchema"] = schema
    if ready is not None:
        skill["ready"] = ready
    return skill


def build_agent_card(
    tools: list[dict[str, Any]],
    config: AgentCardConfig | None = None,
    model_ready: bool = False,
) -> dict[str, Any]:
    """Build an A2A agent card from tool definitions.

    Parameters
    ----------
    tools:
        List of dicts with at minimum ``name`` and ``description`` keys.
        Optionally include ``inputSchema`` for parameter schemas and
        ``requires_model`` (bool) for readiness gating.
    config:
        Agent card configuration. Uses defaults if not provided.
    model_ready:
        Whether the inference model is currently loaded. Skills that
        require a model will have ``ready: false`` when this is False.

    Returns
    -------
    dict
        A2A agent card JSON-serializable dict.
    """
    if config is None:
        config = AgentCardConfig()

    skills = []
    for t in tools:
        requires_model = t.get("requires_model", False)
        ready = True if not requires_model else model_ready
        skills.append(_tool_to_skill(t["name"], t.get("description", ""), t.get("inputSchema"), ready=ready))

    capabilities: dict[str, Any] = {
        "mcp": True,
        "a2a": True,
        "openapi": True,
        "streaming": False,
        "pushNotifications": False,
    }
    capabilities.update(config.extra_capabilities)

    card: dict[str, Any] = {
        "name": config.name,
        "description": config.description,
        "url": config.url,
        "version": config.version,
        "provider": {
            "organization": config.provider,
            "url": config.documentation_url,
        },
        "capabilities": capabilities,
        "skills": skills,
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "authSchemes": [
            {
                "scheme": "apiKey",
                "service_identifier": "Authorization",
                "description": "Bearer token in Authorization header",
            }
        ],
        "readinessUrls": {
            "warmup": f"{config.url}/api/v1/warmup",
            "ready": f"{config.url}/api/v1/ready",
        },
    }

    return card
