"""Server configuration and tool definitions for the HTTP agent server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HTTPServerConfig:
    """Configuration for the HTTP agent server."""

    host: str = "0.0.0.0"
    port: int = 8402
    model: Optional[str] = None
    enable_x402: bool = False
    x402_address: str = ""
    x402_price: str = "1000"  # base units (1000 = $0.001 USDC)
    x402_currency: str = "USDC"
    x402_network: str = "base"
    x402_threshold: int = 1_000_000  # base units = $1 USDC
    settler_url: str = "https://api.settle402.dev"  # settle402 batch settlement service
    settler_token: str = ""  # X-Settler-Token for auth
    base_url: str = ""  # auto-detected if empty


def _get_tool_definitions() -> list[dict[str, Any]]:
    """Return tool definitions for the agent card.

    These are read from the platform_tools module's registered functions.
    """
    from ..prompts import PLATFORM_TOOL_DESCRIPTIONS

    tools: list[dict[str, Any]] = []

    # Platform tools — run_inference requires a model, others don't
    model_required_platform = {"run_inference"}
    for name, desc in PLATFORM_TOOL_DESCRIPTIONS.items():
        tool_def: dict[str, Any] = {"name": name, "description": desc}
        if name in model_required_platform:
            tool_def["requires_model"] = True
        tools.append(tool_def)

    # Code tools (existing 7) — require a loaded model
    code_tools = {
        "generate_code": "Generate code from natural language description using on-device inference",
        "review_code": "Review code for bugs, security issues, and improvements",
        "explain_code": "Explain code in plain English",
        "write_tests": "Generate unit tests for code",
        "general_task": "Free-form prompt through the local model",
        "review_file": "Read a file from disk and review it locally",
        "analyze_files": "Read multiple files and answer a question about them",
    }
    for name, desc in code_tools.items():
        tools.append({"name": name, "description": desc, "requires_model": True})

    return tools
