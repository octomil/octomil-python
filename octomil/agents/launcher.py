"""Launch coding agents powered by local Octomil models."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Recommended models for coding agents, ordered by preference.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommendedModel:
    """A model recommendation shown in the interactive picker."""

    key: str  # catalog key passed to ``octomil serve``
    label: str  # display name
    description: str  # one-line description
    size: str  # approximate download size
    recommended: bool = False  # highlight as top pick


RECOMMENDED_MODELS: list[RecommendedModel] = [
    RecommendedModel(
        key="qwen-coder-7b",
        label="qwen-coder-7b",
        description="Best local coding model, purpose-built for code",
        size="~4.5 GB",
        recommended=True,
    ),
    RecommendedModel(
        key="qwen-coder-3b",
        label="qwen-coder-3b",
        description="Fast coding model, runs on any machine",
        size="~2 GB",
    ),
    RecommendedModel(
        key="qwen-coder-1.5b",
        label="qwen-coder-1.5b",
        description="Ultra-light coding model, instant responses",
        size="~1 GB",
    ),
    RecommendedModel(
        key="qwen-7b",
        label="qwen-7b",
        description="Strong general-purpose with good code ability",
        size="~4.5 GB",
    ),
    RecommendedModel(
        key="llama-8b",
        label="llama-8b",
        description="Meta Llama 3.1, solid all-rounder",
        size="~4.5 GB",
    ),
]


def _is_model_downloaded(key: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        from ..models.catalog import get_model

        entry = get_model(key)
        if entry is None:
            return False

        variant = entry.variants.get(entry.default_quant)
        if variant is None:
            return False

        from ..sources.huggingface import HuggingFaceSource

        hf = HuggingFaceSource()
        if variant.mlx and hf.check_cache(variant.mlx):
            return True
        if variant.gguf and hf.check_cache(variant.gguf.repo, variant.gguf.filename):
            return True
    except Exception:
        pass
    return False


def _select_model() -> str:
    """Show an interactive model picker and return the chosen model key."""
    click.echo("\nModel Configuration\n")
    click.echo("  Recommended")

    for i, m in enumerate(RECOMMENDED_MODELS):
        downloaded = _is_model_downloaded(m.key)
        status = "downloaded" if downloaded else "not downloaded"
        marker = " (Recommended)" if m.recommended else ""
        prefix = "  > " if i == 0 else "    "
        click.echo(f"{prefix}{m.label}{marker}")
        click.echo(f"      {m.description}, {m.size}, ({status})")

    click.echo()

    choices = {str(i + 1): m.key for i, m in enumerate(RECOMMENDED_MODELS)}
    labels = {str(i + 1): m.label for i, m in enumerate(RECOMMENDED_MODELS)}

    hint_parts = [f"{k}={labels[k]}" for k in sorted(choices)]
    hint = ", ".join(hint_parts)

    selection = click.prompt(
        f"Select model [{hint}] or enter a model name",
        default="1",
    )

    if selection in choices:
        return choices[selection]
    return selection


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def is_serve_running(host: str = "localhost", port: int = 8080) -> bool:
    """Check if ``octomil serve`` is already running."""
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def start_serve_background(model: str, port: int = 8080) -> subprocess.Popen:
    """Start ``octomil serve`` in the background and wait until ready."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "octomil", "serve", model, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if is_serve_running(port=port):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("octomil serve failed to start within 60s")


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------


def launch_agent(
    agent_name: str,
    model: Optional[str] = None,
    port: int = 8080,
) -> None:
    """Launch a coding agent with a local model backend.

    1. Ensures the agent binary is installed (offers to install if not).
    2. If no model specified, shows an interactive picker.
    3. Starts ``octomil serve`` in the background if no server is running.
    4. Sets the appropriate env var so the agent talks to the local server.
    5. Execs the agent and tears down the server on exit.
    """
    from .registry import get_agent, is_agent_installed

    agent = get_agent(agent_name)
    if agent is None:
        from .registry import list_agents

        available = ", ".join(a.name for a in list_agents())
        raise click.ClickException(
            f"Unknown agent '{agent_name}'. Available: {available}"
        )

    # Check if agent is installed
    if not is_agent_installed(agent):
        click.echo(f"{agent.display_name} is not installed.")
        click.echo(f"  Install: {agent.install_cmd}")
        if click.confirm("Install now?"):
            subprocess.run(shlex.split(agent.install_cmd), check=True)
        else:
            raise SystemExit(1)

    # Agents that don't need a local model (e.g. Claude Code) are exec'd
    # directly unless the user explicitly passed --model to proxy through
    # the local server.
    use_local_server = agent.needs_local_model or model is not None

    serve_proc: Optional[subprocess.Popen] = None
    env = os.environ.copy()

    if use_local_server:
        base_url = f"http://localhost:{port}/v1"
        if not is_serve_running(port=port):
            if model is None:
                model = _select_model()
            click.echo(f"Starting octomil serve {model}...")
            serve_proc = start_serve_background(model, port=port)
            click.echo(f"Model ready at {base_url}")
        else:
            click.echo(f"Using existing server at {base_url}")

        env[agent.env_key] = base_url
        if agent.env_key.startswith("OPENAI"):
            env["OPENAI_API_KEY"] = "octomil-local"

    try:
        click.echo(f"Launching {agent.display_name}...\n")
        result = subprocess.run(shlex.split(agent.exec_cmd), env=env)
    finally:
        if serve_proc is not None:
            serve_proc.terminate()

    sys.exit(result.returncode)
