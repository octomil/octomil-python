"""Launch coding agents powered by local EdgeML models."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from typing import Optional

import click


def is_serve_running(host: str = "localhost", port: int = 8080) -> bool:
    """Check if ``edgeml serve`` is already running."""
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def start_serve_background(model: str, port: int = 8080) -> subprocess.Popen:
    """Start ``edgeml serve`` in the background and wait until ready."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "edgeml", "serve", model, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if is_serve_running(port=port):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("edgeml serve failed to start within 60s")


def launch_agent(
    agent_name: str,
    model: Optional[str] = None,
    port: int = 8080,
) -> None:
    """Launch a coding agent with a local model backend.

    1. Ensures the agent binary is installed (offers to install if not).
    2. Starts ``edgeml serve`` in the background if no server is running.
    3. Sets the appropriate env var so the agent talks to the local server.
    4. Execs the agent and tears down the server on exit.
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
            subprocess.run(agent.install_cmd, shell=True, check=True)
        else:
            raise SystemExit(1)

    # Start serve if not running
    serve_proc: Optional[subprocess.Popen] = None
    base_url = f"http://localhost:{port}/v1"
    if not is_serve_running(port=port):
        if model is None:
            model = "qwen3"
        click.echo(f"Starting edgeml serve {model}...")
        serve_proc = start_serve_background(model, port=port)
        click.echo(f"Model ready at {base_url}")
    else:
        click.echo(f"Using existing server at {base_url}")

    # Set env and exec
    env = os.environ.copy()
    env[agent.env_key] = base_url
    if agent.env_key.startswith("OPENAI"):
        env["OPENAI_API_KEY"] = "edgeml-local"

    try:
        click.echo(f"Launching {agent.display_name}...\n")
        result = subprocess.run([agent.exec_cmd], env=env)
        sys.exit(result.returncode)
    finally:
        if serve_proc is not None:
            serve_proc.terminate()
