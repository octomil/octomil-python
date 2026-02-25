"""Interactive / demo commands for the Octomil CLI.

Contains: chat, warmup, launch, demo (code-assistant), completions,
and registration of the ``interactive`` command from cli_hw.
"""

from __future__ import annotations

import os
from typing import Optional

import click

from octomil.cli_helpers import _complete_model_name, _get_api_key


# ---------------------------------------------------------------------------
# octomil chat
# ---------------------------------------------------------------------------


@click.command()
@click.argument("model", required=False, default=None, shell_complete=_complete_model_name)
@click.option("--port", "-p", default=8080, help="Port for local server.")
@click.option("--system", "-s", default=None, help="System prompt.")
@click.option(
    "--temperature",
    "-t",
    default=0.7,
    type=float,
    help="Sampling temperature (default: 0.7).",
)
@click.option(
    "--max-tokens",
    default=2048,
    type=int,
    help="Max tokens per response (default: 2048).",
)
def chat(
    model: Optional[str],
    port: int,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> None:
    """Chat with a model locally.

    Starts the server, downloads the model if needed, and opens an
    interactive chat. One command from install to conversation.

    \b
    Examples:
        octomil chat
        octomil chat qwen-coder-7b
        octomil chat llama-8b --system "You are a Python expert."
        octomil chat qwen-coder-3b -t 0.3
    """
    from octomil.agents.launcher import (
        _auto_select_model,
        is_serve_running,
        start_serve_background,
    )
    from octomil.chat import run_chat_repl

    if model is None:
        model = _auto_select_model()

    serve_proc = None
    if not is_serve_running(port=port):
        click.echo(f"Starting octomil serve {model}...")
        serve_proc = start_serve_background(model, port=port)
        click.echo("Ready.\n")
    else:
        click.echo(f"Using existing server at localhost:{port}\n")

    url = f"http://localhost:{port}"
    try:
        run_chat_repl(
            url,
            model,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        if serve_proc is not None:
            serve_proc.terminate()


# ---------------------------------------------------------------------------
# octomil warmup
# ---------------------------------------------------------------------------


@click.command()
def warmup() -> None:
    """Pre-download the recommended model for your device.

    Detects available hardware, picks the best model, and downloads it
    so that subsequent ``octomil launch`` or ``octomil chat`` calls start
    instantly with zero cold-start delay.

    \b
    Examples:
        octomil warmup
    """
    from octomil.agents.launcher import _build_recommendations
    from octomil.sources.huggingface import HuggingFaceSource

    recommendations = _build_recommendations()
    best = recommendations[0]

    if best.downloaded:
        click.echo(f"{best.key} is already downloaded. Nothing to do.")
        return

    click.echo(f"Downloading {best.key} ({best.size})...")

    try:
        from octomil.models.catalog import get_model

        entry = get_model(best.key)
        if entry is None:
            raise click.ClickException(f"Model '{best.key}' not found in catalog.")

        variant = entry.variants.get(entry.default_quant)
        if variant is None:
            raise click.ClickException(f"No default variant for '{best.key}'.")

        hf = HuggingFaceSource()
        if variant.mlx:
            hf.resolve(variant.mlx)
        elif variant.gguf:
            hf.resolve(variant.gguf.repo, variant.gguf.filename)
        else:
            raise click.ClickException(f"No downloadable artifact for '{best.key}'.")

        click.echo(f"{best.key} downloaded successfully.")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# octomil launch
# ---------------------------------------------------------------------------


@click.command()
@click.argument(
    "agent",
    type=click.Choice(["claude", "codex", "openclaw", "aider"], case_sensitive=False),
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to serve (default: auto-select best for device).",
)
@click.option("--port", "-p", default=8080, help="Port for local server.")
@click.option("--select", "-s", is_flag=True, help="Interactively choose a model.")
def launch(agent: str, model: Optional[str], port: int, select: bool) -> None:
    """Launch a coding agent powered by a local model.

    Starts octomil serve in the background (if not already running),
    configures the agent's environment to point at the local
    OpenAI-compatible endpoint, and execs the agent.

    Without --model or --select, auto-picks the best model for your device.

    \b
    Examples:
        octomil launch codex
        octomil launch codex --select
        octomil launch codex --model codestral
        octomil launch aider --model deepseek-coder-v2
    """
    from octomil.agents.launcher import launch_agent

    launch_agent(agent, model=model, port=port, select=select)


# ---------------------------------------------------------------------------
# octomil demo
# ---------------------------------------------------------------------------


@click.group()
def demo() -> None:
    """Run interactive demos showcasing Octomil capabilities."""


@demo.command("code-assistant")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to serve (default: gemma-2b, or OCTOMIL_MODEL env var).",
)
@click.option(
    "--url",
    default=None,
    help="Connect to an existing octomil serve instance instead of auto-starting.",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8099,
    help="Port for auto-started server (default: 8099).",
)
@click.option(
    "--no-auto-start",
    is_flag=True,
    help="Don't auto-start octomil serve; fail if no server is running.",
)
def demo_code_assistant(
    model: Optional[str],
    url: Optional[str],
    port: int,
    no_auto_start: bool,
) -> None:
    """Local code assistant -- 100% on-device, zero cloud, zero cost.

    Starts an interactive chat powered by a local LLM through octomil serve.
    Shows real-time performance metrics: latency, throughput, cost savings.

    \b
    Examples:
        octomil demo code-assistant
        octomil demo code-assistant --model phi-3-mini
        octomil demo code-assistant --url http://localhost:8080
    """
    from octomil.demos.code_assistant import run_demo

    effective_model: str = (
        model if model else os.environ.get("OCTOMIL_MODEL", "gemma-2b")
    )
    effective_url: str = url if url else f"http://localhost:{port}"
    run_demo(url=effective_url, model=effective_model, auto_start=not no_auto_start)


# ---------------------------------------------------------------------------
# octomil completions
# ---------------------------------------------------------------------------


@click.command()
@click.argument("shell", required=False, default=None, type=click.Choice(["bash", "zsh", "fish"]))
def completions(shell: Optional[str]) -> None:
    """Print shell completion setup instructions.

    Prints the eval snippet needed to enable tab completion for octomil
    commands and model names in your current shell.
    """
    if shell is None:
        # Auto-detect from SHELL env
        user_shell = os.path.basename(os.environ.get("SHELL", ""))
        if user_shell in ("bash", "zsh", "fish"):
            shell = user_shell
        else:
            shell = "zsh"

    snippets = {
        "bash": 'eval "$(_OCTOMIL_COMPLETE=bash_source octomil)"',
        "zsh": 'eval "$(_OCTOMIL_COMPLETE=zsh_source octomil)"',
        "fish": '_OCTOMIL_COMPLETE=fish_source octomil | source',
    }
    rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}

    snippet = snippets[shell]
    rc_file = rc_files[shell]

    click.echo(f"Add this to {rc_file}:\n")
    click.secho(f"  {snippet}", bold=True)
    click.echo(f"\nOr run it now to enable for this session:\n")
    click.echo(f"  {snippet}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register all interactive/demo commands onto the main CLI group."""
    cli.add_command(chat)
    cli.add_command(warmup)
    cli.add_command(launch)
    cli.add_command(demo)
    cli.add_command(completions)

    # Register the hardware-aware interactive command from cli_hw
    from octomil.cli_hw import interactive_cmd_factory

    cli.add_command(interactive_cmd_factory(cli), "interactive")
