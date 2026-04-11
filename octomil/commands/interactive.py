"""Interactive / demo commands for the Octomil CLI.

Contains: chat, warmup, launch, demo (code-assistant), completions,
and registration of the ``interactive`` command from cli_hw.
"""

from __future__ import annotations

import os
from typing import Optional

import click

from octomil.cli_helpers import _complete_model_name, cli_header

# ---------------------------------------------------------------------------
# octomil chat
# ---------------------------------------------------------------------------


@click.command()
@click.argument("model", required=False, default=None, shell_complete=_complete_model_name)
@click.option("--app", default=None, help="App context (slug).")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--select", is_flag=True, help="Interactively choose a local model.")
@click.option("--port", "-p", default=None, type=int, hidden=True)
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
    app: Optional[str],
    policy: Optional[str],
    select: bool,
    port: Optional[int],
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> None:
    """Chat with a model locally.

    Downloads the model if needed and opens an interactive chat using the
    same direct local/cloud routing path as ``octomil run``.

    \b
    Examples:
        octomil chat
        octomil chat gemma3-1b
        octomil chat --select
        octomil chat llama-8b --system "You are a Python expert."
        octomil chat gemma3-4b -t 0.3
    """
    from octomil.chat import run_chat_repl, stream_chat_via_kernel
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    if model is None:
        if select:
            from octomil.agents.launcher import _select_model_tui

            model = _select_model_tui()
        else:
            try:
                defaults = kernel.resolve_chat_defaults(policy=policy, app=app)
            except Exception as exc:
                raise click.ClickException(str(exc)) from exc
            model = defaults.model or "gemma3-1b"

    cli_header(f"Chat — {model}")
    if port is not None:
        click.echo("Note: --port is ignored by direct chat. Use `octomil serve` for a local HTTP server.", err=True)
    click.echo("Running through the shared execution kernel. Use `octomil serve` for a local HTTP server.\n")
    run_chat_repl(
        model,
        kernel,
        system_prompt=system,
        temperature=temperature,
        max_tokens=max_tokens,
        policy=policy,
        app=app,
        stream_fn=stream_chat_via_kernel,
    )


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

    cli_header("Warmup")

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
@click.argument("agent", required=False, default=None)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to serve (default: auto-select best for device).",
)
@click.option("--port", "-p", default=8080, help="Port for local server.")
@click.option("--select", "-s", is_flag=True, help="Interactively choose a model.")
def launch(agent: Optional[str], model: Optional[str], port: int, select: bool) -> None:
    """Launch a coding agent powered by a local model.

    Run without arguments to pick an agent interactively.
    Starts octomil serve in the background (if not already running),
    configures the agent's environment to point at the local
    OpenAI-compatible endpoint, and execs the agent.

    Without --model or --select, auto-picks the best model for your device.

    \b
    Examples:
        octomil launch
        octomil launch codex
        octomil launch codex --select
        octomil launch codex --model codestral
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
    cli_header("Demo — Code Assistant")

    from octomil.demos.code_assistant import run_demo

    effective_model: str = model if model else os.environ.get("OCTOMIL_MODEL", "gemma-2b")
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
    cli_header("Completions")

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
        "fish": "_OCTOMIL_COMPLETE=fish_source octomil | source",
    }
    rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}

    snippet = snippets[shell]
    rc_file = rc_files[shell]

    click.echo(f"Add this to {rc_file}:\n")
    click.secho(f"  {snippet}", bold=True)
    click.echo("\nOr run it now to enable for this session:\n")
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
