"""CLI command for ``octomil setup``."""

from __future__ import annotations

import click


def register(cli: click.Group) -> None:
    cli.add_command(setup)


@click.command()
@click.option(
    "--foreground",
    is_flag=True,
    help="Run silently (used by install.sh background process).",
)
@click.option(
    "--status",
    "show_status",
    is_flag=True,
    help="Show current setup progress and exit.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run setup even if already complete.",
)
def setup(foreground: bool, show_status: bool, force: bool) -> None:
    """Set up the native inference engine and download a model.

    \b
    After installing via ``curl -fsSL https://get.octomil.com | sh``,
    this runs automatically in the background. You can also run it
    manually:

    \b
        octomil setup               # interactive, shows progress
        octomil setup --status      # check progress of background setup
        octomil setup --force       # re-run from scratch

    \b
    What it does:
        1. Finds a system Python with venv support
        2. Creates ~/.octomil/engines/venv/
        3. Installs the best engine for your hardware (mlx-lm or llama.cpp)
        4. Downloads the recommended model for your device
    """
    from octomil.setup import (
        load_state,
        run_setup,
    )

    if show_status:
        _print_status(load_state())
        return

    state = run_setup(force=force, foreground=foreground)

    if not foreground and state.phase == "failed":
        raise SystemExit(1)


def _print_status(state) -> None:  # noqa: ANN001
    """Print human-readable setup status."""
    from octomil.setup import SETUP_LOG

    phase_labels = {
        "pending": "Not started",
        "creating_venv": "Creating virtual environment...",
        "installing_engine": "Installing inference engine...",
        "downloading_model": "Downloading model...",
        "complete": "Complete",
        "failed": "Failed",
    }

    label = phase_labels.get(state.phase, state.phase)
    click.echo(f"Setup status: {label}")

    if state.engine:
        click.echo(f"  Engine: {state.engine}")
    if state.package:
        click.echo(f"  Package: {state.package}")
    if state.engine_installed:
        click.echo(click.style("  Engine installed: yes", fg="green"))
    if state.model_key:
        dl = "yes" if state.model_downloaded else "no"
        color = "green" if state.model_downloaded else "yellow"
        click.echo(click.style(f"  Model ({state.model_key}): {dl}", fg=color))
    if state.error:
        click.echo(click.style(f"  Error: {state.error}", fg="red"))
    if state.phase not in ("pending", "complete", "failed"):
        click.echo(f"\n  Log: {SETUP_LOG}")
        click.echo("  Run 'octomil setup --status' to check again.")
