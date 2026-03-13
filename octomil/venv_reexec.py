"""Shared venv re-exec logic for frozen binary commands.

When Octomil is installed as a PyInstaller binary, native Python packages
like ``mlx-lm`` are not bundled. Commands that need native engine access
call ``try_venv_reexec()`` to transparently re-launch into the managed
venv at ``~/.octomil/engines/venv/`` via ``os.execv()``.

Extracted from ``commands/serve.py`` so that ``benchmark``, ``mcp serve``,
and any future engine-dependent commands share the same logic.
"""

from __future__ import annotations

import os
import sys

import click


def needs_venv_reexec() -> bool:
    """Return True if running from a frozen binary with no native engines."""
    return getattr(sys, "frozen", False) is True


def try_venv_reexec() -> bool:
    """Try to re-exec into the managed venv's Python for native engine support.

    When running from a frozen binary with no native engines, this checks
    if ``octomil setup`` has prepared a venv with mlx-lm or llama.cpp.
    If so, ``os.execv()`` replaces this process entirely with the venv's
    Python running the original command. Single process, no proxy.

    If no venv exists and setup hasn't run yet, runs setup inline so that
    the command is a single-command experience with no separate step.

    Returns True if re-exec was initiated (unreachable after os.execv).
    Returns False if setup failed and we should fall through.
    """
    from octomil.setup import (
        get_venv_python,
        is_engine_ready,
        is_setup_in_progress,
        load_state,
        run_setup,
    )

    if is_setup_in_progress():
        wait_for_setup()

    venv_py = get_venv_python()
    if not venv_py or not is_engine_ready():
        # No venv yet — run setup inline so the command just works
        state = load_state()
        if state.phase != "failed":
            click.echo(
                click.style(
                    "\n  First run: setting up native inference engine...\n",
                    fg="cyan",
                )
            )
            result = run_setup()
            if result.phase == "failed":
                click.echo(click.style(f"  Setup failed: {result.error}", fg="red"))
                return False
            # Re-check after setup
            venv_py = get_venv_python()
            if not venv_py or not is_engine_ready():
                return False
        else:
            return False

    # Build the argv for re-exec: venv python -m octomil <original args>
    # Reconstruct original args from sys.argv (frozen binary: ["octomil", "serve", ...])
    # We need everything after the binary name in the original invocation.
    original_args = sys.argv[1:]  # ["serve", "model", "--port", "8080", ...]
    new_argv = [venv_py, "-m", "octomil", *original_args]

    click.echo(
        click.style(
            "\n  Re-launching with native engine via managed venv...\n",
            fg="green",
        )
    )
    os.execv(venv_py, new_argv)
    return True  # unreachable after execv


def wait_for_setup() -> None:
    """Wait for a running ``octomil setup`` to finish, showing progress."""
    import time

    from octomil.setup import is_setup_in_progress, load_state

    click.echo("\n  Engine setup is in progress, waiting...")
    phase_labels = {
        "creating_venv": "creating virtual environment",
        "installing_engine": "installing inference engine",
        "downloading_model": "downloading model",
    }

    last_phase = ""
    waited = 0
    while is_setup_in_progress() and waited < 600:
        state = load_state()
        label = phase_labels.get(state.phase, state.phase)
        if state.phase != last_phase:
            click.echo(f"    {label}...")
            last_phase = state.phase
        time.sleep(2)
        waited += 2

    state = load_state()
    if state.phase == "complete":
        click.echo(click.style("  Setup complete.", fg="green"))
    elif state.phase == "failed":
        click.echo(click.style(f"  Setup failed: {state.error}", fg="red"))
