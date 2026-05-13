"""Shared managed-runtime re-exec logic for frozen binary commands.

Older Octomil binary installs could prepare a managed venv at
``~/.octomil/engines/venv/``. Frozen commands may re-launch into that
runtime if it already exists, but they do not create it implicitly. The
standalone installer must not reach for the user's Python or pip.

Extracted from ``commands/serve.py`` so that ``benchmark``, ``mcp serve``,
and any future engine-dependent commands share the same logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click


def needs_venv_reexec() -> bool:
    """Return True if running from a frozen binary with no native engines."""
    return getattr(sys, "frozen", False) is True


def _is_running_inside_managed_venv(venv_py: str) -> bool:
    try:
        return Path(sys.executable).resolve() == Path(venv_py).resolve()
    except OSError:
        return sys.executable == venv_py


def should_managed_venv_reexec(*, include_non_frozen: bool = False) -> bool:
    """Return True when this process should delegate to the managed engine venv.

    The curl installer creates a lightweight user-facing ``octomil`` entrypoint
    plus a managed engine venv under ``~/.octomil/engines/venv``. Engine-facing
    commands need to inspect the managed venv even when the lightweight
    entrypoint itself is not a frozen PyInstaller binary.
    """
    if os.environ.get("OCTOMIL_DISABLE_MANAGED_VENV_REEXEC") == "1":
        return False
    if os.environ.get("OCTOMIL_MANAGED_VENV_REEXECED") == "1":
        return False
    if needs_venv_reexec():
        return True
    if not include_non_frozen:
        return False

    argv0 = Path(sys.argv[0]).name
    return argv0 == "octomil"


def try_venv_reexec() -> bool:
    """Try to re-exec into an existing managed venv for native engine support.

    When running from a frozen binary with no native engines, this checks
    if ``octomil setup`` has prepared a venv with mlx-lm or llama.cpp.
    If so, ``os.execv()`` replaces this process entirely with the venv's
    Python running the original command. Single process, no proxy.

    If no venv exists, returns False. Set ``OCTOMIL_ALLOW_MANAGED_PYTHON_SETUP=1``
    to opt into the legacy inline setup path for development.

    Returns True if re-exec was initiated (unreachable after os.execv).
    Returns False if no managed runtime is available and we should fall through.
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
    if venv_py and _is_running_inside_managed_venv(venv_py):
        return False
    if not venv_py or not is_engine_ready():
        if os.environ.get("OCTOMIL_ALLOW_MANAGED_PYTHON_SETUP") != "1":
            return False

        state = load_state()
        if state.phase == "failed":
            return False

        click.echo(
            click.style(
                "\n  Setting up legacy managed Python runtime...\n",
                fg="cyan",
            )
        )
        result = run_setup()
        if result.phase == "failed":
            click.echo(click.style(f"  Setup failed: {result.error}", fg="red"))
            return False
        venv_py = get_venv_python()
        if not venv_py or not is_engine_ready():
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
    os.environ["OCTOMIL_MANAGED_VENV_REEXECED"] = "1"
    try:
        os.execv(venv_py, new_argv)
    finally:
        os.environ.pop("OCTOMIL_MANAGED_VENV_REEXECED", None)
    return True  # unreachable after execv


def try_managed_venv_reexec(*, include_non_frozen: bool = False) -> bool:
    """Try managed-venv delegation when appropriate for this entrypoint."""
    if not should_managed_venv_reexec(include_non_frozen=include_non_frozen):
        return False
    return try_venv_reexec()


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
