"""Shell completions setup command."""

from __future__ import annotations

import sys

import click


def register(cli: click.Group) -> None:
    cli.add_command(completions)


_BASH_SCRIPT = """\
eval "$(_OCTOMIL_COMPLETE=bash_source octomil)"
"""

_ZSH_SCRIPT = """\
eval "$(_OCTOMIL_COMPLETE=zsh_source octomil)"
"""

_FISH_SCRIPT = """\
_OCTOMIL_COMPLETE=fish_source octomil | source
"""

_SHELLS = {
    "bash": _BASH_SCRIPT.strip(),
    "zsh": _ZSH_SCRIPT.strip(),
    "fish": _FISH_SCRIPT.strip(),
}


@click.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
@click.option("--install", is_flag=True, help="Append completion source to your shell rc file.")
def completions(shell: str | None, install: bool) -> None:
    """Set up shell tab-completions for octomil.

    \b
    Print the completion script:
        octomil completions bash
        octomil completions zsh
        octomil completions fish

    \b
    Auto-install to your shell rc file:
        octomil completions --install
    """
    if install:
        _install_completions()
        return

    if not shell:
        click.echo("Usage: octomil completions <bash|zsh|fish>")
        click.echo("       octomil completions --install")
        sys.exit(1)

    click.echo(_SHELLS[shell])


def _detect_shell() -> str:
    """Detect the user's current shell."""
    import os

    shell_path = os.environ.get("SHELL", "")
    if "zsh" in shell_path:
        return "zsh"
    if "fish" in shell_path:
        return "fish"
    return "bash"


def _get_rc_path(shell: str) -> str:
    """Return the rc file path for the given shell."""
    import os

    home = os.path.expanduser("~")
    if shell == "zsh":
        return os.path.join(home, ".zshrc")
    if shell == "fish":
        conf_dir = os.path.join(home, ".config", "fish", "conf.d")
        return os.path.join(conf_dir, "octomil.fish")
    return os.path.join(home, ".bashrc")


_MARKER = "# octomil shell completions"


def _install_completions() -> None:
    """Detect shell and append completion source to rc file."""
    import os

    shell = _detect_shell()
    script = _SHELLS[shell]
    rc_path = _get_rc_path(shell)

    # Check if already installed
    if os.path.exists(rc_path):
        with open(rc_path) as f:
            if _MARKER in f.read():
                click.echo(f"Completions already installed in {rc_path}")
                return

    # For fish, ensure conf.d directory exists
    if shell == "fish":
        os.makedirs(os.path.dirname(rc_path), exist_ok=True)

    with open(rc_path, "a") as f:
        f.write(f"\n{_MARKER}\n{script}\n")

    click.echo(f"Completions installed in {rc_path}")
    click.echo(f"Restart your shell or run: source {rc_path}")
