from __future__ import annotations

import click


def interactive_cmd_factory(cli_group: click.Group) -> click.Command:
    """Create the interactive command that needs a reference to the CLI group."""

    @click.command("interactive")
    def interactive() -> None:
        """Open interactive command panel."""
        from .interactive import launch_interactive

        launch_interactive(cli_group)

    return interactive
