"""Click Group subclass that renders --help with sectioned command groups."""

from __future__ import annotations

import click

# Section name -> command names (in display order).
# Commands not listed here appear in an "Other" section at the end.
COMMAND_SECTIONS: dict[str, list[str]] = {
    "Get Started": ["run", "chat", "embed", "transcribe", "launch", "setup"],
    "Compatibility": ["serve"],
    "Models": ["models", "push", "pull", "convert", "check", "list", "warmup"],
    "Deploy": ["deploy", "rollback", "pair", "status", "benchmark", "dashboard"],
    "Account": ["login", "init", "org", "team", "keys"],
    "Advanced": [
        "responses",
        "embeddings",
        "audio",
        "train",
        "federation",
        "mcp",
        "scan",
        "demo",
        "interactive",
        "completions",
    ],
}


class SectionedGroup(click.Group):
    """A Click Group that organises commands into named sections in --help."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Collect all visible commands.
        all_commands: dict[str, click.Command] = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            all_commands[name] = cmd

        if not all_commands:
            return

        # Consistent column alignment across all sections.
        limit = formatter.width - 6 - max(len(n) for n in all_commands)

        placed: set[str] = set()

        for section_name, cmd_names in COMMAND_SECTIONS.items():
            rows: list[tuple[str, str]] = []
            for name in cmd_names:
                if name in all_commands:
                    rows.append((name, all_commands[name].get_short_help_str(limit)))
                    placed.add(name)
            if rows:
                with formatter.section(section_name):
                    formatter.write_dl(rows)

        # Fallback for any unplaced commands.
        remaining = [(n, all_commands[n].get_short_help_str(limit)) for n in sorted(all_commands) if n not in placed]
        if remaining:
            with formatter.section("Other"):
                formatter.write_dl(remaining)
