from __future__ import annotations

import logging
from dataclasses import dataclass

import click

logger = logging.getLogger(__name__)


@dataclass
class _CommandEntry:
    name: str
    description: str
    category: str  # "serve", "deploy", "manage", "hardware", etc.


def _build_command_catalog(group: click.Group) -> list[_CommandEntry]:
    """Extract commands from a Click group into a sorted catalog."""
    # Category mapping based on command names
    CATEGORY_MAP = {
        "serve": "Serve",
        "benchmark": "Serve",
        "deploy": "Deploy",
        "push": "Deploy",
        "pull": "Deploy",
        "convert": "Deploy",
        "check": "Deploy",
        "status": "Deploy",
        "scan": "Deploy",
        "pair": "Deploy",
        "hw": "Hardware",
        "optimize": "Hardware",
        "interactive": "Hardware",
        "login": "Account",
        "init": "Account",
        "org": "Account",
        "team": "Account",
        "keys": "Account",
        "dashboard": "Observe",
        "list": "Observe",
        "agents": "Agents",
    }
    entries: list[_CommandEntry] = []
    for name, cmd in sorted(group.commands.items()):
        help_text = ""
        if isinstance(cmd, click.Command) and cmd.help:
            help_text = cmd.help.split("\n")[0].strip()
        elif isinstance(cmd, click.MultiCommand):
            help_text = cmd.help.split("\n")[0].strip() if cmd.help else ""
        category = CATEGORY_MAP.get(name, "Other")
        entries.append(
            _CommandEntry(name=name, description=help_text, category=category)
        )
    return entries


def _fuzzy_filter(entries: list[_CommandEntry], query: str) -> list[_CommandEntry]:
    """Simple fuzzy filter: all query chars must appear in order in name or description."""
    if not query:
        return entries
    query_lower = query.lower()
    results: list[_CommandEntry] = []
    for entry in entries:
        target = f"{entry.name} {entry.description}".lower()
        qi = 0
        for ch in target:
            if qi < len(query_lower) and ch == query_lower[qi]:
                qi += 1
        if qi == len(query_lower):
            results.append(entry)
    return results


def _fallback_interactive(commands: list[_CommandEntry]) -> str | None:
    """Plain text fallback when prompt_toolkit is not available."""
    # Group by category
    categories: dict[str, list[_CommandEntry]] = {}
    for cmd in commands:
        categories.setdefault(cmd.category, []).append(cmd)

    click.echo("\n  EdgeML Commands\n")
    for cat, cmds in sorted(categories.items()):
        click.secho(f"  {cat}", bold=True)
        for cmd in cmds:
            click.echo(f"    {cmd.name:<20} {cmd.description}")
        click.echo()

    click.echo("  Type a command name to run it, or 'q' to quit.")
    try:
        choice = click.prompt("  >", type=str, default="q")
    except (click.Abort, EOFError):
        return None

    if choice.strip().lower() == "q":
        return None
    return choice.strip()


def launch_interactive(cli_group: click.Group) -> None:
    """Entry point. Falls back to plain list if prompt_toolkit unavailable."""
    commands = _build_command_catalog(cli_group)

    try:
        from prompt_toolkit import Application  # noqa: F401
        from prompt_toolkit.key_binding import KeyBindings  # noqa: F401
        from prompt_toolkit.layout import HSplit, Layout, Window  # noqa: F401
        from prompt_toolkit.layout.controls import FormattedTextControl  # noqa: F401

        _run_tui(cli_group, commands)
    except ImportError:
        logger.debug("prompt_toolkit not installed, using fallback")
        choice = _fallback_interactive(commands)
        if choice and choice in cli_group.commands:
            ctx = click.Context(cli_group)
            cli_group.commands[choice].invoke(ctx)
        elif choice:
            click.echo(f"Unknown command: {choice}", err=True)


def _run_tui(cli_group: click.Group, commands: list[_CommandEntry]) -> None:
    """Run the full TUI panel with prompt_toolkit."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    selected = [0]
    filtered = [list(commands)]
    search_mode = [False]
    search_query = [""]
    result: list[str | None] = [None]  # Will store the selected command name

    def get_display_text():  # type: ignore[no-untyped-def]
        lines: list[tuple[str, str]] = []
        lines.append(("bold", "  EdgeML Interactive\n"))
        lines.append(
            (
                "",
                "  Use \u2191\u2193 to navigate, Enter to run, / to search, q to quit\n\n",
            )
        )

        # Group by category
        categories: dict[str, list[tuple[int, _CommandEntry]]] = {}
        for i, cmd in enumerate(filtered[0]):
            categories.setdefault(cmd.category, []).append((i, cmd))

        for cat, cmds in sorted(categories.items()):
            lines.append(("bold", f"  {cat}\n"))
            for i, cmd in cmds:
                prefix = " \u25b8 " if i == selected[0] else "   "
                style = "reverse" if i == selected[0] else ""
                lines.append((style, f"{prefix}{cmd.name:<20} {cmd.description}\n"))
            lines.append(("", "\n"))

        if search_mode[0]:
            lines.append(("bold", f"  Search: {search_query[0]}\u2588\n"))

        return lines

    control = FormattedTextControl(get_display_text)
    bindings = KeyBindings()

    @bindings.add("up")
    def _up(event):  # type: ignore[no-untyped-def]
        selected[0] = max(0, selected[0] - 1)

    @bindings.add("down")
    def _down(event):  # type: ignore[no-untyped-def]
        selected[0] = min(len(filtered[0]) - 1, selected[0] + 1)

    @bindings.add("enter")
    def _enter(event):  # type: ignore[no-untyped-def]
        if filtered[0]:
            result[0] = filtered[0][selected[0]].name
        event.app.exit()

    @bindings.add("/")
    def _search(event):  # type: ignore[no-untyped-def]
        if not search_mode[0]:
            search_mode[0] = True
            search_query[0] = ""

    @bindings.add("escape")
    def _escape(event):  # type: ignore[no-untyped-def]
        if search_mode[0]:
            search_mode[0] = False
            search_query[0] = ""
            filtered[0] = list(commands)
            selected[0] = 0
        else:
            event.app.exit()

    @bindings.add("q")
    def _quit(event):  # type: ignore[no-untyped-def]
        if not search_mode[0]:
            event.app.exit()
        else:
            search_query[0] += "q"
            filtered[0] = _fuzzy_filter(commands, search_query[0])
            selected[0] = 0

    @bindings.add("c-c")
    def _ctrl_c(event):  # type: ignore[no-untyped-def]
        event.app.exit()

    @bindings.add("backspace")
    def _backspace(event):  # type: ignore[no-untyped-def]
        if search_mode[0] and search_query[0]:
            search_query[0] = search_query[0][:-1]
            filtered[0] = _fuzzy_filter(commands, search_query[0])
            selected[0] = 0

    # Catch all other keys for search mode
    @bindings.add("<any>")
    def _any_key(event):  # type: ignore[no-untyped-def]
        if search_mode[0]:
            search_query[0] += event.data
            filtered[0] = _fuzzy_filter(commands, search_query[0])
            selected[0] = 0

    layout = Layout(Window(content=control))
    app: Application[None] = Application(
        layout=layout, key_bindings=bindings, full_screen=True
    )
    app.run()

    # Execute selected command
    if result[0] and result[0] in cli_group.commands:
        import sys

        sys.argv = ["edgeml", result[0]]
        ctx = click.Context(cli_group)
        try:
            cli_group.commands[result[0]].invoke(ctx)
        except SystemExit:
            pass
