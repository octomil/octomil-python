"""Tests for edgeml.interactive — command catalog, fuzzy filter, fallback UI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click

from edgeml.interactive import (
    _CommandEntry,
    _build_command_catalog,
    _fallback_interactive,
    _fuzzy_filter,
)


# ---------------------------------------------------------------------------
# Helpers — build a Click group for testing
# ---------------------------------------------------------------------------


def _make_cli_group() -> click.Group:
    """Build a Click group with representative commands."""
    group = click.Group(name="edgeml")

    @click.command()
    def serve():
        """Start inference server."""

    @click.command()
    def deploy():
        """Deploy model to device."""

    @click.command()
    def hw():
        """Show hardware info."""

    @click.command()
    def optimize():
        """Optimize model for hardware."""

    @click.command()
    def login():
        """Authenticate with EdgeML."""

    @click.command()
    def benchmark():
        """Run inference benchmarks."""

    @click.command()
    def dashboard():
        """Open observability dashboard."""

    @click.command()
    def scan():
        """Scan network for devices."""

    group.add_command(serve)
    group.add_command(deploy)
    group.add_command(hw)
    group.add_command(optimize)
    group.add_command(login)
    group.add_command(benchmark)
    group.add_command(dashboard)
    group.add_command(scan)

    return group


# ---------------------------------------------------------------------------
# _CommandEntry dataclass
# ---------------------------------------------------------------------------


class Test_CommandEntry:
    def test_fields(self):
        entry = _CommandEntry(
            name="serve", description="Start server.", category="Serve"
        )
        assert entry.name == "serve"
        assert entry.description == "Start server."
        assert entry.category == "Serve"

    def test_equality(self):
        a = _CommandEntry(name="serve", description="desc", category="Serve")
        b = _CommandEntry(name="serve", description="desc", category="Serve")
        assert a == b

    def test_inequality(self):
        a = _CommandEntry(name="serve", description="desc", category="Serve")
        b = _CommandEntry(name="deploy", description="desc", category="Deploy")
        assert a != b


# ---------------------------------------------------------------------------
# _build_command_catalog
# ---------------------------------------------------------------------------


class TestBuildCommandCatalog:
    def test_returns_command_entry_list(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        assert isinstance(catalog, list)
        assert all(isinstance(e, _CommandEntry) for e in catalog)

    def test_extracts_all_commands(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        names = {e.name for e in catalog}
        assert names == {
            "serve",
            "deploy",
            "hw",
            "optimize",
            "login",
            "benchmark",
            "dashboard",
            "scan",
        }

    def test_descriptions_extracted(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["serve"].description == "Start inference server."
        assert by_name["deploy"].description == "Deploy model to device."

    def test_sorted_by_name(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        names = [e.name for e in catalog]
        assert names == sorted(names)

    def test_empty_group(self):
        group = click.Group(name="empty")
        catalog = _build_command_catalog(group)
        assert catalog == []

    def test_command_without_help(self):
        """Command with no help text gets empty description."""
        group = click.Group(name="test")

        @click.command()
        def mystery():
            pass

        group.add_command(mystery)
        catalog = _build_command_catalog(group)
        assert len(catalog) == 1
        assert catalog[0].description == ""

    def test_multiline_help_takes_first_line(self):
        """Only the first line of help text is used."""
        group = click.Group(name="test")

        @click.command(help="First line.\nSecond line with details.")
        def verbose():
            pass

        group.add_command(verbose)
        catalog = _build_command_catalog(group)
        assert catalog[0].description == "First line."

    def test_multi_command_subgroup(self):
        """MultiCommand (Group) subgroup gets its help text."""
        group = click.Group(name="root")
        sub = click.Group(name="sub", help="Subcommand group.")
        group.add_command(sub)

        catalog = _build_command_catalog(group)
        assert len(catalog) == 1
        assert catalog[0].name == "sub"
        assert catalog[0].description == "Subcommand group."


# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------


class TestCategoryMapping:
    def test_serve_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["serve"].category == "Serve"

    def test_benchmark_is_serve_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["benchmark"].category == "Serve"

    def test_deploy_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["deploy"].category == "Deploy"

    def test_scan_is_deploy_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["scan"].category == "Deploy"

    def test_hw_is_hardware_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["hw"].category == "Hardware"

    def test_optimize_is_hardware_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["optimize"].category == "Hardware"

    def test_login_is_account_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["login"].category == "Account"

    def test_dashboard_is_observe_category(self):
        group = _make_cli_group()
        catalog = _build_command_catalog(group)
        by_name = {e.name: e for e in catalog}
        assert by_name["dashboard"].category == "Observe"

    def test_unknown_command_gets_other_category(self):
        """Command not in CATEGORY_MAP → 'Other'."""
        group = click.Group(name="test")

        @click.command()
        def foobar():
            """Unknown command."""

        group.add_command(foobar)
        catalog = _build_command_catalog(group)
        assert catalog[0].category == "Other"


# ---------------------------------------------------------------------------
# _fuzzy_filter
# ---------------------------------------------------------------------------


class TestFuzzyFilter:
    def _entries(self) -> list[_CommandEntry]:
        return [
            _CommandEntry(
                name="serve", description="Start inference server.", category="Serve"
            ),
            _CommandEntry(
                name="deploy", description="Deploy model to device.", category="Deploy"
            ),
            _CommandEntry(
                name="hw", description="Show hardware info.", category="Hardware"
            ),
            _CommandEntry(
                name="optimize", description="Optimize model.", category="Hardware"
            ),
            _CommandEntry(
                name="login", description="Authenticate.", category="Account"
            ),
            _CommandEntry(name="scan", description="Scan network.", category="Deploy"),
        ]

    def test_empty_query_returns_all(self):
        entries = self._entries()
        result = _fuzzy_filter(entries, "")
        assert result == entries

    def test_exact_match(self):
        entries = self._entries()
        result = _fuzzy_filter(entries, "serve")
        names = [e.name for e in result]
        assert "serve" in names

    def test_srv_matches_serve(self):
        """'srv' chars appear in order in 'serve' → match."""
        entries = self._entries()
        result = _fuzzy_filter(entries, "srv")
        names = [e.name for e in result]
        assert "serve" in names

    def test_xyz_no_match(self):
        """'xyz' won't match any command."""
        entries = self._entries()
        result = _fuzzy_filter(entries, "xyz")
        assert result == []

    def test_case_insensitive(self):
        entries = self._entries()
        result = _fuzzy_filter(entries, "SERVE")
        names = [e.name for e in result]
        assert "serve" in names

    def test_matches_in_description(self):
        """Fuzzy filter searches name + description."""
        entries = self._entries()
        # "model" appears in deploy description and optimize description
        result = _fuzzy_filter(entries, "model")
        names = [e.name for e in result]
        assert "deploy" in names or "optimize" in names

    def test_empty_entries_returns_empty(self):
        result = _fuzzy_filter([], "anything")
        assert result == []

    def test_single_char_query(self):
        entries = self._entries()
        # 's' should match several entries (serve, scan, etc.)
        result = _fuzzy_filter(entries, "s")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _fallback_interactive
# ---------------------------------------------------------------------------


class TestFallbackInteractive:
    def test_output_includes_header(self):
        """Fallback prints 'EdgeML Commands' header."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
            _CommandEntry(name="hw", description="Hardware info.", category="Hardware"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "q"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result is None
        # Check that echo was called with the header
        echo_calls = [str(c) for c in mock_click.echo.call_args_list]
        header_found = any("EdgeML Commands" in c for c in echo_calls)
        assert header_found

    def test_output_shows_categories(self):
        """Fallback groups commands by category and prints category headers."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
            _CommandEntry(
                name="deploy", description="Deploy model.", category="Deploy"
            ),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "q"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            _fallback_interactive(commands)

        secho_calls = [str(c) for c in mock_click.secho.call_args_list]
        assert any("Serve" in c for c in secho_calls)
        assert any("Deploy" in c for c in secho_calls)

    def test_returns_user_choice(self):
        """When user types a command name, it is returned."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "serve"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result == "serve"

    def test_returns_none_on_q(self):
        """Typing 'q' returns None."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "q"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result is None

    def test_returns_none_on_abort(self):
        """Ctrl+C (click.Abort) returns None."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.side_effect = click.Abort()
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result is None

    def test_returns_none_on_eof(self):
        """EOFError returns None."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.side_effect = EOFError()
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result is None

    def test_strips_whitespace_from_input(self):
        """User input is stripped of whitespace."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "  serve  "
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            result = _fallback_interactive(commands)

        assert result == "serve"

    def test_shows_command_names_in_output(self):
        """Command names are printed in the listing."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
            _CommandEntry(name="hw", description="Hardware.", category="Hardware"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "q"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            _fallback_interactive(commands)

        echo_calls = [str(c) for c in mock_click.echo.call_args_list]
        all_output = " ".join(echo_calls)
        assert "serve" in all_output
        assert "hw" in all_output

    def test_shows_quit_instruction(self):
        """Output includes instruction about 'q' to quit."""
        commands = [
            _CommandEntry(name="serve", description="Start server.", category="Serve"),
        ]
        with patch("edgeml.interactive.click") as mock_click:
            mock_click.prompt.return_value = "q"
            mock_click.echo = MagicMock()
            mock_click.secho = MagicMock()
            mock_click.Abort = click.Abort

            _fallback_interactive(commands)

        echo_calls = [str(c) for c in mock_click.echo.call_args_list]
        all_output = " ".join(echo_calls)
        assert "quit" in all_output.lower() or "q" in all_output
