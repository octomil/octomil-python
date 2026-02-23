"""Tests for octomil launch command and agent registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from octomil.agents.launcher import (
    is_serve_running,
    launch_agent,
    start_serve_background,
)
from octomil.agents.registry import AGENTS, get_agent, is_agent_installed, list_agents
from octomil.cli import main


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_get_known_agent(self):
        agent = get_agent("claude")
        assert agent is not None
        assert agent.display_name == "Claude Code"
        assert agent.env_key == "ANTHROPIC_BASE_URL"

    def test_get_unknown_agent(self):
        assert get_agent("nonexistent") is None

    def test_list_agents(self):
        agents = list_agents()
        assert len(agents) == len(AGENTS)
        names = [a.name for a in agents]
        assert "claude" in names
        assert "codex" in names
        assert "openclaw" in names
        assert "aider" in names

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_agent_installed(self, mock_which):
        agent = get_agent("claude")
        assert agent is not None
        assert is_agent_installed(agent) is True

    @patch("shutil.which", return_value=None)
    def test_agent_not_installed(self, mock_which):
        agent = get_agent("claude")
        assert agent is not None
        assert is_agent_installed(agent) is False

    def test_all_agents_have_required_fields(self):
        for agent in list_agents():
            assert agent.name
            assert agent.display_name
            assert agent.env_key
            assert agent.install_check
            assert agent.install_cmd
            assert agent.exec_cmd


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestLaunchCLI:
    def test_launch_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--help"])
        assert result.exit_code == 0
        assert "Launch a coding agent" in result.output

    def test_launch_invalid_agent(self):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "invalid"])
        assert result.exit_code != 0

    def test_launch_lists_agent_choices(self):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--help"])
        assert "claude" in result.output
        assert "codex" in result.output
        assert "aider" in result.output
        assert "openclaw" in result.output


# ---------------------------------------------------------------------------
# is_serve_running
# ---------------------------------------------------------------------------


class TestIsServeRunning:
    @patch("octomil.agents.launcher.urllib.request.urlopen")
    def test_returns_true_when_server_responds(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        assert is_serve_running() is True
        mock_urlopen.assert_called_once_with(
            "http://localhost:8080/v1/models", timeout=2
        )

    @patch("octomil.agents.launcher.urllib.request.urlopen")
    def test_returns_false_on_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError
        assert is_serve_running() is False

    @patch("octomil.agents.launcher.urllib.request.urlopen")
    def test_custom_host_port(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        assert is_serve_running(host="0.0.0.0", port=9090) is True
        mock_urlopen.assert_called_once_with("http://0.0.0.0:9090/v1/models", timeout=2)


# ---------------------------------------------------------------------------
# start_serve_background
# ---------------------------------------------------------------------------


class TestStartServeBackground:
    @patch("octomil.agents.launcher.is_serve_running")
    @patch("octomil.agents.launcher.subprocess.Popen")
    def test_starts_and_waits_for_ready(self, mock_popen, mock_running):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        # First call: not ready; second call: ready
        mock_running.side_effect = [False, True]

        with patch("octomil.agents.launcher.time.sleep"):
            proc = start_serve_background("qwen3", port=8080)

        assert proc is mock_proc
        mock_popen.assert_called_once()

    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.launcher.subprocess.Popen")
    def test_raises_on_timeout(self, mock_popen, mock_running):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        with patch("octomil.agents.launcher.time.sleep"):
            with pytest.raises(RuntimeError, match="failed to start"):
                start_serve_background("model", port=8080)

        mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# launch_agent
# ---------------------------------------------------------------------------


class TestLaunchAgent:
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_uses_shlex_split(self, mock_installed, mock_running, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit) as exc_info:
            launch_agent("claude")

        assert exc_info.value.code == 0
        # Verify shlex.split was used (list of args, not a single string)
        call_args = mock_run.call_args
        assert isinstance(call_args[0][0], list)
        assert call_args[0][0][0] == "claude"

    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_starts_serve_when_not_running(
        self, mock_installed, mock_running, mock_serve, mock_run
    ):
        mock_proc = MagicMock()
        mock_serve.return_value = mock_proc
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("claude", model="qwen3")

        mock_serve.assert_called_once_with("qwen3", port=8080)
        mock_proc.terminate.assert_called_once()

    @patch("octomil.agents.launcher.click.confirm", return_value=False)
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=False)
    def test_launch_offers_install(
        self, mock_installed, mock_running, mock_run, mock_confirm
    ):
        """When agent is not installed and user declines, raises SystemExit(1)."""
        with pytest.raises(SystemExit):
            launch_agent("claude")

    def test_launch_unknown_agent_raises(self):
        with pytest.raises(click.ClickException, match="Unknown agent"):
            launch_agent("nonexistent")

    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_sets_openai_api_key_for_openai_agents(
        self, mock_installed, mock_running, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex")

        call_kwargs = mock_run.call_args
        env = call_kwargs[1]["env"]
        assert env["OPENAI_API_KEY"] == "octomil-local"
        assert "OPENAI_BASE_URL" in env
