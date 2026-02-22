"""Tests for edgeml launch command and agent registry."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from edgeml.agents.registry import AGENTS, get_agent, is_agent_installed, list_agents
from edgeml.cli import main


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
