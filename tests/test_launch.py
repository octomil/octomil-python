"""Tests for octomil launch command and agent registry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from octomil.agents.launcher import (
    _PICKER_AGENTS,
    RecommendedModel,
    _auto_select_model,
    _build_serve_cmd,
    _select_agent_fallback,
    _select_model_fallback,
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
        assert "droid" in names
        assert "opencode" in names
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
            assert agent.description
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

    def test_launch_help_shows_examples(self):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--help"])
        assert "octomil launch" in result.output
        assert "octomil launch codex" in result.output

    def test_launch_help_shows_select_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--help"])
        assert "--select" in result.output


# ---------------------------------------------------------------------------
# is_serve_running
# ---------------------------------------------------------------------------


class TestIsServeRunning:
    @patch("octomil.agents.launcher.urllib.request.urlopen")
    def test_returns_true_when_server_responds(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        assert is_serve_running() is True
        mock_urlopen.assert_called_once_with("http://localhost:8080/v1/models", timeout=2)

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


class TestBuildServeCmd:
    def test_unfrozen_uses_python_m(self):
        with patch.object(sys, "frozen", False, create=True):
            cmd = _build_serve_cmd("llama-8b", 8080)
        assert "-m" in cmd
        assert "octomil" in cmd
        assert "serve" in cmd
        assert "llama-8b" in cmd

    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_frozen_skips_python_m(self, mock_venv, mock_ready):
        with patch.object(sys, "frozen", True, create=True):
            cmd = _build_serve_cmd("llama-8b", 8080)
        assert "-m" not in cmd
        assert "serve" in cmd
        assert "llama-8b" in cmd

    def test_port_in_cmd(self):
        cmd = _build_serve_cmd("model", 9090)
        assert "--port" in cmd
        assert "9090" in cmd


class TestStartServeBackground:
    @patch("octomil.agents.launcher.is_serve_running")
    @patch("octomil.agents.launcher.subprocess.Popen")
    def test_starts_and_waits_for_ready(self, mock_popen, mock_running, tmp_path):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        mock_running.side_effect = [False, True]

        log_path = str(tmp_path / "serve.log")
        with (
            patch("octomil.agents.launcher.time.sleep"),
            patch("octomil.agents.launcher.os.path.join", return_value=log_path),
        ):
            proc = start_serve_background("qwen3", port=8080, timeout=5)

        assert proc is mock_proc
        mock_popen.assert_called_once()

    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.launcher.subprocess.Popen")
    def test_raises_on_timeout(self, mock_popen, mock_running, tmp_path):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        log_path = str(tmp_path / "serve.log")
        with (
            patch("octomil.agents.launcher.time.sleep"),
            patch("octomil.agents.launcher.os.path.join", return_value=log_path),
        ):
            with pytest.raises(click.ClickException):
                start_serve_background("model", port=8080, timeout=3)

        mock_proc.terminate.assert_called_once()

    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.launcher.subprocess.Popen")
    def test_raises_on_early_exit(self, mock_popen, mock_running, tmp_path):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # process already exited
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        log_path = str(tmp_path / "serve.log")
        # Write a fake log so tail reads something
        with open(log_path, "w") as f:
            f.write("error: model not found\n")

        with (
            patch("octomil.agents.launcher.time.sleep"),
            patch("octomil.agents.launcher.os.path.join", return_value=log_path),
        ):
            with pytest.raises(click.ClickException):
                start_serve_background("model", port=8080, timeout=5)


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
    def test_launch_starts_serve_when_not_running(self, mock_installed, mock_running, mock_serve, mock_run):
        mock_proc = MagicMock()
        mock_serve.return_value = mock_proc
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex", model="qwen3")

        mock_serve.assert_called_once_with("qwen3", port=8080)
        mock_proc.terminate.assert_called_once()

    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_claude_with_model_starts_serve(self, mock_installed, mock_running, mock_serve, mock_run):
        """Claude Code with --model should start serve and set ANTHROPIC env vars."""
        mock_serve.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("claude", model="qwen-7b")

        mock_serve.assert_called_once_with("qwen-7b", port=8080)
        env = mock_run.call_args[1]["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8080/v1"
        assert env["ANTHROPIC_API_KEY"] == "octomil-local"
        # Verify --model flag is passed to claude
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "qwen-7b" in cmd

    @patch("octomil.agents.launcher.click.confirm", return_value=False)
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=False)
    def test_launch_offers_install(self, mock_installed, mock_running, mock_run, mock_confirm):
        """When agent is not installed and user declines, raises SystemExit(1)."""
        with pytest.raises(SystemExit):
            launch_agent("claude")

    def test_launch_unknown_agent_raises(self):
        with pytest.raises(click.ClickException, match="Unknown agent"):
            launch_agent("nonexistent")

    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_sets_openai_api_key_for_openai_agents(self, mock_installed, mock_running, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex")

        call_kwargs = mock_run.call_args
        env = call_kwargs[1]["env"]
        assert env["OPENAI_API_KEY"] == "octomil-local"
        assert "OPENAI_BASE_URL" in env

    @patch("octomil.agents.launcher._select_model_tui", return_value="qwen-coder-7b")
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_auto_selects_when_no_model(self, mock_installed, mock_running, mock_serve, mock_run, mock_tui):
        mock_serve.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex")

        mock_tui.assert_called_once()
        mock_serve.assert_called_once_with("qwen-coder-7b", port=8080)

    @patch("octomil.agents.launcher._select_model_tui", return_value="llama-8b")
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_shows_tui_with_select_flag(self, mock_installed, mock_running, mock_serve, mock_run, mock_tui):
        mock_serve.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex", select=True)

        mock_tui.assert_called_once()
        mock_serve.assert_called_once_with("llama-8b", port=8080)

    @patch("octomil.agents.launcher._auto_select_model")
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_skips_auto_select_when_model_given(
        self, mock_installed, mock_running, mock_serve, mock_run, mock_auto
    ):
        mock_serve.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent("codex", model="llama-8b")

        mock_auto.assert_not_called()
        mock_serve.assert_called_once_with("llama-8b", port=8080)

    @patch("octomil.agents.launcher._select_agent_tui", return_value="codex")
    @patch("octomil.agents.launcher.subprocess.run")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    @patch("octomil.agents.registry.is_agent_installed", return_value=True)
    def test_launch_no_agent_shows_picker(self, mock_installed, mock_running, mock_run, mock_picker):
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(SystemExit):
            launch_agent()  # no agent_name

        mock_picker.assert_called_once()


# ---------------------------------------------------------------------------
# Agent picker
# ---------------------------------------------------------------------------


class TestAgentPicker:
    def test_picker_agents_exist_in_registry(self):
        for name in _PICKER_AGENTS:
            assert get_agent(name) is not None

    @patch("shutil.which", return_value=None)
    @patch("click.prompt", return_value="1")
    def test_fallback_returns_first_agent(self, mock_prompt, mock_which):
        from octomil.agents.registry import list_agents

        agents = list_agents()
        result = _select_agent_fallback(agents, lambda a: False)
        assert result == agents[0].name

    @patch("shutil.which", return_value=None)
    @patch("click.prompt", return_value="codex")
    def test_fallback_returns_custom_name(self, mock_prompt, mock_which):
        from octomil.agents.registry import list_agents

        agents = list_agents()
        result = _select_agent_fallback(agents, lambda a: False)
        assert result == "codex"


# ---------------------------------------------------------------------------
# Auto-select model
# ---------------------------------------------------------------------------


class TestAutoSelectModel:
    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_returns_first_recommendation(self, mock_recs, mock_budget):
        mock_recs.return_value = [
            RecommendedModel(
                key="qwen-coder-7b",
                label="qwen-coder-7b",
                description="Best small coding model",
                size="~4.5 GB",
                recommended=True,
            ),
        ]
        result = _auto_select_model()
        assert result == "qwen-coder-7b"

    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_prints_usage_hint(self, mock_recs, mock_budget, capsys):
        mock_recs.return_value = [
            RecommendedModel(
                key="llama-8b",
                label="llama-8b",
                description="Meta Llama",
                size="~4.5 GB",
                recommended=True,
            ),
        ]
        _auto_select_model()
        captured = capsys.readouterr()
        assert "llama-8b" in captured.out
        assert "--select" in captured.out


# ---------------------------------------------------------------------------
# Fallback model picker
# ---------------------------------------------------------------------------


class TestSelectModelFallback:
    @patch("octomil.agents.launcher._is_model_downloaded", return_value=False)
    @patch("click.prompt", return_value="1")
    def test_returns_first_when_default(self, mock_prompt, mock_downloaded):
        recs = [
            RecommendedModel(
                key="qwen-coder-7b",
                label="qwen-coder-7b",
                description="Best small coder",
                size="~4.5 GB",
                recommended=True,
            ),
            RecommendedModel(
                key="llama-8b",
                label="llama-8b",
                description="Llama",
                size="~4.5 GB",
            ),
        ]
        result = _select_model_fallback(recs, 16.0)
        assert result == "qwen-coder-7b"

    @patch("octomil.agents.launcher._is_model_downloaded", return_value=False)
    @patch("click.prompt", return_value="custom-model")
    def test_returns_custom_model_name(self, mock_prompt, mock_downloaded):
        recs = [
            RecommendedModel(
                key="qwen-coder-7b",
                label="qwen-coder-7b",
                description="Best small coder",
                size="~4.5 GB",
                recommended=True,
            ),
        ]
        result = _select_model_fallback(recs, 16.0)
        assert result == "custom-model"


# ---------------------------------------------------------------------------
# RecommendedModel.downloaded field
# ---------------------------------------------------------------------------


class TestRecommendedModelDownloadedField:
    def test_recommended_model_has_downloaded_field(self):
        model = RecommendedModel(
            key="test",
            label="test",
            description="test model",
            size="~1 GB",
        )
        assert hasattr(model, "downloaded")
        assert model.downloaded is False

    def test_recommended_model_downloaded_true(self):
        model = RecommendedModel(
            key="test",
            label="test",
            description="test model",
            size="~1 GB",
            downloaded=True,
        )
        assert model.downloaded is True


# ---------------------------------------------------------------------------
# Auto-select prefers downloaded models
# ---------------------------------------------------------------------------


class TestAutoSelectPrefersDownloaded:
    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_auto_select_prefers_downloaded(self, mock_recs, mock_budget):
        """When a smaller model is already downloaded, prefer it over
        the largest fitting model that hasn't been downloaded yet."""
        mock_recs.return_value = [
            RecommendedModel(
                key="llama-8b",
                label="llama-8b",
                description="Meta Llama 3.1",
                size="~4.5 GB",
                recommended=True,
                downloaded=False,
            ),
            RecommendedModel(
                key="qwen-coder-3b",
                label="qwen-coder-3b",
                description="Fast coding model",
                size="~2 GB",
                downloaded=True,
            ),
        ]
        result = _auto_select_model()
        assert result == "qwen-coder-3b"

    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_auto_select_picks_largest_when_none_downloaded(self, mock_recs, mock_budget):
        """When no models are downloaded, pick the largest fitting model."""
        mock_recs.return_value = [
            RecommendedModel(
                key="llama-8b",
                label="llama-8b",
                description="Meta Llama 3.1",
                size="~4.5 GB",
                recommended=True,
                downloaded=False,
            ),
            RecommendedModel(
                key="qwen-coder-3b",
                label="qwen-coder-3b",
                description="Fast coding model",
                size="~2 GB",
                downloaded=False,
            ),
        ]
        result = _auto_select_model()
        assert result == "llama-8b"

    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_auto_select_picks_largest_downloaded(self, mock_recs, mock_budget):
        """When two models are downloaded, pick the larger of the two."""
        mock_recs.return_value = [
            RecommendedModel(
                key="llama-8b",
                label="llama-8b",
                description="Meta Llama 3.1",
                size="~4.5 GB",
                recommended=True,
                downloaded=True,
            ),
            RecommendedModel(
                key="qwen-coder-3b",
                label="qwen-coder-3b",
                description="Fast coding model",
                size="~2 GB",
                downloaded=True,
            ),
        ]
        result = _auto_select_model()
        assert result == "llama-8b"

    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_auto_select_prints_already_downloaded_message(self, mock_recs, mock_budget, capsys):
        """When selecting a downloaded model, prints 'already downloaded'."""
        mock_recs.return_value = [
            RecommendedModel(
                key="qwen-coder-7b",
                label="qwen-coder-7b",
                description="Best small coder",
                size="~4.5 GB",
                recommended=True,
                downloaded=True,
            ),
        ]
        _auto_select_model()
        captured = capsys.readouterr()
        assert "already downloaded" in captured.out

    @patch("octomil.agents.launcher._get_memory_budget_gb", return_value=16.0)
    @patch("octomil.agents.launcher._build_recommendations")
    def test_auto_select_prints_will_download_message(self, mock_recs, mock_budget, capsys):
        """When no model is downloaded, prints 'will download' with size."""
        mock_recs.return_value = [
            RecommendedModel(
                key="qwen-coder-7b",
                label="qwen-coder-7b",
                description="Best small coder",
                size="~4.5 GB",
                recommended=True,
                downloaded=False,
            ),
        ]
        _auto_select_model()
        captured = capsys.readouterr()
        assert "will download" in captured.out
        assert "~4.5 GB" in captured.out
