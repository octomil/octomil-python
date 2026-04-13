"""Tests for ``octomil local`` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from octomil.cli import main

# The CLI commands use lazy imports: `from octomil.local_runner.manager import LocalRunnerManager`.
# We must patch at the source module so the lazy import picks up the mock.
_MGR_PATCH = "octomil.local_runner.manager.LocalRunnerManager"


class TestLocalStatus:
    def test_status_not_running(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.status.return_value = MagicMock(
                running=False,
                pid=None,
                port=None,
                model=None,
                engine=None,
                base_url=None,
                uptime_seconds=None,
                idle_timeout_seconds=None,
            )
            result = runner.invoke(main, ["local", "status"])
            assert result.exit_code == 0
            assert "not running" in result.output.lower()

    def test_status_running(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.status.return_value = MagicMock(
                running=True,
                pid=12345,
                port=51200,
                model="gemma-1b",
                engine="mlx-lm",
                base_url="http://127.0.0.1:51200",
                uptime_seconds=120.5,
                idle_timeout_seconds=1800,
            )
            result = runner.invoke(main, ["local", "status"])
            assert result.exit_code == 0
            assert "12345" in result.output
            assert "gemma-1b" in result.output
            assert "mlx-lm" in result.output

    def test_status_json(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.status.return_value = MagicMock(
                running=True,
                pid=12345,
                port=51200,
                model="gemma-1b",
                engine="mlx-lm",
                base_url="http://127.0.0.1:51200",
                uptime_seconds=60.0,
                idle_timeout_seconds=1800,
            )
            result = runner.invoke(main, ["local", "status", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["running"] is True
            assert data["pid"] == 12345
            assert data["model"] == "gemma-1b"


class TestLocalStop:
    def test_stop_not_running(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.status.return_value = MagicMock(running=False, pid=None, model=None)
            result = runner.invoke(main, ["local", "stop"])
            assert result.exit_code == 0
            assert "not running" in result.output.lower()

    def test_stop_running(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.status.return_value = MagicMock(running=True, pid=12345, model="gemma-1b")
            result = runner.invoke(main, ["local", "stop"])
            assert result.exit_code == 0
            assert "stopped" in result.output.lower()
            mock_mgr.stop.assert_called_once()


class TestLocalEndpoint:
    def test_endpoint_with_model(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.ensure.return_value = MagicMock(
                base_url="http://127.0.0.1:51200",
                port=51200,
                model="gemma-1b",
                engine="mlx-lm",
                pid=12345,
                token="secret",
            )
            mock_mgr._token_path = Path("/tmp/test-token")
            result = runner.invoke(main, ["local", "endpoint", "--model", "gemma-1b"])
            assert result.exit_code == 0
            assert "http://127.0.0.1:51200" in result.output
            assert "secret" not in result.output  # token not shown by default

    def test_endpoint_show_token(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.ensure.return_value = MagicMock(
                base_url="http://127.0.0.1:51200",
                port=51200,
                model="gemma-1b",
                engine="mlx-lm",
                pid=12345,
                token="secret-token-123",
            )
            mock_mgr._token_path = Path("/tmp/test-token")
            result = runner.invoke(main, ["local", "endpoint", "--model", "gemma-1b", "--show-token"])
            assert result.exit_code == 0
            assert "secret-token-123" in result.output

    def test_endpoint_json(self) -> None:
        runner = CliRunner()
        with patch(_MGR_PATCH) as MockMgr:
            mock_mgr = MockMgr.return_value
            mock_mgr.ensure.return_value = MagicMock(
                base_url="http://127.0.0.1:51200",
                port=51200,
                model="gemma-1b",
                engine="mlx-lm",
                pid=12345,
                token="secret",
            )
            mock_mgr._token_path = Path("/tmp/test-token")
            result = runner.invoke(main, ["local", "endpoint", "--model", "gemma-1b", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["base_url"] == "http://127.0.0.1:51200"
            assert data["model"] == "gemma-1b"
            assert "token" not in data  # not shown without --show-token


class TestLocalRunnerServeCommand:
    def test_hidden_command_exists(self) -> None:
        """The _local-runner-serve command should be registered but hidden."""
        runner = CliRunner()
        result = runner.invoke(main, ["_local-runner-serve", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--port" in result.output
        assert "--token-file" in result.output
        assert "--idle-timeout" in result.output

    def test_hidden_command_not_in_help(self) -> None:
        """Hidden commands should not appear in main help."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "_local-runner-serve" not in result.output
