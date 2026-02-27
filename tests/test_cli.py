"""Tests for octomil.cli — Click command-line interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from octomil.cli import main
from octomil.cli_helpers import _get_api_key


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "env-key-123")
        assert _get_api_key() == "env-key-123"

    def test_empty_when_no_env_no_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        monkeypatch.setattr(
            "os.path.expanduser", lambda x: str(tmp_path / ".octomil" / "credentials")
        )
        assert _get_api_key() == ""

    def test_from_credentials_file_json(self, monkeypatch, tmp_path):
        import json

        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        cred_dir = tmp_path / ".octomil"
        cred_dir.mkdir()
        cred_file = cred_dir / "credentials"
        cred_file.write_text(json.dumps({"api_key": "file-key-456", "org": "acme"}))

        monkeypatch.setattr(
            "octomil.cli_helpers.os.path.expanduser", lambda x: str(cred_dir / "credentials")
        )
        assert _get_api_key() == "file-key-456"

    def test_from_credentials_file_legacy(self, monkeypatch, tmp_path):
        """Backward compat: reads legacy key=value format."""
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        cred_dir = tmp_path / ".octomil"
        cred_dir.mkdir()
        cred_file = cred_dir / "credentials"
        cred_file.write_text("api_key=legacy-key-789\n")

        monkeypatch.setattr(
            "octomil.cli_helpers.os.path.expanduser", lambda x: str(cred_dir / "credentials")
        )
        assert _get_api_key() == "legacy-key-789"

    def test_env_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "env-key")
        assert _get_api_key() == "env-key"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


class TestMainGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Octomil" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "2.6.0" in result.output

    def test_no_args_shows_quickstart(self):
        """Bare `octomil` with no args shows focused onboarding screen."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Get started" in result.output
        assert "octomil serve" in result.output
        assert "octomil push" in result.output
        assert "octomil deploy" in result.output
        assert "octomil launch" in result.output
        assert "--help" in result.output

    def test_help_shows_full_commands(self):
        """--help still shows the full Click-generated help with all commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # These commands appear in full help but NOT in the quickstart screen
        assert "benchmark" in result.output
        assert "train" in result.output
        assert "push" in result.output
        assert "pull" in result.output


# ---------------------------------------------------------------------------
# octomil serve
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_basic(self):
        runner = CliRunner()
        with patch("octomil.serve.run_server") as mock_run:
            result = runner.invoke(main, ["serve", "gemma-1b"])
        assert result.exit_code == 0
        assert "Starting Octomil serve" in result.output
        assert "gemma-1b" in result.output
        mock_run.assert_called_once()

    def test_serve_custom_port(self):
        runner = CliRunner()
        with patch("octomil.serve.run_server"):
            result = runner.invoke(main, ["serve", "gemma-1b", "--port", "9000"])
        assert result.exit_code == 0
        assert "9000" in result.output



# ---------------------------------------------------------------------------
# octomil login
# ---------------------------------------------------------------------------


class TestLoginCommand:
    def test_login_saves_credentials(self, tmp_path, monkeypatch):
        import json

        cred_dir = tmp_path / ".octomil"
        monkeypatch.setattr(
            "octomil.cli_helpers.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.octomil" in x else x,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["login", "--api-key", "test-key-789"])
        assert result.exit_code == 0
        assert "API key saved" in result.output

        # Verify file was created with JSON format
        cred_file = cred_dir / "credentials"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text())
        assert data["api_key"] == "test-key-789"


# ---------------------------------------------------------------------------
# octomil dashboard
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    @patch("octomil.commands.deploy.webbrowser.open")
    def test_dashboard_opens_browser(self, mock_open):
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert "Opening dashboard" in result.output
        mock_open.assert_called_once_with("https://app.octomil.com")

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_dashboard_custom_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_DASHBOARD_URL", "https://custom.dashboard.io")
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        mock_open.assert_called_once_with("https://custom.dashboard.io")


# ---------------------------------------------------------------------------
# octomil deploy --phone
# ---------------------------------------------------------------------------


class TestDeployCommand:
    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        # Mock httpx for pairing session creation + polling
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "ABC123",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "ABC123" in result.output
        mock_open.assert_called_once()
        call_url = mock_open.call_args[0][0]
        assert call_url.startswith("octomil://pair?")
        assert "token=ABC123" in call_url
        assert "host=" in call_url

    @patch("octomil.commands.deploy._get_client")
    def test_deploy_rollout(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.deploy.return_value = {"id": "rollout-1", "status": "created"}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main, ["deploy", "sentiment-v1", "--rollout", "10", "--strategy", "canary"]
        )
        assert result.exit_code == 0
        assert "Rollout created" in result.output


# ---------------------------------------------------------------------------
# octomil status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    @patch("octomil.commands.deploy._get_client")
    def test_status(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.status.return_value = {
            "model": {"name": "test-model", "id": "abc", "framework": "pytorch"},
            "active_rollouts": [
                {"version": "1.0.0", "rollout_percentage": 50, "status": "active"}
            ],
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["status", "test-model"])
        assert result.exit_code == 0
        assert "test-model" in result.output
        assert "Active rollouts: 1" in result.output
        assert "v1.0.0" in result.output

    @patch("octomil.commands.deploy._get_client")
    def test_status_no_rollouts(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.status.return_value = {
            "model": {"name": "test-model", "id": "abc", "framework": "pytorch"},
            "active_rollouts": [],
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["status", "test-model"])
        assert result.exit_code == 0
        assert "No active rollouts" in result.output


# ---------------------------------------------------------------------------
# octomil push
# ---------------------------------------------------------------------------


class TestPushCommand:
    @patch("octomil.commands.model_ops._get_client")
    def test_push(self, mock_get_client, tmp_path, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.push.return_value = {"formats": {"onnx": "ok", "coreml": "ok"}}
        mock_get_client.return_value = mock_client

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake model data")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["push", str(model_file), "--model-id", "test-model", "--version", "1.0.0"],
        )
        assert result.exit_code == 0
        assert "test-model v1.0.0" in result.output
        mock_client.push.assert_called_once()


# ---------------------------------------------------------------------------
# octomil pull
# ---------------------------------------------------------------------------


class TestPullCommand:
    @patch("octomil.commands.model_ops._get_client")
    def test_pull(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.pull.return_value = {"model_path": "/tmp/model.onnx"}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main, ["pull", "test-model", "--version", "1.0.0", "--format", "onnx"]
        )
        assert result.exit_code == 0
        assert "Downloaded: /tmp/model.onnx" in result.output


# ---------------------------------------------------------------------------
# octomil train start
# ---------------------------------------------------------------------------


class TestTrainStartCommand:
    @patch("octomil.commands.enterprise._get_client")
    def test_train_start_basic(self, mock_get_client, monkeypatch):
        from octomil.models import TrainingSession

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train.return_value = TrainingSession(
            session_id="tr-123",
            model_name="sentiment-v1",
            group="default",
            strategy="fedavg",
            rounds=10,
            status="created",
        )
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["train", "start", "sentiment-v1"])
        assert result.exit_code == 0
        assert "Training started" in result.output
        assert "tr-123" in result.output
        mock_client.train.assert_called_once()

    @patch("octomil.commands.enterprise._get_client")
    def test_train_start_with_options(self, mock_get_client, monkeypatch):
        from octomil.models import TrainingSession

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train.return_value = TrainingSession(
            session_id="tr-456",
            model_name="sentiment-v1",
            group="default",
            strategy="fedprox",
            rounds=50,
            status="created",
        )
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "start",
                "sentiment-v1",
                "--strategy",
                "fedprox",
                "--rounds",
                "50",
                "--privacy",
                "dp-sgd",
                "--epsilon",
                "1.0",
            ],
        )
        assert result.exit_code == 0
        assert "fedprox" in result.output
        assert "50" in result.output
        assert "dp-sgd" in result.output
        mock_client.train.assert_called_once_with(
            "sentiment-v1",
            strategy="fedprox",
            rounds=50,
            group=None,
            privacy="dp-sgd",
            epsilon=1.0,
            min_devices=2,
        )

    def test_train_start_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["train", "start", "model"])
        assert result.exit_code != 0

    @patch("octomil.commands.enterprise._get_client")
    def test_train_start_with_group(self, mock_get_client, monkeypatch):
        from octomil.models import TrainingSession

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train.return_value = TrainingSession(
            session_id="tr-789",
            model_name="sentiment-v1",
            group="us-west",
            strategy="fedavg",
            rounds=10,
            status="created",
        )
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["train", "start", "sentiment-v1", "--group", "us-west"],
        )
        assert result.exit_code == 0
        mock_client.train.assert_called_once_with(
            "sentiment-v1",
            strategy="fedavg",
            rounds=10,
            group="us-west",
            privacy=None,
            epsilon=None,
            min_devices=2,
        )

    def test_train_start_invalid_strategy(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(
            main, ["train", "start", "model", "--strategy", "invalid"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# octomil train status
# ---------------------------------------------------------------------------


class TestTrainStatusCommand:
    @patch("octomil.commands.enterprise._get_client")
    def test_train_status(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train_status.return_value = {
            "current_round": 23,
            "total_rounds": 50,
            "active_devices": 47,
            "status": "in_progress",
            "loss": 0.342,
            "accuracy": 0.912,
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["train", "status", "sentiment-v1"])
        assert result.exit_code == 0
        assert "23/50" in result.output
        assert "47" in result.output
        assert "in_progress" in result.output
        assert "0.3420" in result.output
        assert "91.2%" in result.output

    @patch("octomil.commands.enterprise._get_client")
    def test_train_status_no_metrics(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train_status.return_value = {
            "current_round": 1,
            "total_rounds": 10,
            "active_devices": 3,
            "status": "starting",
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["train", "status", "my-model"])
        assert result.exit_code == 0
        assert "1/10" in result.output
        assert "Loss" not in result.output
        assert "Accuracy" not in result.output


# ---------------------------------------------------------------------------
# octomil train stop
# ---------------------------------------------------------------------------


class TestTrainStopCommand:
    @patch("octomil.commands.enterprise._get_client")
    def test_train_stop(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train_stop.return_value = {"last_round": 23}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["train", "stop", "sentiment-v1"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
        assert "23" in result.output


# ---------------------------------------------------------------------------
# octomil benchmark
# ---------------------------------------------------------------------------


def _make_benchmark_mocks():
    """Return mocks for _detect_backend, psutil, and GenerationRequest."""
    mock_metrics = MagicMock()
    mock_metrics.tokens_per_second = 42.0
    mock_metrics.ttfc_ms = 10.0
    mock_metrics.prompt_tokens = 5
    mock_metrics.total_tokens = 20

    mock_backend = MagicMock()
    mock_backend.name = "echo"
    mock_backend.generate.return_value = ("hello", mock_metrics)

    mock_process = MagicMock()
    mem_info = MagicMock()
    mem_info.rss = 500 * 1024 * 1024
    mock_process.memory_info.return_value = mem_info

    mock_vm = MagicMock()
    mock_vm.total = 16 * 1024 * 1024 * 1024

    return mock_backend, mock_process, mock_vm


class TestBenchmarkCommand:
    """Tests for the benchmark command's --local / default-share behaviour."""

    def _run_benchmark(self, monkeypatch, cli_args, api_key="test-key"):
        """Helper that patches all heavy benchmark deps and invokes the CLI."""
        if api_key:
            monkeypatch.setenv("OCTOMIL_API_KEY", api_key)
        else:
            monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()

        with (
            patch("octomil.commands.benchmark._get_api_key", return_value=api_key or ""),
            patch("octomil.serve._detect_backend", return_value=mock_backend),
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.virtual_memory", return_value=mock_vm),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["benchmark", *cli_args])
        return result

    # Benchmark no longer requires an API key — removed share tests

    def test_local_flag_skips_share(self, monkeypatch):
        """With --local, no upload attempt is made."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()

        with (
            patch("octomil.commands.benchmark._get_api_key", return_value="test-key"),
            patch("octomil.serve._detect_backend", return_value=mock_backend),
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.virtual_memory", return_value=mock_vm),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main, ["benchmark", "gemma-1b", "--local", "--iterations", "1"]
            )

        assert result.exit_code == 0
        assert "Results kept local (--local)" in result.output
        assert "Sharing anonymous benchmark data" not in result.output

    def test_benchmark_help_shows_local_flag(self):
        """--local flag is documented in help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.output
        # --share should no longer appear
        assert "--share" not in result.output


# ---------------------------------------------------------------------------
# octomil warmup
# ---------------------------------------------------------------------------


class TestWarmupCommand:
    def test_warmup_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["warmup", "--help"])
        assert result.exit_code == 0
        assert "Pre-download" in result.output
