"""Tests for edgeml.cli — Click command-line interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from edgeml.cli import _get_api_key, main


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "env-key-123")
        assert _get_api_key() == "env-key-123"

    def test_empty_when_no_env_no_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        monkeypatch.setattr(
            "os.path.expanduser", lambda x: str(tmp_path / ".edgeml" / "credentials")
        )
        assert _get_api_key() == ""

    def test_from_credentials_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        cred_dir = tmp_path / ".edgeml"
        cred_dir.mkdir()
        cred_file = cred_dir / "credentials"
        cred_file.write_text("api_key=file-key-456\n")

        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_file) if "credentials" in x else x,
        )
        # Need to also patch os.path.exists and open to use the right path
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser", lambda x: str(cred_dir / "credentials")
        )
        assert _get_api_key() == "file-key-456"

    def test_env_takes_priority(self, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "env-key")
        assert _get_api_key() == "env-key"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


class TestMainGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "EdgeML" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


# ---------------------------------------------------------------------------
# edgeml serve
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_basic(self):
        runner = CliRunner()
        with patch("edgeml.serve.run_server") as mock_run:
            result = runner.invoke(main, ["serve", "gemma-1b"])
        assert result.exit_code == 0
        assert "Starting EdgeML serve" in result.output
        assert "gemma-1b" in result.output
        mock_run.assert_called_once()

    def test_serve_custom_port(self):
        runner = CliRunner()
        with patch("edgeml.serve.run_server"):
            result = runner.invoke(main, ["serve", "gemma-1b", "--port", "9000"])
        assert result.exit_code == 0
        assert "9000" in result.output

    def test_serve_share_no_key(self, monkeypatch):
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        runner = CliRunner()
        with patch("edgeml.serve.run_server"):
            result = runner.invoke(main, ["serve", "test", "--share"])
        assert result.exit_code == 0
        assert "--share requires an API key" in result.output


# ---------------------------------------------------------------------------
# edgeml login
# ---------------------------------------------------------------------------


class TestLoginCommand:
    def test_login_saves_credentials(self, tmp_path, monkeypatch):
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["login", "--api-key", "test-key-789"])
        assert result.exit_code == 0
        assert "API key saved" in result.output

        # Verify file was created
        cred_file = cred_dir / "credentials"
        assert cred_file.exists()
        assert "api_key=test-key-789" in cred_file.read_text()


# ---------------------------------------------------------------------------
# edgeml dashboard
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    @patch("edgeml.cli.webbrowser.open")
    def test_dashboard_opens_browser(self, mock_open):
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert "Opening dashboard" in result.output
        mock_open.assert_called_once_with("https://app.edgeml.io")

    @patch("edgeml.cli.webbrowser.open")
    def test_dashboard_custom_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_DASHBOARD_URL", "https://custom.dashboard.io")
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        mock_open.assert_called_once_with("https://custom.dashboard.io")


# ---------------------------------------------------------------------------
# edgeml deploy --phone
# ---------------------------------------------------------------------------


class TestDeployCommand:
    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

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
        assert call_url.startswith("edgeml://pair?")
        assert "token=ABC123" in call_url
        assert "host=" in call_url

    @patch("edgeml.cli._get_client")
    def test_deploy_rollout(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
# edgeml status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    @patch("edgeml.cli._get_client")
    def test_status(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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

    @patch("edgeml.cli._get_client")
    def test_status_no_rollouts(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
# edgeml push
# ---------------------------------------------------------------------------


class TestPushCommand:
    @patch("edgeml.cli._get_client")
    def test_push(self, mock_get_client, tmp_path, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.push.return_value = {"formats": {"onnx": "ok", "coreml": "ok"}}
        mock_get_client.return_value = mock_client

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake model data")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["push", str(model_file), "--name", "test-model", "--version", "1.0.0"],
        )
        assert result.exit_code == 0
        assert "Uploaded: test-model v1.0.0" in result.output
        mock_client.push.assert_called_once()


# ---------------------------------------------------------------------------
# edgeml pull
# ---------------------------------------------------------------------------


class TestPullCommand:
    @patch("edgeml.cli._get_client")
    def test_pull(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
# edgeml train start
# ---------------------------------------------------------------------------


class TestTrainStartCommand:
    @patch("edgeml.cli._get_client")
    def test_train_start_basic(self, mock_get_client, monkeypatch):
        from edgeml.models import TrainingSession

        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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

    @patch("edgeml.cli._get_client")
    def test_train_start_with_options(self, mock_get_client, monkeypatch):
        from edgeml.models import TrainingSession

        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["train", "start", "model"])
        assert result.exit_code != 0

    @patch("edgeml.cli._get_client")
    def test_train_start_with_group(self, mock_get_client, monkeypatch):
        from edgeml.models import TrainingSession

        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(
            main, ["train", "start", "model", "--strategy", "invalid"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# edgeml train status
# ---------------------------------------------------------------------------


class TestTrainStatusCommand:
    @patch("edgeml.cli._get_client")
    def test_train_status(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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

    @patch("edgeml.cli._get_client")
    def test_train_status_no_metrics(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
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
# edgeml train stop
# ---------------------------------------------------------------------------


class TestTrainStopCommand:
    @patch("edgeml.cli._get_client")
    def test_train_stop(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.train_stop.return_value = {"last_round": 23}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["train", "stop", "sentiment-v1"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
        assert "23" in result.output


# ---------------------------------------------------------------------------
# edgeml benchmark
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
            monkeypatch.setenv("EDGEML_API_KEY", api_key)
        else:
            monkeypatch.delenv("EDGEML_API_KEY", raising=False)

        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()

        with (
            patch("edgeml.cli._get_api_key", return_value=api_key or ""),
            patch("edgeml.serve._detect_backend", return_value=mock_backend),
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.virtual_memory", return_value=mock_vm),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["benchmark", *cli_args])
        return result

    def test_default_shares_with_api_key(self, monkeypatch):
        """Without --local and with an API key, benchmark data is shared."""
        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("edgeml.cli._get_api_key", return_value="test-key"),
            patch("edgeml.serve._detect_backend", return_value=mock_backend),
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.virtual_memory", return_value=mock_vm),
            patch("httpx.post", return_value=mock_resp) as mock_post,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["benchmark", "gemma-1b", "--iterations", "1"])

        assert result.exit_code == 0
        assert "Sharing anonymous benchmark data" in result.output
        assert "Benchmark data shared successfully" in result.output
        mock_post.assert_called_once()

    def test_default_shares_warns_no_api_key(self, monkeypatch):
        """Without --local and without an API key, warns about missing key."""
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()

        with (
            patch("edgeml.cli._get_api_key", return_value=""),
            patch("edgeml.serve._detect_backend", return_value=mock_backend),
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.virtual_memory", return_value=mock_vm),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["benchmark", "gemma-1b", "--iterations", "1"])

        assert result.exit_code == 0
        assert "Skipping share: no API key" in result.output
        assert "--local" in result.output

    def test_local_flag_skips_share(self, monkeypatch):
        """With --local, no upload attempt is made."""
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        mock_backend, mock_process, mock_vm = _make_benchmark_mocks()

        with (
            patch("edgeml.cli._get_api_key", return_value="test-key"),
            patch("edgeml.serve._detect_backend", return_value=mock_backend),
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
