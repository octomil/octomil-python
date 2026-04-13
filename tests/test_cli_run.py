"""Tests for `octomil run`, `octomil embed`, and `octomil transcribe` CLI commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from octomil.cli import main
from octomil.commands.inference import RuntimeInstallCandidate
from octomil.config.local import ResolvedExecutionDefaults
from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.kernel import ExecutionResult


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# octomil run
# ---------------------------------------------------------------------------


def _mock_result(text: str = "Hello!") -> ExecutionResult:
    return ExecutionResult(
        id="resp_abc123",
        model="gemma3-1b",
        capability="chat",
        locality="on_device",
        fallback_used=False,
        output_text=text,
        usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    )


def _mock_cloud_fallback_result(text: str = "Hello from cloud!") -> ExecutionResult:
    result = _mock_result(text)
    result.locality = "cloud"
    result.fallback_used = True
    return result


class TestRunCommand:
    def test_run_with_prompt(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream", "Hello!"])
            assert result.exit_code == 0
            assert "Hello!" in result.output

    def test_run_reads_stdin(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream"], input="Hello from stdin!")
            assert result.exit_code == 0

    def test_run_json_output(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--json", "Hello!"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["model"] == "gemma3-1b"
            assert data["locality"] == "on_device"
            assert data["output_text"] == "Hello!"

    def test_run_missing_prompt_fails(self, runner):
        result = runner.invoke(main, ["run", "--no-stream"])
        assert result.exit_code != 0
        assert "Missing prompt" in result.output

    def test_run_with_model_override(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream", "--model", "phi-mini", "Hello!"])
            assert result.exit_code == 0
            call_kwargs = mock_kernel.create_response.call_args
            assert call_kwargs.kwargs.get("model") == "phi-mini" or call_kwargs[1].get("model") == "phi-mini"

    def test_run_model_errors_do_not_traceback(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(
                side_effect=OctomilError(
                    code=OctomilErrorCode.MODEL_NOT_FOUND,
                    message="Unknown model 'gemma-1b'. Did you mean: gemma3-1b?",
                )
            )
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream", "Hello!"])
            assert result.exit_code != 0
            assert "Unknown model 'gemma-1b'" in result.output
            assert "Traceback" not in result.output

    def test_run_runtime_errors_do_not_traceback(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(side_effect=RuntimeError("No runtime available"))
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream", "Hello!"])
            assert result.exit_code != 0
            assert "No inference backend available" in result.output
            assert "pip install 'octomil[mlx]'" in result.output
            assert "Traceback" not in result.output

    def test_run_install_runtime_retries_once(self, runner):
        candidate = RuntimeInstallCandidate(
            engine_name="test-runtime",
            requirement="test-runtime>=1.0",
            extra_name="test",
            description="test local inference",
        )

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(side_effect=[RuntimeError("No runtime available"), _mock_result()])
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference._recommended_runtime_install", return_value=candidate):
                with patch("octomil.commands.inference._runtime_install_command", return_value=["install-runtime"]):
                    with patch("octomil.commands.inference.subprocess.run") as mock_run:
                        result = runner.invoke(main, ["run", "--no-stream", "--install-runtime", "Hello!"])

            assert result.exit_code == 0
            assert "Installing test-runtime runtime" in result.output
            assert "Hello!" in result.output
            assert mock_kernel.create_response.await_count == 2
            mock_run.assert_called_once_with(["install-runtime"], check=True)

    def test_run_interactive_prompt_can_install_runtime(self, runner):
        candidate = RuntimeInstallCandidate(
            engine_name="test-runtime",
            requirement="test-runtime>=1.0",
            extra_name="test",
            description="test local inference",
        )

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(side_effect=[RuntimeError("No runtime available"), _mock_result()])
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference._can_prompt_runtime_install", return_value=True):
                with patch("octomil.commands.inference._recommended_runtime_install", return_value=candidate):
                    with patch("octomil.commands.inference._runtime_install_command", return_value=["install-runtime"]):
                        with patch("octomil.commands.inference.subprocess.run") as mock_run:
                            result = runner.invoke(main, ["run", "--no-stream", "Hello!"], input="y\n")

            assert result.exit_code == 0
            assert "Install test-runtime now" in result.output
            assert "Hello!" in result.output
            assert mock_kernel.create_response.await_count == 2
            mock_run.assert_called_once()

    def test_run_install_runtime_happens_before_cloud_fallback(self, runner):
        candidate = RuntimeInstallCandidate(
            engine_name="test-runtime",
            requirement="test-runtime>=1.0",
            extra_name="test",
            description="test local inference",
        )

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.resolve_chat_defaults = MagicMock(
                return_value=ResolvedExecutionDefaults(
                    model="gemma3-1b",
                    policy_preset="local_first",
                )
            )
            mock_kernel.create_response = AsyncMock(return_value=_mock_cloud_fallback_result())
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference._has_real_local_runtime", return_value=False):
                with patch("octomil.commands.inference._recommended_runtime_install", return_value=candidate):
                    with patch("octomil.commands.inference._runtime_install_command", return_value=["install-runtime"]):
                        with patch("octomil.commands.inference.subprocess.run") as mock_run:
                            result = runner.invoke(main, ["run", "--no-stream", "--install-runtime", "Hello!"])

            assert result.exit_code == 0
            assert "Installing test-runtime runtime" in result.output
            assert "Using hosted cloud fallback" in result.output
            mock_run.assert_called_once_with(["install-runtime"], check=True)

    def test_run_no_install_runtime_disables_env_install(self, runner, monkeypatch):
        monkeypatch.setenv("OCTOMIL_AUTO_INSTALL_RUNTIME", "1")

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(side_effect=RuntimeError("No runtime available"))
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference.subprocess.run") as mock_run:
                result = runner.invoke(main, ["run", "--no-stream", "--no-install-runtime", "Hello!"])

            assert result.exit_code != 0
            assert "No inference backend available" in result.output
            assert "Traceback" not in result.output
            mock_run.assert_not_called()

    def test_run_warns_when_using_cloud_fallback(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_cloud_fallback_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["run", "--no-stream", "Hello!"])
            assert result.exit_code == 0
            assert "Hello from cloud!" in result.output
            assert "Using hosted cloud fallback" in result.output

    def test_run_uses_runner_for_resolved_default_model(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.resolve_chat_defaults = MagicMock(
                return_value=ResolvedExecutionDefaults(
                    model="gemma3-1b",
                    policy_preset="local_first",
                )
            )
            mock_kernel.create_response = AsyncMock(return_value=_mock_result("kernel"))
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference._has_real_local_runtime", return_value=True):
                with patch("octomil.commands.inference._planner_engine_for_runner", return_value="mlx-lm"):
                    with patch(
                        "octomil.commands.inference._try_runner_response",
                        return_value="runner local",
                    ) as mock_runner:
                        result = runner.invoke(main, ["run", "--no-stream", "Hello!"])

            assert result.exit_code == 0
            assert result.output.strip() == "runner local"
            mock_runner.assert_called_once()
            assert mock_runner.call_args.args[:2] == ("gemma3-1b", "Hello!")
            assert mock_runner.call_args.kwargs["engine"] == "mlx-lm"
            mock_kernel.create_response.assert_not_awaited()

    def test_run_cloud_only_skips_local_runner(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.resolve_chat_defaults = MagicMock(
                return_value=ResolvedExecutionDefaults(
                    model="gemma3-1b",
                    policy_preset="cloud_only",
                )
            )
            mock_kernel.create_response = AsyncMock(return_value=_mock_result("cloud path"))
            mock_kf.return_value = mock_kernel

            with patch("octomil.commands.inference._has_real_local_runtime", return_value=True):
                with patch("octomil.commands.inference._try_runner_response") as mock_runner:
                    result = runner.invoke(main, ["run", "--no-stream", "--policy", "cloud_only", "Hello!"])

            assert result.exit_code == 0
            assert "cloud path" in result.output
            mock_runner.assert_not_called()
            mock_kernel.create_response.assert_awaited_once()


# ---------------------------------------------------------------------------
# octomil embed
# ---------------------------------------------------------------------------


def _mock_embed_result() -> ExecutionResult:
    return ExecutionResult(
        id="emb_abc123",
        model="nomic-embed-text-v1.5",
        capability="embedding",
        locality="on_device",
        fallback_used=False,
        embeddings=[[0.1, 0.2, 0.3]],
        dimensions=3,
        usage={"input_tokens": 5, "total_tokens": 5},
    )


class TestEmbedCommand:
    def test_embed_with_text_json(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_embeddings = AsyncMock(return_value=_mock_embed_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["embed", "--json", "On-device inference"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["model"] == "nomic-embed-text-v1.5"
            assert len(data["data"]) == 1

    def test_embed_summary_without_json(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_embeddings = AsyncMock(return_value=_mock_embed_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["embed", "Hello"])
            assert result.exit_code == 0
            assert "Generated 1 embedding(s)" in result.output

    def test_embed_missing_input_fails(self, runner):
        result = runner.invoke(main, ["embed"])
        assert result.exit_code != 0
        assert "Missing input" in result.output


# ---------------------------------------------------------------------------
# octomil transcribe
# ---------------------------------------------------------------------------


def _mock_transcribe_result() -> ExecutionResult:
    return ExecutionResult(
        id="txn_abc123",
        model="whisper-small",
        capability="transcription",
        locality="on_device",
        fallback_used=False,
        output_text="Hello world.",
    )


class TestTranscribeCommand:
    def test_transcribe_file(self, runner, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.transcribe_audio = AsyncMock(return_value=_mock_transcribe_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["transcribe", str(audio_file)])
            assert result.exit_code == 0
            assert "Hello world." in result.output

    def test_transcribe_json_output(self, runner, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.transcribe_audio = AsyncMock(return_value=_mock_transcribe_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["transcribe", "--json", str(audio_file)])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["text"] == "Hello world."
            assert data["locality"] == "on_device"

    def test_transcribe_missing_input_fails(self, runner):
        result = runner.invoke(main, ["transcribe"])
        assert result.exit_code != 0
        assert "Missing audio" in result.output

    def test_transcribe_nonexistent_file_fails(self, runner):
        result = runner.invoke(main, ["transcribe", "/nonexistent/file.wav"])
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# API-exact commands
# ---------------------------------------------------------------------------


class TestResponsesCreate:
    def test_responses_create(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_response = AsyncMock(return_value=_mock_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["responses", "create", "--model", "gemma3-1b", "--input", "Hello!"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["model"] == "gemma3-1b"


class TestEmbeddingsCreate:
    def test_embeddings_create(self, runner):
        with patch("octomil.commands.inference._kernel") as mock_kf:
            mock_kernel = AsyncMock()
            mock_kernel.create_embeddings = AsyncMock(return_value=_mock_embed_result())
            mock_kf.return_value = mock_kernel

            result = runner.invoke(main, ["embeddings", "create", "--model", "nomic", "--input", "test text"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "data" in data
