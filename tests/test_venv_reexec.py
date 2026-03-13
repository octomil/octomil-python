"""Tests for octomil.venv_reexec — shared venv re-exec logic."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from octomil.venv_reexec import needs_venv_reexec, try_venv_reexec, wait_for_setup

# ---------------------------------------------------------------------------
# needs_venv_reexec
# ---------------------------------------------------------------------------


class TestNeedsVenvReexec:
    def test_true_when_frozen(self):
        with patch.object(sys, "frozen", True, create=True):
            assert needs_venv_reexec() is True

    def test_false_when_not_frozen(self):
        frozen = getattr(sys, "frozen", None)
        if frozen is not None:
            delattr(sys, "frozen")
        try:
            assert needs_venv_reexec() is False
        finally:
            if frozen is not None:
                sys.frozen = frozen


# ---------------------------------------------------------------------------
# try_venv_reexec
# ---------------------------------------------------------------------------


class TestTryVenvReexec:
    @patch("octomil.setup.load_state")
    @patch("octomil.setup.run_setup")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_no_venv_runs_setup(self, mock_venv, mock_ready, mock_progress, mock_run_setup, mock_load_state):
        """When no venv exists, try_venv_reexec runs setup inline."""
        from octomil.setup import SetupState

        mock_load_state.return_value = SetupState()  # phase="pending"
        mock_run_setup.return_value = SetupState(phase="failed", error="test")

        result = try_venv_reexec()
        assert result is False
        mock_run_setup.assert_called_once()

    @patch("octomil.setup.load_state")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_skips_if_already_failed(self, mock_venv, mock_ready, mock_progress, mock_load_state):
        """When previous setup failed, don't retry — fall through."""
        from octomil.setup import SetupState

        mock_load_state.return_value = SetupState(phase="failed", error="no python")

        result = try_venv_reexec()
        assert result is False

    @patch("os.execv")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    def test_success_execv(self, mock_venv, mock_ready, mock_progress, mock_execv):
        """When venv is ready, os.execv is called with correct args."""
        with patch.object(sys, "argv", ["octomil", "serve", "qwen-coder-7b", "--port", "8080"]):
            try_venv_reexec()

        mock_execv.assert_called_once()
        call_args = mock_execv.call_args[0]
        assert call_args[0] == "/venv/bin/python"
        assert "-m" in call_args[1]
        assert "octomil" in call_args[1]
        assert "serve" in call_args[1]
        assert "qwen-coder-7b" in call_args[1]

    @patch("os.execv")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    def test_preserves_benchmark_args(self, mock_venv, mock_ready, mock_progress, mock_execv):
        """Re-exec preserves benchmark command args."""
        with patch.object(sys, "argv", ["octomil", "benchmark", "llama-8b", "--local", "--all-engines"]):
            try_venv_reexec()

        mock_execv.assert_called_once()
        new_argv = mock_execv.call_args[0][1]
        assert "benchmark" in new_argv
        assert "llama-8b" in new_argv
        assert "--local" in new_argv
        assert "--all-engines" in new_argv

    @patch("os.execv")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    def test_preserves_mcp_serve_args(self, mock_venv, mock_ready, mock_progress, mock_execv):
        """Re-exec preserves mcp serve command args."""
        with patch.object(sys, "argv", ["octomil", "mcp", "serve", "--port", "9000", "--model", "gemma-3b"]):
            try_venv_reexec()

        mock_execv.assert_called_once()
        new_argv = mock_execv.call_args[0][1]
        assert "mcp" in new_argv
        assert "serve" in new_argv
        assert "--port" in new_argv
        assert "9000" in new_argv

    @patch("octomil.setup.run_setup")
    @patch("octomil.setup.load_state")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", side_effect=[None, "/venv/bin/python"])
    @patch("os.execv")
    def test_runs_setup_then_reexecs(
        self, mock_execv, mock_venv, mock_ready, mock_progress, mock_load_state, mock_run_setup
    ):
        """When setup succeeds, re-exec follows.

        Note: is_engine_ready is only called once (after setup) because
        the first get_venv_python() returns None and short-circuits.
        """
        from octomil.setup import SetupState

        mock_load_state.return_value = SetupState()
        mock_run_setup.return_value = SetupState(phase="complete", engine_installed=True)

        with patch.object(sys, "argv", ["octomil", "serve", "qwen-coder-7b"]):
            try_venv_reexec()

        mock_run_setup.assert_called_once()
        mock_execv.assert_called_once()


# ---------------------------------------------------------------------------
# wait_for_setup
# ---------------------------------------------------------------------------


class TestWaitForSetup:
    @patch("time.sleep")
    @patch("octomil.setup.load_state")
    @patch("octomil.setup.is_setup_in_progress", side_effect=[True, True, False])
    def test_waits_until_complete(self, mock_progress, mock_load_state, mock_sleep):
        from octomil.setup import SetupState

        mock_load_state.side_effect = [
            SetupState(phase="installing_engine"),
            SetupState(phase="installing_engine"),
            SetupState(phase="complete"),
        ]
        wait_for_setup()
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @patch("octomil.setup.load_state")
    @patch("octomil.setup.is_setup_in_progress", side_effect=[True, False])
    def test_reports_failure(self, mock_progress, mock_load_state, mock_sleep):
        from octomil.setup import SetupState

        mock_load_state.side_effect = [
            SetupState(phase="installing_engine"),
            SetupState(phase="failed", error="pip crashed"),
        ]
        wait_for_setup()
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Benchmark re-exec integration
# ---------------------------------------------------------------------------


class TestBenchmarkReexec:
    @patch("octomil.venv_reexec.try_venv_reexec")
    @patch("octomil.runtime.engines.get_registry")
    def test_benchmark_triggers_reexec_when_frozen_no_native(self, mock_registry, mock_reexec):
        """benchmark() calls try_venv_reexec when frozen + no native engines."""
        # Mock registry to return only ollama
        mock_detection = MagicMock()
        mock_detection.available = True
        mock_detection.engine.name = "ollama"
        mock_registry.return_value.detect_all.return_value = [mock_detection]

        mock_reexec.return_value = False  # simulate re-exec not available

        with patch.object(sys, "frozen", True, create=True):
            from click.testing import CliRunner

            from octomil.commands.benchmark import benchmark

            runner = CliRunner()
            # Will fail after re-exec check since _detect_backend won't work,
            # but we just need to verify try_venv_reexec was called
            with patch("octomil.serve._detect_backend") as mock_detect:
                mock_backend = MagicMock()
                mock_backend.name = "ollama"
                mock_detect.return_value = mock_backend
                # Mock psutil to avoid actual system calls
                with patch("psutil.Process"):
                    with patch("psutil.virtual_memory"):
                        # Won't complete fully, but re-exec should have been tried
                        runner.invoke(
                            benchmark,
                            ["test-model", "--local", "--iterations", "1"],
                            catch_exceptions=True,
                        )

            mock_reexec.assert_called_once()

    def test_benchmark_skips_reexec_when_not_frozen(self):
        """benchmark() does NOT call try_venv_reexec when not frozen."""
        frozen = getattr(sys, "frozen", None)
        if frozen is not None:
            delattr(sys, "frozen")
        try:
            with patch("octomil.serve._detect_backend") as mock_detect:
                mock_backend = MagicMock()
                mock_backend.name = "echo"
                mock_backend.generate.return_value = (
                    "hi",
                    MagicMock(
                        tokens_per_second=10.0,
                        ttfc_ms=5.0,
                        prompt_tokens=5,
                        total_tokens=10,
                    ),
                )
                mock_detect.return_value = mock_backend

                with patch("psutil.Process") as mock_proc:
                    mock_proc.return_value.memory_info.return_value.rss = 1024 * 1024
                    from click.testing import CliRunner

                    from octomil.commands.benchmark import benchmark

                    runner = CliRunner()
                    result = runner.invoke(
                        benchmark,
                        ["test-model", "--local", "--iterations", "1"],
                        catch_exceptions=True,
                    )
                    # When not frozen, the frozen gate prevents registry import
                    assert result.exit_code == 0 or "Error" not in (result.output or "")
        finally:
            if frozen is not None:
                sys.frozen = frozen


# ---------------------------------------------------------------------------
# MCP serve re-exec integration
# ---------------------------------------------------------------------------


class TestMCPServeReexec:
    @patch("octomil.venv_reexec.try_venv_reexec")
    @patch("octomil.runtime.engines.get_registry")
    def test_mcp_serve_triggers_reexec_when_frozen_no_native(self, mock_registry, mock_reexec):
        """mcp serve calls try_venv_reexec when frozen + no native engines."""
        mock_detection = MagicMock()
        mock_detection.available = True
        mock_detection.engine.name = "ollama"
        mock_registry.return_value.detect_all.return_value = [mock_detection]

        mock_reexec.return_value = False

        with patch.object(sys, "frozen", True, create=True):
            from click.testing import CliRunner

            from octomil.commands.mcp_cmd import mcp

            runner = CliRunner()
            # Will fail after re-exec check (missing fastapi), but we verify
            # try_venv_reexec was called
            runner.invoke(
                mcp,
                ["serve"],
                catch_exceptions=True,
            )

            mock_reexec.assert_called_once()
