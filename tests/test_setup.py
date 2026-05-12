"""Tests for octomil.setup and octomil.commands.setup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from octomil.setup import (
    PHASE_COMPLETE,
    PHASE_CREATING_VENV,
    PHASE_FAILED,
    PHASE_INSTALLING_ENGINE,
    PHASE_PENDING,
    SetupState,
    _has_venv_module,
    _is_correct_arch,
    _save_state,
    create_managed_venv,
    detect_best_engine,
    download_model,
    find_system_python,
    get_venv_python,
    install_engine,
    is_engine_ready,
    is_setup_complete,
    is_setup_in_progress,
    load_state,
    run_setup,
)

# ---------------------------------------------------------------------------
# SetupState serialization
# ---------------------------------------------------------------------------


class TestSetupState:
    def test_default_state(self):
        state = SetupState()
        assert state.phase == PHASE_PENDING
        assert state.engine is None
        assert state.engine_installed is False
        assert state.model_downloaded is False
        assert state.error is None

    def test_save_and_load(self, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        state = SetupState(
            phase=PHASE_COMPLETE,
            engine="mlx-lm",
            package="octomil[mlx,serve]",
            engine_installed=True,
            model_key="qwen-coder-7b",
            model_downloaded=True,
            started_at=1000.0,
            finished_at=1060.0,
        )
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            loaded = load_state()

        assert loaded.phase == PHASE_COMPLETE
        assert loaded.engine == "mlx-lm"
        assert loaded.engine_installed is True
        assert loaded.model_key == "qwen-coder-7b"
        assert loaded.model_downloaded is True

    def test_load_missing_file(self, tmp_path: Path):
        with patch("octomil.setup.STATE_FILE", tmp_path / "nonexistent.json"):
            state = load_state()
        assert state.phase == PHASE_PENDING

    def test_load_corrupt_json(self, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        state_file.write_text("not valid json")
        with patch("octomil.setup.STATE_FILE", state_file):
            state = load_state()
        assert state.phase == PHASE_PENDING

    def test_load_ignores_unknown_fields(self, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        data = {"phase": "complete", "engine_installed": True, "unknown_field": "value"}
        state_file.write_text(json.dumps(data))
        with patch("octomil.setup.STATE_FILE", state_file):
            state = load_state()
        assert state.phase == PHASE_COMPLETE
        assert state.engine_installed is True


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


class TestPublicQueries:
    def test_get_venv_python_exists(self, tmp_path: Path):
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin" / "python"
        python.touch()
        with patch("octomil.setup.VENV_DIR", venv):
            assert get_venv_python() == str(python)

    def test_get_venv_python_missing(self, tmp_path: Path):
        with patch("octomil.setup.VENV_DIR", tmp_path / "no_venv"):
            assert get_venv_python() is None

    def test_is_engine_ready_true(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state = SetupState(phase=PHASE_COMPLETE, engine_installed=True)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            assert is_engine_ready() is True

    def test_is_engine_ready_false_not_installed(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state = SetupState(phase=PHASE_COMPLETE, engine_installed=False)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            assert is_engine_ready() is False

    def test_is_engine_ready_false_not_complete(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state = SetupState(phase=PHASE_INSTALLING_ENGINE, engine_installed=True)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            assert is_engine_ready() is False

    def test_is_setup_in_progress(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        for phase in [PHASE_CREATING_VENV, PHASE_INSTALLING_ENGINE, "downloading_model"]:
            state = SetupState(phase=phase)
            with patch("octomil.setup.STATE_FILE", state_file):
                _save_state(state)
                assert is_setup_in_progress() is True

    def test_is_setup_not_in_progress(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        for phase in [PHASE_PENDING, PHASE_COMPLETE, PHASE_FAILED]:
            state = SetupState(phase=phase)
            with patch("octomil.setup.STATE_FILE", state_file):
                _save_state(state)
                assert is_setup_in_progress() is False

    def test_is_setup_complete(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state = SetupState(phase=PHASE_COMPLETE)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            assert is_setup_complete() is True

    def test_is_setup_not_complete(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state = SetupState(phase=PHASE_FAILED)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(state)
            assert is_setup_complete() is False


# ---------------------------------------------------------------------------
# System Python detection
# ---------------------------------------------------------------------------


class TestFindSystemPython:
    @patch("octomil.setup._is_correct_arch", return_value=True)
    @patch("octomil.setup._has_venv_module", return_value=True)
    @patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "python3.12" else None)
    def test_finds_python312(self, mock_which, mock_venv, mock_arch):
        result = find_system_python()
        assert result == "/usr/bin/python3.12"

    @patch("octomil.setup._is_correct_arch", return_value=True)
    @patch("octomil.setup._has_venv_module", return_value=True)
    @patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "python3" else None)
    def test_falls_back_to_python3(self, mock_which, mock_venv, mock_arch):
        result = find_system_python()
        assert result == "/usr/bin/python3"

    @patch("shutil.which", return_value=None)
    def test_returns_none_when_no_python(self, mock_which):
        assert find_system_python() is None

    @patch("octomil.setup._is_correct_arch", return_value=True)
    @patch("octomil.setup._has_venv_module", return_value=False)
    @patch("shutil.which", return_value="/usr/bin/python3")
    def test_skips_python_without_venv(self, mock_which, mock_venv, mock_arch):
        assert find_system_python() is None

    @patch("octomil.setup._is_correct_arch", return_value=False)
    @patch("octomil.setup._has_venv_module", return_value=True)
    @patch("shutil.which", return_value="/usr/bin/python3")
    def test_skips_wrong_arch(self, mock_which, mock_venv, mock_arch):
        assert find_system_python() is None


class TestArchDetection:
    @patch("platform.machine", return_value="x86_64")
    def test_non_arm64_always_correct(self, mock_machine):
        assert _is_correct_arch("/usr/bin/python3") is True

    @patch("platform.machine", return_value="arm64")
    @patch(
        "subprocess.run",
        return_value=MagicMock(stdout="/usr/bin/python3: Mach-O 64-bit executable arm64"),
    )
    def test_arm64_native_python(self, mock_run, mock_machine):
        assert _is_correct_arch("/usr/bin/python3") is True

    @patch("platform.machine", return_value="arm64")
    @patch(
        "subprocess.run",
        return_value=MagicMock(stdout="/usr/bin/python3: Mach-O 64-bit executable x86_64"),
    )
    def test_arm64_rosetta_python(self, mock_run, mock_machine):
        assert _is_correct_arch("/usr/bin/python3") is False


class TestHasVenvModule:
    @patch("subprocess.run", return_value=MagicMock(returncode=0))
    def test_has_venv(self, mock_run):
        assert _has_venv_module("/usr/bin/python3") is True

    @patch("subprocess.run", return_value=MagicMock(returncode=1))
    def test_no_venv(self, mock_run):
        assert _has_venv_module("/usr/bin/python3") is False

    @patch("subprocess.run", side_effect=OSError("not found"))
    def test_error(self, mock_run):
        assert _has_venv_module("/usr/bin/python3") is False


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------


class TestDetectBestEngine:
    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Darwin")
    def test_apple_silicon(self, mock_sys, mock_machine):
        engine, package = detect_best_engine()
        assert engine == "mlx-lm"
        assert "mlx" in package

    @patch("platform.machine", return_value="x86_64")
    @patch("platform.system", return_value="Linux")
    def test_linux(self, mock_sys, mock_machine):
        engine, package = detect_best_engine()
        assert engine == "llama.cpp"
        assert "llama" in package

    @patch("platform.machine", return_value="x86_64")
    @patch("platform.system", return_value="Darwin")
    def test_intel_mac(self, mock_sys, mock_machine):
        engine, package = detect_best_engine()
        assert engine == "llama.cpp"


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------


def _make_venv_side_effect(venv_dir: Path):
    """Return a subprocess.run side_effect that recreates venv/bin/python."""

    def _side_effect(*args, **kwargs):
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin" / "python").touch()
        return MagicMock(returncode=0)

    return _side_effect


class TestCreateManagedVenv:
    @patch("octomil.setup._has_uv", return_value=True)
    def test_creates_with_uv(self, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"

        with (
            patch("octomil.setup.VENV_DIR", venv_dir),
            patch("subprocess.run", side_effect=_make_venv_side_effect(venv_dir)),
        ):
            result = create_managed_venv("/usr/bin/python3")

        assert result == str(venv_dir / "bin" / "python")

    @patch("octomil.setup._has_uv", return_value=False)
    def test_creates_with_stdlib_venv(self, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"

        with (
            patch("octomil.setup.VENV_DIR", venv_dir),
            patch("subprocess.run", side_effect=_make_venv_side_effect(venv_dir)) as mock_run,
        ):
            result = create_managed_venv("/usr/bin/python3")

        assert result == str(venv_dir / "bin" / "python")
        args = mock_run.call_args[0][0]
        assert args[0] == "/usr/bin/python3"
        assert args[1] == "-m"
        assert args[2] == "venv"

    @patch("octomil.setup._has_uv", return_value=False)
    def test_removes_existing_venv(self, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()
        (venv_dir / "old_file.txt").touch()

        with (
            patch("octomil.setup.VENV_DIR", venv_dir),
            patch("subprocess.run", side_effect=_make_venv_side_effect(venv_dir)),
        ):
            create_managed_venv("/usr/bin/python3")

        # rmtree is called first; side_effect recreates bin/python but not old_file
        assert not (venv_dir / "old_file.txt").exists()

    @patch("octomil.setup._has_uv", return_value=False)
    def test_raises_if_python_missing(self, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"

        with (
            patch("octomil.setup.VENV_DIR", venv_dir),
            patch("subprocess.run"),  # doesn't create files
            pytest.raises(RuntimeError, match="python not found"),
        ):
            create_managed_venv("/usr/bin/python3")


# ---------------------------------------------------------------------------
# Engine installation
# ---------------------------------------------------------------------------


class TestInstallEngine:
    @patch("octomil.setup._has_uv", return_value=True)
    @patch("subprocess.run")
    def test_installs_with_uv(self, mock_run, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"
        (venv_dir / "bin").mkdir(parents=True)

        with patch("octomil.setup.VENV_DIR", venv_dir):
            install_engine("octomil[mlx,serve]")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "uv"
        assert "octomil[mlx,serve]" in args

    @patch("octomil.setup._has_uv", return_value=False)
    @patch("subprocess.run")
    def test_installs_with_pip(self, mock_run, mock_uv, tmp_path: Path):
        venv_dir = tmp_path / "venv"
        (venv_dir / "bin").mkdir(parents=True)

        with patch("octomil.setup.VENV_DIR", venv_dir):
            install_engine("octomil[llama,serve]")

        # Two calls: pip upgrade + pip install
        assert mock_run.call_count == 2
        install_args = mock_run.call_args_list[1][0][0]
        assert "octomil[llama,serve]" in install_args


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


class TestDownloadModel:
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    @patch("subprocess.run")
    def test_download_success(self, mock_run, mock_venv_py):
        mock_run.return_value = MagicMock(returncode=0, stdout="Downloaded", stderr="")
        download_model("qwen-coder-7b")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "/venv/bin/python"

    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    @patch("subprocess.run")
    def test_download_failure(self, mock_run, mock_venv_py):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ERROR:Model not found")
        with pytest.raises(RuntimeError, match="Model download failed"):
            download_model("nonexistent-model")

    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_download_no_venv(self, mock_venv_py):
        with pytest.raises(RuntimeError, match="Managed venv not found"):
            download_model("qwen-coder-7b")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunSetup:
    @patch("octomil.setup.download_model")
    @patch("octomil.setup.install_engine")
    @patch("octomil.setup.create_managed_venv", return_value="/venv/bin/python")
    @patch("octomil.setup.detect_best_engine", return_value=("mlx-lm", "octomil[mlx,serve]"))
    @patch("octomil.setup.find_system_python", return_value="/usr/bin/python3")
    @patch("octomil.setup._get_recommended_model", return_value="qwen-coder-7b")
    def test_full_success(
        self,
        mock_model,
        mock_python,
        mock_engine,
        mock_venv,
        mock_install,
        mock_download,
        tmp_path: Path,
    ):
        state_file = tmp_path / "setup_state.json"
        with patch("octomil.setup.STATE_FILE", state_file):
            state = run_setup(foreground=True)

        assert state.phase == PHASE_COMPLETE
        assert state.engine == "mlx-lm"
        assert state.engine_installed is True
        assert state.model_key == "qwen-coder-7b"
        assert state.model_downloaded is True
        assert state.error is None
        mock_venv.assert_called_once_with("/usr/bin/python3")
        mock_install.assert_called_once_with("octomil[mlx,serve]")
        mock_download.assert_called_once_with("qwen-coder-7b")

    @patch("octomil.setup.find_system_python", return_value=None)
    def test_no_python_fails(self, mock_python, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        with patch("octomil.setup.STATE_FILE", state_file):
            state = run_setup(foreground=True)

        assert state.phase == PHASE_FAILED
        assert "No suitable Python" in (state.error or "")

    @patch("octomil.setup.download_model")
    @patch("octomil.setup.install_engine", side_effect=RuntimeError("pip failed"))
    @patch("octomil.setup.create_managed_venv", return_value="/venv/bin/python")
    @patch("octomil.setup.detect_best_engine", return_value=("mlx-lm", "octomil[mlx,serve]"))
    @patch("octomil.setup.find_system_python", return_value="/usr/bin/python3")
    def test_install_failure(self, mock_python, mock_engine, mock_venv, mock_install, mock_download, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        with patch("octomil.setup.STATE_FILE", state_file):
            state = run_setup(foreground=True)

        assert state.phase == PHASE_FAILED
        assert "pip failed" in (state.error or "")
        mock_download.assert_not_called()

    @patch("octomil.setup.download_model", side_effect=RuntimeError("network error"))
    @patch("octomil.setup.install_engine")
    @patch("octomil.setup.create_managed_venv", return_value="/venv/bin/python")
    @patch("octomil.setup.detect_best_engine", return_value=("mlx-lm", "octomil[mlx,serve]"))
    @patch("octomil.setup.find_system_python", return_value="/usr/bin/python3")
    @patch("octomil.setup._get_recommended_model", return_value="qwen-coder-7b")
    def test_download_failure_non_fatal(
        self,
        mock_model,
        mock_python,
        mock_engine,
        mock_venv,
        mock_install,
        mock_download,
        tmp_path: Path,
    ):
        state_file = tmp_path / "setup_state.json"
        with patch("octomil.setup.STATE_FILE", state_file):
            state = run_setup(foreground=True)

        # Model download failure is non-fatal
        assert state.phase == PHASE_COMPLETE
        assert state.engine_installed is True
        assert state.model_downloaded is False

    @patch("octomil.setup.find_system_python", return_value="/usr/bin/python3")
    def test_skips_if_already_complete(self, mock_python, tmp_path: Path):
        state_file = tmp_path / "setup_state.json"
        existing = SetupState(phase=PHASE_COMPLETE, engine_installed=True)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(existing)
            state = run_setup(foreground=True)

        assert state.phase == PHASE_COMPLETE
        mock_python.assert_not_called()

    @patch("octomil.setup.download_model")
    @patch("octomil.setup.install_engine")
    @patch("octomil.setup.create_managed_venv", return_value="/venv/bin/python")
    @patch("octomil.setup.detect_best_engine", return_value=("mlx-lm", "octomil[mlx,serve]"))
    @patch("octomil.setup.find_system_python", return_value="/usr/bin/python3")
    @patch("octomil.setup._get_recommended_model", return_value="qwen-coder-7b")
    def test_force_reruns_complete(
        self,
        mock_model,
        mock_python,
        mock_engine,
        mock_venv,
        mock_install,
        mock_download,
        tmp_path: Path,
    ):
        state_file = tmp_path / "setup_state.json"
        existing = SetupState(phase=PHASE_COMPLETE, engine_installed=True)
        with patch("octomil.setup.STATE_FILE", state_file):
            _save_state(existing)
            state = run_setup(force=True, foreground=True)

        assert state.phase == PHASE_COMPLETE
        mock_venv.assert_called_once()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestSetupCLI:
    def test_setup_status_pending(self):
        from octomil.cli import main

        with patch("octomil.setup.load_state", return_value=SetupState()):
            runner = CliRunner()
            result = runner.invoke(main, ["setup", "--status"])

        assert result.exit_code == 0
        assert "Not started" in result.output

    def test_setup_status_complete(self):
        from octomil.cli import main

        complete_state = SetupState(
            phase=PHASE_COMPLETE,
            engine="mlx-lm",
            package="octomil[mlx,serve]",
            engine_installed=True,
            model_key="qwen-coder-7b",
            model_downloaded=True,
        )
        with patch("octomil.setup.load_state", return_value=complete_state):
            runner = CliRunner()
            result = runner.invoke(main, ["setup", "--status"])

        assert result.exit_code == 0
        assert "Complete" in result.output
        assert "mlx-lm" in result.output

    def test_setup_status_failed(self):
        from octomil.cli import main

        failed_state = SetupState(
            phase=PHASE_FAILED,
            error="No suitable Python found.",
        )
        with patch("octomil.setup.load_state", return_value=failed_state):
            runner = CliRunner()
            result = runner.invoke(main, ["setup", "--status"])

        assert result.exit_code == 0
        assert "Failed" in result.output
        assert "No suitable Python" in result.output

    @patch("octomil.setup.run_setup")
    def test_setup_foreground(self, mock_run):
        from octomil.cli import main

        mock_run.return_value = SetupState(phase=PHASE_COMPLETE)
        runner = CliRunner()
        result = runner.invoke(main, ["setup", "--foreground"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(force=False, foreground=True)

    @patch("octomil.setup.run_setup")
    def test_setup_force(self, mock_run):
        from octomil.cli import main

        mock_run.return_value = SetupState(phase=PHASE_COMPLETE)
        runner = CliRunner()
        result = runner.invoke(main, ["setup", "--force"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(force=True, foreground=False)

    @patch("octomil.setup.run_setup")
    def test_setup_failure_exits_nonzero(self, mock_run):
        from octomil.cli import main

        mock_run.return_value = SetupState(phase=PHASE_FAILED, error="broken")
        runner = CliRunner()
        result = runner.invoke(main, ["setup"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Launcher integration
# ---------------------------------------------------------------------------


class TestLauncherIntegration:
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/home/user/.octomil/engines/venv/bin/python")
    def test_build_serve_cmd_uses_venv(self, mock_venv_py, mock_ready):
        from octomil.agents.launcher import _build_serve_cmd

        with patch.object(sys, "frozen", True, create=True):
            cmd = _build_serve_cmd("qwen-coder-7b", 8080)

        assert cmd[0] == "/home/user/.octomil/engines/venv/bin/python"
        assert "-m" in cmd
        assert "octomil" in cmd
        assert "serve" in cmd
        assert "qwen-coder-7b" in cmd

    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_build_serve_cmd_fallback_frozen(self, mock_venv_py, mock_ready):
        from octomil.agents.launcher import _build_serve_cmd

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", "/usr/local/bin/octomil"):
                cmd = _build_serve_cmd("qwen-coder-7b", 8080)

        assert cmd[0] == "/usr/local/bin/octomil"
        assert "serve" in cmd
        assert "-m" not in cmd

    def test_build_serve_cmd_not_frozen(self):
        from octomil.agents.launcher import _build_serve_cmd

        # Ensure frozen is not set (normal python)
        frozen = getattr(sys, "frozen", None)
        if frozen:
            delattr(sys, "frozen")
        try:
            cmd = _build_serve_cmd("qwen-coder-7b", 8080)
            assert "-m" in cmd
            assert "octomil" in cmd
        finally:
            if frozen is not None:
                sys.frozen = frozen


# ---------------------------------------------------------------------------
# Serve re-exec integration
# ---------------------------------------------------------------------------


class TestServeReexec:
    """Re-exec tests now delegate to octomil.venv_reexec.

    See tests/test_venv_reexec.py for comprehensive tests.
    These remain as integration smoke tests verifying the serve command
    still reaches the shared helper.
    """

    @patch("octomil.setup.load_state")
    @patch("octomil.setup.run_setup")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_try_venv_reexec_no_venv_does_not_run_setup_by_default(
        self, mock_venv, mock_ready, mock_progress, mock_run_setup, mock_load_state
    ):
        """When no venv exists, try_venv_reexec falls through without user Python."""
        from octomil.setup import SetupState
        from octomil.venv_reexec import try_venv_reexec

        mock_load_state.return_value = SetupState()  # phase="pending"
        mock_run_setup.return_value = SetupState(phase="failed", error="test")

        result = try_venv_reexec()
        assert result is False
        mock_run_setup.assert_not_called()

    @patch("octomil.setup.load_state")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    def test_try_venv_reexec_skips_if_already_failed(self, mock_venv, mock_ready, mock_progress, mock_load_state):
        """When previous setup failed, don't retry — fall through."""
        from octomil.setup import SetupState
        from octomil.venv_reexec import try_venv_reexec

        mock_load_state.return_value = SetupState(phase="failed", error="no python")

        result = try_venv_reexec()
        assert result is False

    @patch("os.execv")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    def test_try_venv_reexec_success(self, mock_venv, mock_ready, mock_progress, mock_execv):
        from octomil.venv_reexec import try_venv_reexec

        with patch.object(sys, "argv", ["octomil", "serve", "qwen-coder-7b", "--port", "8080"]):
            try_venv_reexec()

        mock_execv.assert_called_once()
        call_args = mock_execv.call_args[0]
        assert call_args[0] == "/venv/bin/python"
        assert "-m" in call_args[1]
        assert "octomil" in call_args[1]
        assert "serve" in call_args[1]
        assert "qwen-coder-7b" in call_args[1]


# ---------------------------------------------------------------------------
# _ensure_engine_ready (launcher auto-setup)
# ---------------------------------------------------------------------------


class TestEnsureEngineReady:
    @patch("octomil.setup.load_state")
    @patch("octomil.setup.run_setup")
    @patch("octomil.setup.is_engine_ready", return_value=False)
    @patch("octomil.setup.get_venv_python", return_value=None)
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    def test_runs_setup_when_no_venv(self, mock_progress, mock_venv, mock_ready, mock_run_setup, mock_load_state):
        from octomil.agents.launcher import _ensure_engine_ready
        from octomil.setup import SetupState

        mock_load_state.return_value = SetupState()
        mock_run_setup.return_value = SetupState(phase="complete")

        with patch.object(sys, "frozen", True, create=True):
            _ensure_engine_ready()

        mock_run_setup.assert_called_once()

    @patch("octomil.setup.is_engine_ready", return_value=True)
    @patch("octomil.setup.get_venv_python", return_value="/venv/bin/python")
    @patch("octomil.setup.is_setup_in_progress", return_value=False)
    def test_skips_when_already_ready(self, mock_progress, mock_venv, mock_ready):
        from octomil.agents.launcher import _ensure_engine_ready

        with patch.object(sys, "frozen", True, create=True):
            _ensure_engine_ready()
        # No setup call — already ready

    def test_noop_when_not_frozen(self):
        from octomil.agents.launcher import _ensure_engine_ready

        _ensure_engine_ready()  # should not import octomil.setup at all


# ---------------------------------------------------------------------------
# OpenClaw configure_local
# ---------------------------------------------------------------------------


class TestOpenClawConfigureLocal:
    @patch("subprocess.run")
    def test_configure_openclaw_runs_config_commands(self, mock_run):
        from octomil.agents.registry import _configure_openclaw

        mock_run.return_value = MagicMock(returncode=0)

        result = _configure_openclaw("http://localhost:8080/v1", "qwen-coder-7b")

        assert result == {}
        # _configure_openclaw now runs 2 commands:
        # 1. openclaw config set models.providers.octomil <json>
        # 2. openclaw config set agents.defaults.model.primary octomil/<model>
        assert mock_run.call_count == 2
        # Check that openclaw config set was called with correct provider
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("models.providers.octomil" in cmd for cmd in calls)
        assert any("octomil/qwen-coder-7b" in cmd for cmd in calls)
