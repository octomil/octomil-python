"""Tests for ROCm/hipBLAS detection in llama.cpp engine (EDG-116)."""

from __future__ import annotations

from unittest.mock import patch

from octomil.engines.llamacpp_engine import LlamaCppEngine


# ---------------------------------------------------------------------------
# _has_rocm() detection
# ---------------------------------------------------------------------------


class TestHasRocm:
    """Test the _has_rocm() static method."""

    def test_detected_via_env_var(self) -> None:
        """HIP_VISIBLE_DEVICES env var triggers ROCm detection."""
        with patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "0"}):
            with patch("os.path.isdir", return_value=False):
                with patch("shutil.which", return_value=None):
                    assert LlamaCppEngine._has_rocm() is True

    def test_detected_via_opt_rocm(self) -> None:
        """/opt/rocm directory triggers ROCm detection."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.path.isdir", return_value=True) as mock_isdir:
                assert LlamaCppEngine._has_rocm() is True
                mock_isdir.assert_called_once_with("/opt/rocm")

    def test_detected_via_rocminfo(self) -> None:
        """rocminfo on PATH triggers ROCm detection."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.path.isdir", return_value=False):
                with patch("shutil.which", return_value="/usr/bin/rocminfo"):
                    assert LlamaCppEngine._has_rocm() is True

    def test_not_detected_when_nothing_present(self) -> None:
        """No ROCm indicators means _has_rocm() returns False."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.path.isdir", return_value=False):
                with patch("shutil.which", return_value=None):
                    assert LlamaCppEngine._has_rocm() is False

    def test_empty_env_var_not_detected(self) -> None:
        """Empty HIP_VISIBLE_DEVICES should not trigger detection."""
        with patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": ""}):
            with patch("os.path.isdir", return_value=False):
                with patch("shutil.which", return_value=None):
                    assert LlamaCppEngine._has_rocm() is False


# ---------------------------------------------------------------------------
# detect_info() with ROCm
# ---------------------------------------------------------------------------


class TestDetectInfoRocm:
    """Test detect_info() returns correct string when ROCm is available."""

    def test_rocm_on_linux(self) -> None:
        """Non-macOS with ROCm (and no CUDA) reports CPU + ROCm."""
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch.object(LlamaCppEngine, "_has_cuda", return_value=False):
                with patch.object(LlamaCppEngine, "_has_rocm", return_value=True):
                    assert engine.detect_info() == "CPU + ROCm"

    def test_cuda_takes_priority_over_rocm(self) -> None:
        """When both CUDA and ROCm are available, CUDA wins."""
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch.object(LlamaCppEngine, "_has_cuda", return_value=True):
                with patch.object(LlamaCppEngine, "_has_rocm", return_value=True):
                    assert engine.detect_info() == "CPU + CUDA"

    def test_metal_takes_priority_on_macos(self) -> None:
        """macOS always reports Metal regardless of ROCm."""
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Darwin"
        ):
            assert engine.detect_info() == "CPU + Metal"

    def test_cpu_fallback_no_gpu(self) -> None:
        """No GPU acceleration falls back to CPU only."""
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch.object(LlamaCppEngine, "_has_cuda", return_value=False):
                with patch.object(LlamaCppEngine, "_has_rocm", return_value=False):
                    assert engine.detect_info() == "CPU"


# ---------------------------------------------------------------------------
# display_name with ROCm
# ---------------------------------------------------------------------------


class TestDisplayNameRocm:
    """Test display_name property reflects ROCm when detected."""

    def test_rocm_display_name(self) -> None:
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch(
                "octomil.engines.llamacpp_engine.platform.machine",
                return_value="x86_64",
            ):
                with patch.object(LlamaCppEngine, "_has_cuda", return_value=False):
                    with patch.object(LlamaCppEngine, "_has_rocm", return_value=True):
                        assert engine.display_name == "llama.cpp (ROCm, x86_64)"

    def test_cuda_display_name_over_rocm(self) -> None:
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch(
                "octomil.engines.llamacpp_engine.platform.machine",
                return_value="x86_64",
            ):
                with patch.object(LlamaCppEngine, "_has_cuda", return_value=True):
                    with patch.object(LlamaCppEngine, "_has_rocm", return_value=True):
                        assert engine.display_name == "llama.cpp (CUDA, x86_64)"

    def test_cpu_display_name_fallback(self) -> None:
        engine = LlamaCppEngine()
        with patch(
            "octomil.engines.llamacpp_engine.platform.system", return_value="Linux"
        ):
            with patch(
                "octomil.engines.llamacpp_engine.platform.machine",
                return_value="x86_64",
            ):
                with patch.object(LlamaCppEngine, "_has_cuda", return_value=False):
                    with patch.object(LlamaCppEngine, "_has_rocm", return_value=False):
                        assert engine.display_name == "llama.cpp (CPU, x86_64)"
