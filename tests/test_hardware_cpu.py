"""Tests for edgeml.hardware._cpu — CPU feature detection."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch


from edgeml.hardware._cpu import (
    _detect_x86_features_linux,
    _detect_x86_features_macos,
    _get_cpu_brand,
    detect_cpu,
)


def _completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    return m


# ---------------------------------------------------------------------------
# detect_cpu
# ---------------------------------------------------------------------------


class TestDetectCpu:
    def test_returns_cpu_info(self) -> None:
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.side_effect = lambda logical: 16 if logical else 8
        mock_freq = MagicMock()
        mock_freq.current = 3200.0
        mock_psutil.cpu_freq.return_value = mock_freq

        with (
            patch.dict("sys.modules", {"psutil": mock_psutil}),
            patch("edgeml.hardware._cpu._get_cpu_brand", return_value="Test CPU"),
            patch("edgeml.hardware._cpu.platform") as mock_platform,
            patch("edgeml.hardware._cpu.sys") as mock_sys,
        ):
            mock_platform.machine.return_value = "x86_64"
            mock_sys.platform = "linux"
            with patch(
                "edgeml.hardware._cpu._detect_x86_features_linux",
                return_value=(True, False),
            ):
                result = detect_cpu()

        assert result.brand == "Test CPU"
        assert result.cores == 8
        assert result.threads == 16
        assert result.architecture == "x86_64"
        assert result.has_avx2 is True
        assert result.has_avx512 is False

    def test_neon_on_arm64(self) -> None:
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.side_effect = lambda logical: 10 if logical else 10
        mock_psutil.cpu_freq.return_value = None

        with (
            patch.dict("sys.modules", {"psutil": mock_psutil}),
            patch("edgeml.hardware._cpu._get_cpu_brand", return_value="Apple M4"),
            patch("edgeml.hardware._cpu.platform") as mock_platform,
            patch("edgeml.hardware._cpu.sys") as mock_sys,
        ):
            mock_platform.machine.return_value = "arm64"
            mock_sys.platform = "darwin"
            with patch(
                "edgeml.hardware._cpu._detect_x86_features_macos",
                return_value=(False, False),
            ):
                result = detect_cpu()

        assert result.has_neon is True
        assert result.has_avx2 is False
        assert result.base_speed_ghz == 0.0
        # GFLOPS uses 1.0 as floor when freq is 0
        assert result.estimated_gflops > 0

    def test_gflops_with_avx512(self) -> None:
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.side_effect = lambda logical: 32 if logical else 16
        mock_freq = MagicMock()
        mock_freq.current = 2500.0
        mock_psutil.cpu_freq.return_value = mock_freq

        with (
            patch.dict("sys.modules", {"psutil": mock_psutil}),
            patch("edgeml.hardware._cpu._get_cpu_brand", return_value="Intel Xeon"),
            patch("edgeml.hardware._cpu.platform") as mock_platform,
            patch("edgeml.hardware._cpu.sys") as mock_sys,
        ):
            mock_platform.machine.return_value = "x86_64"
            mock_sys.platform = "linux"
            with patch(
                "edgeml.hardware._cpu._detect_x86_features_linux",
                return_value=(True, True),
            ):
                result = detect_cpu()

        assert result.has_avx512 is True
        # 16 cores * 2.5 GHz * 64 = 2560 GFLOPS
        assert result.estimated_gflops == 2560.0


# ---------------------------------------------------------------------------
# _get_cpu_brand
# ---------------------------------------------------------------------------


class TestGetCpuBrand:
    def test_darwin_sysctl(self) -> None:
        with (
            patch("edgeml.hardware._cpu.sys") as mock_sys,
            patch("edgeml.hardware._cpu.subprocess.run") as mock_run,
        ):
            mock_sys.platform = "darwin"
            mock_run.return_value = _completed("Apple M4 Pro\n")
            assert _get_cpu_brand() == "Apple M4 Pro"

    def test_fallback_to_platform(self) -> None:
        with (
            patch("edgeml.hardware._cpu.sys") as mock_sys,
            patch("edgeml.hardware._cpu.platform") as mock_platform,
        ):
            mock_sys.platform = "linux"
            mock_platform.processor.return_value = "Intel Core i9"
            assert _get_cpu_brand() == "Intel Core i9"

    def test_fallback_to_machine(self) -> None:
        with (
            patch("edgeml.hardware._cpu.sys") as mock_sys,
            patch("edgeml.hardware._cpu.platform") as mock_platform,
        ):
            mock_sys.platform = "linux"
            mock_platform.processor.return_value = ""
            mock_platform.machine.return_value = "x86_64"
            assert _get_cpu_brand() == "x86_64"

    def test_sysctl_failure_fallback(self) -> None:
        with (
            patch("edgeml.hardware._cpu.sys") as mock_sys,
            patch("edgeml.hardware._cpu.subprocess.run") as mock_run,
            patch("edgeml.hardware._cpu.platform") as mock_platform,
        ):
            mock_sys.platform = "darwin"
            mock_run.side_effect = FileNotFoundError
            mock_platform.processor.return_value = "arm"
            assert _get_cpu_brand() == "arm"


# ---------------------------------------------------------------------------
# _detect_x86_features_linux
# ---------------------------------------------------------------------------


class TestLinuxFeatures:
    def test_avx2_and_avx512(self) -> None:
        content = "flags : fpu avx2 avx512f sse4_2\n"
        with patch("builtins.open", mock_open(read_data=content)):
            avx2, avx512 = _detect_x86_features_linux()
            assert avx2 is True
            assert avx512 is True

    def test_avx2_only(self) -> None:
        content = "flags : fpu avx2 sse4_2\n"
        with patch("builtins.open", mock_open(read_data=content)):
            avx2, avx512 = _detect_x86_features_linux()
            assert avx2 is True
            assert avx512 is False

    def test_no_avx(self) -> None:
        content = "flags : fpu sse4_2\n"
        with patch("builtins.open", mock_open(read_data=content)):
            avx2, avx512 = _detect_x86_features_linux()
            assert avx2 is False
            assert avx512 is False

    def test_file_not_found(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            avx2, avx512 = _detect_x86_features_linux()
            assert avx2 is False
            assert avx512 is False


# ---------------------------------------------------------------------------
# _detect_x86_features_macos
# ---------------------------------------------------------------------------


class TestMacosFeatures:
    def test_avx2_from_sysctl(self) -> None:
        with patch("edgeml.hardware._cpu.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed("FPU VME SSE AVX2\n"),  # machdep.cpu.features
                _completed("RDWRFSGS\n"),  # leaf7_features (no avx512)
            ]
            avx2, avx512 = _detect_x86_features_macos()
            assert avx2 is True
            assert avx512 is False

    def test_avx512_from_leaf7(self) -> None:
        with patch("edgeml.hardware._cpu.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed("FPU VME SSE AVX2\n"),
                _completed("AVX512BW AVX512CD\n"),
            ]
            avx2, avx512 = _detect_x86_features_macos()
            assert avx2 is True
            assert avx512 is True

    def test_sysctl_failure(self) -> None:
        with patch("edgeml.hardware._cpu.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            avx2, avx512 = _detect_x86_features_macos()
            assert avx2 is False
            assert avx512 is False
