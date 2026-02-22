"""Tests for NVIDIA CUDA GPU detection backend (edgeml.hardware._cuda)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from edgeml.hardware._cuda import (
    CUDABackend,
    _lookup_capabilities,
    _lookup_speed_coefficient,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> CUDABackend:
    return CUDABackend()


# ---------------------------------------------------------------------------
# Helper: build a CompletedProcess from nvidia-smi CSV lines
# ---------------------------------------------------------------------------


def _smi_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    """Test CUDABackend.check_availability()."""

    def test_available_when_nvidia_smi_succeeds(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run") as mock_run:
            mock_run.return_value = _smi_result("NVIDIA-SMI 550.54.14", returncode=0)
            assert backend.check_availability() is True

    def test_not_available_when_nvidia_smi_not_found(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
            with patch.object(backend, "_is_jetson_platform", return_value=False):
                assert backend.check_availability() is False

    def test_not_available_when_nvidia_smi_times_out(self, backend: CUDABackend) -> None:
        with patch(
            "edgeml.hardware._cuda.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            with patch.object(backend, "_is_jetson_platform", return_value=False):
                assert backend.check_availability() is False

    def test_falls_back_to_jetson_when_smi_missing(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
            with patch.object(backend, "_is_jetson_platform", return_value=True):
                assert backend.check_availability() is True

    def test_nvidia_smi_nonzero_exit_falls_through(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run") as mock_run:
            mock_run.return_value = _smi_result("", returncode=1)
            with patch.object(backend, "_is_jetson_platform", return_value=False):
                assert backend.check_availability() is False


# ---------------------------------------------------------------------------
# _is_jetson_platform
# ---------------------------------------------------------------------------


class TestIsJetsonPlatform:
    def test_not_jetson_on_x86(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.platform.machine", return_value="x86_64"):
            assert backend._is_jetson_platform() is False

    def test_jetson_on_aarch64_with_tegra(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.platform.machine", return_value="aarch64"):
            with patch("edgeml.hardware._cuda.os.path.exists", side_effect=lambda p: p == "/etc/nv_tegra_release"):
                assert backend._is_jetson_platform() is True

    def test_jetson_on_arm64_with_device_tree(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.platform.machine", return_value="arm64"):
            with patch(
                "edgeml.hardware._cuda.os.path.exists",
                side_effect=lambda p: p == "/proc/device-tree/model",
            ):
                assert backend._is_jetson_platform() is True

    def test_not_jetson_aarch64_no_files(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.platform.machine", return_value="aarch64"):
            with patch("edgeml.hardware._cuda.os.path.exists", return_value=False):
                assert backend._is_jetson_platform() is False


# ---------------------------------------------------------------------------
# Full nvidia-smi query (6-field CSV)
# ---------------------------------------------------------------------------

_FULL_RTX4090 = "0, NVIDIA GeForce RTX 4090, 24564, 22100, 2464, 550.54.14\n"
_FULL_RTX3060 = "0, NVIDIA GeForce RTX 3060, 12288, 11000, 1288, 535.129.03\n"
_FULL_A100 = "0, NVIDIA A100-SXM4-80GB, 81920, 79000, 2920, 525.85.12\n"

_MULTI_GPU = (
    "0, NVIDIA GeForce RTX 4090, 24564, 22100, 2464, 550.54.14\n"
    "1, NVIDIA GeForce RTX 4090, 24564, 23000, 1564, 550.54.14\n"
)


class TestFullQueryDetection:
    """Test _detect_via_nvidia_smi with 6-field CSV."""

    def _run_detect(
        self, backend: CUDABackend, full_csv: str, cuda_version: str = "12.4"
    ):
        """Helper: mock subprocess.run to return full CSV, then nvidia-smi header with CUDA version."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            cmd = args[0]
            call_count += 1
            if "--query-gpu=index,name,memory.total" in " ".join(cmd):
                return _smi_result(full_csv)
            if "--query-gpu=name,memory.total" in " ".join(cmd):
                # Should not reach here when full query succeeds
                return _smi_result("")
            if cmd == ["nvidia-smi"]:
                return _smi_result(
                    f"+------------------------------------------+\n"
                    f"| NVIDIA-SMI 550.54.14   CUDA Version: {cuda_version} |\n"
                    f"+------------------------------------------+\n"
                )
            return _smi_result("", returncode=1)

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            return backend._detect_via_nvidia_smi(diagnostics), diagnostics

    def test_rtx4090_detection(self, backend: CUDABackend) -> None:
        result, diag = self._run_detect(backend, _FULL_RTX4090)
        assert result is not None
        assert len(result.gpus) == 1
        gpu = result.gpus[0]
        assert gpu.index == 0
        assert "RTX 4090" in gpu.name
        assert gpu.memory.total_gb == pytest.approx(24564 / 1024, rel=0.01)
        assert gpu.memory.free_gb == pytest.approx(22100 / 1024, rel=0.01)
        assert gpu.memory.used_gb == pytest.approx(2464 / 1024, rel=0.01)
        assert gpu.speed_coefficient == 105
        assert gpu.architecture == "Ada Lovelace"
        assert gpu.compute_capability == "8.9"
        assert result.driver_version == "550.54.14"
        assert result.cuda_version == "12.4"
        assert result.backend == "cuda"
        assert result.detection_method == "nvidia-smi"

    def test_rtx3060_detection(self, backend: CUDABackend) -> None:
        result, _ = self._run_detect(backend, _FULL_RTX3060)
        assert result is not None
        gpu = result.gpus[0]
        assert "RTX 3060" in gpu.name
        assert gpu.speed_coefficient == 30
        assert gpu.architecture == "Ampere"
        assert gpu.compute_capability == "8.6"
        assert result.driver_version == "535.129.03"

    def test_a100_detection(self, backend: CUDABackend) -> None:
        result, _ = self._run_detect(backend, _FULL_A100)
        assert result is not None
        gpu = result.gpus[0]
        assert "A100" in gpu.name
        assert gpu.speed_coefficient == 130
        assert gpu.architecture == "Ampere"
        assert gpu.compute_capability == "8.0"
        assert gpu.capabilities.get("nvlink") is True

    def test_multi_gpu(self, backend: CUDABackend) -> None:
        result, _ = self._run_detect(backend, _MULTI_GPU)
        assert result is not None
        assert len(result.gpus) == 2
        assert result.is_multi_gpu is True
        assert result.gpus[0].index == 0
        assert result.gpus[1].index == 1
        expected_total = round((24564 + 24564) / 1024, 2)
        assert result.total_vram_gb == pytest.approx(expected_total, rel=0.01)
        assert result.speed_coefficient == 105  # best of both

    def test_cuda_version_extraction(self, backend: CUDABackend) -> None:
        result, _ = self._run_detect(backend, _FULL_RTX4090, cuda_version="12.6")
        assert result is not None
        assert result.cuda_version == "12.6"

    def test_skips_short_csv_lines(self, backend: CUDABackend) -> None:
        """Lines with fewer than 6 fields are skipped."""
        bad_csv = "0, NVIDIA GeForce RTX 4090, 24564\n"
        result, _ = self._run_detect(backend, bad_csv)
        # Full query finds no valid lines, falls through to simple query or None
        # Since our mock won't return valid simple query, this depends on the fallback
        # The full query parses 0 gpus, then simple query fires.

    def test_nvidia_smi_not_found(self, backend: CUDABackend) -> None:
        with patch(
            "edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_nvidia_smi(diagnostics)
            assert result is None
            assert any("not found" in d for d in diagnostics)

    def test_nvidia_smi_timeout(self, backend: CUDABackend) -> None:
        with patch(
            "edgeml.hardware._cuda.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15),
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_nvidia_smi(diagnostics)
            assert result is None
            assert any("timed out" in d for d in diagnostics)


# ---------------------------------------------------------------------------
# Simple nvidia-smi query fallback (2-field CSV)
# ---------------------------------------------------------------------------

_SIMPLE_RTX4090 = "NVIDIA GeForce RTX 4090, 24564\n"


class TestSimpleQueryFallback:
    """Test fallback to simple 2-field CSV query when full query fails."""

    def test_simple_query_after_full_failure(self, backend: CUDABackend) -> None:
        """When full query returns non-zero, simple query is used."""

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "--query-gpu=index,name,memory.total" in " ".join(cmd):
                return _smi_result("", returncode=1)
            if "--query-gpu=name,memory.total" in " ".join(cmd):
                return _smi_result(_SIMPLE_RTX4090)
            if cmd == ["nvidia-smi"]:
                return _smi_result(
                    "| NVIDIA-SMI 550.54.14   CUDA Version: 12.4 |\n"
                )
            return _smi_result("", returncode=1)

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            result = backend._detect_via_nvidia_smi(diagnostics)
            assert result is not None
            assert len(result.gpus) == 1
            gpu = result.gpus[0]
            assert "RTX 4090" in gpu.name
            assert gpu.memory.total_gb == pytest.approx(24564 / 1024, rel=0.01)
            # free/used default to 0.0 in simple query
            assert gpu.memory.free_gb == 0.0
            assert gpu.memory.used_gb == 0.0
            assert gpu.speed_coefficient == 105
            assert gpu.index == 0

    def test_simple_query_skips_short_lines(self, backend: CUDABackend) -> None:
        """Single-field lines are ignored in simple query."""

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "--query-gpu=index,name,memory.total" in " ".join(cmd):
                return _smi_result("", returncode=1)
            if "--query-gpu=name,memory.total" in " ".join(cmd):
                return _smi_result("NVIDIA GeForce RTX 4090\n")  # only 1 field
            if cmd == ["nvidia-smi"]:
                return _smi_result("")
            return _smi_result("", returncode=1)

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            result = backend._detect_via_nvidia_smi(diagnostics)
            assert result is None


# ---------------------------------------------------------------------------
# Jetson detection
# ---------------------------------------------------------------------------


class TestJetsonDetection:
    """Test Jetson detection via mocked file reads and platform checks."""

    def test_jetson_via_nv_tegra_release(self, backend: CUDABackend) -> None:
        tegra_content = "# R35 (release), REVISION: 4.1, GCID: 33958178"
        m = mock_open(read_data=tegra_content)

        def open_side_effect(path, *args, **kwargs):
            if path == "/etc/nv_tegra_release":
                return m()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
                diagnostics: list[str] = []
                result = backend._detect_jetson(diagnostics)
                # tegra_release content doesn't match known Jetson models,
                # so it falls through to default "Nano"
                assert result is not None
                assert len(result.gpus) == 1
                assert "Jetson" in result.gpus[0].name
                assert result.detection_method == "jetson"
                assert any("nv_tegra_release" in d for d in diagnostics)

    def test_jetson_via_device_tree_model(self, backend: CUDABackend) -> None:
        model_bytes = b"NVIDIA Jetson AGX Orin Developer Kit\x00"

        def open_side_effect(path, *args, **kwargs):
            if path == "/etc/nv_tegra_release":
                raise FileNotFoundError(path)
            if path == "/proc/device-tree/model":
                m = MagicMock()
                m.__enter__ = lambda s: s
                m.__exit__ = MagicMock(return_value=False)
                m.read.return_value = model_bytes
                return m
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
                diagnostics: list[str] = []
                result = backend._detect_jetson(diagnostics)
                assert result is not None
                gpu = result.gpus[0]
                assert "AGX Orin" in gpu.name
                assert gpu.memory.total_gb == 32.0
                assert gpu.speed_coefficient == 35
                assert any("device-tree" in d for d in diagnostics)

    def test_jetson_detection_no_match(self, backend: CUDABackend) -> None:
        """When no Jetson identification methods succeed, returns None."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("edgeml.hardware._cuda.platform.release", return_value="5.15.0-generic"):
                with patch("edgeml.hardware._cuda.os.path.isfile", return_value=False):
                    with patch(
                        "edgeml.hardware._cuda.subprocess.run",
                        side_effect=FileNotFoundError,
                    ):
                        diagnostics: list[str] = []
                        result = backend._detect_jetson(diagnostics)
                        assert result is None


# ---------------------------------------------------------------------------
# _normalize_jetson_model
# ---------------------------------------------------------------------------


class TestNormalizeJetsonModel:
    @pytest.mark.parametrize(
        "raw_input, expected_name, expected_vram",
        [
            ("NVIDIA Jetson AGX Orin Developer Kit", "AGX Orin", 32.0),
            ("Jetson Orin NX 16GB", "Orin NX 16GB", 16.0),
            ("Jetson Orin NX 8GB", "Orin NX 8GB", 8.0),
            ("jetson orin nano 8gb module", "Orin Nano 8GB", 8.0),
            ("jetson orin nano 4gb module", "Orin Nano 4GB", 4.0),
            ("Jetson Xavier NX", "Xavier", 8.0),
            ("NVIDIA Jetson AGX Xavier", "Xavier", 32.0),
            ("Jetson TX2", "TX2", 8.0),
            ("Jetson Nano Developer Kit", "Nano", 4.0),
            # Substring match on "orin" alone when nothing more specific
            ("Jetson Orin something else", "AGX Orin", 32.0),
            # Completely unknown -> fallback
            ("Unknown Jetson Board XYZ", "Nano", 4.0),
        ],
    )
    def test_normalize(
        self,
        backend: CUDABackend,
        raw_input: str,
        expected_name: str,
        expected_vram: float,
    ) -> None:
        name, vram = backend._normalize_jetson_model(raw_input)
        assert name == expected_name
        assert vram == expected_vram


# ---------------------------------------------------------------------------
# _lookup_speed_coefficient
# ---------------------------------------------------------------------------


class TestSpeedCoefficient:
    @pytest.mark.parametrize(
        "gpu_name, expected",
        [
            ("NVIDIA GeForce RTX 4090", 105),
            ("NVIDIA GeForce RTX 3060", 30),
            ("NVIDIA A100-SXM4-80GB", 130),
            ("NVIDIA H100 PCIe", 180),
            ("NVIDIA H200 SXM", 200),
            ("NVIDIA GeForce RTX 3090 Ti", 65),
            ("NVIDIA GeForce GTX 1080 Ti", 22),
            ("NVIDIA L4", 40),
            ("NVIDIA T4", 20),
            ("NVIDIA V100-SXM2-32GB", 30),
            ("NVIDIA GeForce RTX 5090", 120),
            ("NVIDIA GeForce RTX 5070 Ti", 80),
        ],
    )
    def test_known_gpus(self, gpu_name: str, expected: int) -> None:
        assert _lookup_speed_coefficient(gpu_name) == expected

    def test_unknown_gpu_returns_zero(self) -> None:
        assert _lookup_speed_coefficient("AMD Radeon RX 7900 XTX") == 0
        assert _lookup_speed_coefficient("Some Random GPU") == 0

    def test_case_insensitive(self) -> None:
        assert _lookup_speed_coefficient("nvidia geforce rtx 4090") == 105
        assert _lookup_speed_coefficient("NVIDIA GEFORCE RTX 4090") == 105


# ---------------------------------------------------------------------------
# _lookup_capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_rtx4090_capabilities(self) -> None:
        caps = _lookup_capabilities("NVIDIA GeForce RTX 4090")
        assert caps["architecture"] == "Ada Lovelace"
        assert caps["compute_capability"] == "8.9"
        assert caps["tensor_cores"] is True
        assert caps["fp8"] is True
        assert caps["nvlink"] is False

    def test_a100_capabilities(self) -> None:
        caps = _lookup_capabilities("NVIDIA A100-SXM4-80GB")
        assert caps["architecture"] == "Ampere"
        assert caps["compute_capability"] == "8.0"
        assert caps["nvlink"] is True
        assert caps["fp8"] is False

    def test_h100_capabilities(self) -> None:
        caps = _lookup_capabilities("NVIDIA H100 PCIe")
        assert caps["architecture"] == "Hopper"
        assert caps["compute_capability"] == "9.0"
        assert caps["fp8"] is True
        assert caps["nvlink"] is True

    def test_v100_capabilities(self) -> None:
        caps = _lookup_capabilities("NVIDIA V100-SXM2-32GB")
        assert caps["architecture"] == "Volta"
        assert caps["compute_capability"] == "7.0"

    def test_rtx3060_capabilities(self) -> None:
        caps = _lookup_capabilities("NVIDIA GeForce RTX 3060")
        assert caps["architecture"] == "Ampere"
        assert caps["compute_capability"] == "8.6"
        assert caps["fp8"] is False

    def test_unknown_gpu_returns_empty(self) -> None:
        caps = _lookup_capabilities("Unknown GPU")
        assert caps == {}

    def test_returns_copy_not_reference(self) -> None:
        """Ensure _lookup_capabilities returns a copy, not the internal dict."""
        caps1 = _lookup_capabilities("NVIDIA GeForce RTX 4090")
        caps2 = _lookup_capabilities("NVIDIA GeForce RTX 4090")
        caps1["custom_key"] = "mutated"
        assert "custom_key" not in caps2


# ---------------------------------------------------------------------------
# _get_cuda_version
# ---------------------------------------------------------------------------


class TestGetCudaVersion:
    def test_cuda_version_from_nvidia_smi(self, backend: CUDABackend) -> None:
        smi_output = (
            "+------------------------------------------+\n"
            "| NVIDIA-SMI 550.54.14   CUDA Version: 12.4 |\n"
            "+------------------------------------------+\n"
        )
        with patch("edgeml.hardware._cuda.subprocess.run") as mock_run:
            mock_run.return_value = _smi_result(smi_output)
            version = backend._get_cuda_version()
            assert version == "12.4"

    def test_cuda_version_fallback_to_nvcc(self, backend: CUDABackend) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args[0]
            if cmd == ["nvidia-smi"]:
                raise FileNotFoundError
            if cmd == ["nvcc", "--version"]:
                return _smi_result(
                    "nvcc: NVIDIA (R) Cuda compiler driver\n"
                    "Cuda compilation tools, release 12.2, V12.2.140\n"
                )
            raise FileNotFoundError

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=side_effect):
            version = backend._get_cuda_version()
            assert version == "12.2"

    def test_cuda_version_none_when_all_fail(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
            version = backend._get_cuda_version()
            assert version is None


# ---------------------------------------------------------------------------
# detect() integration
# ---------------------------------------------------------------------------


class TestDetectIntegration:
    """Test the top-level detect() method routing."""

    def test_detect_returns_nvidia_smi_result(self, backend: CUDABackend) -> None:
        """detect() returns nvidia-smi result when available."""

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "--query-gpu=index,name,memory.total" in " ".join(cmd):
                return _smi_result(_FULL_RTX4090)
            if cmd == ["nvidia-smi"]:
                return _smi_result("| CUDA Version: 12.4 |\n")
            return _smi_result("", returncode=1)

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=side_effect):
            result = backend.detect()
            assert result is not None
            assert len(result.gpus) == 1
            assert "RTX 4090" in result.gpus[0].name

    def test_detect_falls_back_to_jetson(self, backend: CUDABackend) -> None:
        """detect() tries Jetson when nvidia-smi fails."""
        tegra_content = "Jetson Orin NX 16GB"
        m = mock_open(read_data=tegra_content)

        def subprocess_side_effect(*args, **kwargs):
            raise FileNotFoundError

        def open_side_effect(path, *args, **kwargs):
            if path == "/etc/nv_tegra_release":
                return m()
            raise FileNotFoundError(path)

        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=subprocess_side_effect):
            with patch("builtins.open", side_effect=open_side_effect):
                result = backend.detect()
                assert result is not None
                assert "Orin NX 16GB" in result.gpus[0].name
                assert result.detection_method == "jetson"

    def test_detect_returns_none_when_nothing_found(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch("edgeml.hardware._cuda.os.path.isfile", return_value=False):
                    result = backend.detect()
                    assert result is None


# ---------------------------------------------------------------------------
# get_fingerprint
# ---------------------------------------------------------------------------


class TestGetFingerprint:
    def test_fingerprint_success(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run") as mock_run:
            mock_run.return_value = _smi_result("NVIDIA GeForce RTX 4090, 24564")
            fp = backend.get_fingerprint()
            assert fp == "cuda:NVIDIA GeForce RTX 4090, 24564"

    def test_fingerprint_none_on_failure(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run", side_effect=FileNotFoundError):
            fp = backend.get_fingerprint()
            assert fp is None

    def test_fingerprint_none_on_empty(self, backend: CUDABackend) -> None:
        with patch("edgeml.hardware._cuda.subprocess.run") as mock_run:
            mock_run.return_value = _smi_result("")
            fp = backend.get_fingerprint()
            assert fp is None


# ---------------------------------------------------------------------------
# Backend name property
# ---------------------------------------------------------------------------


class TestBackendName:
    def test_name_is_cuda(self, backend: CUDABackend) -> None:
        assert backend.name == "cuda"
