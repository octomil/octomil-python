"""Tests for AMD ROCm GPU detection backend (edgeml.hardware._rocm)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from edgeml.hardware._rocm import (
    ROCmBackend,
    _AMD_PCI_DEVICES,
    _build_rocm_result,
    _list_drm_cards,
    _lookup_amd_speed,
)
from edgeml.hardware._types import GPUInfo, GPUMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> ROCmBackend:
    return ROCmBackend()


# ---------------------------------------------------------------------------
# Helper: build subprocess results
# ---------------------------------------------------------------------------


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cmd"], returncode=returncode, stdout=stdout, stderr=""
    )


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    def test_available_via_rocm_smi(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed("rocm-smi version 6.0.0")
            assert backend.check_availability() is True

    def test_available_via_rocminfo(self, backend: ROCmBackend) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args[0]
            if cmd == ["rocm-smi", "--version"]:
                raise FileNotFoundError
            if cmd == ["rocminfo"]:
                return _completed("ROCk module loaded")
            raise FileNotFoundError

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=side_effect):
            assert backend.check_availability() is True

    def test_available_via_sysfs(self, backend: ROCmBackend) -> None:
        """Falls back to sysfs vendor file when CLI tools missing."""

        def run_side_effect(*args, **kwargs):
            raise FileNotFoundError

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=run_side_effect):
            with patch(
                "edgeml.hardware._rocm._list_drm_cards",
                return_value=["/sys/class/drm/card0"],
            ):
                with patch("edgeml.hardware._rocm.os.path.isfile", return_value=True):
                    with patch(
                        "builtins.open", mock_open(read_data="0x1002\n")
                    ):
                        assert backend.check_availability() is True

    def test_not_available_nothing_present(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run", side_effect=FileNotFoundError
        ):
            with patch("edgeml.hardware._rocm._list_drm_cards", return_value=[]):
                assert backend.check_availability() is False

    def test_not_available_sysfs_non_amd_vendor(self, backend: ROCmBackend) -> None:
        """sysfs with Intel vendor (0x8086) does not trigger availability."""

        def run_side_effect(*args, **kwargs):
            raise FileNotFoundError

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=run_side_effect):
            with patch(
                "edgeml.hardware._rocm._list_drm_cards",
                return_value=["/sys/class/drm/card0"],
            ):
                with patch("edgeml.hardware._rocm.os.path.isfile", return_value=True):
                    with patch("builtins.open", mock_open(read_data="0x8086\n")):
                        assert backend.check_availability() is False

    def test_rocm_smi_timeout(self, backend: ROCmBackend) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise subprocess.TimeoutExpired(cmd="rocm-smi", timeout=5)

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=side_effect):
            with patch("edgeml.hardware._rocm._list_drm_cards", return_value=[]):
                assert backend.check_availability() is False


# ---------------------------------------------------------------------------
# _detect_via_rocm_smi
# ---------------------------------------------------------------------------

_ROCM_SMI_VERSION_OUTPUT = """\
======================= ROCm System Management Interface =======================
========================== Version of SMI ======================================
ROCM-SMI version: 6.0.0
"""

_ROCM_SMI_DETAIL_OUTPUT = """\
========================= GPU0 =========================
GPU[0]          : Card series:         Radeon RX 7900 XTX
GPU[0]          : VRAM Total:          24576 MB
GPU[0]          : VRAM Used:           512 MB
========================= GPU1 =========================
GPU[1]          : Card series:         Radeon RX 7900 XT
GPU[1]          : VRAM Total:          20480 MB
GPU[1]          : VRAM Used:           256 MB
"""


class TestDetectViaRocmSmi:
    def test_two_gpus_detected(self, backend: ROCmBackend) -> None:
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd == ["rocm-smi", "--version"]:
                return _completed(_ROCM_SMI_VERSION_OUTPUT)
            if "--showid" in cmd:
                return _completed(_ROCM_SMI_DETAIL_OUTPUT)
            return _completed("", returncode=1)

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            result = backend._detect_via_rocm_smi(diagnostics)
            assert result is not None
            assert len(result.gpus) == 2
            assert result.is_multi_gpu is True
            assert result.backend == "rocm"
            assert result.rocm_version == "6.0.0"
            assert result.detection_method == "rocm-smi"
            # GPU 0
            assert "7900 XTX" in result.gpus[0].name
            assert result.gpus[0].memory.total_gb == pytest.approx(24576 / 1024, rel=0.01)
            assert result.gpus[0].memory.used_gb == pytest.approx(512 / 1024, rel=0.01)
            # GPU 1
            assert "7900 XT" in result.gpus[1].name
            assert result.gpus[1].memory.total_gb == pytest.approx(20480 / 1024, rel=0.01)

    def test_rocm_smi_not_found(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run", side_effect=FileNotFoundError
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_rocm_smi(diagnostics)
            assert result is None
            assert any("not found" in d for d in diagnostics)

    def test_rocm_smi_timeout(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="rocm-smi", timeout=5),
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_rocm_smi(diagnostics)
            assert result is None
            assert any("timed out" in d for d in diagnostics)

    def test_rocm_smi_nonzero_exit(self, backend: ROCmBackend) -> None:
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd == ["rocm-smi", "--version"]:
                return _completed("version: 6.0.0")
            if "--showid" in cmd:
                return _completed("", returncode=1)
            return _completed("", returncode=1)

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            result = backend._detect_via_rocm_smi(diagnostics)
            assert result is None

    def test_vram_in_gb_units(self, backend: ROCmBackend) -> None:
        """Test VRAM parsing when rocm-smi reports in GB."""
        detail = """\
========================= GPU0 =========================
GPU[0]          : Card series:         Instinct MI300X
GPU[0]          : VRAM Total:          192 GB
GPU[0]          : VRAM Used:           10 GB
"""

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd == ["rocm-smi", "--version"]:
                return _completed("version: 6.2.0")
            if "--showid" in cmd:
                return _completed(detail)
            return _completed("", returncode=1)

        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=side_effect):
            diagnostics: list[str] = []
            result = backend._detect_via_rocm_smi(diagnostics)
            assert result is not None
            assert result.gpus[0].memory.total_gb == pytest.approx(192.0, rel=0.01)
            assert result.gpus[0].memory.used_gb == pytest.approx(10.0, rel=0.01)


# ---------------------------------------------------------------------------
# _detect_via_rocminfo
# ---------------------------------------------------------------------------

_ROCMINFO_OUTPUT = """\
*******
Agent Type: CPU
Name: Intel(R) Core(TM) i9-13900K
*******
Agent Type: GPU
Name: gfx1100
  Pool 1
    Size: 25165824 (0x1800000) KB
*******
Agent Type: GPU
Name: gfx1100
  Pool 1
    Size: 20971520 (0x1400000) KB
"""


class TestDetectViaRocminfo:
    def test_two_gpu_agents_detected(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(_ROCMINFO_OUTPUT)
            diagnostics: list[str] = []
            result = backend._detect_via_rocminfo(diagnostics)
            assert result is not None
            assert len(result.gpus) == 2
            assert result.gpus[0].capabilities.get("gfx_version") == "gfx1100"
            assert result.detection_method == "rocminfo"

    def test_skips_cpu_agents(self, backend: ROCmBackend) -> None:
        cpu_only = """\
*******
Agent Type: CPU
Name: AMD Ryzen 9 7950X
"""
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(cpu_only)
            diagnostics: list[str] = []
            result = backend._detect_via_rocminfo(diagnostics)
            assert result is None

    def test_rocminfo_not_found(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run", side_effect=FileNotFoundError
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_rocminfo(diagnostics)
            assert result is None
            assert any("not found" in d for d in diagnostics)

    def test_rocminfo_timeout(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="rocminfo", timeout=15),
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_rocminfo(diagnostics)
            assert result is None
            assert any("timed out" in d for d in diagnostics)


# ---------------------------------------------------------------------------
# _detect_via_lspci
# ---------------------------------------------------------------------------

_LSPCI_OUTPUT = """\
00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:a780]
06:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]
07:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT] [1002:7448]
"""

_LSPCI_UNKNOWN_DEVICE = """\
06:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Unknown Device [1002:ffff]
"""


class TestDetectViaLspci:
    def test_two_amd_gpus_from_lspci(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(_LSPCI_OUTPUT)
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is not None
            assert len(result.gpus) == 2
            assert result.gpus[0].name == "Radeon RX 7900 XTX"
            assert result.gpus[0].memory.total_gb == 24.0
            assert result.gpus[1].name == "Radeon RX 7900 XT"
            assert result.gpus[1].memory.total_gb == 20.0
            assert result.detection_method == "lspci"
            assert result.is_multi_gpu is True

    def test_unknown_device_id(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(_LSPCI_UNKNOWN_DEVICE)
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is not None
            assert len(result.gpus) == 1
            # Unknown device: VRAM estimated as 0.0
            assert result.gpus[0].memory.total_gb == 0.0
            # Name from lspci description line
            assert "Advanced Micro Devices" in result.gpus[0].name or "AMD" in result.gpus[0].name

    def test_ignores_non_vga_lines(self, backend: ROCmBackend) -> None:
        """Lines without VGA/3D/Display are skipped even if they have AMD IDs."""
        non_vga = "06:00.0 Audio device [0403]: Advanced Micro Devices, Inc. [AMD/ATI] [1002:ab38]\n"
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(non_vga)
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is None

    def test_lspci_not_found(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run", side_effect=FileNotFoundError
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is None
            assert any("not found" in d for d in diagnostics)

    def test_lspci_timeout(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="lspci", timeout=10),
        ):
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is None
            assert any("timed out" in d for d in diagnostics)

    def test_intel_only_lines_ignored(self, backend: ROCmBackend) -> None:
        intel_only = "00:02.0 VGA compatible controller [0300]: Intel Corporation [8086:a780]\n"
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed(intel_only)
            diagnostics: list[str] = []
            result = backend._detect_via_lspci(diagnostics)
            assert result is None


# ---------------------------------------------------------------------------
# _detect_via_sysfs
# ---------------------------------------------------------------------------


class TestDetectViaSysfs:
    def test_single_amd_card(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm._list_drm_cards",
            return_value=["/sys/class/drm/card0"],
        ):
            def isfile_side(path: str) -> bool:
                return path in (
                    "/sys/class/drm/card0/device/vendor",
                    "/sys/class/drm/card0/device/device",
                    "/sys/class/drm/card0/device/mem_info_vram_total",
                )

            def open_side(path, *args, **kwargs):
                data_map = {
                    "/sys/class/drm/card0/device/vendor": "0x1002\n",
                    "/sys/class/drm/card0/device/device": "0x744c\n",
                    "/sys/class/drm/card0/device/mem_info_vram_total": "25769803776\n",
                }
                if path in data_map:
                    return mock_open(read_data=data_map[path])()
                raise FileNotFoundError(path)

            with patch("edgeml.hardware._rocm.os.path.isfile", side_effect=isfile_side):
                with patch("builtins.open", side_effect=open_side):
                    diagnostics: list[str] = []
                    result = backend._detect_via_sysfs(diagnostics)
                    assert result is not None
                    assert len(result.gpus) == 1
                    assert result.gpus[0].name == "Radeon RX 7900 XTX"
                    expected_vram = 25769803776 / (1024**3)
                    assert result.gpus[0].memory.total_gb == pytest.approx(
                        expected_vram, rel=0.01
                    )
                    assert result.detection_method == "sysfs"

    def test_non_amd_vendor_skipped(self, backend: ROCmBackend) -> None:
        with patch(
            "edgeml.hardware._rocm._list_drm_cards",
            return_value=["/sys/class/drm/card0"],
        ):
            with patch("edgeml.hardware._rocm.os.path.isfile", return_value=True):
                with patch("builtins.open", mock_open(read_data="0x10de\n")):
                    diagnostics: list[str] = []
                    result = backend._detect_via_sysfs(diagnostics)
                    assert result is None

    def test_no_drm_cards(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm._list_drm_cards", return_value=[]):
            diagnostics: list[str] = []
            result = backend._detect_via_sysfs(diagnostics)
            assert result is None

    def test_unknown_device_id_sysfs(self, backend: ROCmBackend) -> None:
        """Unknown PCI device ID results in fallback name."""
        with patch(
            "edgeml.hardware._rocm._list_drm_cards",
            return_value=["/sys/class/drm/card0"],
        ):
            def isfile_side(path: str) -> bool:
                return path in (
                    "/sys/class/drm/card0/device/vendor",
                    "/sys/class/drm/card0/device/device",
                )

            def open_side(path, *args, **kwargs):
                data_map = {
                    "/sys/class/drm/card0/device/vendor": "0x1002\n",
                    "/sys/class/drm/card0/device/device": "0xdead\n",
                }
                if path in data_map:
                    return mock_open(read_data=data_map[path])()
                raise FileNotFoundError(path)

            with patch("edgeml.hardware._rocm.os.path.isfile", side_effect=isfile_side):
                with patch("builtins.open", side_effect=open_side):
                    diagnostics: list[str] = []
                    result = backend._detect_via_sysfs(diagnostics)
                    assert result is not None
                    assert "dead" in result.gpus[0].name or "AMD" in result.gpus[0].name

    def test_sysfs_vram_fallback_from_pci_table(self, backend: ROCmBackend) -> None:
        """When mem_info_vram_total is missing, fall back to PCI table estimate."""
        with patch(
            "edgeml.hardware._rocm._list_drm_cards",
            return_value=["/sys/class/drm/card0"],
        ):
            def isfile_side(path: str) -> bool:
                return path in (
                    "/sys/class/drm/card0/device/vendor",
                    "/sys/class/drm/card0/device/device",
                )

            def open_side(path, *args, **kwargs):
                data_map = {
                    "/sys/class/drm/card0/device/vendor": "0x1002\n",
                    "/sys/class/drm/card0/device/device": "0x744c\n",
                }
                if path in data_map:
                    return mock_open(read_data=data_map[path])()
                raise FileNotFoundError(path)

            with patch("edgeml.hardware._rocm.os.path.isfile", side_effect=isfile_side):
                with patch("builtins.open", side_effect=open_side):
                    diagnostics: list[str] = []
                    result = backend._detect_via_sysfs(diagnostics)
                    assert result is not None
                    # Falls back to PCI table's 24.0 GB for 744c
                    assert result.gpus[0].memory.total_gb == 24.0


# ---------------------------------------------------------------------------
# PCI device ID table coverage (parametrize)
# ---------------------------------------------------------------------------


class TestPCIDeviceIDTable:
    @pytest.mark.parametrize(
        "device_id, expected_name, expected_vram",
        [
            ("7551", "Radeon RX 9070 XT", 16.0),
            ("7552", "Radeon RX 9070", 16.0),
            ("744c", "Radeon RX 7900 XTX", 24.0),
            ("7448", "Radeon RX 7900 XT", 20.0),
            ("7480", "Radeon RX 7800 XT", 16.0),
            ("740f", "Instinct MI300X", 192.0),
            ("740c", "Instinct MI300A", 128.0),
            ("7408", "Instinct MI250X", 128.0),
            ("738e", "Instinct MI100", 32.0),
            ("73bf", "Radeon RX 6900 XT", 16.0),
            ("73ff", "Radeon RX 6600 XT", 8.0),
        ],
    )
    def test_pci_device_lookup(
        self, device_id: str, expected_name: str, expected_vram: float
    ) -> None:
        assert device_id in _AMD_PCI_DEVICES
        name, vram = _AMD_PCI_DEVICES[device_id]
        assert name == expected_name
        assert vram == expected_vram


# ---------------------------------------------------------------------------
# _lookup_amd_speed
# ---------------------------------------------------------------------------


class TestLookupAmdSpeed:
    @pytest.mark.parametrize(
        "gpu_name, expected_speed",
        [
            ("Radeon RX 7900 XTX", 55),
            ("Radeon RX 7900 XT", 48),
            ("Radeon RX 9070 XT", 60),
            ("Radeon RX 9070", 50),
            ("Instinct MI300X", 150),
            ("Instinct MI300A", 120),
            ("Instinct MI250X", 90),
            ("Instinct MI100", 40),
            ("Radeon RX 7800 XT", 38),
            ("Radeon RX 6600 XT", 18),
            ("Radeon RX 6600", 15),
        ],
    )
    def test_known_gpus(self, gpu_name: str, expected_speed: int) -> None:
        assert _lookup_amd_speed(gpu_name) == expected_speed

    def test_unknown_gpu_returns_zero(self) -> None:
        assert _lookup_amd_speed("NVIDIA GeForce RTX 4090") == 0
        assert _lookup_amd_speed("Some Random GPU") == 0
        assert _lookup_amd_speed("") == 0

    def test_case_insensitive(self) -> None:
        assert _lookup_amd_speed("radeon rx 7900 xtx") == 55
        assert _lookup_amd_speed("RADEON RX 7900 XTX") == 55

    def test_longer_match_wins(self) -> None:
        """'RX 9070 XT' (longer) should match before 'RX 9070'."""
        assert _lookup_amd_speed("Radeon RX 9070 XT") == 60
        assert _lookup_amd_speed("Radeon RX 9070") == 50


# ---------------------------------------------------------------------------
# _list_drm_cards
# ---------------------------------------------------------------------------


class TestListDrmCards:
    def test_returns_card_dirs(self) -> None:
        with patch("edgeml.hardware._rocm.os.path.isdir", return_value=True):
            with patch(
                "edgeml.hardware._rocm.os.listdir",
                return_value=["card0", "card1", "card0-HDMI-A-1", "card1-DP-1", "renderD128"],
            ):
                result = _list_drm_cards()
                assert "/sys/class/drm/card0" in result
                assert "/sys/class/drm/card1" in result
                # Connectors and render nodes excluded
                assert "/sys/class/drm/card0-HDMI-A-1" not in result
                assert "/sys/class/drm/card1-DP-1" not in result
                assert "/sys/class/drm/renderD128" not in result

    def test_returns_empty_when_no_drm_dir(self) -> None:
        with patch("edgeml.hardware._rocm.os.path.isdir", return_value=False):
            assert _list_drm_cards() == []

    def test_returns_empty_when_no_cards(self) -> None:
        with patch("edgeml.hardware._rocm.os.path.isdir", return_value=True):
            with patch("edgeml.hardware._rocm.os.listdir", return_value=[]):
                assert _list_drm_cards() == []


# ---------------------------------------------------------------------------
# _build_rocm_result
# ---------------------------------------------------------------------------


class TestBuildRocmResult:
    def test_single_gpu_result(self) -> None:
        gpu = GPUInfo(
            index=0,
            name="Radeon RX 7900 XTX",
            memory=GPUMemory(total_gb=24.0, free_gb=20.0, used_gb=4.0),
            speed_coefficient=55,
        )
        diagnostics = ["rocm: detected 1 GPU(s) via lspci"]
        result = _build_rocm_result([gpu], "6.0.0", "lspci", diagnostics)
        assert result.backend == "rocm"
        assert result.total_vram_gb == 24.0
        assert result.is_multi_gpu is False
        assert result.speed_coefficient == 55
        assert result.rocm_version == "6.0.0"
        assert result.detection_method == "lspci"

    def test_multi_gpu_vram_aggregation(self) -> None:
        gpu0 = GPUInfo(
            index=0,
            name="Radeon RX 7900 XTX",
            memory=GPUMemory(total_gb=24.0),
            speed_coefficient=55,
        )
        gpu1 = GPUInfo(
            index=1,
            name="Radeon RX 7900 XT",
            memory=GPUMemory(total_gb=20.0),
            speed_coefficient=48,
        )
        result = _build_rocm_result([gpu0, gpu1], None, "rocm-smi", [])
        assert result.total_vram_gb == 44.0
        assert result.is_multi_gpu is True
        assert result.speed_coefficient == 55  # best of both

    def test_rocm_version_none(self) -> None:
        gpu = GPUInfo(
            index=0,
            name="Radeon RX 7800 XT",
            memory=GPUMemory(total_gb=16.0),
            speed_coefficient=38,
        )
        result = _build_rocm_result([gpu], None, "sysfs", [])
        assert result.rocm_version is None


# ---------------------------------------------------------------------------
# detect() fallback chain integration
# ---------------------------------------------------------------------------


class TestDetectFallbackChain:
    """Test that detect() tries methods in order: rocm-smi -> rocminfo -> lspci -> sysfs."""

    def test_rocm_smi_success_skips_others(self, backend: ROCmBackend) -> None:
        with patch.object(
            backend,
            "_detect_via_rocm_smi",
            return_value=_build_rocm_result(
                [GPUInfo(index=0, name="RX 7900 XTX", memory=GPUMemory(total_gb=24.0), speed_coefficient=55)],
                "6.0.0",
                "rocm-smi",
                [],
            ),
        ) as mock_smi:
            with patch.object(backend, "_detect_via_rocminfo") as mock_rocminfo:
                with patch.object(backend, "_detect_via_lspci") as mock_lspci:
                    result = backend.detect()
                    assert result is not None
                    assert result.detection_method == "rocm-smi"
                    mock_smi.assert_called_once()
                    mock_rocminfo.assert_not_called()
                    mock_lspci.assert_not_called()

    def test_falls_to_rocminfo(self, backend: ROCmBackend) -> None:
        with patch.object(backend, "_detect_via_rocm_smi", return_value=None):
            with patch.object(
                backend,
                "_detect_via_rocminfo",
                return_value=_build_rocm_result(
                    [GPUInfo(index=0, name="gfx1100", memory=GPUMemory(total_gb=24.0), speed_coefficient=0)],
                    None,
                    "rocminfo",
                    [],
                ),
            ):
                with patch.object(backend, "_detect_via_lspci") as mock_lspci:
                    result = backend.detect()
                    assert result is not None
                    assert result.detection_method == "rocminfo"
                    mock_lspci.assert_not_called()

    def test_falls_to_lspci(self, backend: ROCmBackend) -> None:
        with patch.object(backend, "_detect_via_rocm_smi", return_value=None):
            with patch.object(backend, "_detect_via_rocminfo", return_value=None):
                with patch.object(
                    backend,
                    "_detect_via_lspci",
                    return_value=_build_rocm_result(
                        [GPUInfo(index=0, name="Radeon RX 7900 XTX", memory=GPUMemory(total_gb=24.0), speed_coefficient=55)],
                        None,
                        "lspci",
                        [],
                    ),
                ):
                    with patch.object(backend, "_detect_via_sysfs") as mock_sysfs:
                        result = backend.detect()
                        assert result is not None
                        assert result.detection_method == "lspci"
                        mock_sysfs.assert_not_called()

    def test_falls_to_sysfs(self, backend: ROCmBackend) -> None:
        with patch.object(backend, "_detect_via_rocm_smi", return_value=None):
            with patch.object(backend, "_detect_via_rocminfo", return_value=None):
                with patch.object(backend, "_detect_via_lspci", return_value=None):
                    with patch.object(
                        backend,
                        "_detect_via_sysfs",
                        return_value=_build_rocm_result(
                            [GPUInfo(index=0, name="Radeon RX 7900 XTX", memory=GPUMemory(total_gb=24.0), speed_coefficient=55)],
                            None,
                            "sysfs",
                            [],
                        ),
                    ):
                        result = backend.detect()
                        assert result is not None
                        assert result.detection_method == "sysfs"

    def test_all_methods_fail_returns_none(self, backend: ROCmBackend) -> None:
        with patch.object(backend, "_detect_via_rocm_smi", return_value=None):
            with patch.object(backend, "_detect_via_rocminfo", return_value=None):
                with patch.object(backend, "_detect_via_lspci", return_value=None):
                    with patch.object(backend, "_detect_via_sysfs", return_value=None):
                        result = backend.detect()
                        assert result is None


# ---------------------------------------------------------------------------
# get_fingerprint
# ---------------------------------------------------------------------------


class TestGetFingerprint:
    def test_fingerprint_success(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Card series: Radeon RX 7900 XTX")
            fp = backend.get_fingerprint()
            assert fp is not None
            assert fp.startswith("rocm:")
            assert "7900 XTX" in fp

    def test_fingerprint_none_on_failure(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run", side_effect=FileNotFoundError):
            assert backend.get_fingerprint() is None

    def test_fingerprint_none_on_empty(self, backend: ROCmBackend) -> None:
        with patch("edgeml.hardware._rocm.subprocess.run") as mock_run:
            mock_run.return_value = _completed("")
            assert backend.get_fingerprint() is None


# ---------------------------------------------------------------------------
# Backend name property
# ---------------------------------------------------------------------------


class TestBackendName:
    def test_name_is_rocm(self, backend: ROCmBackend) -> None:
        assert backend.name == "rocm"
