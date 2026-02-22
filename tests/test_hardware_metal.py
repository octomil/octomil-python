"""Tests for edgeml.hardware._metal — Apple Silicon Metal GPU detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from edgeml.hardware._metal import MetalBackend, _M_SERIES_SKUS


def _completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    return m


@pytest.fixture
def backend() -> MetalBackend:
    return MetalBackend()


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    def test_available_on_darwin_arm64(self, backend: MetalBackend) -> None:
        with (
            patch("edgeml.hardware._metal.sys") as mock_sys,
            patch("edgeml.hardware._metal.platform") as mock_platform,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            assert backend.check_availability() is True

    def test_not_available_on_linux(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert backend.check_availability() is False

    def test_not_available_on_darwin_x86(self, backend: MetalBackend) -> None:
        with (
            patch("edgeml.hardware._metal.sys") as mock_sys,
            patch("edgeml.hardware._metal.platform") as mock_platform,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "x86_64"
            assert backend.check_availability() is False


# ---------------------------------------------------------------------------
# _get_chip_name
# ---------------------------------------------------------------------------


class TestGetChipName:
    def test_m4_pro_from_sysctl(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Apple M4 Pro\n")
            diag: list[str] = []
            assert backend._get_chip_name(diag) == "M4 Pro"

    def test_m1_from_sysctl(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Apple M1\n")
            diag: list[str] = []
            assert backend._get_chip_name(diag) == "M1"

    def test_m3_max_from_sysctl(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Apple M3 Max\n")
            diag: list[str] = []
            assert backend._get_chip_name(diag) == "M3 Max"

    def test_fallback_to_system_profiler(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            # sysctl fails, system_profiler succeeds
            mock_run.side_effect = [
                _completed("", returncode=1),  # sysctl fails
                _completed("Hardware Overview:\n  Chip: Apple M2 Ultra\n"),
            ]
            diag: list[str] = []
            assert backend._get_chip_name(diag) == "M2 Ultra"

    def test_returns_none_when_all_fail(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            diag: list[str] = []
            assert backend._get_chip_name(diag) is None
            assert len(diag) >= 1

    def test_non_m_series_brand(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Apple Something New\n")
            diag: list[str] = []
            result = backend._get_chip_name(diag)
            assert result == "Apple Something New"


# ---------------------------------------------------------------------------
# _get_total_memory
# ---------------------------------------------------------------------------


class TestGetTotalMemory:
    def test_memory_from_sysctl(self, backend: MetalBackend) -> None:
        # 16 GB in bytes
        mem_bytes = str(16 * 1024**3)
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed(f"{mem_bytes}\n")
            diag: list[str] = []
            result = backend._get_total_memory(diag)
            assert abs(result - 16.0) < 0.01

    def test_fallback_to_psutil(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("", returncode=1)
            mock_psutil = MagicMock()
            mock_psutil.virtual_memory.return_value.total = 32 * 1024**3
            with patch.dict("sys.modules", {"psutil": mock_psutil}):
                diag: list[str] = []
                result = backend._get_total_memory(diag)
                assert abs(result - 32.0) < 0.01

    def test_returns_zero_when_all_fail(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            with patch.dict("sys.modules", {"psutil": None}):
                diag: list[str] = []
                result = backend._get_total_memory(diag)
                assert result == 0.0


# ---------------------------------------------------------------------------
# _match_sku
# ---------------------------------------------------------------------------


class TestMatchSku:
    @pytest.mark.parametrize(
        "chip_name,expected_key",
        [
            ("M4 Pro", "M4 Pro"),
            ("M1", "M1"),
            ("M3 Max", "M3 Max"),
            ("M2 Ultra", "M2 Ultra"),
            ("Apple M4", "M4"),
            ("Apple M1 Pro", "M1 Pro"),
        ],
    )
    def test_known_skus(
        self, backend: MetalBackend, chip_name: str, expected_key: str
    ) -> None:
        assert backend._match_sku(chip_name) == expected_key

    def test_unknown_chip_returns_none(self, backend: MetalBackend) -> None:
        assert backend._match_sku("Unknown Chip X99") is None


# ---------------------------------------------------------------------------
# detect (integration)
# ---------------------------------------------------------------------------


class TestDetect:
    def test_m4_pro_16gb(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mem_bytes = str(16 * 1024**3)
            mock_run.side_effect = [
                _completed("Apple M4 Pro\n"),  # chip name
                _completed(f"{mem_bytes}\n"),  # memory
            ]
            result = backend.detect()
            assert result is not None
            assert result.backend == "metal"
            assert len(result.gpus) == 1
            assert "M4 Pro" in result.gpus[0].name
            assert result.gpus[0].speed_coefficient == 60
            assert abs(result.total_vram_gb - 16.0) < 0.1
            assert result.gpus[0].capabilities["unified_memory"] is True

    def test_returns_none_when_chip_unknown(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            result = backend.detect()
            assert result is None

    def test_returns_none_when_memory_zero(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed("Apple M4\n"),  # chip name
                _completed("0\n"),  # memory = 0
            ]
            result = backend.detect()
            assert result is None

    def test_unknown_chip_conservative_estimates(self, backend: MetalBackend) -> None:
        mem_bytes = str(32 * 1024**3)
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed("Apple Something Future\n"),
                _completed(f"{mem_bytes}\n"),
            ]
            result = backend.detect()
            assert result is not None
            # Conservative speed_coefficient for unknown
            assert result.speed_coefficient == 20
            assert result.gpus[0].capabilities["gpu_cores"] == 8


# ---------------------------------------------------------------------------
# get_fingerprint
# ---------------------------------------------------------------------------


class TestGetFingerprint:
    def test_fingerprint_success(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.return_value = _completed("Apple M4 Pro")
            fp = backend.get_fingerprint()
            assert fp == "metal:Apple M4 Pro"

    def test_fingerprint_none_on_failure(self, backend: MetalBackend) -> None:
        with patch("edgeml.hardware._metal.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            assert backend.get_fingerprint() is None


# ---------------------------------------------------------------------------
# name
# ---------------------------------------------------------------------------


class TestBackendName:
    def test_name_is_metal(self, backend: MetalBackend) -> None:
        assert backend.name == "metal"


# ---------------------------------------------------------------------------
# SKU table coverage
# ---------------------------------------------------------------------------


class TestSkuTable:
    def test_all_skus_have_positive_values(self) -> None:
        for key, (cores, bw, speed) in _M_SERIES_SKUS.items():
            assert cores > 0, f"{key} has non-positive GPU cores"
            assert bw > 0, f"{key} has non-positive bandwidth"
            assert speed > 0, f"{key} has non-positive speed coefficient"

    def test_sku_table_has_m1_through_m4(self) -> None:
        for gen in ["M1", "M2", "M3", "M4"]:
            assert gen in _M_SERIES_SKUS, f"Missing {gen} base SKU"
            assert f"{gen} Pro" in _M_SERIES_SKUS, f"Missing {gen} Pro"
            assert f"{gen} Max" in _M_SERIES_SKUS, f"Missing {gen} Max"
            assert f"{gen} Ultra" in _M_SERIES_SKUS, f"Missing {gen} Ultra"
