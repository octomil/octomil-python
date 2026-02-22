"""Tests for edgeml.hardware._unified and edgeml.hardware._base."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from edgeml.hardware._base import GPUBackend, GPUBackendRegistry, reset_gpu_registry
from edgeml.hardware._types import (
    CPUInfo,
    GPUDetectionResult,
    GPUInfo,
    GPUMemory,
    HardwareProfile,
)
from edgeml.hardware._unified import UnifiedDetector, detect_hardware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cpu(**overrides) -> CPUInfo:
    defaults = dict(
        brand="Test CPU",
        cores=8,
        threads=16,
        base_speed_ghz=3.0,
        architecture="x86_64",
        has_avx2=False,
        has_avx512=False,
        has_neon=False,
        estimated_gflops=192.0,
    )
    defaults.update(overrides)
    return CPUInfo(**defaults)


def _make_gpu_result(backend: str = "cuda", vram_gb: float = 24.0) -> GPUDetectionResult:
    return GPUDetectionResult(
        gpus=[
            GPUInfo(
                index=0,
                name="Test GPU",
                memory=GPUMemory(total_gb=vram_gb),
                speed_coefficient=100,
            )
        ],
        backend=backend,
        total_vram_gb=vram_gb,
        speed_coefficient=100,
    )


def _mock_virtual_memory(total_gb: float = 16.0, available_gb: float = 12.0) -> MagicMock:
    """Build a mock psutil virtual_memory result."""
    mock_vm = MagicMock()
    mock_vm.total = int(total_gb * (1024**3))
    mock_vm.available = int(available_gb * (1024**3))
    return mock_vm


class _FakeBackend(GPUBackend):
    """Concrete GPUBackend for testing."""

    def __init__(
        self,
        name: str = "fake",
        available: bool = True,
        result: GPUDetectionResult | None = None,
        raise_on_detect: Exception | None = None,
    ) -> None:
        self._name = name
        self._available = available
        self._result = result
        self._raise_on_detect = raise_on_detect

    @property
    def name(self) -> str:
        return self._name

    def check_availability(self) -> bool:
        return self._available

    def detect(self) -> GPUDetectionResult | None:
        if self._raise_on_detect:
            raise self._raise_on_detect
        return self._result


# ---------------------------------------------------------------------------
# GPUBackendRegistry
# ---------------------------------------------------------------------------


class TestGPUBackendRegistry:
    def test_register_adds_backend(self):
        registry = GPUBackendRegistry()
        backend = _FakeBackend(name="cuda")
        registry.register(backend)
        assert len(registry.backends) == 1
        assert registry.backends[0].name == "cuda"

    def test_register_deduplicates_by_name(self):
        registry = GPUBackendRegistry()
        backend_a = _FakeBackend(name="cuda", available=True)
        backend_b = _FakeBackend(name="cuda", available=False)
        registry.register(backend_a)
        registry.register(backend_b)
        assert len(registry.backends) == 1
        # The first registration wins
        assert registry.backends[0].check_availability() is True

    def test_register_multiple_different_backends(self):
        registry = GPUBackendRegistry()
        registry.register(_FakeBackend(name="cuda"))
        registry.register(_FakeBackend(name="rocm"))
        registry.register(_FakeBackend(name="metal"))
        assert len(registry.backends) == 3

    def test_detect_best_returns_first_successful_backend(self):
        registry = GPUBackendRegistry()
        cuda_result = _make_gpu_result("cuda", 24.0)
        metal_result = _make_gpu_result("metal", 16.0)

        registry.register(_FakeBackend(name="cuda", available=True, result=cuda_result))
        registry.register(_FakeBackend(name="metal", available=True, result=metal_result))

        result, diagnostics = registry.detect_best()
        assert result is not None
        assert result.backend == "cuda"
        assert len(diagnostics) == 0

    def test_detect_best_skips_unavailable_backends(self):
        registry = GPUBackendRegistry()
        metal_result = _make_gpu_result("metal", 16.0)

        registry.register(_FakeBackend(name="cuda", available=False))
        registry.register(_FakeBackend(name="metal", available=True, result=metal_result))

        result, diagnostics = registry.detect_best()
        assert result is not None
        assert result.backend == "metal"
        assert any("cuda: not available" in d for d in diagnostics)

    def test_detect_best_returns_none_when_all_fail(self):
        registry = GPUBackendRegistry()
        registry.register(_FakeBackend(name="cuda", available=False))
        registry.register(_FakeBackend(name="rocm", available=False))

        result, diagnostics = registry.detect_best()
        assert result is None
        assert len(diagnostics) == 2
        assert all("not available" in d for d in diagnostics)

    def test_detect_best_handles_detection_exception(self):
        registry = GPUBackendRegistry()
        registry.register(
            _FakeBackend(
                name="cuda",
                available=True,
                raise_on_detect=RuntimeError("driver crash"),
            )
        )

        result, diagnostics = registry.detect_best()
        assert result is None
        assert len(diagnostics) == 1
        assert "detection failed" in diagnostics[0]
        assert "driver crash" in diagnostics[0]

    def test_detect_best_skips_backend_with_no_gpus(self):
        registry = GPUBackendRegistry()
        empty_result = GPUDetectionResult(
            gpus=[], backend="cuda", total_vram_gb=0.0, speed_coefficient=0,
        )
        registry.register(_FakeBackend(name="cuda", available=True, result=empty_result))

        result, diagnostics = registry.detect_best()
        assert result is None
        assert any("no GPUs detected" in d for d in diagnostics)

    def test_backends_property_returns_copy(self):
        registry = GPUBackendRegistry()
        registry.register(_FakeBackend(name="cuda"))
        backends = registry.backends
        backends.append(_FakeBackend(name="extra"))
        # Original registry unaffected
        assert len(registry.backends) == 1


# ---------------------------------------------------------------------------
# UnifiedDetector
# ---------------------------------------------------------------------------


class TestUnifiedDetector:
    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_cuda_detected_uses_cuda(self, mock_vm_fn, mock_get_registry, mock_detect_cpu):
        """When CUDA GPU is detected, best_backend should be 'cuda'."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(32.0, 24.0)

        cuda_result = _make_gpu_result("cuda", 24.0)
        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (cuda_result, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.best_backend == "cuda"
        assert profile.gpu is not None
        assert profile.gpu.backend == "cuda"

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_all_backends_fail_cpu_avx512_fallback(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """No GPU detected + has_avx512 -> best_backend = 'cpu_avx512'."""
        mock_detect_cpu.return_value = _make_cpu(has_avx512=True, has_avx2=True)
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, ["cuda: not available", "rocm: not available"])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.best_backend == "cpu_avx512"
        assert profile.gpu is None

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_all_backends_fail_cpu_avx2_fallback(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """No GPU + has_avx2 (no avx512) -> best_backend = 'cpu_avx2'."""
        mock_detect_cpu.return_value = _make_cpu(has_avx2=True, has_avx512=False)
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.best_backend == "cpu_avx2"

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_all_backends_fail_cpu_neon_fallback(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """No GPU + ARM with NEON -> best_backend = 'cpu_neon'."""
        mock_detect_cpu.return_value = _make_cpu(
            architecture="arm64", has_neon=True, has_avx2=False, has_avx512=False,
        )
        mock_vm_fn.return_value = _mock_virtual_memory(8.0, 6.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.best_backend == "cpu_neon"

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_all_backends_fail_plain_cpu_fallback(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """No GPU + no SIMD features -> best_backend = 'cpu'."""
        mock_detect_cpu.return_value = _make_cpu(
            has_avx2=False, has_avx512=False, has_neon=False,
        )
        mock_vm_fn.return_value = _mock_virtual_memory(8.0, 6.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.best_backend == "cpu"

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_cache_within_ttl_returns_same_object(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """Second call within CACHE_TTL returns the cached profile."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        first = detector.detect()
        second = detector.detect()

        assert first is second
        # detect_cpu should only have been called once
        assert mock_detect_cpu.call_count == 1

    @patch("edgeml.hardware._unified.time")
    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_cache_expiry_redetects(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu, mock_time,
    ):
        """After TTL expires, detect() should re-run detection."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        # Control monotonic() and time()
        mock_time.monotonic.side_effect = [0.0, 301.0]
        mock_time.time.side_effect = [1000.0, 2000.0]

        detector = UnifiedDetector()
        first = detector.detect()
        second = detector.detect()

        assert first is not second
        assert mock_detect_cpu.call_count == 2

    @patch("edgeml.hardware._unified.time")
    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_force_bypasses_cache(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu, mock_time,
    ):
        """force=True always re-runs detection even within TTL."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        mock_time.monotonic.side_effect = [0.0, 1.0]
        mock_time.time.side_effect = [1000.0, 1001.0]

        detector = UnifiedDetector()
        first = detector.detect()
        second = detector.detect(force=True)

        assert first is not second
        assert mock_detect_cpu.call_count == 2

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_diagnostics_aggregated_from_backends(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """Diagnostics from the registry are stored on the profile."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        expected_diag = ["cuda: not available", "rocm: detection failed -- boom"]
        mock_registry.detect_best.return_value = (None, expected_diag)
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.diagnostics == expected_diag

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_ram_values_computed_from_psutil(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """total_ram_gb and available_ram_gb come from psutil."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(31.5, 20.25)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        detector = UnifiedDetector()
        profile = detector.detect()

        assert profile.total_ram_gb == 31.5
        assert profile.available_ram_gb == 20.25


# ---------------------------------------------------------------------------
# Global detect_hardware() function
# ---------------------------------------------------------------------------


class TestDetectHardwareGlobal:
    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_detect_hardware_returns_profile(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """Global detect_hardware() creates a singleton detector and returns a profile."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        # Reset the module-level singleton
        import edgeml.hardware._unified as unified_mod
        unified_mod._detector = None

        profile = detect_hardware()

        assert isinstance(profile, HardwareProfile)

    @patch("edgeml.hardware._unified.detect_cpu")
    @patch("edgeml.hardware._unified.get_gpu_registry")
    @patch("psutil.virtual_memory")
    def test_detect_hardware_reuses_singleton(
        self, mock_vm_fn, mock_get_registry, mock_detect_cpu,
    ):
        """Calling detect_hardware() twice reuses the same UnifiedDetector."""
        mock_detect_cpu.return_value = _make_cpu()
        mock_vm_fn.return_value = _mock_virtual_memory(16.0, 12.0)

        mock_registry = MagicMock()
        mock_registry.detect_best.return_value = (None, [])
        mock_get_registry.return_value = mock_registry

        import edgeml.hardware._unified as unified_mod
        unified_mod._detector = None

        first = detect_hardware()
        second = detect_hardware()

        # Same cached profile (same detector, within TTL)
        assert first is second


# ---------------------------------------------------------------------------
# reset_gpu_registry()
# ---------------------------------------------------------------------------


class TestResetGpuRegistry:
    def test_reset_clears_singleton(self):
        """reset_gpu_registry() sets the module-level _registry to None."""
        import edgeml.hardware._base as base_mod

        # Make sure there is a registry
        base_mod._registry = GPUBackendRegistry()
        assert base_mod._registry is not None

        reset_gpu_registry()
        assert base_mod._registry is None

    def test_get_gpu_registry_after_reset_creates_new(self):
        """After reset, get_gpu_registry creates a fresh registry."""
        from edgeml.hardware._base import get_gpu_registry

        reset_gpu_registry()
        with patch("edgeml.hardware._base._auto_register"):
            registry = get_gpu_registry()
        assert isinstance(registry, GPUBackendRegistry)


# ---------------------------------------------------------------------------
# GPUBackend abstract class
# ---------------------------------------------------------------------------


class TestGPUBackend:
    def test_get_fingerprint_default_returns_none(self):
        """Default get_fingerprint() returns None."""
        backend = _FakeBackend(name="test")
        assert backend.get_fingerprint() is None
