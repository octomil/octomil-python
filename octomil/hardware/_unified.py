"""Unified hardware detector with caching."""

from __future__ import annotations

import logging
import sys
import time

from ._base import get_gpu_registry
from ._cpu import detect_cpu
from ._types import HardwareProfile

logger = logging.getLogger(__name__)


class UnifiedDetector:
    CACHE_TTL: float = 300.0  # 5 minutes

    def __init__(self) -> None:
        self._cache: HardwareProfile | None = None
        self._cache_time: float = 0.0

    def detect(self, force: bool = False) -> HardwareProfile:
        """Detect hardware. Uses cache if within TTL unless force=True."""
        now = time.monotonic()
        if not force and self._cache and (now - self._cache_time) < self.CACHE_TTL:
            return self._cache

        import psutil

        cpu = detect_cpu()
        ram = psutil.virtual_memory()
        total_ram_gb = round(ram.total / (1024**3), 2)
        available_ram_gb = round(ram.available / (1024**3), 2)

        registry = get_gpu_registry()
        gpu_result, diagnostics = registry.detect_best()

        # Determine best backend
        if gpu_result:
            best_backend = gpu_result.backend
        elif cpu.has_avx512:
            best_backend = "cpu_avx512"
        elif cpu.has_avx2:
            best_backend = "cpu_avx2"
        elif cpu.has_neon:
            best_backend = "cpu_neon"
        else:
            best_backend = "cpu"

        profile = HardwareProfile(
            gpu=gpu_result,
            cpu=cpu,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            platform=sys.platform,
            best_backend=best_backend,
            diagnostics=diagnostics,
            timestamp=time.time(),
        )

        self._cache = profile
        self._cache_time = now
        return profile


_detector: UnifiedDetector | None = None


def detect_hardware(force: bool = False) -> HardwareProfile:
    """One-liner API: detect all hardware capabilities."""
    global _detector
    if _detector is None:
        _detector = UnifiedDetector()
    return _detector.detect(force=force)
