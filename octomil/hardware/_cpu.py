"""CPU feature detection using psutil and platform."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys

from ._types import CPUInfo

logger = logging.getLogger(__name__)


def detect_cpu() -> CPUInfo:
    """Detect CPU capabilities."""
    import psutil

    brand = _get_cpu_brand()
    arch = platform.machine()
    cores = psutil.cpu_count(logical=False) or 1
    threads = psutil.cpu_count(logical=True) or 1
    freq = psutil.cpu_freq()
    base_ghz = (freq.current / 1000) if freq else 0.0

    has_avx2 = False
    has_avx512 = False
    has_neon = arch in ("arm64", "aarch64")

    if sys.platform == "linux":
        has_avx2, has_avx512 = _detect_x86_features_linux()
    elif sys.platform == "darwin":
        has_avx2, has_avx512 = _detect_x86_features_macos()

    # Estimated GFLOPS: cores * GHz * SIMD width multiplier
    multiplier = 64 if has_avx512 else 32 if has_avx2 else 16 if has_neon else 8
    estimated_gflops = cores * max(base_ghz, 1.0) * multiplier

    return CPUInfo(
        brand=brand,
        cores=cores,
        threads=threads,
        base_speed_ghz=round(base_ghz, 2),
        architecture=arch,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        has_neon=has_neon,
        estimated_gflops=round(estimated_gflops, 1),
    )


def _get_cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _detect_x86_features_linux() -> tuple[bool, bool]:
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read().lower()
        has_avx2 = "avx2" in content
        has_avx512 = "avx512f" in content
        return has_avx2, has_avx512
    except Exception:
        return False, False


def _detect_x86_features_macos() -> tuple[bool, bool]:
    has_avx2 = False
    has_avx512 = False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            features = result.stdout.upper()
            has_avx2 = "AVX2" in features
        result2 = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result2.returncode == 0:
            leaf7 = result2.stdout.upper()
            if "AVX512" in leaf7:
                has_avx512 = True
    except Exception:
        pass
    return has_avx2, has_avx512
