"""Apple Silicon / Metal GPU detection backend."""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import sys

from ._base import GPUBackend
from ._types import GPUDetectionResult, GPUInfo, GPUMemory

logger = logging.getLogger(__name__)

# M-series SKU table: (gpu_cores, memory_bandwidth_gbps, speed_coefficient)
_M_SERIES_SKUS: dict[str, tuple[int, float, int]] = {
    # M1 family
    "M1": (8, 68.25, 25),
    "M1 Pro": (16, 200.0, 45),
    "M1 Max": (32, 400.0, 75),
    "M1 Ultra": (64, 800.0, 130),
    # M2 family
    "M2": (10, 100.0, 30),
    "M2 Pro": (19, 200.0, 50),
    "M2 Max": (38, 400.0, 80),
    "M2 Ultra": (76, 800.0, 140),
    # M3 family
    "M3": (10, 100.0, 35),
    "M3 Pro": (18, 150.0, 50),
    "M3 Max": (40, 400.0, 85),
    "M3 Ultra": (80, 800.0, 150),
    # M4 family
    "M4": (10, 120.0, 40),
    "M4 Pro": (20, 273.0, 60),
    "M4 Max": (40, 546.0, 100),
    "M4 Ultra": (80, 819.0, 160),
}


class MetalBackend(GPUBackend):
    @property
    def name(self) -> str:
        return "metal"

    def check_availability(self) -> bool:
        return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")

    def detect(self) -> GPUDetectionResult | None:
        diagnostics: list[str] = []

        chip_name = self._get_chip_name(diagnostics)
        if not chip_name:
            diagnostics.append("metal: could not determine Apple Silicon chip name")
            return None

        total_mem_gb = self._get_total_memory(diagnostics)
        if total_mem_gb <= 0:
            diagnostics.append("metal: could not determine system memory")
            return None

        # Unified memory: reserve ~4GB for OS
        os_reserve_gb = 4.0
        available_vram_gb = max(total_mem_gb - os_reserve_gb, 0.0)

        sku_key = self._match_sku(chip_name)
        gpu_cores = 0
        bandwidth_gbps = 0.0
        speed_coeff = 0

        if sku_key:
            gpu_cores, bandwidth_gbps, speed_coeff = _M_SERIES_SKUS[sku_key]
            diagnostics.append(
                f"metal: matched SKU '{sku_key}' — {gpu_cores} GPU cores, "
                f"{bandwidth_gbps} GB/s bandwidth"
            )
        else:
            diagnostics.append(
                f"metal: unknown chip '{chip_name}', using conservative estimates"
            )
            gpu_cores = 8
            speed_coeff = 20

        memory = GPUMemory(
            total_gb=round(total_mem_gb, 2),
            free_gb=round(available_vram_gb, 2),
            used_gb=round(os_reserve_gb, 2),
        )

        capabilities: dict[str, object] = {
            "unified_memory": True,
            "gpu_cores": gpu_cores,
            "memory_bandwidth_gbps": bandwidth_gbps,
            "metal_family": "apple",
        }

        gpu = GPUInfo(
            index=0,
            name=f"Apple {chip_name}"
            if not chip_name.startswith("Apple")
            else chip_name,
            memory=memory,
            speed_coefficient=speed_coeff,
            capabilities=capabilities,
            architecture="apple_silicon",
        )

        return GPUDetectionResult(
            gpus=[gpu],
            backend="metal",
            total_vram_gb=round(total_mem_gb, 2),
            is_multi_gpu=False,
            speed_coefficient=speed_coeff,
            detection_method="sysctl",
            diagnostics=diagnostics,
        )

    def get_fingerprint(self) -> str | None:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"metal:{result.stdout.strip()}"
        except Exception:
            pass
        return None

    def _get_chip_name(self, diagnostics: list[str]) -> str | None:
        """Get Apple Silicon chip name via sysctl."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                brand = result.stdout.strip()
                # Extract M-series chip name from brand string
                # e.g. "Apple M2 Pro" -> "M2 Pro", "Apple M1" -> "M1"
                match = re.search(r"(M[1-9]\d*(?:\s+(?:Pro|Max|Ultra))?)", brand)
                if match:
                    return match.group(1)
                return brand
        except Exception as exc:
            diagnostics.append(f"metal: sysctl brand_string failed — {exc}")

        # Fallback: try system_profiler
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Chip" in line:
                        # "Chip: Apple M2 Pro"
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            chip = parts[1].strip()
                            match = re.search(
                                r"(M[1-9]\d*(?:\s+(?:Pro|Max|Ultra))?)", chip
                            )
                            if match:
                                return match.group(1)
                            return chip
        except Exception as exc:
            diagnostics.append(f"metal: system_profiler fallback failed — {exc}")

        return None

    def _get_total_memory(self, diagnostics: list[str]) -> float:
        """Get total system memory in GB via sysctl hw.memsize."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                mem_bytes = int(result.stdout.strip())
                return mem_bytes / (1024**3)
        except Exception as exc:
            diagnostics.append(f"metal: sysctl hw.memsize failed — {exc}")

        # Fallback: psutil
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except Exception as exc:
            diagnostics.append(f"metal: psutil memory fallback failed — {exc}")

        return 0.0

    def _match_sku(self, chip_name: str) -> str | None:
        """Match a chip name to an SKU table key."""
        # Try exact match first
        if chip_name in _M_SERIES_SKUS:
            return chip_name

        # Try removing "Apple " prefix
        clean = chip_name.replace("Apple ", "").strip()
        if clean in _M_SERIES_SKUS:
            return clean

        # Try matching progressively: "M4 Max" -> "M4 Max", "M4 Pro" -> "M4 Pro"
        # Sort keys longest-first to match "M1 Ultra" before "M1"
        for key in sorted(_M_SERIES_SKUS.keys(), key=len, reverse=True):
            if key in clean:
                return key

        return None
