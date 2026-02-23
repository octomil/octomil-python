"""NVIDIA CUDA GPU detection backend."""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess

from ._base import GPUBackend
from ._types import GPUDetectionResult, GPUInfo, GPUMemory

logger = logging.getLogger(__name__)

# Speed coefficients: tok/s per B params at Q4_K_M
_SPEED_COEFFICIENTS: dict[str, int] = {
    # RTX 50 series
    "RTX 5090": 120,
    "RTX 5080": 95,
    "RTX 5070 Ti": 80,
    "RTX 5070": 70,
    # RTX 40 series
    "RTX 4090": 105,
    "RTX 4080 SUPER": 85,
    "RTX 4080": 80,
    "RTX 4070 Ti SUPER": 70,
    "RTX 4070 Ti": 65,
    "RTX 4070 SUPER": 60,
    "RTX 4070": 55,
    "RTX 4060 Ti": 45,
    "RTX 4060": 40,
    # RTX 30 series
    "RTX 3090 Ti": 65,
    "RTX 3090": 60,
    "RTX 3080 Ti": 55,
    "RTX 3080": 50,
    "RTX 3070 Ti": 42,
    "RTX 3070": 38,
    "RTX 3060 Ti": 35,
    "RTX 3060": 30,
    # RTX 20 series
    "RTX 2080 Ti": 35,
    "RTX 2080 SUPER": 30,
    "RTX 2080": 28,
    "RTX 2070 SUPER": 25,
    "RTX 2070": 22,
    "RTX 2060 SUPER": 20,
    "RTX 2060": 18,
    # GTX 16/10 series
    "GTX 1080 Ti": 22,
    "GTX 1080": 18,
    "GTX 1070 Ti": 16,
    "GTX 1070": 14,
    "GTX 1660 Ti": 15,
    "GTX 1660 SUPER": 14,
    "GTX 1650": 8,
    # Datacenter
    "H100": 180,
    "H200": 200,
    "A100": 130,
    "A6000": 80,
    "A5000": 60,
    "A4000": 45,
    "L40S": 90,
    "L40": 80,
    "L4": 40,
    "T4": 20,
    "V100": 30,
    # Jetson
    "AGX Orin": 35,
    "Orin NX 16GB": 25,
    "Orin NX 8GB": 18,
    "Orin Nano 8GB": 15,
    "Orin Nano 4GB": 10,
    "Xavier": 12,
    "TX2": 5,
    "Nano": 4,
}

# Capabilities: compute_capability, architecture, tensor_cores, fp8, nvlink
_CAPABILITIES: dict[str, dict[str, object]] = {
    # RTX 50 series (Blackwell)
    "RTX 5090": {
        "compute_capability": "10.0",
        "architecture": "Blackwell",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 5080": {
        "compute_capability": "10.0",
        "architecture": "Blackwell",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 5070 Ti": {
        "compute_capability": "10.0",
        "architecture": "Blackwell",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 5070": {
        "compute_capability": "10.0",
        "architecture": "Blackwell",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    # RTX 40 series (Ada Lovelace)
    "RTX 4090": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4080 SUPER": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4080": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4070 Ti SUPER": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4070 Ti": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4070 SUPER": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4070": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4060 Ti": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "RTX 4060": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    # RTX 30 series (Ampere)
    "RTX 3090 Ti": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 3090": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 3080 Ti": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 3080": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 3070 Ti": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 3070": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 3060 Ti": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 3060": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    # RTX 20 series (Turing)
    "RTX 2080 Ti": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 2080 SUPER": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 2080": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "RTX 2070 SUPER": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 2070": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 2060 SUPER": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "RTX 2060": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    # Datacenter
    "H200": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": True,
    },
    "H100": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": True,
    },
    "A100": {
        "compute_capability": "8.0",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "A6000": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
    "A5000": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "A4000": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "L40S": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "L40": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "L4": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "tensor_cores": True,
        "fp8": True,
        "nvlink": False,
    },
    "T4": {
        "compute_capability": "7.5",
        "architecture": "Turing",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": False,
    },
    "V100": {
        "compute_capability": "7.0",
        "architecture": "Volta",
        "tensor_cores": True,
        "fp8": False,
        "nvlink": True,
    },
}

# Jetson model normalization map
_JETSON_MODELS: dict[str, tuple[str, float]] = {
    "agx orin": ("AGX Orin", 32.0),
    "orin nx 16": ("Orin NX 16GB", 16.0),
    "orin nx 8": ("Orin NX 8GB", 8.0),
    "orin nx": ("Orin NX 16GB", 16.0),
    "orin nano 8": ("Orin Nano 8GB", 8.0),
    "orin nano 4": ("Orin Nano 4GB", 4.0),
    "orin nano": ("Orin Nano 8GB", 8.0),
    "orin": ("AGX Orin", 32.0),
    "xavier nx": ("Xavier", 8.0),
    "agx xavier": ("Xavier", 32.0),
    "xavier": ("Xavier", 16.0),
    "tx2": ("TX2", 8.0),
    "nano": ("Nano", 4.0),
}


class CUDABackend(GPUBackend):
    @property
    def name(self) -> str:
        return "cuda"

    def check_availability(self) -> bool:
        # Check if nvidia-smi exists
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Jetson devices may not have nvidia-smi
        if self._is_jetson_platform():
            return True

        return False

    def detect(self) -> GPUDetectionResult | None:
        diagnostics: list[str] = []

        # Try nvidia-smi first (desktop/server GPUs)
        result = self._detect_via_nvidia_smi(diagnostics)
        if result and result.gpus:
            return result

        # Try Jetson detection
        result = self._detect_jetson(diagnostics)
        if result and result.gpus:
            return result

        if diagnostics:
            logger.debug("cuda: all detection methods failed: %s", diagnostics)
        return None

    def get_fingerprint(self) -> str | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"cuda:{result.stdout.strip()}"
        except Exception:
            pass
        return None

    def _detect_via_nvidia_smi(
        self, diagnostics: list[str]
    ) -> GPUDetectionResult | None:
        """Primary detection via nvidia-smi CSV query."""
        gpus: list[GPUInfo] = []
        driver_version: str | None = None
        cuda_version: str | None = None

        # Try full query first
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue

                    idx = int(parts[0])
                    gpu_name = parts[1]
                    mem_total_mb = float(parts[2])
                    mem_free_mb = float(parts[3])
                    mem_used_mb = float(parts[4])
                    driver_version = parts[5]

                    memory = GPUMemory(
                        total_gb=round(mem_total_mb / 1024, 2),
                        free_gb=round(mem_free_mb / 1024, 2),
                        used_gb=round(mem_used_mb / 1024, 2),
                    )

                    speed_coeff = _lookup_speed_coefficient(gpu_name)
                    caps = _lookup_capabilities(gpu_name)
                    compute_cap = str(caps.get("compute_capability", "")) or None
                    arch = str(caps.get("architecture", "")) or None

                    gpus.append(
                        GPUInfo(
                            index=idx,
                            name=gpu_name,
                            memory=memory,
                            speed_coefficient=speed_coeff,
                            capabilities=caps,
                            compute_capability=compute_cap,
                            architecture=arch,
                        )
                    )

                cuda_version = self._get_cuda_version()
                diagnostics.append(
                    f"cuda: detected {len(gpus)} GPU(s) via nvidia-smi full query"
                )
        except FileNotFoundError:
            diagnostics.append("cuda: nvidia-smi not found")
            return None
        except subprocess.TimeoutExpired:
            diagnostics.append("cuda: nvidia-smi timed out")
            return None
        except Exception as exc:
            diagnostics.append(f"cuda: nvidia-smi full query failed — {exc}")

        # Fallback: simpler query
        if not gpus:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    for idx, line in enumerate(result.stdout.strip().splitlines()):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 2:
                            continue
                        gpu_name = parts[0]
                        mem_total_mb = float(parts[1])
                        memory = GPUMemory(total_gb=round(mem_total_mb / 1024, 2))
                        speed_coeff = _lookup_speed_coefficient(gpu_name)
                        caps = _lookup_capabilities(gpu_name)
                        compute_cap = str(caps.get("compute_capability", "")) or None
                        arch = str(caps.get("architecture", "")) or None

                        gpus.append(
                            GPUInfo(
                                index=idx,
                                name=gpu_name,
                                memory=memory,
                                speed_coefficient=speed_coeff,
                                capabilities=caps,
                                compute_capability=compute_cap,
                                architecture=arch,
                            )
                        )
                    cuda_version = self._get_cuda_version()
                    diagnostics.append(
                        f"cuda: detected {len(gpus)} GPU(s) via nvidia-smi simple query"
                    )
            except Exception as exc:
                diagnostics.append(f"cuda: nvidia-smi simple query failed — {exc}")

        if not gpus:
            return None

        total_vram = sum(g.memory.total_gb for g in gpus)
        best_speed = max(g.speed_coefficient for g in gpus) if gpus else 0

        return GPUDetectionResult(
            gpus=gpus,
            backend="cuda",
            total_vram_gb=round(total_vram, 2),
            is_multi_gpu=len(gpus) > 1,
            speed_coefficient=best_speed,
            driver_version=driver_version,
            cuda_version=cuda_version,
            detection_method="nvidia-smi",
            diagnostics=diagnostics,
        )

    def _detect_jetson(self, diagnostics: list[str]) -> GPUDetectionResult | None:
        """Detect NVIDIA Jetson devices via 6-method fallback chain."""
        model_str: str | None = None

        # Method 1: /etc/nv_tegra_release
        try:
            with open("/etc/nv_tegra_release") as f:
                content = f.read()
            if content.strip():
                model_str = content.strip()
                diagnostics.append("cuda/jetson: detected via /etc/nv_tegra_release")
        except (FileNotFoundError, PermissionError):
            pass

        # Method 2: device-tree model
        if not model_str:
            for dt_path in (
                "/proc/device-tree/model",
                "/sys/firmware/devicetree/base/model",
            ):
                try:
                    with open(dt_path, "rb") as f:
                        raw = f.read().rstrip(b"\x00").decode("utf-8", errors="replace")
                    if raw.strip():
                        model_str = raw.strip()
                        diagnostics.append(f"cuda/jetson: detected via {dt_path}")
                        break
                except (FileNotFoundError, PermissionError):
                    continue

        # Method 3: kernel release containing "tegra"
        if not model_str:
            try:
                release = platform.release().lower()
                if "tegra" in release:
                    model_str = f"tegra ({release})"
                    diagnostics.append(
                        "cuda/jetson: detected via kernel release string"
                    )
            except Exception:
                pass

        # Method 4: tegra utility files
        if not model_str:
            for util_path in ("/usr/bin/tegrastats", "/usr/sbin/nvpmodel"):
                if os.path.isfile(util_path):
                    model_str = "Jetson (tegra utilities present)"
                    diagnostics.append(f"cuda/jetson: detected via {util_path}")
                    break

        # Method 5: /proc/cpuinfo for nvidia/tegra
        if not model_str:
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read().lower()
                if "nvidia" in cpuinfo or "tegra" in cpuinfo:
                    model_str = "Jetson (cpuinfo)"
                    diagnostics.append("cuda/jetson: detected via /proc/cpuinfo")
            except (FileNotFoundError, PermissionError):
                pass

        # Method 6: nvcc --version
        if not model_str:
            try:
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and "cuda" in result.stdout.lower():
                    # nvcc present but no nvidia-smi — likely Jetson
                    model_str = "Jetson (nvcc present)"
                    diagnostics.append("cuda/jetson: detected via nvcc")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        if not model_str:
            return None

        normalized_name, estimated_vram = self._normalize_jetson_model(model_str)
        speed_coeff = _SPEED_COEFFICIENTS.get(normalized_name, 10)

        memory = GPUMemory(total_gb=estimated_vram)
        gpu = GPUInfo(
            index=0,
            name=f"NVIDIA Jetson {normalized_name}",
            memory=memory,
            speed_coefficient=speed_coeff,
            capabilities={"jetson": True, "unified_memory": True},
            architecture="jetson",
        )

        cuda_version = self._get_cuda_version()

        return GPUDetectionResult(
            gpus=[gpu],
            backend="cuda",
            total_vram_gb=estimated_vram,
            is_multi_gpu=False,
            speed_coefficient=speed_coeff,
            cuda_version=cuda_version,
            detection_method="jetson",
            diagnostics=diagnostics,
        )

    def _normalize_jetson_model(self, raw: str) -> tuple[str, float]:
        """Map raw Jetson identification string to a known model name and VRAM."""
        lower = raw.lower()
        # Try matching known Jetson model substrings (longest first)
        for key in sorted(_JETSON_MODELS.keys(), key=len, reverse=True):
            if key in lower:
                return _JETSON_MODELS[key]
        # Unknown Jetson — conservative fallback
        return ("Nano", 4.0)

    def _get_cuda_version(self) -> str | None:
        """Get CUDA version from nvidia-smi output or nvcc."""
        # Try nvidia-smi first (it prints CUDA version in header)
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass

        # Try nvcc
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                match = re.search(r"release\s+([\d.]+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass

        return None

    def _is_jetson_platform(self) -> bool:
        """Quick check if we're running on a Jetson-like platform."""
        if platform.machine() not in ("aarch64", "arm64"):
            return False
        for path in (
            "/etc/nv_tegra_release",
            "/proc/device-tree/model",
            "/usr/bin/tegrastats",
        ):
            if os.path.exists(path):
                return True
        return False


def _lookup_speed_coefficient(gpu_name: str) -> int:
    """Look up speed coefficient for a GPU name. Tries longest match first."""
    upper = gpu_name.upper()
    for key in sorted(_SPEED_COEFFICIENTS.keys(), key=len, reverse=True):
        if key.upper() in upper:
            return _SPEED_COEFFICIENTS[key]
    return 0


def _lookup_capabilities(gpu_name: str) -> dict[str, object]:
    """Look up known capabilities for a GPU name."""
    upper = gpu_name.upper()
    for key in sorted(_CAPABILITIES.keys(), key=len, reverse=True):
        if key.upper() in upper:
            return dict(_CAPABILITIES[key])
    return {}
