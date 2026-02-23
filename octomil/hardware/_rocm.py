"""AMD ROCm GPU detection backend with 4-method fallback chain."""

from __future__ import annotations

import logging
import os
import re
import subprocess

from ._base import GPUBackend
from ._types import GPUDetectionResult, GPUInfo, GPUMemory

logger = logging.getLogger(__name__)

# AMD PCI device ID -> (name, estimated_vram_gb)
_AMD_PCI_DEVICES: dict[str, tuple[str, float]] = {
    # RDNA 4
    "7551": ("Radeon RX 9070 XT", 16.0),
    "7552": ("Radeon RX 9070", 16.0),
    # RDNA 3
    "744c": ("Radeon RX 7900 XTX", 24.0),
    "7448": ("Radeon RX 7900 XT", 20.0),
    "745e": ("Radeon RX 7900 GRE", 16.0),
    "7480": ("Radeon RX 7800 XT", 16.0),
    "7470": ("Radeon RX 7700 XT", 12.0),
    "7460": ("Radeon RX 7600 XT", 16.0),
    "7422": ("Radeon RX 7600", 8.0),
    # RDNA 2
    "73bf": ("Radeon RX 6900 XT", 16.0),
    "73a5": ("Radeon RX 6950 XT", 16.0),
    "73df": ("Radeon RX 6700 XT", 12.0),
    "73ff": ("Radeon RX 6600 XT", 8.0),
    "73e3": ("Radeon RX 6600", 8.0),
    # CDNA
    "740f": ("Instinct MI300X", 192.0),
    "740c": ("Instinct MI300A", 128.0),
    "7408": ("Instinct MI250X", 128.0),
    "7388": ("Instinct MI250", 128.0),
    "738c": ("Instinct MI210", 64.0),
    "7380": ("Instinct MI200", 64.0),
    "738e": ("Instinct MI100", 32.0),
}

# Speed coefficients for AMD GPUs (tok/s per B params at Q4_K_M)
_AMD_SPEED_COEFFICIENTS: dict[str, int] = {
    # CDNA
    "MI300X": 150,
    "MI300A": 120,
    "MI250X": 90,
    "MI250": 80,
    "MI210": 60,
    "MI200": 55,
    "MI100": 40,
    # RDNA 4
    "RX 9070 XT": 60,
    "RX 9070": 50,
    # RDNA 3
    "RX 7900 XTX": 55,
    "RX 7900 XT": 48,
    "RX 7900 GRE": 40,
    "RX 7800 XT": 38,
    "RX 7700 XT": 30,
    "RX 7600 XT": 25,
    "RX 7600": 20,
    # RDNA 2
    "RX 6950 XT": 35,
    "RX 6900 XT": 32,
    "RX 6700 XT": 22,
    "RX 6600 XT": 18,
    "RX 6600": 15,
}


class ROCmBackend(GPUBackend):
    @property
    def name(self) -> str:
        return "rocm"

    def check_availability(self) -> bool:
        # Check for rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for rocminfo
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check sysfs for AMD vendor
        try:
            for card_dir in sorted(_list_drm_cards()):
                vendor_path = os.path.join(card_dir, "device", "vendor")
                if os.path.isfile(vendor_path):
                    with open(vendor_path) as f:
                        vendor = f.read().strip()
                    if vendor == "0x1002":
                        return True
        except Exception:
            pass

        return False

    def detect(self) -> GPUDetectionResult | None:
        diagnostics: list[str] = []

        # Method 1: rocm-smi
        result = self._detect_via_rocm_smi(diagnostics)
        if result and result.gpus:
            return result

        # Method 2: rocminfo
        result = self._detect_via_rocminfo(diagnostics)
        if result and result.gpus:
            return result

        # Method 3: lspci
        result = self._detect_via_lspci(diagnostics)
        if result and result.gpus:
            return result

        # Method 4: sysfs
        result = self._detect_via_sysfs(diagnostics)
        if result and result.gpus:
            return result

        if diagnostics:
            logger.debug("rocm: all detection methods failed: %s", diagnostics)
        return None

    def get_fingerprint(self) -> str | None:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"rocm:{result.stdout.strip()[:200]}"
        except Exception:
            pass
        return None

    def _detect_via_rocm_smi(self, diagnostics: list[str]) -> GPUDetectionResult | None:
        """Method 1: Parse rocm-smi output for GPU info."""
        gpus: list[GPUInfo] = []
        rocm_version: str | None = None

        try:
            # Get ROCm version
            ver_result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ver_result.returncode == 0:
                for line in ver_result.stdout.splitlines():
                    if "version" in line.lower():
                        match = re.search(r"([\d.]+)", line)
                        if match:
                            rocm_version = match.group(1)
                            break

            # Get GPU details via rocm-smi --showallinfo or individual queries
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                diagnostics.append(
                    f"rocm: rocm-smi query returned exit code {result.returncode}"
                )
                return None

            output = result.stdout

            # Parse GPU blocks — rocm-smi output format varies by version,
            # but typically has sections per GPU.
            # Try to extract GPU index, name, and VRAM info.
            current_gpu_idx = -1
            gpu_names: dict[int, str] = {}
            gpu_vram_total: dict[int, float] = {}
            gpu_vram_used: dict[int, float] = {}

            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Match GPU index headers like "GPU[0]" or "=============== GPU0 ==============="
                idx_match = re.search(r"GPU\[?(\d+)\]?", line, re.IGNORECASE)
                if idx_match:
                    current_gpu_idx = int(idx_match.group(1))

                # Match product name
                if "card series" in line.lower() or "product name" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) == 2 and current_gpu_idx >= 0:
                        gpu_names[current_gpu_idx] = parts[1].strip()

                # Match VRAM total / used (in bytes or MB)
                if "vram total" in line.lower():
                    mem_match = re.search(
                        r"([\d.]+)\s*(MB|GB|bytes)?", line.split(":", 1)[-1]
                    )
                    if mem_match and current_gpu_idx >= 0:
                        val = float(mem_match.group(1))
                        unit = (mem_match.group(2) or "bytes").upper()
                        if unit == "GB":
                            gpu_vram_total[current_gpu_idx] = val
                        elif unit == "MB":
                            gpu_vram_total[current_gpu_idx] = val / 1024
                        else:
                            gpu_vram_total[current_gpu_idx] = val / (1024**3)

                if "vram used" in line.lower():
                    mem_match = re.search(
                        r"([\d.]+)\s*(MB|GB|bytes)?", line.split(":", 1)[-1]
                    )
                    if mem_match and current_gpu_idx >= 0:
                        val = float(mem_match.group(1))
                        unit = (mem_match.group(2) or "bytes").upper()
                        if unit == "GB":
                            gpu_vram_used[current_gpu_idx] = val
                        elif unit == "MB":
                            gpu_vram_used[current_gpu_idx] = val / 1024
                        else:
                            gpu_vram_used[current_gpu_idx] = val / (1024**3)

            # Build GPUInfo from parsed data
            all_indices = sorted(set(gpu_names.keys()) | set(gpu_vram_total.keys()))
            for idx in all_indices:
                gpu_name = gpu_names.get(idx, f"AMD GPU {idx}")
                total = round(gpu_vram_total.get(idx, 0.0), 2)
                used = round(gpu_vram_used.get(idx, 0.0), 2)
                free = round(max(total - used, 0.0), 2)

                memory = GPUMemory(total_gb=total, free_gb=free, used_gb=used)
                speed_coeff = _lookup_amd_speed(gpu_name)

                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=gpu_name,
                        memory=memory,
                        speed_coefficient=speed_coeff,
                    )
                )

            if gpus:
                diagnostics.append(f"rocm: detected {len(gpus)} GPU(s) via rocm-smi")

        except FileNotFoundError:
            diagnostics.append("rocm: rocm-smi not found")
            return None
        except subprocess.TimeoutExpired:
            diagnostics.append("rocm: rocm-smi timed out")
            return None
        except Exception as exc:
            diagnostics.append(f"rocm: rocm-smi parsing failed — {exc}")
            return None

        if not gpus:
            return None

        return _build_rocm_result(gpus, rocm_version, "rocm-smi", diagnostics)

    def _detect_via_rocminfo(self, diagnostics: list[str]) -> GPUDetectionResult | None:
        """Method 2: Parse rocminfo agent information."""
        gpus: list[GPUInfo] = []

        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                diagnostics.append(
                    f"rocm: rocminfo returned exit code {result.returncode}"
                )
                return None

            output = result.stdout

            # rocminfo lists agents — GPU agents have "Agent Type: GPU"
            # Parse agent blocks
            agents = output.split("*******")
            gpu_idx = 0
            for agent_block in agents:
                if "Agent Type:" not in agent_block:
                    continue

                # Check if this is a GPU agent
                agent_type_match = re.search(r"Agent Type:\s*(\w+)", agent_block)
                if not agent_type_match or agent_type_match.group(1).upper() != "GPU":
                    continue

                # Extract name
                name_match = re.search(r"Name:\s*(.+)", agent_block)
                gpu_name = (
                    name_match.group(1).strip() if name_match else f"AMD GPU {gpu_idx}"
                )

                # Extract gfx version for architecture info
                gfx_match = re.search(r"(gfx\w+)", agent_block)
                gfx_version = gfx_match.group(1) if gfx_match else None

                # Extract pool size for VRAM estimation
                # Look for "Pool" sections with "Size" — GPU pool is typically VRAM
                vram_gb = 0.0
                pool_matches = re.finditer(
                    r"Pool\s*\d*.*?Size:\s*(\d+)\s*\(.*?\)\s*KB",
                    agent_block,
                    re.DOTALL,
                )
                for pool_match in pool_matches:
                    size_kb = int(pool_match.group(1))
                    size_gb = size_kb / (1024**2)
                    if size_gb > vram_gb:
                        vram_gb = size_gb

                # Fallback: look for any memory size line
                if vram_gb == 0.0:
                    mem_matches = re.findall(
                        r"Size:\s*(\d+)\s*\(.*?\)\s*KB", agent_block
                    )
                    for mem_str in mem_matches:
                        size_gb = int(mem_str) / (1024**2)
                        if size_gb > vram_gb:
                            vram_gb = size_gb

                memory = GPUMemory(total_gb=round(vram_gb, 2))
                speed_coeff = _lookup_amd_speed(gpu_name)

                capabilities: dict[str, object] = {}
                if gfx_version:
                    capabilities["gfx_version"] = gfx_version

                gpus.append(
                    GPUInfo(
                        index=gpu_idx,
                        name=gpu_name,
                        memory=memory,
                        speed_coefficient=speed_coeff,
                        capabilities=capabilities,
                    )
                )
                gpu_idx += 1

            if gpus:
                diagnostics.append(f"rocm: detected {len(gpus)} GPU(s) via rocminfo")

        except FileNotFoundError:
            diagnostics.append("rocm: rocminfo not found")
            return None
        except subprocess.TimeoutExpired:
            diagnostics.append("rocm: rocminfo timed out")
            return None
        except Exception as exc:
            diagnostics.append(f"rocm: rocminfo parsing failed — {exc}")
            return None

        if not gpus:
            return None

        return _build_rocm_result(gpus, None, "rocminfo", diagnostics)

    def _detect_via_lspci(self, diagnostics: list[str]) -> GPUDetectionResult | None:
        """Method 3: Parse lspci output for AMD GPU PCI device IDs."""
        gpus: list[GPUInfo] = []

        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                diagnostics.append(
                    f"rocm: lspci returned exit code {result.returncode}"
                )
                return None

            # Match VGA/3D controllers with AMD vendor (1002)
            # Pattern: [1002:XXXX] where XXXX is device ID
            gpu_idx = 0
            for line in result.stdout.splitlines():
                if "1002:" not in line:
                    continue
                if "VGA" not in line and "3D" not in line and "Display" not in line:
                    continue

                # Extract device ID
                dev_match = re.search(r"\[1002:([0-9a-fA-F]{4})\]", line)
                if not dev_match:
                    continue

                device_id = dev_match.group(1).lower()
                if device_id in _AMD_PCI_DEVICES:
                    gpu_name, estimated_vram = _AMD_PCI_DEVICES[device_id]
                else:
                    # Unknown AMD GPU — extract name from lspci description
                    name_match = re.search(r"(?:VGA|3D|Display).*?:\s*(.+?)\s*\[", line)
                    gpu_name = (
                        name_match.group(1).strip()
                        if name_match
                        else f"AMD GPU [{device_id}]"
                    )
                    estimated_vram = 0.0

                memory = GPUMemory(total_gb=estimated_vram)
                speed_coeff = _lookup_amd_speed(gpu_name)

                gpus.append(
                    GPUInfo(
                        index=gpu_idx,
                        name=gpu_name,
                        memory=memory,
                        speed_coefficient=speed_coeff,
                        capabilities={"pci_device_id": device_id},
                    )
                )
                gpu_idx += 1

            if gpus:
                diagnostics.append(f"rocm: detected {len(gpus)} GPU(s) via lspci")

        except FileNotFoundError:
            diagnostics.append("rocm: lspci not found")
            return None
        except subprocess.TimeoutExpired:
            diagnostics.append("rocm: lspci timed out")
            return None
        except Exception as exc:
            diagnostics.append(f"rocm: lspci parsing failed — {exc}")
            return None

        if not gpus:
            return None

        return _build_rocm_result(gpus, None, "lspci", diagnostics)

    def _detect_via_sysfs(self, diagnostics: list[str]) -> GPUDetectionResult | None:
        """Method 4: Read /sys/class/drm/card* for AMD vendor 0x1002."""
        gpus: list[GPUInfo] = []
        gpu_idx = 0

        try:
            for card_dir in sorted(_list_drm_cards()):
                device_dir = os.path.join(card_dir, "device")
                vendor_path = os.path.join(device_dir, "vendor")

                if not os.path.isfile(vendor_path):
                    continue

                with open(vendor_path) as f:
                    vendor = f.read().strip()
                if vendor != "0x1002":
                    continue

                # Read device ID
                device_id_path = os.path.join(device_dir, "device")
                device_id = ""
                if os.path.isfile(device_id_path):
                    with open(device_id_path) as f:
                        # Format: 0xXXXX
                        raw = f.read().strip().lower()
                        device_id = raw.replace("0x", "")

                # Try to read VRAM info
                vram_gb = 0.0
                vram_path = os.path.join(device_dir, "mem_info_vram_total")
                if os.path.isfile(vram_path):
                    with open(vram_path) as f:
                        vram_bytes_str = f.read().strip()
                    try:
                        vram_gb = int(vram_bytes_str) / (1024**3)
                    except ValueError:
                        pass

                # Look up device name
                if device_id in _AMD_PCI_DEVICES:
                    gpu_name, fallback_vram = _AMD_PCI_DEVICES[device_id]
                    if vram_gb == 0.0:
                        vram_gb = fallback_vram
                else:
                    gpu_name = (
                        f"AMD GPU [{device_id}]" if device_id else f"AMD GPU {gpu_idx}"
                    )

                memory = GPUMemory(total_gb=round(vram_gb, 2))
                speed_coeff = _lookup_amd_speed(gpu_name)

                gpus.append(
                    GPUInfo(
                        index=gpu_idx,
                        name=gpu_name,
                        memory=memory,
                        speed_coefficient=speed_coeff,
                        capabilities={"pci_device_id": device_id, "sysfs": True},
                    )
                )
                gpu_idx += 1

            if gpus:
                diagnostics.append(f"rocm: detected {len(gpus)} GPU(s) via sysfs")

        except Exception as exc:
            diagnostics.append(f"rocm: sysfs detection failed — {exc}")
            return None

        if not gpus:
            return None

        return _build_rocm_result(gpus, None, "sysfs", diagnostics)


def _list_drm_cards() -> list[str]:
    """List /sys/class/drm/card* directories (only cardN, not cardN-*)."""
    drm_base = "/sys/class/drm"
    if not os.path.isdir(drm_base):
        return []
    entries = []
    for entry in os.listdir(drm_base):
        # Match "card0", "card1", etc. but not "card0-HDMI-A-1"
        if re.fullmatch(r"card\d+", entry):
            entries.append(os.path.join(drm_base, entry))
    return entries


def _lookup_amd_speed(gpu_name: str) -> int:
    """Look up speed coefficient for an AMD GPU name."""
    upper = gpu_name.upper()
    for key in sorted(_AMD_SPEED_COEFFICIENTS.keys(), key=len, reverse=True):
        if key.upper() in upper:
            return _AMD_SPEED_COEFFICIENTS[key]
    return 0


def _build_rocm_result(
    gpus: list[GPUInfo],
    rocm_version: str | None,
    method: str,
    diagnostics: list[str],
) -> GPUDetectionResult:
    """Build a GPUDetectionResult from detected AMD GPUs."""
    total_vram = sum(g.memory.total_gb for g in gpus)
    best_speed = max(g.speed_coefficient for g in gpus) if gpus else 0

    return GPUDetectionResult(
        gpus=gpus,
        backend="rocm",
        total_vram_gb=round(total_vram, 2),
        is_multi_gpu=len(gpus) > 1,
        speed_coefficient=best_speed,
        rocm_version=rocm_version,
        detection_method=method,
        diagnostics=diagnostics,
    )
