"""Collect device runtime profile for planner requests."""

from __future__ import annotations

import logging
import os
import platform
import subprocess

from .schemas import DeviceRuntimeProfile, InstalledRuntime

logger = logging.getLogger(__name__)


def _get_sdk_version() -> str:
    """Return the octomil SDK version string."""
    try:
        from octomil import __version__

        return __version__
    except Exception:
        return "unknown"


def _get_chip() -> str | None:
    """Try to detect the CPU chip name (macOS-only via sysctl)."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _get_ram_total_bytes() -> int | None:
    """Try to detect total system RAM in bytes."""
    # POSIX approach
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and page_count > 0:
            return page_size * page_count
    except (ValueError, OSError, AttributeError):
        pass

    # Fallback to psutil if available
    try:
        import psutil

        return psutil.virtual_memory().total
    except (ImportError, Exception):
        pass

    return None


def _detect_installed_runtimes(*, exclude_echo: bool = True) -> list[InstalledRuntime]:
    """Detect locally-installed inference engines via the engine registry.

    Uses the same detection mechanism as engine_bridge.py but returns
    lightweight InstalledRuntime descriptors instead of full EnginePlugin
    instances.
    """
    runtimes: list[InstalledRuntime] = []
    try:
        from octomil.runtime.engines import get_registry

        registry = get_registry()
        detections = registry.detect_all()
        for det in detections:
            if exclude_echo and det.engine.name in {"echo", "ollama"}:
                continue
            if not det.available:
                continue
            runtimes.append(
                InstalledRuntime(
                    engine=det.engine.name,
                    version=None,  # version info not available from detect()
                    available=True,
                    accelerator=None,
                    metadata={"info": det.info} if det.info else {},
                )
            )
    except Exception:
        logger.debug("Failed to detect installed runtimes", exc_info=True)
    return runtimes


def _detect_accelerators() -> list[str]:
    """Detect available hardware accelerators."""
    accels: list[str] = []

    # Apple Silicon / Metal
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        accels.append("metal")

    # CUDA
    try:
        import torch

        if torch.cuda.is_available():
            accels.append("cuda")
    except ImportError:
        pass

    return accels


def collect_device_runtime_profile(
    *,
    model_id: str | None = None,
    exclude_echo: bool = True,
) -> DeviceRuntimeProfile:
    """Collect a full device runtime profile for planner requests.

    Parameters
    ----------
    model_id:
        Optional model to pass to engine detection (for model-specific
        availability checks).
    exclude_echo:
        Whether to exclude the echo engine from installed_runtimes.
    """
    return DeviceRuntimeProfile(
        sdk="python",
        sdk_version=_get_sdk_version(),
        platform=platform.system(),
        arch=platform.machine(),
        os_version=platform.release(),
        chip=_get_chip(),
        ram_total_bytes=_get_ram_total_bytes(),
        gpu_core_count=None,
        accelerators=_detect_accelerators(),
        installed_runtimes=_detect_installed_runtimes(exclude_echo=exclude_echo),
    )
