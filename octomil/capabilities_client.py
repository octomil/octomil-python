"""Capabilities namespace -- ``client.capabilities.current()``.

**Tier: Core Contract (MUST)**

Surfaces device hardware information that ``DeviceInfo`` collects
internally, along with available engine runtimes, through a clean
``client.capabilities`` sub-API::

    profile = client.capabilities.current()
    print(profile.device_class, profile.memory_mb, profile.accelerators)
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client import OctomilClient

logger = logging.getLogger(__name__)


@dataclass
class CapabilityProfile:
    """Snapshot of the current device's capabilities."""

    device_class: str
    """Classification: ``flagship``, ``high``, ``mid``, or ``low``."""

    available_runtimes: list[str]
    """Engine names detected on this device (e.g. ``["mlx-lm", "llama.cpp"]``)."""

    memory_mb: int | None
    """Total system RAM in megabytes, or ``None`` if unavailable."""

    storage_mb: int | None
    """Free disk storage in megabytes, or ``None`` if unavailable."""

    platform: str
    """OS platform string (e.g. ``"darwin"``, ``"linux"``, ``"win32"``)."""

    accelerators: list[str] = field(default_factory=list)
    """Detected hardware accelerators (e.g. ``["metal"]``, ``["cuda"]``)."""


def _classify_device(memory_mb: int | None, has_gpu: bool) -> str:
    """Heuristic device classification matching the contract's DeviceClass enum."""
    if memory_mb is None:
        return "mid"
    if memory_mb >= 16_000 and has_gpu:
        return "flagship"
    if memory_mb >= 8_000 and has_gpu:
        return "high"
    if memory_mb >= 4_000:
        return "mid"
    return "low"


def _detect_accelerators() -> list[str]:
    """Detect available hardware accelerators."""
    accelerators: list[str] = []

    # Metal (macOS Apple Silicon)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        accelerators.append("metal")

    # CUDA
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"],  # noqa: S603, S607
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            accelerators.append("cuda")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # ROCm
    try:
        import subprocess

        result = subprocess.run(
            ["rocm-smi"],  # noqa: S603, S607
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            accelerators.append("rocm")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return accelerators


class CapabilitiesClient:
    """Device capabilities namespace.

    Provides ``current()`` to snapshot the device's hardware profile
    including available runtimes, memory, and accelerators.
    """

    def __init__(self, client: Optional[OctomilClient] = None) -> None:
        self._client = client

    def current(self) -> CapabilityProfile:
        """Return the current device capability profile.

        Collects information from ``DeviceInfo``, the engine registry,
        and hardware detection.  Results are not cached — each call
        re-probes the system.
        """
        from .device_info import get_memory_info, get_storage_info

        memory_mb = get_memory_info()
        storage_mb = get_storage_info()

        # Detect available engines
        available_runtimes: list[str] = []
        try:
            from .engines import get_registry

            registry = get_registry()
            for det in registry.detect_all():
                if det.available:
                    available_runtimes.append(det.engine.name)
        except Exception:
            logger.debug("Failed to detect engine runtimes", exc_info=True)

        accelerators = _detect_accelerators()

        has_gpu = len(accelerators) > 0
        device_class = _classify_device(memory_mb, has_gpu)

        return CapabilityProfile(
            device_class=device_class,
            available_runtimes=available_runtimes,
            memory_mb=memory_mb,
            storage_mb=storage_mb,
            platform=platform.system().lower(),
            accelerators=accelerators,
        )
