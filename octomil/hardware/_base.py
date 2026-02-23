"""Abstract GPU backend base class and registry singleton."""

from __future__ import annotations

import abc
import logging

from ._types import GPUDetectionResult

logger = logging.getLogger(__name__)


class GPUBackend(abc.ABC):
    """Base class for GPU detection backends."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def check_availability(self) -> bool: ...

    @abc.abstractmethod
    def detect(self) -> GPUDetectionResult | None: ...

    def get_fingerprint(self) -> str | None:
        """Optional hardware fingerprint for caching."""
        return None


class GPUBackendRegistry:
    """Singleton registry for GPU detection backends."""

    def __init__(self) -> None:
        self._backends: list[GPUBackend] = []

    def register(self, backend: GPUBackend) -> None:
        for existing in self._backends:
            if existing.name == backend.name:
                return
        self._backends.append(backend)

    @property
    def backends(self) -> list[GPUBackend]:
        return list(self._backends)

    def detect_best(self) -> tuple[GPUDetectionResult | None, list[str]]:
        """Try all backends, return best result + aggregated diagnostics."""
        diagnostics: list[str] = []
        for backend in self._backends:
            try:
                if not backend.check_availability():
                    diagnostics.append(f"{backend.name}: not available")
                    continue
                result = backend.detect()
                if result and result.gpus:
                    return result, diagnostics
                diagnostics.append(f"{backend.name}: available but no GPUs detected")
            except Exception as exc:
                diagnostics.append(f"{backend.name}: detection failed — {exc}")
        return None, diagnostics


_registry: GPUBackendRegistry | None = None


def get_gpu_registry() -> GPUBackendRegistry:
    global _registry
    if _registry is None:
        _registry = GPUBackendRegistry()
        _auto_register(_registry)
    return _registry


def reset_gpu_registry() -> None:
    global _registry
    _registry = None


def _auto_register(registry: GPUBackendRegistry) -> None:
    from ._cuda import CUDABackend
    from ._metal import MetalBackend
    from ._rocm import ROCmBackend

    registry.register(CUDABackend())
    registry.register(ROCmBackend())
    registry.register(MetalBackend())
