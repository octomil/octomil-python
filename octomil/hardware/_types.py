"""Shared dataclasses for hardware detection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPUMemory:
    total_gb: float
    free_gb: float = 0.0
    used_gb: float = 0.0


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory: GPUMemory
    speed_coefficient: int = 0  # tok/s per B params at Q4
    capabilities: dict[str, object] = field(default_factory=dict)
    compute_capability: str | None = None
    architecture: str | None = None


@dataclass(frozen=True)
class GPUDetectionResult:
    gpus: list[GPUInfo]
    backend: str  # "cuda", "rocm", "metal", "cpu"
    total_vram_gb: float = 0.0
    is_multi_gpu: bool = False
    speed_coefficient: int = 0
    driver_version: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    detection_method: str | None = None
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CPUInfo:
    brand: str
    cores: int
    threads: int
    base_speed_ghz: float
    architecture: str  # "x86_64", "arm64"
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False
    estimated_gflops: float = 0.0


@dataclass
class HardwareProfile:
    gpu: GPUDetectionResult | None
    cpu: CPUInfo
    total_ram_gb: float
    available_ram_gb: float
    platform: str  # "darwin", "linux", "win32"
    best_backend: str  # "cuda", "rocm", "metal", "cpu_avx512", etc.
    diagnostics: list[str] = field(default_factory=list)
    timestamp: float = 0.0
