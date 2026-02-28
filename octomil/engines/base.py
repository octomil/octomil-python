"""Abstract engine plugin interface for octomil serve auto-benchmark system.

Each inference engine (mlx-lm, llama.cpp, MNN, ONNX Runtime, etc.) implements
this interface so the registry can detect, benchmark, and select the fastest.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkResult:
    """Result from a quick engine benchmark."""

    engine_name: str
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    memory_mb: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.tokens_per_second > 0


@dataclass
class ProfileResult:
    """Result from hardware utilization profiling."""

    engine_name: str
    accelerator_used: str  # "metal", "cuda", "cpu", "ane", "nnapi"
    utilization_pct: float  # 0-100
    memory_peak_mb: float
    ops_on_accelerator: int
    ops_total: int
    metadata: dict[str, Any] = field(default_factory=dict)


class EnginePlugin(abc.ABC):
    """Base class for inference engine plugins.

    Each engine plugin provides three methods:
    - detect(): is this engine available on the current system?
    - benchmark(): run a quick inference test and return tok/s
    - create_backend(): create the InferenceBackend for actual serving
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short engine identifier (e.g. 'mlx-lm', 'llama.cpp', 'mnn')."""

    @property
    def display_name(self) -> str:
        """Human-readable engine name for terminal output."""
        return self.name

    @property
    def priority(self) -> int:
        """Default priority when benchmark results are tied. Lower = higher priority."""
        return 100

    @property
    def manages_own_download(self) -> bool:
        """Whether create_backend() handles model downloading internally.

        Engines that load directly from HuggingFace (mlx-lm) or manage their
        own model cache (ollama) should return True so that OctomilClient.load_model()
        skips the redundant registry pull step.
        """
        return False

    @abc.abstractmethod
    def detect(self) -> bool:
        """Check if this engine is available on the current system.

        Should check for required libraries, hardware capabilities, etc.
        Must not raise exceptions — return False if unavailable.
        """

    def detect_info(self) -> str:
        """Short description of detected hardware/config (e.g. 'Apple Silicon M2 Pro')."""
        return ""

    @abc.abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Check if this engine can serve the given model."""

    @abc.abstractmethod
    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Run a quick inference benchmark.

        Loads the model, generates n_tokens, and measures throughput.
        Returns BenchmarkResult with tokens_per_second (or error).
        """

    @abc.abstractmethod
    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        """Create an InferenceBackend instance for actual serving.

        The returned object must implement the InferenceBackend interface
        from octomil.serve (load_model, generate, generate_stream, list_models).
        """

    def profile(self, model_name: str, n_tokens: int = 8) -> ProfileResult | None:
        """Profile hardware utilization during inference.

        Returns accelerator usage stats, or None if profiling is not
        supported by this engine.
        """
        return None

    def estimate_memory_mb(
        self, model_size_b: float, quantization: str, context_length: int = 4096
    ) -> float:
        """Estimate memory usage in MB for a model configuration.

        Default implementation delegates to model_optimizer.estimate_memory_mb.
        Engines can override for more precise, engine-specific estimates.
        """
        from octomil.model_optimizer import estimate_memory_mb

        return estimate_memory_mb(model_size_b, quantization, context_length)
