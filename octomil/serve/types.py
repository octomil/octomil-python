"""Core types for the serve package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..telemetry import TelemetryReporter


@dataclass
class GenerationRequest:
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    grammar: Optional[str] = None
    json_mode: bool = False


@dataclass
class GenerationChunk:
    text: str
    token_count: int = 0
    tokens_per_second: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""

    ttfc_ms: float = 0.0
    prompt_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    total_duration_ms: float = 0.0
    cache_hit: bool = False
    attention_backend: str = "standard"
    early_exit_tokens: int = 0
    avg_layers_used: float = 0.0


class InferenceBackend:
    """Base class for inference backends.

    Provides a shared ``ThreadPoolExecutor`` for sync->async bridging and
    a ``warmup()`` hook that subclasses can override to pre-compile GPU
    kernels and prime caches at model load time.
    """

    name: str = "base"
    attention_backend: str = "standard"

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self.name}-inference",
        )
        # Pre-spawn the executor thread so the first streaming request
        # doesn't pay OS thread creation cost (~50ms measured).
        self._executor.submit(lambda: None).result()

    def load_model(self, model_name: str) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        """Override to run a cheap prefill after model load.

        This compiles GPU shaders/kernels and primes caches so the first
        real request doesn't pay cold-start cost.  Called automatically
        at the end of ``load_model`` in subclasses that opt in.
        """

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        raise NotImplementedError

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
        yield  # pragma: no cover -- makes this an async generator

    def list_models(self) -> list[str]:
        raise NotImplementedError


@runtime_checkable
class StreamableState(Protocol):
    """Protocol satisfied by both ServerState and _MultiModelStateAdapter.

    Used as the type for the ``state`` parameter in streaming functions
    to break the circular dependency between streaming.py and multi_model.py.
    """

    backend: Optional[InferenceBackend]
    reporter: Optional["TelemetryReporter"]
