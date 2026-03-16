"""Benchmark runner — measures TTFT p95 using the local echo engine.

Designed to run in CI without real inference hardware. The echo engine
introduces a deterministic ~20ms delay per token, so TTFT measurements
reflect SDK overhead (request parsing, engine dispatch, streaming setup)
rather than model inference speed.

For real hardware benchmarks, use octomil-server-bench.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    iterations: int = 30
    warmup_iterations: int = 3
    model_name: str = "echo-bench"
    max_tokens: int = 16


@dataclass
class BenchmarkSample:
    """Single benchmark measurement."""

    ttft_ms: float
    e2e_ms: float
    token_count: int


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result."""

    samples: list[BenchmarkSample] = field(default_factory=list)
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    ttft_mean_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    model_name: str = ""
    iterations: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.samples) > 0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation on sorted values."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    n = len(sorted_values)
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _compute_stats(result: BenchmarkResult) -> None:
    """Compute percentile stats from collected samples."""
    if not result.samples:
        return
    ttft_values = sorted(s.ttft_ms for s in result.samples)
    e2e_values = sorted(s.e2e_ms for s in result.samples)
    result.ttft_mean_ms = round(sum(ttft_values) / len(ttft_values), 3)
    result.ttft_p50_ms = round(_percentile(ttft_values, 50), 3)
    result.ttft_p95_ms = round(_percentile(ttft_values, 95), 3)
    result.ttft_p99_ms = round(_percentile(ttft_values, 99), 3)
    result.e2e_p50_ms = round(_percentile(e2e_values, 50), 3)
    result.e2e_p95_ms = round(_percentile(e2e_values, 95), 3)


async def _run_single_iteration(config: BenchmarkConfig) -> BenchmarkSample:
    """Run a single benchmark iteration using the echo backend."""
    from octomil.serve import EchoBackend, GenerationRequest

    backend = EchoBackend()
    backend.load_model(config.model_name)

    request = GenerationRequest(
        model=config.model_name,
        messages=[{"role": "user", "content": "Benchmark test prompt for TTFT measurement"}],
        max_tokens=config.max_tokens,
        temperature=0.0,
    )

    start_ns = time.monotonic_ns()
    first_token_ns: Optional[int] = None
    token_count = 0

    async for _chunk in backend.generate_stream(request):
        if first_token_ns is None:
            first_token_ns = time.monotonic_ns()
        token_count += 1

    end_ns = time.monotonic_ns()

    ttft_ms = (first_token_ns - start_ns) / 1e6 if first_token_ns is not None else 0.0
    e2e_ms = (end_ns - start_ns) / 1e6

    return BenchmarkSample(ttft_ms=round(ttft_ms, 3), e2e_ms=round(e2e_ms, 3), token_count=token_count)


async def _run_async(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the full benchmark suite asynchronously."""
    result = BenchmarkResult(model_name=config.model_name, iterations=config.iterations)

    try:
        # Warmup
        for _ in range(config.warmup_iterations):
            await _run_single_iteration(config)

        # Measure
        for _ in range(config.iterations):
            sample = await _run_single_iteration(config)
            result.samples.append(sample)

        _compute_stats(result)
    except Exception as exc:
        result.error = str(exc)

    return result


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
    """Run the benchmark suite synchronously. Entry point for CLI and CI.

    Returns a BenchmarkResult with TTFT p95 and other percentile stats.
    """
    if config is None:
        config = BenchmarkConfig()
    return asyncio.run(_run_async(config))
