"""Benchmark regression gate for release CI.

Measures TTFT p95 for standard inference requests and compares against
a stored baseline to block releases that regress beyond a threshold.
"""

from octomil.benchmark.compare import check_regression, load_baseline, save_baseline
from octomil.benchmark.runner import BenchmarkConfig, BenchmarkSample, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSample",
    "check_regression",
    "load_baseline",
    "run_benchmark",
    "save_baseline",
]
