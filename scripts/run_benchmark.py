#!/usr/bin/env python3
"""Run benchmark regression gate locally or in CI.

Usage:
    # Run benchmark and compare against baseline:
    python scripts/run_benchmark.py

    # Run benchmark and update baseline on success:
    python scripts/run_benchmark.py --update-baseline

    # Custom threshold (default: 15%):
    python scripts/run_benchmark.py --threshold 0.20

    # Custom iteration count:
    python scripts/run_benchmark.py --iterations 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from octomil.benchmark.compare import check_regression, load_baseline, save_baseline  # noqa: E402
from octomil.benchmark.runner import BenchmarkConfig, run_benchmark  # noqa: E402

BASELINE_PATH = _repo_root / "benchmarks" / "baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark regression gate")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help=f"Path to baseline JSON file (default: {BASELINE_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Regression threshold as a fraction (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline file with current results on success",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI parsing)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    print(f"Running benchmark ({config.iterations} iterations, {config.warmup_iterations} warmup)...")
    result = run_benchmark(config)

    if not result.ok:
        print(f"Benchmark failed: {result.error}", file=sys.stderr)
        return 1

    print(f"  TTFT p50:  {result.ttft_p50_ms:.3f}ms")
    print(f"  TTFT p95:  {result.ttft_p95_ms:.3f}ms")
    print(f"  TTFT p99:  {result.ttft_p99_ms:.3f}ms")
    print(f"  TTFT mean: {result.ttft_mean_ms:.3f}ms")
    print(f"  E2E p95:   {result.e2e_p95_ms:.3f}ms")
    print(f"  Samples:   {len(result.samples)}")

    # Load baseline and compare
    try:
        baseline = load_baseline(args.baseline)
    except FileNotFoundError:
        print(f"\nNo baseline file found at {args.baseline}.")
        if args.update_baseline:
            print("Creating initial baseline...")
            save_baseline(
                args.baseline,
                ttft_p95_ms=result.ttft_p95_ms,
                ttft_p50_ms=result.ttft_p50_ms,
                ttft_mean_ms=result.ttft_mean_ms,
                e2e_p95_ms=result.e2e_p95_ms,
                model_name=result.model_name,
                iterations=result.iterations,
            )
            print(f"Baseline saved to {args.baseline}")
            return 0
        print("Run with --update-baseline to create one, or add a baseline.json file.")
        return 1

    regression = check_regression(baseline, result.ttft_p95_ms, args.threshold)
    print(f"\n{regression.message}")

    if args.json:
        import json

        output = {
            "passed": regression.passed,
            "baseline_ttft_p95_ms": regression.baseline_ttft_p95_ms,
            "current_ttft_p95_ms": regression.current_ttft_p95_ms,
            "delta_pct": regression.delta_pct,
            "threshold_pct": regression.threshold_pct,
        }
        print(json.dumps(output, indent=2))

    if regression.passed and args.update_baseline:
        save_baseline(
            args.baseline,
            ttft_p95_ms=result.ttft_p95_ms,
            ttft_p50_ms=result.ttft_p50_ms,
            ttft_mean_ms=result.ttft_mean_ms,
            e2e_p95_ms=result.e2e_p95_ms,
            model_name=result.model_name,
            iterations=result.iterations,
        )
        print(f"Baseline updated at {args.baseline}")

    return 0 if regression.passed else 1


if __name__ == "__main__":
    sys.exit(main())
