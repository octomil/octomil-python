"""Baseline comparison logic for benchmark regression gate.

Loads a JSON baseline file, compares current TTFT p95 against it, and
reports pass/fail based on a configurable regression threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Default regression threshold: 15% slower than baseline = fail
DEFAULT_REGRESSION_THRESHOLD = 0.15

# Minimum TTFT p95 in ms below which regressions are ignored.
# Prevents noise from causing false failures when absolute values are tiny.
MIN_TTFT_THRESHOLD_MS = 1.0


@dataclass
class BaselineEntry:
    """A single metric entry from the baseline file."""

    metric: str
    value: float
    unit: str


@dataclass
class Baseline:
    """Parsed baseline containing TTFT p95 and optional extra metrics."""

    ttft_p95_ms: float
    ttft_p50_ms: float = 0.0
    ttft_mean_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    model_name: str = ""
    iterations: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class RegressionResult:
    """Result of a regression check."""

    passed: bool
    baseline_ttft_p95_ms: float
    current_ttft_p95_ms: float
    delta_pct: float  # positive = regression (slower), negative = improvement
    threshold_pct: float
    message: str


def load_baseline(path: Path) -> Baseline:
    """Load baseline from a JSON file.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Baseline file must contain a JSON object, got {type(data).__name__}")

    ttft_p95 = data.get("ttft_p95_ms")
    if ttft_p95 is None:
        raise ValueError("Baseline file missing required key: ttft_p95_ms")

    try:
        ttft_p95_val = float(ttft_p95)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ttft_p95_ms must be a number, got {ttft_p95!r}") from exc

    return Baseline(
        ttft_p95_ms=ttft_p95_val,
        ttft_p50_ms=float(data.get("ttft_p50_ms", 0.0)),
        ttft_mean_ms=float(data.get("ttft_mean_ms", 0.0)),
        e2e_p95_ms=float(data.get("e2e_p95_ms", 0.0)),
        model_name=data.get("model_name", ""),
        iterations=int(data.get("iterations", 0)),
        metadata=data.get("metadata"),
    )


def save_baseline(
    path: Path,
    *,
    ttft_p95_ms: float,
    ttft_p50_ms: float = 0.0,
    ttft_mean_ms: float = 0.0,
    e2e_p95_ms: float = 0.0,
    model_name: str = "",
    iterations: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Write a baseline JSON file."""
    data: dict[str, Any] = {
        "ttft_p95_ms": round(ttft_p95_ms, 3),
        "ttft_p50_ms": round(ttft_p50_ms, 3),
        "ttft_mean_ms": round(ttft_mean_ms, 3),
        "e2e_p95_ms": round(e2e_p95_ms, 3),
        "model_name": model_name,
        "iterations": iterations,
    }
    if metadata:
        data["metadata"] = metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def check_regression(
    baseline: Baseline,
    current_ttft_p95_ms: float,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> RegressionResult:
    """Compare current TTFT p95 against baseline.

    Returns RegressionResult with pass/fail status.

    A regression is detected when ``current_ttft_p95_ms`` exceeds the baseline
    by more than ``threshold`` (as a fraction, e.g. 0.15 = 15%).

    If both values are below MIN_TTFT_THRESHOLD_MS, the check passes regardless,
    since sub-millisecond noise is not meaningful.
    """
    base_val = baseline.ttft_p95_ms
    curr_val = current_ttft_p95_ms

    # Both below noise floor — always pass
    if base_val < MIN_TTFT_THRESHOLD_MS and curr_val < MIN_TTFT_THRESHOLD_MS:
        return RegressionResult(
            passed=True,
            baseline_ttft_p95_ms=base_val,
            current_ttft_p95_ms=curr_val,
            delta_pct=0.0,
            threshold_pct=threshold * 100,
            message=f"Both values below noise floor ({MIN_TTFT_THRESHOLD_MS}ms). PASS.",
        )

    # Avoid division by zero
    if base_val == 0:
        delta_pct = 0.0 if curr_val == 0 else 100.0
    else:
        delta_pct = ((curr_val - base_val) / base_val) * 100.0

    passed = delta_pct <= threshold * 100

    if passed:
        if delta_pct <= 0:
            message = f"TTFT p95: {curr_val:.1f}ms vs baseline {base_val:.1f}ms ({delta_pct:+.1f}% — improved). PASS."
        else:
            message = (
                f"TTFT p95: {curr_val:.1f}ms vs baseline {base_val:.1f}ms "
                f"({delta_pct:+.1f}% — within {threshold * 100:.0f}% threshold). PASS."
            )
    else:
        message = (
            f"TTFT p95: {curr_val:.1f}ms vs baseline {base_val:.1f}ms "
            f"({delta_pct:+.1f}% — exceeds {threshold * 100:.0f}% threshold). FAIL."
        )

    return RegressionResult(
        passed=passed,
        baseline_ttft_p95_ms=base_val,
        current_ttft_p95_ms=curr_val,
        delta_pct=round(delta_pct, 2),
        threshold_pct=threshold * 100,
        message=message,
    )
