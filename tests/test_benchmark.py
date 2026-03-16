"""Tests for the benchmark regression gate (GAP-01).

Covers:
- Baseline loading/saving (round-trip, missing file, malformed)
- Regression comparison logic (pass, fail, edge cases)
- Percentile computation
- Benchmark runner (mocked echo backend)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from octomil.benchmark.compare import (
    DEFAULT_REGRESSION_THRESHOLD,
    Baseline,
    check_regression,
    load_baseline,
    save_baseline,
)
from octomil.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSample,
    _compute_stats,
    _percentile,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0

    def test_two_values_p50(self):
        result = _percentile([10.0, 20.0], 50)
        assert result == 15.0

    def test_known_p95(self):
        values = sorted(list(range(1, 101)))
        vals_float = [float(v) for v in values]
        p95 = _percentile(vals_float, 95)
        assert 94.0 <= p95 <= 96.0

    def test_p0_returns_first(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_returns_last(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


# ---------------------------------------------------------------------------
# Compute stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_empty_samples_no_crash(self):
        result = BenchmarkResult()
        _compute_stats(result)
        assert result.ttft_p95_ms == 0.0

    def test_single_sample(self):
        result = BenchmarkResult(samples=[BenchmarkSample(ttft_ms=5.0, e2e_ms=100.0, token_count=10)])
        _compute_stats(result)
        assert result.ttft_p50_ms == 5.0
        assert result.ttft_p95_ms == 5.0
        assert result.ttft_mean_ms == 5.0
        assert result.e2e_p50_ms == 100.0

    def test_multiple_samples(self):
        samples = [BenchmarkSample(ttft_ms=float(i), e2e_ms=float(i * 10), token_count=10) for i in range(1, 21)]
        result = BenchmarkResult(samples=samples)
        _compute_stats(result)
        assert result.ttft_mean_ms == 10.5
        assert result.ttft_p50_ms > 0
        assert result.ttft_p95_ms > result.ttft_p50_ms


# ---------------------------------------------------------------------------
# BenchmarkResult properties
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_ok_with_samples(self):
        result = BenchmarkResult(samples=[BenchmarkSample(ttft_ms=1.0, e2e_ms=10.0, token_count=5)])
        assert result.ok is True

    def test_not_ok_with_error(self):
        result = BenchmarkResult(error="something broke")
        assert result.ok is False

    def test_not_ok_with_no_samples(self):
        result = BenchmarkResult()
        assert result.ok is False


# ---------------------------------------------------------------------------
# Baseline loading / saving
# ---------------------------------------------------------------------------


class TestLoadBaseline:
    def test_load_valid_baseline(self, tmp_path: Path):
        data = {"ttft_p95_ms": 50.0, "ttft_p50_ms": 25.0, "model_name": "echo-bench", "iterations": 30}
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(data))

        baseline = load_baseline(path)
        assert baseline.ttft_p95_ms == 50.0
        assert baseline.ttft_p50_ms == 25.0
        assert baseline.model_name == "echo-bench"
        assert baseline.iterations == 30

    def test_load_minimal_baseline(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"ttft_p95_ms": 42.0}))

        baseline = load_baseline(path)
        assert baseline.ttft_p95_ms == 42.0
        assert baseline.ttft_p50_ms == 0.0

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")

    def test_load_missing_key_raises(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"model_name": "test"}))

        with pytest.raises(ValueError, match="missing required key"):
            load_baseline(path)

    def test_load_invalid_value_raises(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"ttft_p95_ms": "not-a-number"}))

        with pytest.raises(ValueError, match="must be a number"):
            load_baseline(path)

    def test_load_non_object_raises(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps([1, 2, 3]))

        with pytest.raises(ValueError, match="JSON object"):
            load_baseline(path)


class TestSaveBaseline:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        save_baseline(
            path,
            ttft_p95_ms=50.123456,
            ttft_p50_ms=25.0,
            ttft_mean_ms=28.0,
            e2e_p95_ms=400.0,
            model_name="echo-bench",
            iterations=30,
        )

        baseline = load_baseline(path)
        assert baseline.ttft_p95_ms == 50.123
        assert baseline.model_name == "echo-bench"

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "baseline.json"
        save_baseline(path, ttft_p95_ms=10.0)
        assert path.exists()

    def test_saves_metadata(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        save_baseline(path, ttft_p95_ms=10.0, metadata={"runner": "ci"})

        with open(path) as f:
            data = json.load(f)
        assert data["metadata"] == {"runner": "ci"}


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------


class TestCheckRegression:
    def test_pass_when_improved(self):
        baseline = Baseline(ttft_p95_ms=50.0)
        result = check_regression(baseline, 40.0)
        assert result.passed is True
        assert result.delta_pct < 0
        assert "improved" in result.message.lower()

    def test_pass_within_threshold(self):
        baseline = Baseline(ttft_p95_ms=50.0)
        # 10% regression, within 15% threshold
        result = check_regression(baseline, 55.0)
        assert result.passed is True
        assert 0 < result.delta_pct <= 15.0
        assert "within" in result.message.lower()

    def test_pass_at_exact_threshold(self):
        baseline = Baseline(ttft_p95_ms=100.0)
        # Exactly 15% regression
        result = check_regression(baseline, 115.0)
        assert result.passed is True

    def test_fail_beyond_threshold(self):
        baseline = Baseline(ttft_p95_ms=50.0)
        # 20% regression, exceeds 15% threshold
        result = check_regression(baseline, 60.0)
        assert result.passed is False
        assert result.delta_pct > 15.0
        assert "exceeds" in result.message.lower()

    def test_custom_threshold(self):
        baseline = Baseline(ttft_p95_ms=100.0)
        # 25% regression, within 30% threshold
        result = check_regression(baseline, 125.0, threshold=0.30)
        assert result.passed is True
        assert result.threshold_pct == 30.0

    def test_both_below_noise_floor_passes(self):
        baseline = Baseline(ttft_p95_ms=0.5)
        result = check_regression(baseline, 0.9)
        assert result.passed is True
        assert "noise floor" in result.message.lower()

    def test_zero_baseline_current_nonzero(self):
        baseline = Baseline(ttft_p95_ms=0.0)
        result = check_regression(baseline, 50.0)
        assert result.passed is False
        assert result.delta_pct == 100.0

    def test_zero_baseline_zero_current(self):
        baseline = Baseline(ttft_p95_ms=0.0)
        result = check_regression(baseline, 0.0)
        # Both are 0 but 0 < MIN_TTFT_THRESHOLD_MS, so noise floor applies
        assert result.passed is True

    def test_result_fields_populated(self):
        baseline = Baseline(ttft_p95_ms=100.0)
        result = check_regression(baseline, 110.0)
        assert result.baseline_ttft_p95_ms == 100.0
        assert result.current_ttft_p95_ms == 110.0
        assert result.delta_pct == 10.0
        assert result.threshold_pct == DEFAULT_REGRESSION_THRESHOLD * 100

    def test_identical_values_pass(self):
        baseline = Baseline(ttft_p95_ms=50.0)
        result = check_regression(baseline, 50.0)
        assert result.passed is True
        assert result.delta_pct == 0.0

    def test_large_improvement(self):
        baseline = Baseline(ttft_p95_ms=100.0)
        result = check_regression(baseline, 10.0)
        assert result.passed is True
        assert result.delta_pct == -90.0


# ---------------------------------------------------------------------------
# Benchmark runner (integration, uses real echo backend)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_runs_successfully(self):
        config = BenchmarkConfig(iterations=5, warmup_iterations=1, max_tokens=4)
        result = run_benchmark(config)
        assert result.ok is True
        assert len(result.samples) == 5
        assert result.ttft_p95_ms > 0
        assert result.e2e_p95_ms > 0

    def test_all_samples_have_tokens(self):
        config = BenchmarkConfig(iterations=3, warmup_iterations=0, max_tokens=4)
        result = run_benchmark(config)
        for sample in result.samples:
            assert sample.token_count > 0
            assert sample.ttft_ms >= 0
            assert sample.e2e_ms > 0

    def test_ttft_less_than_e2e(self):
        config = BenchmarkConfig(iterations=5, warmup_iterations=1, max_tokens=8)
        result = run_benchmark(config)
        assert result.ttft_p95_ms <= result.e2e_p95_ms
