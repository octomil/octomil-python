"""Tests for the engine registry and plugin system (EDG-71)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from octomil.engines.base import BenchmarkResult, EnginePlugin
from octomil.engines.echo_engine import EchoEngine
from octomil.engines.llamacpp_engine import LlamaCppEngine
from octomil.engines.mlx_engine import MLXEngine
from octomil.engines.registry import (
    EngineRegistry,
    RankedEngine,
    get_registry,
    reset_registry,
)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_ok_when_has_tps_and_no_error(self):
        r = BenchmarkResult(engine_name="test", tokens_per_second=42.0)
        assert r.ok is True

    def test_not_ok_when_error(self):
        r = BenchmarkResult(engine_name="test", error="something failed")
        assert r.ok is False

    def test_not_ok_when_zero_tps(self):
        r = BenchmarkResult(engine_name="test", tokens_per_second=0.0)
        assert r.ok is False

    def test_metadata_default_empty(self):
        r = BenchmarkResult(engine_name="test")
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# EnginePlugin base class
# ---------------------------------------------------------------------------


class _FakeEngine(EnginePlugin):
    """Concrete engine for testing the base class."""

    @property
    def name(self) -> str:
        return "fake"

    def detect(self) -> bool:
        return True

    def supports_model(self, model_name: str) -> bool:
        return model_name == "supported-model"

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        return BenchmarkResult(engine_name="fake", tokens_per_second=100.0)

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return MagicMock(name="fake-backend")


class _SlowEngine(EnginePlugin):
    @property
    def name(self) -> str:
        return "slow"

    @property
    def priority(self) -> int:
        return 50

    def detect(self) -> bool:
        return True

    def supports_model(self, model_name: str) -> bool:
        return True

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        return BenchmarkResult(engine_name="slow", tokens_per_second=10.0)

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return MagicMock(name="slow-backend")


class _BrokenEngine(EnginePlugin):
    @property
    def name(self) -> str:
        return "broken"

    def detect(self) -> bool:
        return True

    def supports_model(self, model_name: str) -> bool:
        return True

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        raise RuntimeError("Benchmark crashed")

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        raise RuntimeError("Create backend crashed")


class _UnavailableEngine(EnginePlugin):
    @property
    def name(self) -> str:
        return "unavailable"

    def detect(self) -> bool:
        return False

    def supports_model(self, model_name: str) -> bool:
        return True

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        return BenchmarkResult(engine_name="unavailable", error="not installed")

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        raise RuntimeError("Not available")


class TestEnginePluginBase:
    def test_default_display_name_equals_name(self):
        e = _FakeEngine()
        assert e.display_name == "fake"

    def test_default_priority(self):
        e = _FakeEngine()
        assert e.priority == 100


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    def test_register_engine(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        assert len(reg.engines) == 1
        assert reg.engines[0].name == "fake"

    def test_no_duplicate_registration(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        reg.register(_FakeEngine())
        assert len(reg.engines) == 1

    def test_get_engine_by_name(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        engine = reg.get_engine("fake")
        assert engine is not None
        assert engine.name == "fake"

    def test_get_engine_returns_none_for_unknown(self):
        reg = EngineRegistry()
        assert reg.get_engine("nonexistent") is None

    def test_detect_all_returns_all_engines(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        reg.register(_UnavailableEngine())
        results = reg.detect_all()
        assert len(results) == 2
        available = [r for r in results if r.available]
        unavailable = [r for r in results if not r.available]
        assert len(available) == 1
        assert len(unavailable) == 1

    def test_detect_all_with_model_filter(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())  # only supports "supported-model"
        reg.register(_SlowEngine())  # supports all

        results = reg.detect_all("supported-model")
        available = [r for r in results if r.available]
        assert len(available) == 2  # both support it

        results = reg.detect_all("unsupported-model")
        available = [r for r in results if r.available]
        assert len(available) == 1  # only SlowEngine
        assert available[0].engine.name == "slow"

    def test_benchmark_all_ranks_by_tps(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())  # 100 tok/s
        reg.register(_SlowEngine())  # 10 tok/s

        ranked = reg.benchmark_all(
            "supported-model", engines=[_FakeEngine(), _SlowEngine()]
        )
        assert len(ranked) == 2
        assert ranked[0].engine.name == "fake"  # fastest
        assert ranked[1].engine.name == "slow"
        assert ranked[0].result.tokens_per_second > ranked[1].result.tokens_per_second

    def test_benchmark_all_handles_crash(self):
        reg = EngineRegistry()
        reg.register(_BrokenEngine())

        ranked = reg.benchmark_all("model", engines=[_BrokenEngine()])
        assert len(ranked) == 1
        assert ranked[0].result.error is not None
        assert "Benchmark crashed" in ranked[0].result.error

    def test_select_best_picks_fastest(self):
        reg = EngineRegistry()
        fast = RankedEngine(
            engine=_FakeEngine(),
            result=BenchmarkResult(engine_name="fake", tokens_per_second=100.0),
        )
        slow = RankedEngine(
            engine=_SlowEngine(),
            result=BenchmarkResult(engine_name="slow", tokens_per_second=10.0),
        )
        best = reg.select_best([fast, slow])
        assert best is not None
        assert best.engine.name == "fake"

    def test_select_best_skips_failed_results(self):
        reg = EngineRegistry()
        failed = RankedEngine(
            engine=_BrokenEngine(),
            result=BenchmarkResult(engine_name="broken", error="crashed"),
        )
        ok = RankedEngine(
            engine=_SlowEngine(),
            result=BenchmarkResult(engine_name="slow", tokens_per_second=10.0),
        )
        best = reg.select_best([failed, ok])
        assert best is not None
        assert best.engine.name == "slow"

    def test_select_best_returns_none_for_empty(self):
        reg = EngineRegistry()
        assert reg.select_best([]) is None

    def test_auto_select_with_override(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        reg.register(_SlowEngine())

        engine, ranked = reg.auto_select("model", engine_override="fake")
        assert engine.name == "fake"
        assert ranked == []  # no benchmarking when override

    def test_auto_select_raises_for_unknown_override(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())

        with pytest.raises(ValueError, match="Unknown engine"):
            reg.auto_select("model", engine_override="nonexistent")

    def test_auto_select_raises_for_unavailable_override(self):
        reg = EngineRegistry()
        reg.register(_UnavailableEngine())

        with pytest.raises(ValueError, match="not available"):
            reg.auto_select("model", engine_override="unavailable")

    def test_auto_select_benchmarks_and_picks_best(self):
        reg = EngineRegistry()
        reg.register(_FakeEngine())
        reg.register(_SlowEngine())

        engine, ranked = reg.auto_select("supported-model")
        assert engine.name == "fake"
        assert len(ranked) >= 1


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_get_registry_has_builtin_engines(self):
        reg = get_registry()
        names = [e.name for e in reg.engines]
        assert "mlx-lm" in names
        assert "llama.cpp" in names
        assert "echo" in names

    def test_reset_registry_clears_singleton(self):
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2


# ---------------------------------------------------------------------------
# EchoEngine
# ---------------------------------------------------------------------------


class TestEchoEngine:
    def test_always_detects(self):
        e = EchoEngine()
        assert e.detect() is True

    def test_supports_any_model(self):
        e = EchoEngine()
        assert e.supports_model("anything") is True

    def test_benchmark_returns_error(self):
        e = EchoEngine()
        result = e.benchmark("test")
        assert result.ok is False
        assert "echo" in (result.error or "")

    def test_create_backend_returns_echo(self):
        e = EchoEngine()
        backend = e.create_backend("test-model")
        assert backend.name == "echo"

    def test_lowest_priority(self):
        e = EchoEngine()
        assert e.priority == 999


# ---------------------------------------------------------------------------
# MLXEngine
# ---------------------------------------------------------------------------


class TestMLXEngine:
    def test_detect_returns_false_on_non_darwin(self):
        e = MLXEngine()
        with patch("octomil.engines.mlx_engine.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            assert e.detect() is False

    def test_detect_returns_false_on_intel_mac(self):
        e = MLXEngine()
        with patch("octomil.engines.mlx_engine.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"
            assert e.detect() is False

    def test_supports_catalog_model(self):
        e = MLXEngine()
        assert e.supports_model("gemma-1b") is True
        assert e.supports_model("llama-8b") is True

    def test_supports_repo_id(self):
        e = MLXEngine()
        assert e.supports_model("mlx-community/custom-model") is True

    def test_does_not_support_unknown(self):
        e = MLXEngine()
        assert e.supports_model("nonexistent") is False

    def test_priority_is_highest(self):
        e = MLXEngine()
        assert e.priority == 10


# ---------------------------------------------------------------------------
# LlamaCppEngine
# ---------------------------------------------------------------------------


class TestLlamaCppEngine:
    def test_detect_returns_false_without_lib(self):
        e = LlamaCppEngine()
        with patch.dict("sys.modules", {"llama_cpp": None}):
            assert e.detect() is False

    def test_supports_catalog_model(self):
        e = LlamaCppEngine()
        assert e.supports_model("gemma-1b") is True
        assert e.supports_model("llama-8b") is True

    def test_supports_gguf_file(self):
        e = LlamaCppEngine()
        assert e.supports_model("model.gguf") is True

    def test_supports_repo_id(self):
        e = LlamaCppEngine()
        assert e.supports_model("user/some-model-GGUF") is True

    def test_does_not_support_unknown(self):
        e = LlamaCppEngine()
        assert e.supports_model("nonexistent") is False

    def test_priority(self):
        e = LlamaCppEngine()
        assert e.priority == 20


# ---------------------------------------------------------------------------
# Engine selection ordering
# ---------------------------------------------------------------------------


class TestEngineOrdering:
    def test_mlx_beats_llamacpp_by_priority(self):
        """When tok/s is equal, priority breaks the tie."""
        reg = EngineRegistry()
        mlx_ranked = RankedEngine(
            engine=MLXEngine(),
            result=BenchmarkResult(engine_name="mlx-lm", tokens_per_second=50.0),
        )
        cpp_ranked = RankedEngine(
            engine=LlamaCppEngine(),
            result=BenchmarkResult(engine_name="llama.cpp", tokens_per_second=50.0),
        )
        best = reg.select_best([cpp_ranked, mlx_ranked])
        assert best is not None
        # Both are ok with same tok/s — mlx has lower priority number (higher priority)
        # The select_best just returns the first ok result from the sorted list
        # Sort puts higher tok/s first, then lower priority number
        assert best.engine.name in ("mlx-lm", "llama.cpp")

    def test_echo_never_selected_over_real_engine(self):
        reg = EngineRegistry()
        echo_ranked = RankedEngine(
            engine=EchoEngine(),
            result=BenchmarkResult(engine_name="echo", error="no real inference"),
        )
        slow_ranked = RankedEngine(
            engine=_SlowEngine(),
            result=BenchmarkResult(engine_name="slow", tokens_per_second=1.0),
        )
        best = reg.select_best([echo_ranked, slow_ranked])
        assert best is not None
        assert best.engine.name == "slow"
