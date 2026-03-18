"""Tests for BenchmarkStore — CRUD keyed on model identity."""

from __future__ import annotations

import pytest

from octomil.device_agent.benchmark_store import BenchmarkResult, BenchmarkStore
from octomil.device_agent.db.local_db import LocalDB


@pytest.fixture
def setup():
    db = LocalDB(":memory:")
    store = BenchmarkStore(db)
    yield db, store
    db.close()


class TestRecord:
    def test_insert_single_result(self, setup) -> None:
        _, store = setup
        result = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
            latency_ms=12.5,
            throughput_tps=80.0,
            memory_bytes=1024 * 1024,
        )
        store.record(result)

        results = store.get_results("m1", "v1")
        assert len(results) == 1
        assert results[0].engine == "coreml"
        assert results[0].latency_ms == 12.5
        assert results[0].throughput_tps == 80.0
        assert results[0].memory_bytes == 1024 * 1024

    def test_upsert_overwrites_existing(self, setup) -> None:
        _, store = setup
        r1 = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
            latency_ms=12.5,
        )
        store.record(r1)

        r2 = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
            latency_ms=8.0,
        )
        store.record(r2)

        results = store.get_results("m1", "v1")
        assert len(results) == 1
        assert results[0].latency_ms == 8.0

    def test_multiple_engines(self, setup) -> None:
        _, store = setup
        for engine, latency in [("coreml", 10.0), ("llamacpp", 15.0), ("onnxruntime", 20.0)]:
            store.record(
                BenchmarkResult(
                    model_id="m1",
                    model_version="v1",
                    device_class="arm64",
                    sdk_version="0.5.0",
                    engine=engine,
                    latency_ms=latency,
                )
            )

        results = store.get_results("m1", "v1")
        assert len(results) == 3
        # Should be ordered by latency ASC
        assert results[0].engine == "coreml"
        assert results[1].engine == "llamacpp"
        assert results[2].engine == "onnxruntime"

    def test_metadata_roundtrip(self, setup) -> None:
        _, store = setup
        result = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
            latency_ms=10.0,
            metadata={"batch_size": 4, "quantized": True},
        )
        store.record(result)

        results = store.get_results("m1", "v1")
        assert results[0].metadata == {"batch_size": 4, "quantized": True}


class TestGetResults:
    def test_filter_by_device_class(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
                latency_ms=10.0,
            )
        )
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="x86_64",
                sdk_version="0.5.0",
                engine="onnxruntime",
                latency_ms=20.0,
            )
        )

        arm_results = store.get_results("m1", "v1", device_class="arm64")
        assert len(arm_results) == 1
        assert arm_results[0].engine == "coreml"

    def test_filter_by_sdk_version(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
                latency_ms=10.0,
            )
        )
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.6.0",
                engine="coreml",
                latency_ms=8.0,
            )
        )

        results = store.get_results("m1", "v1", sdk_version="0.6.0")
        assert len(results) == 1
        assert results[0].latency_ms == 8.0

    def test_empty_results(self, setup) -> None:
        _, store = setup
        results = store.get_results("nonexistent", "v1")
        assert results == []


class TestGetBestEngine:
    def test_returns_lowest_latency(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="llamacpp",
                latency_ms=15.0,
            )
        )
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
                latency_ms=8.0,
            )
        )

        best = store.get_best_engine("m1", "v1")
        assert best == "coreml"

    def test_returns_none_when_no_results(self, setup) -> None:
        _, store = setup
        assert store.get_best_engine("nonexistent", "v1") is None

    def test_returns_first_when_no_latency_data(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
            )
        )
        best = store.get_best_engine("m1", "v1")
        assert best == "coreml"


class TestDelete:
    def test_delete_specific_engine(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
                latency_ms=10.0,
            )
        )
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="llamacpp",
                latency_ms=15.0,
            )
        )

        deleted = store.delete("m1", "v1", engine="coreml")
        assert deleted == 1

        results = store.get_results("m1", "v1")
        assert len(results) == 1
        assert results[0].engine == "llamacpp"

    def test_delete_all_for_model_version(self, setup) -> None:
        _, store = setup
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="coreml",
                latency_ms=10.0,
            )
        )
        store.record(
            BenchmarkResult(
                model_id="m1",
                model_version="v1",
                device_class="arm64",
                sdk_version="0.5.0",
                engine="llamacpp",
                latency_ms=15.0,
            )
        )

        deleted = store.delete("m1", "v1")
        assert deleted == 2

        results = store.get_results("m1", "v1")
        assert len(results) == 0

    def test_delete_nonexistent(self, setup) -> None:
        _, store = setup
        deleted = store.delete("nonexistent", "v1")
        assert deleted == 0


class TestBenchmarkResultDataclass:
    def test_to_dict(self) -> None:
        r = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
            latency_ms=10.0,
        )
        d = r.to_dict()
        assert d["model_id"] == "m1"
        assert d["engine"] == "coreml"
        assert d["metadata"] == {}

    def test_defaults(self) -> None:
        r = BenchmarkResult(
            model_id="m1",
            model_version="v1",
            device_class="arm64",
            sdk_version="0.5.0",
            engine="coreml",
        )
        assert r.latency_ms is None
        assert r.throughput_tps is None
        assert r.memory_bytes is None
        assert r.metadata == {}
