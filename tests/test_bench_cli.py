"""Tests for ``octomil bench`` CLI — v0.5 PR D.

CliRunner-based tests against ``octomil.commands.bench``. Each test
constructs a fresh tmp cache root and exercises the verb directly via
``click.testing.CliRunner``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from octomil.commands.bench import bench_cmd
from octomil.runtime.bench.cache import (
    CacheKey,
    CacheStore,
    DispatchShape,
    HardwareFingerprint,
    Result,
    Winner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hardware() -> HardwareFingerprint:
    """Use the real device fingerprint so the CLI's `_open_store`
    (which calls `HardwareFingerprint.detect`) lands on the same
    on-disk subdirectory the test seeds. The runtime_build_tag is
    overridden to keep the test's writer identifiable in cache logs."""
    return HardwareFingerprint.detect(runtime_build_tag="octomil-cli")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "bench-cache"


@pytest.fixture
def seeded_store(cache_root: Path, hardware: HardwareFingerprint) -> CacheStore:
    """A cache store with two committed winners across two models."""
    store = CacheStore(cache_root=cache_root, hardware=hardware)

    def make_key(model_id: str, voice: str) -> CacheKey:
        return CacheKey(
            capability="tts",
            model_id=model_id,
            model_digest="sha256:" + "a" * 64,
            quantization_preference="fp32",
            candidate_set_version="1.0",
            reference_workload_version="1.0",
            dispatch_shape=DispatchShape(
                fields={
                    "language": "en",
                    "sample_format": "pcm_s16le",
                    "sample_rate_out": 24000,
                    "voice_family": voice,
                }
            ),
        )

    def make_result(key: CacheKey, score: float) -> Result:
        return Result(
            cache_key=key,
            hardware_fingerprint=hardware.full_digest(),
            hardware_descriptor=hardware.descriptor_dict(),
            writer_runtime_build_tag=hardware.runtime_build_tag,
            winner=Winner(
                engine="sherpa-onnx",
                provider="coreml",
                config={"num_threads": 1},
                score=score,
                first_chunk_ms=10.0,
                total_latency_ms=20.0,
                quality_metrics={"avg_speaker_embedding_cosine": 0.95},
            ),
            runners_up=(),
            disqualified=(),
            incomplete=False,
            confidence="high",
        )

    store.put(make_result(make_key("kokoro-en-v0_19", "kokoro_en"), score=18.0))
    store.put(make_result(make_key("piper-en-v1_0", "piper_en"), score=22.0))
    return store


# ---------------------------------------------------------------------------
# `bench list`
# ---------------------------------------------------------------------------


def test_list_shows_all_committed_entries(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root)])
    assert res.exit_code == 0, res.output
    assert "kokoro-en-v0_19" in res.output
    assert "piper-en-v1_0" in res.output
    assert "cap=tts" in res.output
    assert "incomplete=False" in res.output


def test_list_filter_by_capability(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root), "--capability", "embeddings"])
    assert res.exit_code == 0
    assert "kokoro-en-v0_19" not in res.output
    assert "No cached entries found" in res.output


def test_list_empty_cache(cache_root: Path, hardware: HardwareFingerprint):
    """Empty cache exits clean with a friendly message on stderr."""
    CacheStore(cache_root=cache_root, hardware=hardware)  # creates the dir
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root)])
    assert res.exit_code == 0
    assert "No cached entries found" in res.output


# ---------------------------------------------------------------------------
# `bench show`
# ---------------------------------------------------------------------------


def test_show_emits_json_per_leaf(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["show", "kokoro-en-v0_19", "--cache-root", str(cache_root)])
    assert res.exit_code == 0, res.output
    # JSON-parseable when split on '---'.
    blocks = [b.strip() for b in res.output.split("\n---\n") if b.strip()]
    assert len(blocks) == 1
    payload = json.loads(blocks[0])
    assert payload["cache_key"]["model_id"] == "kokoro-en-v0_19"
    assert payload["winner"]["engine"] == "sherpa-onnx"


def test_show_unknown_model_exits_one(cache_root: Path, hardware: HardwareFingerprint):
    CacheStore(cache_root=cache_root, hardware=hardware)
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["show", "no-such-model", "--cache-root", str(cache_root)])
    assert res.exit_code == 1
    assert "No cache entries" in res.output


# ---------------------------------------------------------------------------
# `bench reset`
# ---------------------------------------------------------------------------


def test_reset_one_model(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["reset", "kokoro-en-v0_19", "--cache-root", str(cache_root), "--yes"])
    assert res.exit_code == 0, res.output
    assert "Cleared" in res.output

    # Confirm: only kokoro is gone; piper still there.
    res2 = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root)])
    assert "kokoro-en-v0_19" not in res2.output
    assert "piper-en-v1_0" in res2.output


def test_reset_all(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["reset", "--all", "--cache-root", str(cache_root), "--yes"])
    assert res.exit_code == 0
    assert "Cleared 2 cache entries" in res.output

    res2 = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root)])
    assert "No cached entries found" in res2.output


def test_reset_requires_target(cache_root: Path, hardware: HardwareFingerprint):
    CacheStore(cache_root=cache_root, hardware=hardware)
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["reset", "--cache-root", str(cache_root), "--yes"])
    assert res.exit_code != 0
    assert "Pass a <model> arg or --all" in res.output


def test_reset_rejects_both(cache_root: Path, hardware: HardwareFingerprint):
    CacheStore(cache_root=cache_root, hardware=hardware)
    runner = CliRunner()
    res = runner.invoke(
        bench_cmd,
        ["reset", "kokoro-en-v0_19", "--all", "--cache-root", str(cache_root), "--yes"],
    )
    assert res.exit_code != 0
    assert "Pass <model> OR --all" in res.output


def test_reset_aborts_without_yes(seeded_store: CacheStore, cache_root: Path):
    """Without --yes the prompt aborts when the user types n."""
    runner = CliRunner()
    res = runner.invoke(
        bench_cmd,
        ["reset", "kokoro-en-v0_19", "--cache-root", str(cache_root)],
        input="n\n",
    )
    assert res.exit_code != 0
    # Cache is untouched.
    res2 = runner.invoke(bench_cmd, ["list", "--cache-root", str(cache_root)])
    assert "kokoro-en-v0_19" in res2.output


# ---------------------------------------------------------------------------
# `bench run`
# ---------------------------------------------------------------------------


def test_run_v05_stub_exits_two(cache_root: Path, hardware: HardwareFingerprint):
    """The v0.5 `run` verb exits 2 with a clear message — wiring lands
    in PR C2."""
    CacheStore(cache_root=cache_root, hardware=hardware)
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["run", "kokoro-en-v0_19", "--cache-root", str(cache_root)])
    assert res.exit_code == 2
    assert "candidate-enumeration wiring" in res.output


def test_run_rejects_unsupported_capability(cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(
        bench_cmd, ["run", "embedding-v1", "--capability", "embeddings", "--cache-root", str(cache_root)]
    )
    assert res.exit_code != 0
    assert "v0.5 supports capability=tts only" in res.output


# ---------------------------------------------------------------------------
# `bench status`
# ---------------------------------------------------------------------------


def test_status_emits_json(seeded_store: CacheStore, cache_root: Path):
    runner = CliRunner()
    res = runner.invoke(bench_cmd, ["status", "--cache-root", str(cache_root)])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["entry_count_total"] == 2
    assert payload["entry_count_committed"] == 2
    assert payload["entry_count_incomplete"] == 0
    assert "OCTOMIL_RUNTIME_BENCH" in payload["env"]
    assert payload["cache_root"]
