"""Tests for the cache bench harness (octomil.runtime.bench.cache_bench).

Covers:
  1. Valid fixture dir → measured proof (JSON shape, required fields).
  2. Absent staged artifact → skipped proof (anti-pattern guard).
  3. No first-request synthesis fallback: monkeypatches fixture_dir to
     /nonexistent and asserts the harness emits skipped=True rather than
     running a live measurement.
  4. skip_reason is required when skipped=True.
  5. CacheBenchProof.to_dict() shape mirrors the contracts schema.
  6. Privacy: no input contents in proof artifact.
  7. StubCacheAdapter always returns miss (Phase 1 regression guard).
"""

from __future__ import annotations

import json
import math
import os

import pytest

from octomil.runtime.bench.cache_bench import (
    SKIP_STAGED_ARTIFACT_ABSENT,
    CacheBenchConfig,
    CacheBenchProof,
    StubCacheAdapter,
    run_cache_bench,
)

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def staged_fixture_dir(tmp_path):
    """A minimal staged fixture set — an existing directory with a stub
    index file. The StubCacheAdapter doesn't read any specific files, so
    an empty dir (+ sentinel) is sufficient to prove the directory exists."""
    d = tmp_path / "fixtures" / "chat.completion.kv"
    d.mkdir(parents=True)
    # Sentinel so operators can verify the fixture set is real.
    (d / "index.json").write_text(json.dumps({"version": "0.1.11-stub", "entries": 0}), encoding="utf-8")
    return d


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "proofs"


def _make_config(fixture_dir, output_dir, cache_id="chat.completion.kv", capability="chat.completion"):
    return CacheBenchConfig(
        cache_id=cache_id,
        capability=capability,
        fixture_dir=fixture_dir,
        output_dir=output_dir,
        n_cold_iters=5,
        n_warm_iters=10,
        n_hit_iters=10,
    )


# ────────────────────────────────────────────────────────────────────────────
# Test 1: measured proof shape
# ────────────────────────────────────────────────────────────────────────────


def test_measured_proof_has_required_fields(staged_fixture_dir, output_dir):
    """When the fixture dir exists, harness emits a measured proof."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())

    assert proof.skipped is False
    assert proof.skip_reason is None
    assert proof.cache_id == "chat.completion.kv"
    assert proof.capability == "chat.completion"
    assert proof.measured_at.endswith("Z")
    # Metrics must be present and finite.
    for field_name in ("cold_p50_ms", "cold_p95_ms", "warm_p50_ms", "warm_p95_ms", "hit_ratio"):
        val = getattr(proof, field_name)
        assert val is not None, f"{field_name} must not be None when measured"
        assert math.isfinite(val), f"{field_name}={val!r} must be finite"
    assert proof.hit_ratio >= 0 and proof.hit_ratio <= 1


def test_measured_proof_written_to_disk(staged_fixture_dir, output_dir):
    """Proof file must be written at the deterministic path."""
    config = _make_config(staged_fixture_dir, output_dir)
    run_cache_bench(config, adapter=StubCacheAdapter())

    expected = output_dir / "chat.completion.kv_proof.json"
    assert expected.exists(), f"proof not written to {expected}"
    doc = json.loads(expected.read_text())
    assert doc["$schema_version"] == 1
    assert doc["schema_version"] == 1
    assert doc["cache_id"] == "chat.completion.kv"
    assert doc["skipped"] is False


def test_measured_proof_to_dict_shape(staged_fixture_dir, output_dir):
    """to_dict() must contain all required fields from the schema."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())
    d = proof.to_dict()
    required = [
        "$schema_version",
        "schema_version",
        "cache_id",
        "capability",
        "measured_at",
        "skipped",
        "runtime_digest",
        "model_digest",
        "adapter_version",
        "staged_artifact_ref",
        "cold_p50_ms",
        "cold_p95_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "hit_ratio",
        "entries",
        "bytes_overhead",
        "parity_status",
        "writer",
    ]
    for r in required:
        assert r in d, f"to_dict() missing required field: {r}"


# ────────────────────────────────────────────────────────────────────────────
# Test 2 & 3: absent staged artifact → skipped proof (anti-pattern guard)
# ────────────────────────────────────────────────────────────────────────────


def test_absent_fixture_dir_emits_skipped_proof(output_dir):
    """No fixture dir → skipped proof with staged_artifact_absent."""
    config = _make_config(fixture_dir=None, output_dir=output_dir)
    proof = run_cache_bench(config)

    assert proof.skipped is True
    assert proof.skip_reason == SKIP_STAGED_ARTIFACT_ABSENT


def test_nonexistent_fixture_path_emits_skipped_proof(tmp_path, output_dir):
    """Monkeypatch: fixture_dir points to /nonexistent → skipped proof.

    This is the explicit no-first-request-synthesis regression test.
    The harness must NOT run any live measurement when the staged
    artifact is absent — it must emit skipped=True instead.
    """
    nonexistent = tmp_path / "does_not_exist" / "fixtures"
    # Path does not exist on disk.
    assert not nonexistent.exists()

    config = _make_config(fixture_dir=nonexistent, output_dir=output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())

    # The harness must have skipped, not measured.
    assert proof.skipped is True, (
        "REGRESSION: harness ran a live measurement when staged artifact "
        "was absent — this is the first-request-synthesis anti-pattern. "
        "The harness MUST emit skipped=True instead."
    )
    assert proof.skip_reason == SKIP_STAGED_ARTIFACT_ABSENT
    # All metric fields must be null.
    for field_name in ("cold_p50_ms", "cold_p95_ms", "warm_p50_ms", "warm_p95_ms", "hit_ratio"):
        val = getattr(proof, field_name)
        assert val is None, (
            f"REGRESSION: {field_name}={val!r} is not None when harness skipped. "
            "Metric fields must be null when skipped=True."
        )


def test_skipped_proof_written_to_disk(output_dir):
    """Even a skipped proof must be written to disk so CI can verify it."""
    config = _make_config(fixture_dir=None, output_dir=output_dir)
    run_cache_bench(config)

    expected = output_dir / "chat.completion.kv_proof.json"
    assert expected.exists(), "skipped proof must still be written to disk"
    doc = json.loads(expected.read_text())
    assert doc["skipped"] is True
    assert doc["skip_reason"] == SKIP_STAGED_ARTIFACT_ABSENT


# ────────────────────────────────────────────────────────────────────────────
# Test 4: CacheBenchProof invariants
# ────────────────────────────────────────────────────────────────────────────


def test_skipped_proof_without_skip_reason_raises():
    """Constructor must reject skipped=True without skip_reason."""
    with pytest.raises(ValueError, match="skip_reason"):
        CacheBenchProof(
            cache_id="chat.completion.kv",
            capability="chat.completion",
            measured_at="2026-05-09T00:00:00Z",
            skipped=True,
            skip_reason=None,  # Missing — must raise.
            runtime_digest="sha256:" + "a" * 64,
            model_digest="sha256:" + "b" * 64,
            adapter_version="0.1.11-stub",
            staged_artifact_ref="/nonexistent",
        )


def test_unknown_skip_reason_raises():
    """Constructor must reject skip_reason values not in the closed enum."""
    with pytest.raises(ValueError, match="skip_reason"):
        CacheBenchProof(
            cache_id="chat.completion.kv",
            capability="chat.completion",
            measured_at="2026-05-09T00:00:00Z",
            skipped=True,
            skip_reason="unknown_reason",
            runtime_digest="sha256:" + "a" * 64,
            model_digest="sha256:" + "b" * 64,
            adapter_version="0.1.11-stub",
            staged_artifact_ref="/nonexistent",
        )


def test_non_skipped_proof_without_metrics_raises():
    """Constructor must reject skipped=False with null metric fields."""
    with pytest.raises(ValueError, match="cold_p50_ms"):
        CacheBenchProof(
            cache_id="chat.completion.kv",
            capability="chat.completion",
            measured_at="2026-05-09T00:00:00Z",
            skipped=False,
            runtime_digest="sha256:" + "a" * 64,
            model_digest="sha256:" + "b" * 64,
            adapter_version="0.1.11",
            staged_artifact_ref="/fixtures",
            cold_p50_ms=None,  # Missing — must raise.
        )


def test_non_finite_metric_raises():
    """Constructor must reject NaN/Inf metrics."""
    with pytest.raises(ValueError, match="cold_p50_ms"):
        CacheBenchProof(
            cache_id="chat.completion.kv",
            capability="chat.completion",
            measured_at="2026-05-09T00:00:00Z",
            skipped=False,
            runtime_digest="sha256:" + "a" * 64,
            model_digest="sha256:" + "b" * 64,
            adapter_version="0.1.11",
            staged_artifact_ref="/fixtures",
            cold_p50_ms=math.nan,
            cold_p95_ms=1.0,
            warm_p50_ms=1.0,
            warm_p95_ms=1.0,
            hit_ratio=0.5,
        )


# ────────────────────────────────────────────────────────────────────────────
# Test 5: Privacy enforcement
# ────────────────────────────────────────────────────────────────────────────


def test_proof_contains_no_raw_input_content(staged_fixture_dir, output_dir):
    """Proof artifact must not contain raw input content fields."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())
    d = proof.to_dict()

    # These fields must NOT appear in the proof.
    banned_keys = {"input_text", "prompt", "content", "text", "audio", "response"}
    present_banned = banned_keys & set(d.keys())
    assert not present_banned, (
        f"Proof contains banned input-content keys: {present_banned}. "
        "Privacy rule: proof artifacts MUST NOT include input contents."
    )


def test_input_digest_is_sha256_prefixed(staged_fixture_dir, output_dir):
    """input_digest must be 'sha256:<hex64>' when present."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())
    if proof.input_digest is not None:
        assert proof.input_digest.startswith(
            "sha256:"
        ), "input_digest must be 'sha256:<hex>' — raw content is not allowed"
        assert len(proof.input_digest) == len("sha256:") + 64, "input_digest must be 'sha256:' + 64 hex chars"


# ────────────────────────────────────────────────────────────────────────────
# Test 6: StubCacheAdapter contract (Phase 1 regression guard)
# ────────────────────────────────────────────────────────────────────────────


def test_stub_adapter_always_returns_miss(tmp_path):
    """StubCacheAdapter must always return hit=False (Phase 1 stub)."""
    adapter = StubCacheAdapter()
    for i in range(10):
        hit, latency_ms = adapter.warm_lookup(f"key_{i}")
        assert not hit, f"StubCacheAdapter.warm_lookup returned hit=True on iter {i}"
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0.0


def test_stub_adapter_populate_returns_zero_entries(tmp_path):
    """StubCacheAdapter.populate must return (0, 0) for Phase 1."""
    adapter = StubCacheAdapter()
    entries, bytes_overhead = adapter.populate(tmp_path)
    assert entries == 0
    assert bytes_overhead == 0


# ────────────────────────────────────────────────────────────────────────────
# Test 7: writer block in proof dict
# ────────────────────────────────────────────────────────────────────────────


def test_writer_block_in_proof(staged_fixture_dir, output_dir):
    """Proof must contain a writer block with process_kind and runtime_build_tag."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())
    d = proof.to_dict()
    assert "writer" in d
    writer = d["writer"]
    assert "process_kind" in writer
    assert "runtime_build_tag" in writer
    assert writer["process_kind"] == "python_sdk"


def test_writer_pid_present(staged_fixture_dir, output_dir):
    """Proof writer block must include pid (diagnostic only)."""
    config = _make_config(staged_fixture_dir, output_dir)
    proof = run_cache_bench(config, adapter=StubCacheAdapter())
    assert proof.writer_pid == os.getpid()
