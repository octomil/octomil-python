"""Tests for ``octomil.runtime.bench.cache`` — v0.5 PR A.

Tests the cache R/W foundation in isolation. There is no benchmark
harness yet (PR B); these tests construct ``Result`` objects by hand
and verify the cache's read / write / list / clear / rebuild
contracts.

Coverage strategy:

  * Canonical-JSON serialization is byte-identical for equivalent
    inputs (cross-SDK contract).
  * Leaf filenames encode the FULL cache_key (the strategy doc P1
    fix: not just model_digest).
  * Atomic writes never leave half-written files visible.
  * Schema-version mismatch surfaces as cache miss, not crash.
  * ``incomplete=True`` results are written to disk but ignored on
    lookup. Operator promotion path (``accept-partial``) is PR D.
  * Concurrent writers coordinate via the advisory file lock.
  * Index file is rebuildable from leaves.
  * Hardware-fingerprint scope is enforced: a leaf written under
    fingerprint A is invisible when the store is queried under
    fingerprint B.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from octomil.runtime.bench.cache import (
    CACHE_SCHEMA_VERSION,
    CacheKey,
    CacheStore,
    DispatchShape,
    HardwareFingerprint,
    Result,
    Winner,
    _canonical_json_bytes,
    default_cache_root,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hardware() -> HardwareFingerprint:
    return HardwareFingerprint(
        machine="arm64",
        processor="Apple M2",
        cpu_count=8,
        ram_gb=16,
        os_version="macOS 15.1",
        runtime_build_tag="octomil-python:test",
    )


@pytest.fixture
def store(tmp_path: Path, hardware: HardwareFingerprint) -> CacheStore:
    return CacheStore(cache_root=tmp_path / "octomil" / "runtime_bench_v0_5", hardware=hardware)


@pytest.fixture
def tts_dispatch() -> DispatchShape:
    return DispatchShape(
        fields={
            "language": "en",
            "sample_format": "pcm_s16le",
            "sample_rate_out": 24000,
            "voice_family": "kokoro_en",
        }
    )


@pytest.fixture
def tts_cache_key(tts_dispatch: DispatchShape) -> CacheKey:
    return CacheKey(
        capability="tts",
        model_id="kokoro-en-v0_19",
        model_digest="sha256:" + "d" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=tts_dispatch,
    )


@pytest.fixture
def winner_committed() -> Winner:
    return Winner(
        engine="sherpa-onnx",
        provider="coreml",
        config={"num_threads": 1},
        score=145.2,
        first_chunk_ms=132.1,
        total_latency_ms=280.4,
        quality_metrics={"alignment_dtw_correlation": 0.94},
    )


def _committed_result(
    cache_key: CacheKey,
    winner: Winner,
    hardware: HardwareFingerprint,
) -> Result:
    return Result(
        cache_key=cache_key,
        hardware_fingerprint=hardware.full_digest(),
        hardware_descriptor=hardware.descriptor_dict(),
        writer_runtime_build_tag=hardware.runtime_build_tag,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# Canonical-JSON contract — cross-SDK byte-equality
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_at_every_level():
    a = {"b": 1, "a": {"y": 2, "x": [3, 1, 2]}}
    b = {"a": {"x": [3, 1, 2], "y": 2}, "b": 1}
    assert _canonical_json_bytes(a) == _canonical_json_bytes(b)


def test_canonical_json_no_whitespace():
    payload = {"a": 1, "b": [2, 3]}
    encoded = _canonical_json_bytes(payload)
    assert b" " not in encoded
    assert b"\n" not in encoded


def test_canonical_json_utf8_for_non_ascii():
    """Cross-SDK rule: UTF-8 source bytes, NOT ``\\uXXXX`` escapes.
    The contract pins this so non-ASCII model ids hash identically
    across SDK bindings."""
    payload = {"voice": "héllo"}
    encoded = _canonical_json_bytes(payload)
    assert "héllo".encode("utf-8") in encoded
    assert b"\\u" not in encoded


def test_canonical_json_rejects_nan_and_inf():
    """NaN/Inf would produce non-portable JSON. ``allow_nan=False``
    in the implementation; the canonicalization MUST raise."""
    with pytest.raises(ValueError):
        _canonical_json_bytes({"x": float("nan")})


# ---------------------------------------------------------------------------
# Leaf filename encodes the FULL cache_key (strategy P1 fix)
# ---------------------------------------------------------------------------


def test_leaf_filename_changes_when_capability_changes(tts_dispatch: DispatchShape):
    base = CacheKey(
        capability="tts",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=tts_dispatch,
    )
    other = CacheKey(
        capability="transcription",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(fields={"sample_rate_in": 16000, "language": "en", "beam_size_preference": 1}),
    )
    assert base.leaf_filename() != other.leaf_filename()


def test_leaf_filename_changes_when_dispatch_shape_changes():
    """The Eternum-class scenario: same model_digest, different
    sample rate. Pre-strategy-P1 path-only-by-digest would collapse
    these; cache_key-hash-encoded path keeps them distinct."""
    a = CacheKey(
        capability="tts",
        model_id="kokoro-en-v0_19",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "sample_rate_out": 24000,
                "sample_format": "pcm_s16le",
                "voice_family": "kokoro_en",
                "language": "en",
            }
        ),
    )
    b = CacheKey(
        capability="tts",
        model_id="kokoro-en-v0_19",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "sample_rate_out": 48000,
                "sample_format": "pcm_s16le",
                "voice_family": "kokoro_en",
                "language": "en",
            }
        ),
    )
    assert a.leaf_filename() != b.leaf_filename()


def test_leaf_filename_changes_when_quantization_changes(tts_dispatch: DispatchShape):
    fp32 = CacheKey(
        capability="tts",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=tts_dispatch,
    )
    int8 = CacheKey(
        capability="tts",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="int8",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=tts_dispatch,
    )
    assert fp32.leaf_filename() != int8.leaf_filename()


def test_leaf_filename_stable_under_dispatch_field_reordering():
    """A consumer that builds the dispatch_shape with fields in
    different order MUST get the same leaf — canonical JSON sorts."""
    a = CacheKey(
        capability="tts",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "sample_rate_out": 24000,
                "sample_format": "pcm_s16le",
                "voice_family": "kokoro_en",
                "language": "en",
            }
        ),
    )
    b = CacheKey(
        capability="tts",
        model_id="m",
        model_digest="sha256:" + "a" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "language": "en",
                "voice_family": "kokoro_en",
                "sample_format": "pcm_s16le",
                "sample_rate_out": 24000,
            }
        ),
    )
    assert a.leaf_filename() == b.leaf_filename()


# ---------------------------------------------------------------------------
# CacheKey validation
# ---------------------------------------------------------------------------


def test_cache_key_rejects_unknown_capability(tts_dispatch: DispatchShape):
    with pytest.raises(ValueError, match="capability"):
        CacheKey(
            capability="oracle",
            model_id="m",
            model_digest="sha256:" + "a" * 64,
            quantization_preference="fp32",
            candidate_set_version="1.0",
            reference_workload_version="1.0",
            dispatch_shape=tts_dispatch,
        )


def test_cache_key_rejects_malformed_digest(tts_dispatch: DispatchShape):
    with pytest.raises(ValueError, match="model_digest"):
        CacheKey(
            capability="tts",
            model_id="m",
            model_digest="abc123",
            quantization_preference="fp32",
            candidate_set_version="1.0",
            reference_workload_version="1.0",
            dispatch_shape=tts_dispatch,
        )


# ---------------------------------------------------------------------------
# Result commit invariant
# ---------------------------------------------------------------------------


def test_result_incomplete_true_forbids_winner(tts_cache_key, hardware, winner_committed):
    with pytest.raises(ValueError, match="incomplete=True forbids"):
        Result(
            cache_key=tts_cache_key,
            hardware_fingerprint=hardware.full_digest(),
            hardware_descriptor=hardware.descriptor_dict(),
            writer_runtime_build_tag=hardware.runtime_build_tag,
            incomplete=True,
            winner=winner_committed,
        )


def test_result_incomplete_false_requires_winner(tts_cache_key, hardware):
    with pytest.raises(ValueError, match="incomplete=False requires"):
        Result(
            cache_key=tts_cache_key,
            hardware_fingerprint=hardware.full_digest(),
            hardware_descriptor=hardware.descriptor_dict(),
            writer_runtime_build_tag=hardware.runtime_build_tag,
            incomplete=False,
            winner=None,
        )


# ---------------------------------------------------------------------------
# Round-trip: put → get returns the same Winner
# ---------------------------------------------------------------------------


def test_put_then_get_round_trips(store, tts_cache_key, winner_committed, hardware):
    result = _committed_result(tts_cache_key, winner_committed, hardware)
    leaf_path = store.put(result)
    assert leaf_path.is_file()

    cached = store.get(tts_cache_key)
    assert cached is not None
    assert cached.winner is not None
    assert cached.winner.engine == "sherpa-onnx"
    assert cached.winner.provider == "coreml"
    assert cached.winner.score == pytest.approx(145.2)
    assert cached.confidence == "high"


def test_get_returns_none_on_miss(store, tts_cache_key):
    assert store.get(tts_cache_key) is None


# ---------------------------------------------------------------------------
# Incomplete results are written but invisible on lookup
# ---------------------------------------------------------------------------


def test_incomplete_result_is_written_but_lookup_returns_none(store, tts_cache_key, hardware, winner_committed):
    """Reviewer P2 from the strategy review: a timeout/interruption
    persists partial observations so a resume can continue from
    them. Lookup MUST ignore those entries — operators promote via
    the future CLI."""
    partial = Result(
        cache_key=tts_cache_key,
        hardware_fingerprint=hardware.full_digest(),
        hardware_descriptor=hardware.descriptor_dict(),
        writer_runtime_build_tag=hardware.runtime_build_tag,
        incomplete=True,
        partial_observations=(winner_committed,),
        confidence="low",
    )
    leaf = store.put(partial)
    assert leaf.is_file(), "partial result MUST be persisted to disk for resume"
    assert store.get(tts_cache_key) is None, "lookup MUST ignore incomplete=True entries"


# ---------------------------------------------------------------------------
# Schema-version mismatch is a cache miss, not a crash
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_returns_none(store, tts_cache_key, hardware, winner_committed):
    result = _committed_result(tts_cache_key, winner_committed, hardware)
    leaf = store.put(result)
    # Tamper the on-disk schema_version; reader MUST treat as miss.
    data = json.loads(leaf.read_text())
    data["$schema_version"] = CACHE_SCHEMA_VERSION + 99
    leaf.write_text(json.dumps(data))
    assert store.get(tts_cache_key) is None


def test_malformed_json_returns_none(store, tts_cache_key, hardware, winner_committed):
    result = _committed_result(tts_cache_key, winner_committed, hardware)
    leaf = store.put(result)
    leaf.write_text("{ this is not valid JSON")
    assert store.get(tts_cache_key) is None


# ---------------------------------------------------------------------------
# Hardware-fingerprint scope
# ---------------------------------------------------------------------------


def test_get_under_different_hardware_fingerprint_returns_none(tmp_path, tts_cache_key, winner_committed, hardware):
    """Two stores at the same cache root with different fingerprints
    must not share entries. Defense-in-depth: even if path prefixes
    happen to truncate-collide, the result file's full digest is
    checked at read time."""
    cache_root = tmp_path / "octomil" / "runtime_bench_v0_5"
    other_hardware = HardwareFingerprint(
        machine="x86_64",
        processor="Intel",
        cpu_count=4,
        ram_gb=8,
        os_version="Linux 6.0",
        runtime_build_tag="octomil-python:test",
    )

    write_store = CacheStore(cache_root=cache_root, hardware=hardware)
    read_store = CacheStore(cache_root=cache_root, hardware=other_hardware)

    write_store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    assert read_store.get(tts_cache_key) is None


# ---------------------------------------------------------------------------
# Atomic writes — no half-written file visible
# ---------------------------------------------------------------------------


def test_writes_never_leave_partial_file(store, tts_cache_key, winner_committed, hardware):
    """Strategy doc concurrency rule: temp-file-plus-rename. Verify
    by inspecting the model_dir mid-flight isn't possible from a
    single thread; instead, assert the temp-file invariant by
    construction — the only ``.json`` files in the dir are
    finalized leaves + the index."""
    store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    model_dir = store.cache_root / hardware.path_component() / tts_cache_key.model_id
    files = sorted(p.name for p in model_dir.iterdir())
    # Only the final leaf and the index, plus the lock sidecar that
    # _try_writer_lock leaves behind.
    leaf = tts_cache_key.leaf_filename()
    expected = {leaf, "index.json", leaf + ".lock"}
    assert set(files) == expected, f"unexpected files in model_dir: {files}"


# ---------------------------------------------------------------------------
# Index sidecar
# ---------------------------------------------------------------------------


def test_index_is_updated_on_put(store, tts_cache_key, winner_committed, hardware):
    store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    entries = store.list_cache_keys(model_id=tts_cache_key.model_id)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["leaf_filename"] == tts_cache_key.leaf_filename()
    assert entry["incomplete"] is False
    assert "sherpa-onnx" in entry["winner_summary"]


def test_rebuild_index_recovers_from_corruption(store, tts_cache_key, winner_committed, hardware):
    leaf = store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    model_dir = leaf.parent
    index_path = model_dir / "index.json"
    # Corrupt the index.
    index_path.write_text("{ corrupt")
    rebuilt = store.rebuild_index(model_id=tts_cache_key.model_id)
    assert rebuilt == 1
    entries = store.list_cache_keys(model_id=tts_cache_key.model_id)
    assert len(entries) == 1
    assert entries[0]["leaf_filename"] == tts_cache_key.leaf_filename()


def test_rebuild_index_skips_invalid_leaves(store, tts_cache_key, winner_committed, hardware):
    store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    model_dir = store.cache_root / hardware.path_component() / tts_cache_key.model_id
    # Drop a malformed leaf in the same directory.
    (model_dir / ("0" * 64 + ".json")).write_text("{ not json")
    rebuilt = store.rebuild_index(model_id=tts_cache_key.model_id)
    assert rebuilt == 1, "rebuild must skip malformed leaves"


# ---------------------------------------------------------------------------
# Filename integrity check
# ---------------------------------------------------------------------------


def test_renaming_a_leaf_to_a_different_hash_is_a_cache_miss(
    store, tts_cache_key, winner_committed, hardware, tmp_path
):
    """Defense-in-depth: if a leaf ended up at the wrong filename
    (writer bug, manual file move), the reader recomputes the
    filename from the body's cache_key and rejects mismatches."""
    leaf = store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    bogus_name = leaf.parent / ("f" * 64 + ".json")
    leaf.rename(bogus_name)
    assert store.get(tts_cache_key) is None


# ---------------------------------------------------------------------------
# Clear behavior
# ---------------------------------------------------------------------------


def test_clear_model_removes_leaves_and_returns_count(store, tts_cache_key, winner_committed, hardware):
    store.put(_committed_result(tts_cache_key, winner_committed, hardware))
    removed = store.clear_model(model_id=tts_cache_key.model_id)
    assert removed == 1
    assert store.get(tts_cache_key) is None


def test_clear_all_removes_every_model(store, hardware, winner_committed, tts_dispatch):
    keys = [
        CacheKey(
            capability="tts",
            model_id=m,
            model_digest="sha256:" + ("a" if i == 0 else "b") * 64,
            quantization_preference="fp32",
            candidate_set_version="1.0",
            reference_workload_version="1.0",
            dispatch_shape=tts_dispatch,
        )
        for i, m in enumerate(("kokoro-en-v0_19", "piper-en-amy"))
    ]
    for k in keys:
        store.put(_committed_result(k, winner_committed, hardware))
    assert sorted(store.list_models()) == ["kokoro-en-v0_19", "piper-en-amy"]
    store.clear_all()
    assert store.list_models() == []


# ---------------------------------------------------------------------------
# Default cache root resolution
# ---------------------------------------------------------------------------


def test_default_cache_root_honors_xdg_cache_home(tmp_path):
    root = default_cache_root(env={"XDG_CACHE_HOME": str(tmp_path)})
    assert root == tmp_path / "octomil" / "runtime_bench_v0_5"


def test_default_cache_root_falls_back_to_home_dot_cache(tmp_path):
    root = default_cache_root(env={"HOME": str(tmp_path)})
    assert root == tmp_path / ".cache" / "octomil" / "runtime_bench_v0_5"


# ---------------------------------------------------------------------------
# Concurrency — two threads racing the same put
# ---------------------------------------------------------------------------


def test_concurrent_writers_dont_corrupt_leaf(store, tts_cache_key, winner_committed, hardware):
    """Two threads call put() with different winners for the same
    cache_key. After the dust settles, the leaf is one valid file —
    not a half-written corruption.

    The contract doesn't require any particular winner wins (peer-
    wait or fall-through both produce a valid file); it just
    requires the file is parseable and matches its filename hash.
    """
    threads = []
    errors: list[Exception] = []

    def worker(score: float):
        winner = Winner(
            engine="sherpa-onnx",
            config={"num_threads": int(score)},
            score=score,
            provider="cpu",
        )
        result = _committed_result(tts_cache_key, winner, hardware)
        try:
            store.put(result)
        except Exception as exc:  # pragma: no cover — defensive
            errors.append(exc)

    for s in (100.0, 200.0, 300.0):
        t = threading.Thread(target=worker, args=(s,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    cached = store.get(tts_cache_key)
    assert cached is not None
    assert cached.winner is not None
    # The committed winner came from one of the threads; just verify
    # the score is one of the values we wrote.
    assert cached.winner.score in (100.0, 200.0, 300.0)


# ---------------------------------------------------------------------------
# Hardware fingerprint detection (smoke)
# ---------------------------------------------------------------------------


def test_hardware_fingerprint_detect_returns_stable_path_component():
    fp1 = HardwareFingerprint.detect(runtime_build_tag="octomil-python:test")
    fp2 = HardwareFingerprint.detect(runtime_build_tag="octomil-python:test")
    assert fp1.path_component() == fp2.path_component()
    # 16 hex chars per the truncation constant.
    assert len(fp1.path_component()) == 16
    assert all(c in "0123456789abcdef" for c in fp1.path_component())


def test_hardware_fingerprint_path_component_changes_with_runtime_build_tag():
    fp1 = HardwareFingerprint.detect(runtime_build_tag="octomil-python:1.0.0")
    fp2 = HardwareFingerprint.detect(runtime_build_tag="octomil-python:2.0.0")
    assert fp1.path_component() != fp2.path_component()
