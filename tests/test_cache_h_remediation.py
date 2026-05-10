"""Regression tests for Lane H Codex remediation (PR #569).

Covers Codex blockers:

* B1 — embeddings_backend.embed([]) bypasses INVALID_INPUT.
* B2 — result-cache hits drop prompt_tokens / pooling_type / n_dim.
* B3 — cache returns/stores mutable lists; caller mutation corrupts hits.
* B6 — bench input_digest hashes only the path, not the fixture content.
* B7 — TTS frontend cache normalizes before lookup, never uses cached_value.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

from octomil.runtime.native.embeddings_cache import (
    CachedEmbedding,
    CachePolicy,
    EmbeddingsCacheManager,
    EmbeddingsLruCache,
    cache_get_vector,
    cache_put_vector,
)

# --- B3: defensive copy on cache get/put -----------------------------------


def test_b3_cache_get_returns_defensive_copy(monkeypatch) -> None:
    """Mutating the returned list must not corrupt cached state."""
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")
    cache = EmbeddingsLruCache(max_bytes=1024)
    key = b"k" * 32
    original = [1.0, 2.0, 3.0]
    cache_put_vector(cache, key, original, privacy_allowed=True)

    got = cache_get_vector(cache, key, privacy_allowed=True)
    assert got == [1.0, 2.0, 3.0]
    assert got is not None
    got[0] = 999.0  # caller mutation

    second = cache_get_vector(cache, key, privacy_allowed=True)
    assert second == [1.0, 2.0, 3.0], f"caller mutation corrupted cache: got {second}"


def test_b3_cache_put_takes_defensive_copy(monkeypatch) -> None:
    """Mutating the source list after put must not change cached value."""
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")
    cache = EmbeddingsLruCache(max_bytes=1024)
    key = b"x" * 32
    src = [0.1, 0.2, 0.3]
    cache_put_vector(cache, key, src, privacy_allowed=True)

    src[0] = 9.9  # mutate after put

    got = cache_get_vector(cache, key, privacy_allowed=True)
    assert got == [0.1, 0.2, 0.3], f"post-put source mutation corrupted cache: got {got}"


def test_b3_record_round_trip_preserves_metadata(monkeypatch) -> None:
    """B2: get_record returns the stored prompt_tokens/n_dim/pooling_type."""
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")
    mgr = EmbeddingsCacheManager(
        model_digest="sha256:" + "a" * 64,
        adapter_version="test",
    )
    policy = CachePolicy.policy_allowed()
    rec = CachedEmbedding(
        vector=[0.5, 0.6, 0.7, 0.8],
        prompt_tokens=42,
        n_dim=4,
        pooling_type=2,
        is_normalized=True,
    )
    mgr.put_record("hello", rec, policy=policy)

    got = mgr.get_record("hello", policy=policy)
    assert got is not None
    assert got.vector == [0.5, 0.6, 0.7, 0.8]
    assert got.prompt_tokens == 42
    assert got.n_dim == 4
    assert got.pooling_type == 2
    assert got.is_normalized is True

    # Mutating the returned vector must not corrupt the cache.
    got.vector[0] = 999.0
    again = mgr.get_record("hello", policy=policy)
    assert again is not None
    assert again.vector == [0.5, 0.6, 0.7, 0.8]


# --- B6: input_digest reflects fixture content ------------------------------


def test_b6_input_digest_changes_when_fixture_content_changes(
    tmp_path: pathlib.Path,
) -> None:
    """Same path, different content → different digest."""
    from octomil.runtime.bench.cache_bench import StagedFixtureSet

    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "a.txt").write_text("hello")

    s = StagedFixtureSet(
        fixture_dir=fixture,
        cache_id="chat.completion.kv",
        capability="chat.completion",
    )
    digest_v1 = s.input_digest()

    # Mutate content at the same path.
    (fixture / "a.txt").write_text("world")
    digest_v2 = s.input_digest()

    assert digest_v1 != digest_v2, "fixture content changed but input_digest stayed the same — " "Codex B6 regression"


def test_b6_input_digest_changes_when_file_added(
    tmp_path: pathlib.Path,
) -> None:
    """Adding a fixture file at the same path → different digest."""
    from octomil.runtime.bench.cache_bench import StagedFixtureSet

    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "a.txt").write_text("hello")

    s = StagedFixtureSet(
        fixture_dir=fixture,
        cache_id="chat.completion.kv",
        capability="chat.completion",
    )
    before = s.input_digest()

    (fixture / "b.txt").write_text("more data")
    after = s.input_digest()

    assert before != after, "adding a fixture file did not change input_digest"


def test_b6_input_digest_stable_when_nothing_changes(
    tmp_path: pathlib.Path,
) -> None:
    """Determinism: same content twice → same digest."""
    from octomil.runtime.bench.cache_bench import StagedFixtureSet

    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "a.txt").write_text("hello")
    (fixture / "b.txt").write_text("more")

    s = StagedFixtureSet(
        fixture_dir=fixture,
        cache_id="chat.completion.kv",
        capability="chat.completion",
    )
    assert s.input_digest() == s.input_digest()


# --- B7: TTS frontend cache reuses cached normalization ---------------------


def test_b7_tts_raw_text_key_round_trip() -> None:
    """build_raw_text_cache_key produces stable 32-byte keys."""
    from octomil.runtime.native.tts_frontend_cache import (
        build_raw_text_cache_key,
    )

    k1 = build_raw_text_cache_key(
        model_digest_hex="0" * 64,
        raw_text="hello world",
        voice="0",
        speed_x1000=1000,
        language="en-US",
        adapter_version="test",
    )
    k2 = build_raw_text_cache_key(
        model_digest_hex="0" * 64,
        raw_text="hello world",
        voice="0",
        speed_x1000=1000,
        language="en-US",
        adapter_version="test",
    )
    assert k1 == k2
    assert len(k1) == 32

    # Different raw text → different key.
    k3 = build_raw_text_cache_key(
        model_digest_hex="0" * 64,
        raw_text="goodbye",
        voice="0",
        speed_x1000=1000,
        language="en-US",
        adapter_version="test",
    )
    assert k1 != k3


def test_b7_tts_raw_text_key_differs_from_normalized_key() -> None:
    """Raw-text key MUST be in a different namespace from the normalized
    key so the two cannot collide."""
    from octomil.runtime.native.tts_frontend_cache import (
        build_frontend_cache_key,
        build_raw_text_cache_key,
    )

    # Even if raw == normalized, the keys must differ.
    raw = build_raw_text_cache_key(
        model_digest_hex="0" * 64,
        raw_text="hello",
        voice="0",
        speed_x1000=1000,
        language="en-US",
        adapter_version="t",
    )
    norm = build_frontend_cache_key(
        model_digest_hex="0" * 64,
        normalized_text="hello",
        voice="0",
        speed_x1000=1000,
        language="en-US",
        adapter_version="t",
    )
    assert raw != norm


# --- B1: embeddings_backend.embed([]) goes through validation ---------------


def test_b1_embed_empty_list_does_not_short_circuit_via_cache() -> None:
    """The cache precheck must NOT swallow embed([]). Check the
    public guard: empty input list should not produce an empty
    EmbeddingsResult with n_dim=0; instead it must reach send_embed
    which enforces INVALID_INPUT.
    """
    from octomil.errors import OctomilError
    from octomil.runtime.native.embeddings_backend import (
        NativeEmbeddingsBackend,
    )

    backend = NativeEmbeddingsBackend()
    backend._runtime = MagicMock()
    backend._model = MagicMock()
    backend._model_name = "test-model"
    backend._cache_manager = MagicMock()
    backend._default_deadline_ms = 30_000

    # send_embed should be reached (not short-circuited by cache
    # precheck). We make open_session raise an explicit INVALID_INPUT
    # so the test does not depend on full runtime wiring; the
    # important property is that the cache precheck did NOT return
    # early.
    # NativeRuntimeError is exported from the loader module the
    # backend imports from; locate it via the backend.
    from octomil.runtime.native.embeddings_backend import NativeRuntimeError

    err = NativeRuntimeError(status=2, message="invalid empty input")
    backend._runtime.open_session.side_effect = err

    with pytest.raises((OctomilError, NativeRuntimeError)):
        backend.embed([])

    # If the cache precheck had short-circuited, we'd return an
    # EmbeddingsResult and never call open_session.
    assert backend._runtime.open_session.called, "embed([]) silently returned via cache precheck — Codex B1 regression"
