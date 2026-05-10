"""test_embeddings_cache_cross_cut.py — regression tests for the
v0.1.11 Lane B/H cross-cut cache-impl bugs surfaced by the Codex Lane
H sweep on PR #569.

Each test is named after the Codex blocking-issue id (B1 / B2 / B3) so
the regression mapping is one-glance obvious.  See
``fix(cache-impl): post-hoc Lane B/C cross-cut bugs from Lane H sweep``
PR body for the full bug list and per-bug fix summary.
"""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend
from octomil.runtime.native.embeddings_cache import EmbeddingsLruCache

# ---------------------------------------------------------------------------
# B3 — defensive copy on get/put.
# Codex eval round-1: octomil/runtime/native/embeddings_cache.py:215
# "Cache get returns the stored mutable list object directly; the
# backend stores the same vectors it returns to callers."
# ---------------------------------------------------------------------------


def test_b3_get_returns_defensive_copy_of_stored_list() -> None:
    """Caller mutation of the returned list MUST NOT affect the cache."""
    cache = EmbeddingsLruCache()
    key = b"\x00" * 32
    cache.put(key, [1.0, 2.0, 3.0], 3)

    out_first = cache.get(key)
    assert isinstance(out_first, list)
    assert out_first == [1.0, 2.0, 3.0]
    out_first.append(999.0)  # caller mutation
    out_first[0] = -1.0

    out_second = cache.get(key)
    assert out_second == [1.0, 2.0, 3.0], "cache.get must return a defensive copy"


def test_b3_put_snapshots_caller_list() -> None:
    """Caller mutation of the put-time reference MUST NOT affect the cache."""
    cache = EmbeddingsLruCache()
    key = b"\x01" * 32
    src = [10, 20, 30]
    cache.put(key, src, 3)
    src.append(999)  # caller mutates after put
    src[0] = -1

    out = cache.get(key)
    assert out == [10, 20, 30], "cache.put must snapshot the caller's list; " "post-put mutation leaked into cache"


def test_b3_two_gets_return_independent_lists() -> None:
    """Two callers must not be able to corrupt each other's vectors."""
    cache = EmbeddingsLruCache()
    key = b"\x02" * 32
    cache.put(key, [0.1, 0.2, 0.3], 3)

    a = cache.get(key)
    b = cache.get(key)
    assert a is not b, "each get() must return a fresh list, not a shared reference"
    assert isinstance(a, list)
    assert isinstance(b, list)
    a[0] = 999.0
    assert b == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# B1 — empty input list must reject INVALID_INPUT before cache pre-check.
# Codex eval round-1: octomil/runtime/native/embeddings_backend.py:357
# "The result-cache precheck treats an empty input list as an all-hit
#  batch and returns an empty EmbeddingsResult before send_embed
#  validates it."
# ---------------------------------------------------------------------------


def test_b1_embed_empty_list_raises_invalid_input() -> None:
    """embed([]) MUST raise INVALID_INPUT regardless of cache state.

    The bug: the result-cache pre-check iterates over inputs and, for
    a zero-length list, declares all_hit=True with zero entries —
    returning EmbeddingsResult(n_dim=0, pooling_type=0, embeddings=[])
    instead of letting send_embed's INVALID_INPUT validator run.

    This test exercises the pre-runtime validation path: it does NOT
    require a loaded model since the empty-input guard runs after
    deadline validation but before any runtime/session call.
    """
    backend = NativeEmbeddingsBackend()
    # NOTE: load_model is intentionally NOT called.  The bug surfaces
    # in the cache pre-check path which sits before runtime open.  We
    # need to skirt the load_model gate (which raises
    # RUNTIME_UNAVAILABLE).  Set the runtime/model handles to non-None
    # sentinels so the gate clears; the empty-input guard below the
    # gate is what we are exercising.
    backend._runtime = object()  # type: ignore[assignment]
    backend._model = object()
    try:
        with pytest.raises(OctomilError) as ei:
            backend.embed([])
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT
        assert "non-empty" in str(ei.value)
    finally:
        # Clear sentinels so backend.close()'s teardown logs do not
        # fire AttributeError noise on the test's gc.
        backend._runtime = None
        backend._model = None
