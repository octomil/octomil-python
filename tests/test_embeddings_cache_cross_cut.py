"""test_embeddings_cache_cross_cut.py — regression tests for the
v0.1.11 Lane B/H cross-cut cache-impl bugs surfaced by the Codex Lane
H sweep on PR #569.

Each test is named after the Codex blocking-issue id (B1 / B2 / B3) so
the regression mapping is one-glance obvious.  See
``fix(cache-impl): post-hoc Lane B/C cross-cut bugs from Lane H sweep``
PR body for the full bug list and per-bug fix summary.
"""

from __future__ import annotations

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
