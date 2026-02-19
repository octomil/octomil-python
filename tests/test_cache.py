"""Tests for edgeml.cache — KV cache manager with LRU eviction and prefix matching."""

from __future__ import annotations


import pytest

from edgeml.cache import (
    CachedPrefix,
    CacheStats,
    KVCacheManager,
    _estimate_size_bytes,
    _hash_token_prefix,
)


# ---------------------------------------------------------------------------
# Token hashing
# ---------------------------------------------------------------------------


class TestHashTokenPrefix:
    def test_deterministic(self):
        tokens = [1, 2, 3, 4, 5]
        h1 = _hash_token_prefix(tokens, 5)
        h2 = _hash_token_prefix(tokens, 5)
        assert h1 == h2

    def test_different_prefix_lengths(self):
        tokens = [1, 2, 3, 4, 5]
        h4 = _hash_token_prefix(tokens, 4)
        h5 = _hash_token_prefix(tokens, 5)
        assert h4 != h5

    def test_different_tokens(self):
        h1 = _hash_token_prefix([1, 2, 3, 4], 4)
        h2 = _hash_token_prefix([5, 6, 7, 8], 4)
        assert h1 != h2

    def test_same_prefix_different_suffix(self):
        """Tokens that share a prefix should hash the prefix identically."""
        tokens_a = [10, 20, 30, 40, 50]
        tokens_b = [10, 20, 30, 40, 99]
        assert _hash_token_prefix(tokens_a, 4) == _hash_token_prefix(tokens_b, 4)

    def test_hash_length(self):
        h = _hash_token_prefix([1, 2, 3, 4], 4)
        assert len(h) == 16  # SHA-256 truncated to 16 hex chars

    def test_negative_tokens(self):
        """Negative token IDs should be handled without error."""
        h = _hash_token_prefix([-1, -2, 3, 4], 4)
        assert isinstance(h, str) and len(h) == 16


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


class TestEstimateSizeBytes:
    def test_none(self):
        assert _estimate_size_bytes(None) == 0

    def test_list_of_ints(self):
        size = _estimate_size_bytes([1, 2, 3])
        assert size > 0

    def test_dict(self):
        size = _estimate_size_bytes({"a": [1, 2], "b": [3, 4]})
        assert size > 0

    def test_ndarray_nbytes(self):
        """Objects with .nbytes attribute should use that."""

        class FakeArray:
            nbytes = 4096

        assert _estimate_size_bytes(FakeArray()) == 4096

    def test_nested_structure(self):
        """Lists of objects with .nbytes should sum correctly."""

        class FakeArray:
            nbytes = 100

        total = _estimate_size_bytes([FakeArray(), FakeArray()])
        assert total == 200


# ---------------------------------------------------------------------------
# KVCacheManager — basic operations
# ---------------------------------------------------------------------------


class TestKVCacheManagerBasic:
    def test_miss_on_empty_cache(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        result = cache.get_cached_prefix([1, 2, 3, 4, 5])
        assert result is None

    def test_store_and_retrieve(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        tokens = [10, 20, 30, 40, 50]
        fake_kv = {"layer_0": [1.0, 2.0]}

        cache.store_prefix(tokens, fake_kv)
        result = cache.get_cached_prefix(tokens)

        assert result is not None
        assert isinstance(result, CachedPrefix)
        assert result.prefix_length == 5
        assert result.kv_state == fake_kv
        assert result.created_at > 0

    def test_miss_for_different_tokens(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3, 4, 5], "kv_state_a")

        result = cache.get_cached_prefix([9, 8, 7, 6, 5])
        assert result is None

    def test_prefix_match(self):
        """Storing tokens [A,B,C,D,E] should be retrievable from [A,B,C,D,E,F]."""
        cache = KVCacheManager(max_cache_size_mb=10)
        prefix = [100, 200, 300, 400, 500]
        cache.store_prefix(prefix, "cached_state")

        longer = prefix + [600, 700]
        result = cache.get_cached_prefix(longer)
        assert result is not None
        assert result.prefix_length == 5
        assert result.kv_state == "cached_state"

    def test_longest_prefix_wins(self):
        """When multiple prefixes match, the longest should be returned."""
        cache = KVCacheManager(max_cache_size_mb=10)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Store a short prefix and a long prefix
        cache.store_prefix(tokens[:5], "short_kv")
        cache.store_prefix(tokens[:7], "long_kv")

        result = cache.get_cached_prefix(tokens)
        assert result is not None
        # Should match the 7-token prefix (the longest stored prefix
        # that matches the start of the query)
        assert result.prefix_length == 7
        assert result.kv_state == "long_kv"

    def test_too_short_tokens_not_stored(self):
        """Tokens shorter than 4 should not be stored."""
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3], "tiny")
        assert cache.stats().entries == 0

    def test_too_short_tokens_miss(self):
        """Querying with tokens shorter than 4 should return None."""
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3, 4, 5], "state")
        result = cache.get_cached_prefix([1, 2, 3])
        assert result is None


# ---------------------------------------------------------------------------
# KVCacheManager — stats
# ---------------------------------------------------------------------------


class TestKVCacheManagerStats:
    def test_initial_stats(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        s = cache.stats()
        assert isinstance(s, CacheStats)
        assert s.hits == 0
        assert s.misses == 0
        assert s.hit_rate == 0.0
        assert s.entries == 0
        assert s.memory_mb == 0.0

    def test_stats_after_hits_and_misses(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        tokens = [1, 2, 3, 4, 5]
        cache.store_prefix(tokens, "state")

        # One hit
        cache.get_cached_prefix(tokens)
        # Two misses
        cache.get_cached_prefix([9, 8, 7, 6])
        cache.get_cached_prefix([5, 4, 3, 2])

        s = cache.stats()
        assert s.hits == 1
        assert s.misses == 2
        assert s.hit_rate == pytest.approx(1 / 3)
        assert s.entries == 1

    def test_stats_memory_mb(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        # Store something with known size
        cache.store_prefix([1, 2, 3, 4, 5], [1, 2, 3])
        s = cache.stats()
        assert s.memory_mb > 0


# ---------------------------------------------------------------------------
# KVCacheManager — LRU eviction
# ---------------------------------------------------------------------------


class TestKVCacheManagerEviction:
    def test_eviction_on_size_limit(self):
        """When cache exceeds max size, oldest entries are evicted."""
        # Use a very small cache (1 byte effective)
        cache = KVCacheManager(max_cache_size_mb=0)
        # max_bytes = 0 so everything should be evicted immediately
        cache.store_prefix([1, 2, 3, 4], "state_a")
        assert cache.stats().entries == 0

    def test_lru_order(self):
        """Most recently accessed entries should survive eviction."""

        class SizedObj:
            """Object with controllable size."""

            def __init__(self, size: int) -> None:
                self.nbytes = size

        # 1 KB cache limit
        cache = KVCacheManager(max_cache_size_mb=0)
        # Override max bytes directly for fine-grained test
        cache._max_bytes = 1024

        # Store two 400-byte entries (total 800, under 1024)
        cache.store_prefix([1, 2, 3, 4], SizedObj(400))
        cache.store_prefix([5, 6, 7, 8], SizedObj(400))
        assert cache.stats().entries == 2

        # Store a third 400-byte entry — pushes total to 1200,
        # which should evict the first (LRU)
        cache.store_prefix([9, 10, 11, 12], SizedObj(400))
        assert cache.stats().entries == 2

        # First entry should be evicted
        assert cache.get_cached_prefix([1, 2, 3, 4]) is None
        # Remaining entries should still be present
        assert cache.get_cached_prefix([5, 6, 7, 8]) is not None
        assert cache.get_cached_prefix([9, 10, 11, 12]) is not None

    def test_manual_evict(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3, 4], "state")
        assert cache.stats().entries == 1

        # Shrink max and manually evict
        cache._max_bytes = 0
        cache.evict()
        assert cache.stats().entries == 0


# ---------------------------------------------------------------------------
# KVCacheManager — clear
# ---------------------------------------------------------------------------


class TestKVCacheManagerClear:
    def test_clear(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3, 4], "a")
        cache.store_prefix([5, 6, 7, 8], "b")
        assert cache.stats().entries == 2

        cache.clear()
        s = cache.stats()
        assert s.entries == 0
        assert s.memory_mb == 0.0

    def test_clear_preserves_stats(self):
        """Clear should not reset hit/miss counters."""
        cache = KVCacheManager(max_cache_size_mb=10)
        cache.store_prefix([1, 2, 3, 4], "state")
        cache.get_cached_prefix([1, 2, 3, 4])  # hit
        cache.get_cached_prefix([9, 9, 9, 9])  # miss

        cache.clear()
        s = cache.stats()
        assert s.hits == 1
        assert s.misses == 1
        assert s.entries == 0


# ---------------------------------------------------------------------------
# KVCacheManager — replacing existing keys
# ---------------------------------------------------------------------------


class TestKVCacheManagerReplace:
    def test_replace_existing_key(self):
        cache = KVCacheManager(max_cache_size_mb=10)
        tokens = [1, 2, 3, 4, 5]

        cache.store_prefix(tokens, "old_state")
        cache.store_prefix(tokens, "new_state")

        result = cache.get_cached_prefix(tokens)
        assert result is not None
        assert result.kv_state == "new_state"
        assert cache.stats().entries == 1
