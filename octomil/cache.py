"""KV cache persistence and prefix caching for inference backends.

Provides LRU-evicted KV cache storage keyed by token sequence hashes,
enabling multi-turn conversations and repeated system prompts to skip
recomputation of already-processed prefixes.

Usage::

    cache = KVCacheManager(max_cache_size_mb=2048)
    hit = cache.get_cached_prefix(tokens)
    if hit:
        # resume generation from hit.prefix_length with hit.kv_state
        ...
    cache.store_prefix(tokens, kv_state)
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedPrefix:
    """A cached KV state for a token prefix."""

    prefix_length: int
    kv_state: Any
    created_at: float


@dataclass
class CacheStats:
    """Aggregate cache statistics."""

    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    entries: int = 0
    memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_token_prefix(tokens: list[int], length: int) -> str:
    """Hash the first *length* tokens into a compact cache key.

    Uses SHA-256 truncated to 16 hex chars for fast lookup with
    negligible collision probability in practice.
    """
    # Encode token ids as little-endian int32 bytes for speed
    raw = b"".join(t.to_bytes(4, "little", signed=True) for t in tokens[:length])
    return hashlib.sha256(raw).hexdigest()[:16]


def _estimate_size_bytes(obj: Any) -> int:
    """Best-effort size estimate for arbitrary Python objects.

    For numpy / mlx arrays this uses nbytes; for everything else falls
    back to sys.getsizeof with a recursive walk of common containers.
    """
    if obj is None:
        return 0

    # numpy / mlx arrays expose .nbytes
    if hasattr(obj, "nbytes"):
        return int(obj.nbytes)

    # Tuples / lists of arrays (common KV cache shape)
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_size_bytes(item) for item in obj)

    # Dicts
    if isinstance(obj, dict):
        return sum(
            _estimate_size_bytes(k) + _estimate_size_bytes(v) for k, v in obj.items()
        )

    return sys.getsizeof(obj)


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Internal bookkeeping for a single cached prefix."""

    kv_state: Any
    prefix_length: int
    created_at: float
    size_bytes: int


class KVCacheManager:
    """LRU cache for KV states keyed by token-sequence prefix hashes.

    Thread-safe: all public methods are serialised via an internal lock.

    Parameters
    ----------
    max_cache_size_mb:
        Maximum total size of cached KV states in megabytes.  When the
        cache exceeds this limit, the least-recently-used entries are
        evicted.
    """

    def __init__(self, max_cache_size_mb: int = 2048) -> None:
        self._max_bytes = max_cache_size_mb * 1024 * 1024
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._current_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached_prefix(self, tokens: list[int]) -> Optional[CachedPrefix]:
        """Find the longest cached prefix matching the start of *tokens*.

        Returns a ``CachedPrefix`` on hit (and promotes the entry in the
        LRU), or ``None`` on miss.
        """
        with self._lock:
            # Walk from the full token list down to length-1, checking
            # progressively shorter prefixes.  We keep a minimum prefix
            # length of 4 tokens to avoid degenerate matches.
            for length in range(len(tokens), 3, -1):
                key = _hash_token_prefix(tokens, length)
                entry = self._entries.get(key)
                if entry is not None:
                    # Promote to most-recently-used
                    self._entries.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache HIT: prefix_length=%d, key=%s", length, key)
                    return CachedPrefix(
                        prefix_length=entry.prefix_length,
                        kv_state=entry.kv_state,
                        created_at=entry.created_at,
                    )

            self._misses += 1
            return None

    def store_prefix(self, tokens: list[int], kv_state: Any) -> None:
        """Store a KV state for the given token prefix.

        If the key already exists it is replaced (and the entry promoted).
        After storing, the cache is evicted down to ``max_cache_size_mb``.
        """
        length = len(tokens)
        if length < 4:
            return  # too short to be useful

        key = _hash_token_prefix(tokens, length)
        size = _estimate_size_bytes(kv_state)
        now = time.time()

        with self._lock:
            # Remove existing entry with same key if present
            if key in self._entries:
                old = self._entries.pop(key)
                self._current_bytes -= old.size_bytes

            entry = _CacheEntry(
                kv_state=kv_state,
                prefix_length=length,
                created_at=now,
                size_bytes=size,
            )
            self._entries[key] = entry
            self._current_bytes += size

            self._evict_unlocked()

    def evict(self) -> None:
        """Manually trigger LRU eviction to fit within the size limit."""
        with self._lock:
            self._evict_unlocked()

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                hit_rate=self._hits / total if total > 0 else 0.0,
                entries=len(self._entries),
                memory_mb=self._current_bytes / (1024 * 1024),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_unlocked(self) -> None:
        """Evict LRU entries until cache fits within the size budget.

        Caller **must** hold ``self._lock``.
        """
        while self._current_bytes > self._max_bytes and self._entries:
            _key, entry = self._entries.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            logger.debug("Cache EVICT: key=%s, freed=%d bytes", _key, entry.size_bytes)
