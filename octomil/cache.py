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


def _hash_token_prefix(tokens: list[int], length: int) -> int:
    """Hash the first *length* tokens into a compact cache key.

    Uses Python's built-in tuple hash (C-level, ~100x faster than SHA-256)
    which is sufficient for an in-process LRU cache.
    """
    return hash(tuple(tokens[:length]))


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
        return sum(_estimate_size_bytes(k) + _estimate_size_bytes(v) for k, v in obj.items())

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
    max_entries:
        Maximum number of entries in the cache.  ``None`` means unlimited
        (eviction is based on size only).  When set, entry-count eviction
        fires *before* size-based eviction.
    """

    def __init__(self, max_cache_size_mb: int = 2048, max_entries: int | None = None) -> None:
        self._max_bytes = max_cache_size_mb * 1024 * 1024
        self._max_entries = max_entries
        self._entries: OrderedDict[int, _CacheEntry] = OrderedDict()
        self._current_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()
        self._cached_lengths: set[int] = set()  # prefix lengths with entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached_prefix(self, tokens: list[int]) -> Optional[CachedPrefix]:
        """Find the longest cached prefix matching the start of *tokens*.

        Returns a ``CachedPrefix`` on hit (and promotes the entry in the
        LRU), or ``None`` on miss.
        """
        with self._lock:
            if not self._cached_lengths:
                self._misses += 1
                return None

            # Only try lengths that actually exist in cache, longest first
            n = len(tokens)
            for length in sorted((cl for cl in self._cached_lengths if 4 <= cl <= n), reverse=True):
                key = _hash_token_prefix(tokens, length)
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache HIT: prefix_length=%d", length)
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
            self._cached_lengths.add(length)

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
            self._cached_lengths.clear()

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
        """Evict LRU entries until cache fits within the size and entry budgets.

        Entry-count eviction fires before size-based eviction.
        Caller **must** hold ``self._lock``.
        """
        evicted = False

        # Entry count eviction (fires first)
        if self._max_entries is not None:
            while len(self._entries) > self._max_entries and self._entries:
                _key, entry = self._entries.popitem(last=False)
                self._current_bytes -= entry.size_bytes
                evicted = True
                logger.debug(
                    "Cache EVICT (max_entries): key=%s, freed=%d bytes",
                    _key,
                    entry.size_bytes,
                )

        # Size-based eviction
        while self._current_bytes > self._max_bytes and self._entries:
            _key, entry = self._entries.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            evicted = True
            logger.debug("Cache EVICT: key=%s, freed=%d bytes", _key, entry.size_bytes)

        if evicted:
            self._cached_lengths = {e.prefix_length for e in self._entries.values()}


# ---------------------------------------------------------------------------
# Native runtime cache clear / introspection — Lane G skeleton
# ---------------------------------------------------------------------------
# These four functions delegate to octomil.runtime.native.cache which
# calls the real C ABI entry points (oct_runtime_cache_clear_all etc.).
#
# All four are stubs until Lanes B/C/F wire real cache implementations.
# While stubbed, they raise CacheNotImplementedError.  introspect() is
# special: it also attaches a `snapshot` attribute to the raised
# exception so tests can inspect the stub-shaped output.
#
# Usage::
#
#     from octomil.cache import clear_all, clear_capability, clear_scope, introspect
#     from octomil.cache import SCOPE_RUNTIME, CacheNotImplementedError
#
#     try:
#         snap = introspect(my_runtime_handle)
#     except CacheNotImplementedError as exc:
#         # stub path — safe to inspect exc.snapshot.is_stub
#         snap = exc.snapshot
#
# Capability "cache.introspect" registered in octomil-contracts v1.25.0
# (#129); cache scope codes match enums/cache_scope.yaml from v1.24.0 (#123).


def clear_all(runtime_handle: Any) -> None:
    """Clear ALL caches across ALL capabilities.

    Idempotent — calling on an empty cache is always safe.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.

    Raises:
        CacheNotImplementedError: Stub (OCT_STATUS_UNSUPPORTED).
        NativeRuntimeError: Hard ABI error.
        ValueError: ``runtime_handle`` is ``None``.

    .. note::
        Capability ``cache.introspect`` is registered in
        octomil-contracts v1.25.0 (#129); cache scope codes
        ``request|session|runtime|app`` match
        ``enums/cache_scope.yaml`` from octomil-contracts v1.24.0
        (#123).
    """
    from octomil.runtime.native.cache import clear_all as _clear_all  # lazy

    _clear_all(runtime_handle)


def clear_capability(runtime_handle: Any, capability_name: str) -> None:
    """Clear caches for a single canonical capability name.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.
        capability_name: Canonical string, e.g. ``"chat.completion"``.

    Raises:
        CacheNotImplementedError: Stub.
        NativeRuntimeError: Hard ABI error.
        ValueError: ``runtime_handle`` is ``None`` or
            ``capability_name`` is empty.

    .. note::
        Capability ``cache.introspect`` is registered in
        octomil-contracts v1.25.0 (#129); cache scope codes
        ``request|session|runtime|app`` match
        ``enums/cache_scope.yaml`` from octomil-contracts v1.24.0
        (#123).
    """
    from octomil.runtime.native.cache import (  # lazy
        clear_capability as _clear_capability,
    )

    _clear_capability(runtime_handle, capability_name)


def clear_scope(runtime_handle: Any, scope: int) -> None:
    """Clear caches at a given scope level.

    Broader scopes subsume narrower: ``SCOPE_APP > SCOPE_RUNTIME >
    SCOPE_SESSION > SCOPE_REQUEST``.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.
        scope: One of :data:`SCOPE_REQUEST`, :data:`SCOPE_SESSION`,
            :data:`SCOPE_RUNTIME`, :data:`SCOPE_APP`.

    Raises:
        CacheNotImplementedError: Stub.
        NativeRuntimeError: Hard ABI error.
        ValueError: ``runtime_handle`` is ``None`` or ``scope`` is
            not a valid scope constant.

    .. note::
        Capability ``cache.introspect`` is registered in
        octomil-contracts v1.25.0 (#129); cache scope codes
        ``request|session|runtime|app`` match
        ``enums/cache_scope.yaml`` from octomil-contracts v1.24.0
        (#123).
    """
    from octomil.runtime.native.cache import clear_scope as _clear_scope  # lazy

    _clear_scope(runtime_handle, scope)


def introspect(runtime_handle: Any) -> "CacheSnapshot":  # noqa: F821
    """Return a privacy-safe snapshot of current cache state.

    The returned :class:`CacheSnapshot` contains **only** bounded
    numeric / enum fields.  No keys, hashes, paths, or text.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.

    Returns:
        A :class:`CacheSnapshot`.

    Raises:
        CacheNotImplementedError: Stub — ``exc.snapshot`` carries the
            stub-shaped ``CacheSnapshot`` (``is_stub=True``,
            ``entries=[]``).
        NativeRuntimeError: Hard ABI error.
        ValueError: ``runtime_handle`` is ``None`` or the ABI returned
            JSON that fails the privacy schema check.

    .. note::
        Capability ``cache.introspect`` is registered in
        octomil-contracts v1.25.0 (#129); cache scope codes
        ``request|session|runtime|app`` match
        ``enums/cache_scope.yaml`` from octomil-contracts v1.24.0
        (#123).
    """
    from octomil.runtime.native.cache import introspect as _introspect  # lazy

    return _introspect(runtime_handle)


# ---------------------------------------------------------------------------
# Re-export scope constants and error types for ergonomic import
# ---------------------------------------------------------------------------
# Cache scope codes match enums/cache_scope.yaml from octomil-contracts
# v1.24.0 (#123); capability "cache.introspect" registered in v1.25.0 (#129).

from octomil.runtime.native.cache import (  # noqa: E402, I001
    SCOPE_APP as SCOPE_APP,
    SCOPE_REQUEST as SCOPE_REQUEST,
    SCOPE_RUNTIME as SCOPE_RUNTIME,
    SCOPE_SESSION as SCOPE_SESSION,
    CacheEntrySnapshot as CacheEntrySnapshot,
    CacheNotImplementedError as CacheNotImplementedError,
    CacheSnapshot as CacheSnapshot,
)
