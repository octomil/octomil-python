"""embeddings_cache.py — Lane B v0.1.11 Python-side embeddings cache.

Two caches, both controlled by env flags and a per-session CachePolicy:

  OCT_EMBEDDINGS_TOKENIZATION_CACHE (default ON)
    Caches tokenization output keyed by sha256(model_digest |
    normalized_text_hash | tokenizer_version | adapter_version).
    Stores token-ID lists. Token IDs are treated as opaque values;
    they are NOT logged or exposed in metrics.

  OCT_EMBEDDINGS_RESULT_CACHE (default OFF)
    Caches embedding float vectors keyed by sha256(model_digest |
    normalized_text_hash | embedding_options | adapter_version).
    Enabled only when the env var is "1" AND the session's
    CachePolicy.result_cache_allowed is True.

PRIVACY INVARIANTS (enforced by-construction):
  1. The cache API NEVER receives raw text.  The caller hashes the
     normalized text with SHA-256 before calling any cache method.
  2. Token IDs and float vectors are stored as opaque Python objects;
     they are never str-ified, logged, or included in any event
     payload or metric.
  3. Cache keys are 32-byte SHA-256 digests.  They are never printed
     or emitted in any event.
  4. The only observable external signal is a boolean hit/miss, surfaced
     as metrics embeddings.cache_hit_total and cache.miss_total.

METRICS (canonical — registered in octomil-contracts Lane A v1.24.0):
  embeddings.cache_hit_total — counter, incremented on each cache hit
                                (capability-prefixed cache metric per
                                schemas/core/runtime_metric.json).
  cache.miss_total           — counter, incremented on each cache miss
                                (generic cache metric; no capability-
                                prefixed embeddings miss exists).
  cache.lookup_ms            — gauge, lookup latency in ms (most recent
                                lookup, not cumulative).

PROTOTYPE STATUS:
  v0.1.11 Lane B prototype.  Metric names are now canonical (Lane A
  merged at contracts cfffaf8 / runtime 3bcb061).  No envelope label
  fields are emitted from these helpers — the canonical OCT_EVENT_METRIC
  envelope reserves a closed allowlist (request_id, route_id, trace_id,
  engine_version, adapter_version, accelerator, artifact_digest,
  cache_was_hit) and that wiring is left to the SDK telemetry sink at
  the C ABI boundary, not these Python emit helpers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default LRU capacity in bytes (logical estimate: 4 * n_elements).
DEFAULT_MAX_BYTES: int = 32 * 1024 * 1024  # 32 MiB

#: Cache key length: 32 bytes (SHA-256 digest).
KEY_BYTES: int = 32

# ---------------------------------------------------------------------------
# Env-flag queries
# ---------------------------------------------------------------------------


def tokenization_cache_enabled() -> bool:
    """Return True when OCT_EMBEDDINGS_TOKENIZATION_CACHE is unset or '1'.

    Default ON: tokenization cache is safe to enable by default because
    token IDs are no more sensitive than the text the caller controls,
    and this cache never stores text.
    """
    v = os.environ.get("OCT_EMBEDDINGS_TOKENIZATION_CACHE")
    if v is None:
        return True
    return v == "1"


def result_cache_enabled() -> bool:
    """Return True when OCT_EMBEDDINGS_RESULT_CACHE == '1'.

    Default OFF: result vectors encode semantic content and require
    explicit operator opt-in plus a matching CachePolicy.
    """
    return os.environ.get("OCT_EMBEDDINGS_RESULT_CACHE") == "1"


# ---------------------------------------------------------------------------
# CachePolicy — session-level knob
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachePolicy:
    """Session-level cache policy.

    Attributes
    ----------
    tokenization_cache_allowed
        If True and OCT_EMBEDDINGS_TOKENIZATION_CACHE is enabled,
        the tokenization cache is consulted for this session.
        Default True.

    result_cache_allowed
        If True AND OCT_EMBEDDINGS_RESULT_CACHE == '1', the result
        cache is consulted.  Default False (requires explicit opt-in).
    """

    tokenization_cache_allowed: bool = True
    result_cache_allowed: bool = False

    @classmethod
    def strict(cls) -> "CachePolicy":
        """Strict privacy mode: both caches disabled.  The default
        when no explicit policy is supplied."""
        return cls(tokenization_cache_allowed=False, result_cache_allowed=False)

    @classmethod
    def tokenization_only(cls) -> "CachePolicy":
        """Enable tokenization cache only (the safe default for most
        use-cases where token IDs are not considered sensitive)."""
        return cls(tokenization_cache_allowed=True, result_cache_allowed=False)

    @classmethod
    def policy_allowed(cls) -> "CachePolicy":
        """Enable both caches.  Use only when your data-handling policy
        explicitly permits caching semantic embedding vectors."""
        return cls(tokenization_cache_allowed=True, result_cache_allowed=True)


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def derive_cache_key(
    model_digest: str,
    text_sha256_hex: str,
    salt: str,
) -> bytes:
    """Derive a 32-byte cache key.

    key = SHA-256(model_digest_utf8 | text_sha256_hex_utf8 | salt_utf8)

    Parameters
    ----------
    model_digest
        The 'sha256:<64-hex>' artifact digest from the model handle.
    text_sha256_hex
        Hex string (64 chars) of SHA-256(normalized_text). The raw
        text MUST NOT be passed here; caller hashes first.
    salt
        Distinguishes cache domains (tokenization vs result) and
        carries version bytes (adapter version, embedding options).

    Returns
    -------
    bytes
        32-byte opaque cache key.  Never logged or emitted.
    """
    h = hashlib.sha256()
    h.update(model_digest.encode("utf-8"))
    h.update(text_sha256_hex.encode("utf-8"))
    h.update(salt.encode("utf-8"))
    return h.digest()  # 32 bytes


def hash_text(text: str) -> str:
    """Compute SHA-256 of normalized text; return as hex string.

    PRIVACY: this is the ONLY place where raw text is allowed to touch
    the cache layer.  The returned hex string (not the text) is what
    flows into derive_cache_key().
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class EmbeddingsLruCache:
    """Thread-safe in-process LRU cache for embeddings data.

    Capacity is measured in logical bytes: 4 * len(payload) for
    both int lists (token IDs) and float lists (vectors).

    Privacy: payloads are stored as-is (list of int or list of float).
    They are never converted to strings, logged, or emitted in events.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._bytes_used = 0
        self._cache: OrderedDict[bytes, object] = OrderedDict()
        self._sizes: dict[bytes, int] = {}
        self._lock = threading.Lock()

    def get(self, key: bytes) -> Optional[object]:
        """Return the cached value or None on miss.  Promotes to MRU.

        Defensive copy: returns a fresh ``list`` constructed from the
        stored payload so caller mutation cannot corrupt the cache or
        cross-contaminate other consumers of the same key.  This is
        the v0.1.11 Lane B+H follow-up fix for the Codex-flagged
        mutable-list aliasing bug (see fix-cache-impl-cross-cut PR).
        """
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            stored = self._cache[key]
            # int and float lists are the only payload types in this
            # cache (token IDs, embedding vectors).  list(...) yields
            # a shallow copy that is independent of the stored object.
            if isinstance(stored, list):
                return list(stored)
            return stored

    def put(self, key: bytes, value: object, n_elements: int) -> None:
        """Insert or replace.  Evicts LRU entries to satisfy capacity.

        Defensive copy: stores a fresh ``list`` so subsequent caller
        mutation of the put-time reference cannot affect the cached
        entry.  Pairs with the defensive copy in :meth:`get`.
        """
        byte_cost = 4 * n_elements
        # Snapshot the payload before taking the lock.  list(value)
        # is independent of the caller's reference; mutating either
        # side after put() returns leaves the cached entry untouched.
        if isinstance(value, list):
            stored_value: object = list(value)
        else:
            stored_value = value
        with self._lock:
            if key in self._cache:
                self._bytes_used -= self._sizes[key]
                del self._cache[key]
                del self._sizes[key]
            # Evict LRU until there is room.
            while self._cache and self._bytes_used + byte_cost > self._max_bytes:
                old_key, _ = self._cache.popitem(last=False)
                self._bytes_used -= self._sizes.pop(old_key, 0)
            self._cache[key] = stored_value
            self._sizes[key] = byte_cost
            self._bytes_used += byte_cost

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._bytes_used = 0

    def size(self) -> int:
        """Number of entries currently held."""
        with self._lock:
            return len(self._cache)

    def bytes_used(self) -> int:
        """Logical byte consumption."""
        with self._lock:
            return self._bytes_used


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


def cache_get_tokens(
    cache: EmbeddingsLruCache,
    key: bytes,
) -> Optional[list[int]]:
    """Retrieve token IDs from the tokenization cache.

    Returns None on miss.  Caller MUST treat returned token IDs as
    opaque — do not log or emit them.
    """
    return cache.get(key)  # type: ignore[return-value]


def cache_put_tokens(
    cache: EmbeddingsLruCache,
    key: bytes,
    token_ids: list[int],
) -> None:
    """Store token IDs in the tokenization cache.

    PRIVACY: token_ids are opaque — this function does not log them.
    """
    cache.put(key, token_ids, len(token_ids))


def cache_get_vector(
    cache: EmbeddingsLruCache,
    key: bytes,
    *,
    privacy_allowed: bool,
) -> Optional[list[float]]:
    """Retrieve an embedding vector from the result cache.

    Returns None on miss OR when privacy_allowed is False.
    The returned floats are never logged.
    """
    if not privacy_allowed:
        return None
    return cache.get(key)  # type: ignore[return-value]


def cache_put_vector(
    cache: EmbeddingsLruCache,
    key: bytes,
    vector: list[float],
    *,
    privacy_allowed: bool,
) -> None:
    """Store an embedding vector in the result cache.

    PRIVACY: vector bytes are opaque — this function does not log them.
    No-ops when privacy_allowed is False.
    """
    if not privacy_allowed:
        return
    cache.put(key, vector, len(vector))


# ---------------------------------------------------------------------------
# Result-cache entry: vector + replayable metadata.
# ---------------------------------------------------------------------------
# v0.1.11 Lane B+H follow-up: when the result cache replaces a runtime
# call we MUST replay the original per-input metadata (n_dim,
# pooling_type, is_normalized, n_input_tokens).  Storing only the
# vector caused the embeddings backend to return n_dim=0,
# pooling_type=0, total_tokens=0 on full-cache hit, which silently
# changes API-visible usage and pooling semantics for identical
# requests.  This entry struct + helper pair fixes that.
# Privacy: the metadata fields are integers / bools.  No raw text or
# vector content is exposed beyond what the cache already holds.


@dataclass(frozen=True)
class CachedVectorEntry:
    """One result-cache entry: the embedding vector plus the runtime
    metadata needed to reconstruct an :class:`EmbeddingsResult`
    identical to the cold (non-cached) response.

    ``vector`` is stored as a defensive copy at put time and returned
    as a defensive copy at get time (see :class:`EmbeddingsLruCache`).
    """

    vector: list[float]
    n_dim: int
    pooling_type: int
    is_normalized: bool
    n_input_tokens: int


def cache_get_vector_entry(
    cache: EmbeddingsLruCache,
    key: bytes,
    *,
    privacy_allowed: bool,
) -> Optional[CachedVectorEntry]:
    """Retrieve a full :class:`CachedVectorEntry` (vector + metadata).

    Returns None on miss OR when ``privacy_allowed`` is False.

    Defensive copy: rebuilds the entry with a fresh ``list`` for
    ``vector`` so caller mutation of the returned vector cannot
    corrupt the cached payload.  The dataclass itself is frozen so
    only the vector field needs copying.
    """
    if not privacy_allowed:
        return None
    raw = cache.get(key)
    if raw is None:
        return None
    if isinstance(raw, CachedVectorEntry):
        # cache.get already defensive-copies plain lists; for the
        # dataclass we return a fresh entry with a copied vector.
        return CachedVectorEntry(
            vector=list(raw.vector),
            n_dim=raw.n_dim,
            pooling_type=raw.pooling_type,
            is_normalized=raw.is_normalized,
            n_input_tokens=raw.n_input_tokens,
        )
    # Backward compat: callers that put a bare list[float] still get a
    # vector-only response; metadata fields default to 0/False.  This
    # path is only exercised by legacy put paths; new put paths use
    # ``cache_put_vector_entry`` and store the dataclass.
    if isinstance(raw, list):
        return CachedVectorEntry(
            vector=raw,  # cache.get already defensive-copied this list
            n_dim=len(raw),
            pooling_type=0,
            is_normalized=False,
            n_input_tokens=0,
        )
    return None


def cache_put_vector_entry(
    cache: EmbeddingsLruCache,
    key: bytes,
    entry: CachedVectorEntry,
    *,
    privacy_allowed: bool,
) -> None:
    """Store a :class:`CachedVectorEntry`.  No-ops when not allowed.

    Size accounting uses ``len(vector)`` — the four small int/bool
    metadata fields contribute negligibly versus the float payload.

    Defensive copy: stores the entry with a fresh ``list`` for
    ``vector`` so subsequent caller mutation cannot affect the
    cached payload.
    """
    if not privacy_allowed:
        return
    snapshot = CachedVectorEntry(
        vector=list(entry.vector),
        n_dim=entry.n_dim,
        pooling_type=entry.pooling_type,
        is_normalized=entry.is_normalized,
        n_input_tokens=entry.n_input_tokens,
    )
    cache.put(key, snapshot, len(snapshot.vector))


# ---------------------------------------------------------------------------
# EmbeddingsCacheManager — high-level wrapper for use by the backend
# ---------------------------------------------------------------------------


class EmbeddingsCacheManager:
    """Manage the two per-model embeddings caches.

    Instantiated once per NativeEmbeddingsBackend (i.e., once per
    loaded GGUF).  Cleared on model close or backend close.

    Parameters
    ----------
    model_digest
        The 'sha256:<64-hex>' artifact digest for cache key namespacing.
    adapter_version
        Adapter version string (used as part of the salt so keys are
        invalidated if the tokenizer or embedder changes).
    tok_max_bytes, result_max_bytes
        LRU capacity ceilings.  Defaults to DEFAULT_MAX_BYTES each.
    """

    def __init__(
        self,
        *,
        model_digest: str,
        adapter_version: str,
        tok_max_bytes: int = DEFAULT_MAX_BYTES,
        result_max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._model_digest = model_digest
        self._adapter_version = adapter_version
        self._tok_salt = f"tok|{adapter_version}"
        self._result_salt = f"vec|{adapter_version}"
        self._tok_cache = EmbeddingsLruCache(max_bytes=tok_max_bytes)
        self._result_cache = EmbeddingsLruCache(max_bytes=result_max_bytes)

    def close(self) -> None:
        """Clear both caches.  Call on model close or runtime close."""
        self._tok_cache.clear()
        self._result_cache.clear()

    # ------------------------------------------------------------------
    # Tokenization cache
    # ------------------------------------------------------------------

    def get_tokens(
        self,
        text: str,
        *,
        policy: CachePolicy,
    ) -> Optional[list[int]]:
        """Look up token IDs for ``text``.

        PRIVACY: ``text`` is hashed here; the hash (not the text)
        enters the cache key.  Returns None on miss or when the cache
        is disabled.
        """
        if not policy.tokenization_cache_allowed:
            return None
        if not tokenization_cache_enabled():
            return None
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._tok_salt)
        return cache_get_tokens(self._tok_cache, key)

    def put_tokens(
        self,
        text: str,
        token_ids: list[int],
        *,
        policy: CachePolicy,
    ) -> None:
        """Store token IDs for ``text``.

        PRIVACY: ``text`` is hashed here; token_ids are never logged.
        """
        if not policy.tokenization_cache_allowed:
            return
        if not tokenization_cache_enabled():
            return
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._tok_salt)
        cache_put_tokens(self._tok_cache, key, token_ids)

    # ------------------------------------------------------------------
    # Result cache
    # ------------------------------------------------------------------

    def get_vector(
        self,
        text: str,
        *,
        policy: CachePolicy,
    ) -> Optional[list[float]]:
        """Look up an embedding vector for ``text``.

        PRIVACY: ``text`` is hashed; float values are never logged.
        Returns None when the result cache is disabled (env or policy).
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return None
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        return cache_get_vector(self._result_cache, key, privacy_allowed=True)

    def put_vector(
        self,
        text: str,
        vector: list[float],
        *,
        policy: CachePolicy,
    ) -> None:
        """Store an embedding vector for ``text``.

        PRIVACY: ``text`` is hashed; float bytes are never logged.
        No-ops when the result cache is disabled.
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        cache_put_vector(self._result_cache, key, vector, privacy_allowed=True)

    # ------------------------------------------------------------------
    # Result cache — vector + replayable metadata (preferred path)
    # ------------------------------------------------------------------

    def get_vector_entry(
        self,
        text: str,
        *,
        policy: CachePolicy,
    ) -> Optional[CachedVectorEntry]:
        """Look up a full :class:`CachedVectorEntry` for ``text``.

        Returns the vector AND the metadata needed to reconstruct an
        :class:`EmbeddingsResult` identical to the cold response
        (n_dim, pooling_type, is_normalized, n_input_tokens).
        Preferred over :meth:`get_vector` for v0.1.11 cache hits.

        PRIVACY: ``text`` is hashed; vector and metadata are opaque.
        Returns None when the result cache is disabled (env or policy).
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return None
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        return cache_get_vector_entry(self._result_cache, key, privacy_allowed=True)

    def put_vector_entry(
        self,
        text: str,
        entry: CachedVectorEntry,
        *,
        policy: CachePolicy,
    ) -> None:
        """Store a :class:`CachedVectorEntry` for ``text``.

        PRIVACY: ``text`` is hashed; vector and metadata are opaque.
        No-ops when the result cache is disabled.
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        cache_put_vector_entry(self._result_cache, key, entry, privacy_allowed=True)

    # ------------------------------------------------------------------
    # Stats (opaque — for metrics, NOT for logging raw data)
    # ------------------------------------------------------------------

    def tok_cache_size(self) -> int:
        return self._tok_cache.size()

    def result_cache_size(self) -> int:
        return self._result_cache.size()

    def tok_cache_bytes(self) -> int:
        return self._tok_cache.bytes_used()

    def result_cache_bytes(self) -> int:
        return self._result_cache.bytes_used()


# ---------------------------------------------------------------------------
# Metric emission helpers
# ---------------------------------------------------------------------------
# Canonical metric names — Lane A merged at octomil-contracts cfffaf8
# (schemas/core/runtime_metric.json v1.24.0).  Names below are the closed
# enum constants and are validated by ci/validate_cache_metric_payload.py.
# `cache_layer` is intentionally NOT emitted as an envelope field — the
# canonical envelope allowlist (request_id, route_id, trace_id,
# engine_version, adapter_version, accelerator, artifact_digest,
# cache_was_hit) does not include free-form labels, and the metric NAME
# already namespaces the emission point.

_METRIC_CACHE_HIT = "embeddings.cache_hit_total"  # canonical (Lane A)
_METRIC_CACHE_MISS = "cache.miss_total"  # canonical (Lane A)
_METRIC_LOOKUP_MS = "cache.lookup_ms"  # canonical (Lane A)


def emit_cache_hit(metric_sink: object, cache_layer: str) -> None:
    """Emit embeddings.cache_hit_total metric.

    PRIVACY: only the canonical metric name + value is emitted — no
    input text, no token IDs, no vector bytes, no free-form label.
    The ``cache_layer`` argument is accepted for source-call-site
    documentation only; it is NOT included in the metric event,
    because the canonical envelope reserves a closed allowlist of
    fields and free-form labels would expand cardinality.
    Best-effort: never raises.
    """
    del cache_layer  # name-only emission; see canonical envelope
    try:
        if metric_sink is not None and hasattr(metric_sink, "emit"):
            metric_sink.emit(_METRIC_CACHE_HIT, 1.0, {})
    except Exception:  # noqa: BLE001
        pass


def emit_cache_miss(metric_sink: object, cache_layer: str) -> None:
    """Emit cache.miss_total metric.

    PRIVACY: same constraints as emit_cache_hit. ``cache_layer`` is
    intentionally not forwarded.
    Best-effort: never raises.
    """
    del cache_layer  # name-only emission; see canonical envelope
    try:
        if metric_sink is not None and hasattr(metric_sink, "emit"):
            metric_sink.emit(_METRIC_CACHE_MISS, 1.0, {})
    except Exception:  # noqa: BLE001
        pass


def emit_lookup_ms(metric_sink: object, elapsed_ms: float) -> None:
    """Emit cache.lookup_ms metric.

    PRIVACY: only the numeric latency value is emitted.
    Best-effort: never raises.
    """
    try:
        if metric_sink is not None and hasattr(metric_sink, "emit"):
            metric_sink.emit(_METRIC_LOOKUP_MS, elapsed_ms, {})
    except Exception:  # noqa: BLE001
        pass
