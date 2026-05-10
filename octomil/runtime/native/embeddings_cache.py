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
# Cached record — what we actually store in the result cache
# ---------------------------------------------------------------------------


@dataclass
class CachedEmbedding:
    """Full embedding record stored in the result cache.

    Codex B2: cache hits replay every API-visible metadata field
    (prompt_tokens, n_dim, pooling_type, is_normalized) — not just the
    vector — so enabling the result cache does NOT change the
    EmbeddingsResult shape callers see for identical inputs.

    The vector list is defensively copied on read and write
    (see B3) so caller mutation cannot corrupt cached state.
    """

    vector: list[float]
    prompt_tokens: int
    n_dim: int
    pooling_type: int
    is_normalized: bool


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
        """Return the cached value or None on miss.  Promotes to MRU."""
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: bytes, value: object, n_elements: int) -> None:
        """Insert or replace.  Evicts LRU entries to satisfy capacity."""
        byte_cost = 4 * n_elements
        with self._lock:
            if key in self._cache:
                self._bytes_used -= self._sizes[key]
                del self._cache[key]
                del self._sizes[key]
            # Evict LRU until there is room.
            while self._cache and self._bytes_used + byte_cost > self._max_bytes:
                old_key, _ = self._cache.popitem(last=False)
                self._bytes_used -= self._sizes.pop(old_key, 0)
            self._cache[key] = value
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

    Codex B3: returns a defensive copy. The cache stores the same list
    instance the backend handed us via ``put_vector`` (private to the
    cache); without a copy on read, a caller mutating the returned list
    would silently corrupt every subsequent hit on the same key.
    """
    if not privacy_allowed:
        return None
    cached = cache.get(key)
    if cached is None:
        return None
    # Defensive: hand callers a fresh list so they cannot mutate the
    # stored entry. ``list(...)`` over a list of floats is O(n) — same
    # order of magnitude as the rest of the lookup work — and protects
    # the cache invariant that put(x); a=get(x); b=get(x) implies a == b.
    return list(cached)  # type: ignore[call-overload]


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

    Codex B3: stores a defensive copy of ``vector``. Without the copy
    a caller that mutates ``vector`` after handing it to the cache
    (e.g. in-place normalization) would silently change the cached
    value seen by every later hit.
    """
    if not privacy_allowed:
        return
    cache.put(key, list(vector), len(vector))


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

        Codex B2 note: prefer ``get_record`` over this thin shim — the
        bare-vector form drops prompt_tokens / pooling_type / n_dim and
        is kept only for callers that legitimately do not need those
        fields.
        """
        record = self.get_record(text, policy=policy)
        if record is None:
            return None
        return record.vector

    def put_vector(
        self,
        text: str,
        vector: list[float],
        *,
        policy: CachePolicy,
    ) -> None:
        """Store an embedding vector for ``text`` (no metadata).

        PRIVACY: ``text`` is hashed; float bytes are never logged.
        No-ops when the result cache is disabled.

        Codex B2 note: prefer ``put_record`` so metadata is preserved.
        This shim stores zeroed metadata and exists for back-compat.
        """
        self.put_record(
            text,
            CachedEmbedding(
                vector=vector,
                prompt_tokens=0,
                n_dim=len(vector),
                pooling_type=0,
                is_normalized=False,
            ),
            policy=policy,
        )

    # Codex B2: full-record get/put so the backend can replay every
    # API-visible field on a result-cache hit instead of fabricating
    # n_dim=0 / pooling_type=0 / prompt_tokens=0 like the v0.1.11 PR
    # #569 skeleton did. Stored values are defensive-copied; see B3.
    def get_record(
        self,
        text: str,
        *,
        policy: CachePolicy,
    ) -> Optional["CachedEmbedding"]:
        """Look up the full cached embedding record for ``text``.

        Returns None on miss or when the result cache is disabled.
        The returned object is a defensive copy: callers may mutate
        ``.vector`` without corrupting the cache.
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return None
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        cached = self._result_cache.get(key)
        if cached is None:
            return None
        # Stored value is a CachedEmbedding; defensive-copy the vector
        # so caller mutation can't corrupt future hits (B3).
        rec: CachedEmbedding = cached  # type: ignore[assignment]
        return CachedEmbedding(
            vector=list(rec.vector),
            prompt_tokens=rec.prompt_tokens,
            n_dim=rec.n_dim,
            pooling_type=rec.pooling_type,
            is_normalized=rec.is_normalized,
        )

    def put_record(
        self,
        text: str,
        record: "CachedEmbedding",
        *,
        policy: CachePolicy,
    ) -> None:
        """Store the full cached embedding record for ``text``.

        Codex B2: stores prompt_tokens / n_dim / pooling_type /
        is_normalized alongside the vector so cache hits replay the
        same API-visible metadata as the cold runtime call.

        Codex B3: stores a defensive copy of ``record.vector`` so a
        caller mutating ``record`` after put cannot mutate the cached
        entry.
        """
        privacy_allowed = policy.result_cache_allowed and result_cache_enabled()
        if not privacy_allowed:
            return
        text_hex = hash_text(text)
        key = derive_cache_key(self._model_digest, text_hex, self._result_salt)
        # Defensive copy of the vector list. Other fields are
        # immutable scalars and don't need copying.
        copied = CachedEmbedding(
            vector=list(record.vector),
            prompt_tokens=record.prompt_tokens,
            n_dim=record.n_dim,
            pooling_type=record.pooling_type,
            is_normalized=record.is_normalized,
        )
        self._result_cache.put(key, copied, len(copied.vector))

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
