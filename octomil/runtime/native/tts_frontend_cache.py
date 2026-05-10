"""TTS frontend cache — v0.1.11 Lane C prototype.

Caches normalized-text bytes (the output of
:func:`octomil.audio.text_normalize.normalize_for_profile`) keyed by
a 32-byte SHA-256 digest of:

    sha256( model_digest | NUL | sha256(normalized_text) | NUL
          | voice | NUL | speed_str | NUL | language | NUL
          | adapter_version )

"Phoneme tokens" are treated as the opaque normalized-text bytes.
Deeper phoneme extraction (piper-phonemize internal output) is
blocked-on-sherpa-fork — flagged as TODO(lane-a).

Privacy contract:
  - Keys are 32 raw bytes (SHA-256). No prefix structure.
  - The cache NEVER receives raw user text. The caller hashes it
    before calling :func:`build_frontend_cache_key`.
  - Voice / speed / language are folded into the key only — they do
    NOT appear in any metric label, log line, or event payload.
  - Values are opaque bytes. The cache never logs them.
  - No metric carries text or phoneme content; metric names are from
    the canonical Lane A closed set (contracts#123 / runtime#51).

Env gate: ``OCT_TTS_FRONTEND_CACHE`` — absent or ``"1"`` → ON (default);
``"0"`` → OFF.

Capacity: LRU with ``cache_max_bytes`` (default 16 MiB). Evicts LRU
entries on overflow.

Lifecycle: :meth:`TtsFrontendCache.clear` is called on model-close and
session-group close. Call it from the backend's ``close()`` method.

Metrics (canonical names, Lane A contracts#123):
  - ``tts.frontend_cache_hit_total``  — counter (1.0 per hit)
  - ``tts.audio_cache_miss_total``    — counter (1.0 per miss; reused
    as the generic frontend-cache miss counter for v0.1.11)
  - ``cache.lookup_ms``               — gauge ms of each lookup call

Note: a provisional ``tts.frontend_saved_ms`` metric was considered
but is NOT in Lane A's canonical enum. It has been dropped here; if
the wall-clock-saved gauge becomes useful it can be canonicalized
separately under octomil-contracts#128.

These names are emitted via the Python-side metric sink at debug-log
level (``_emit_metric``); they are NOT yet forwarded to
OCT_EVENT_METRIC from Python (the runtime-side emission in
``sherpa_onnx_tts_batch_session.cpp`` is the authoritative source).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OCT_TTS_FRONTEND_CACHE env var — default ON.
_ENV_CACHE_FLAG = "OCT_TTS_FRONTEND_CACHE"

# Maximum value size.  A normalized text string for any realistic TTS
# utterance fits in 16 KB.  Any value larger is rejected — it would
# indicate audio bytes were passed by mistake.
MAX_PHONEME_TOKEN_BYTES: int = 16_384  # 16 KB

# Default LRU capacity.
DEFAULT_CACHE_MAX_BYTES: int = 16 * 1024 * 1024  # 16 MiB

# Canonical metric names — Lane A contracts#123 / runtime#51.
_METRIC_HIT = "tts.frontend_cache_hit_total"
_METRIC_MISS = "tts.audio_cache_miss_total"
_METRIC_LOOKUP_MS = "cache.lookup_ms"


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def build_frontend_cache_key(
    model_digest_hex: str,
    normalized_text: str,
    voice: str,
    speed_x1000: int,
    language: str,
    adapter_version: str,
) -> bytes:
    """Build the 32-byte cache key.

    Parameters
    ----------
    model_digest_hex
        64-char lowercase hex SHA-256 of the model artifact.  No
        ``sha256:`` prefix.
    normalized_text
        UTF-8 normalized text (output of ``normalize_for_profile``).
        This argument is SHA-256 hashed inside this function; it is
        NEVER stored or logged.
    voice
        Numeric speaker-id string (e.g. ``"0"``).
    speed_x1000
        Speed multiplier × 1000 as an integer (1000 = 1.0×).
    language
        BCP-47 language tag (e.g. ``"en-US"``).
    adapter_version
        Version token from the adapter (e.g. ``"octomil-runtime/dev"``).

    Returns
    -------
    bytes
        32 raw bytes (SHA-256). No prefix structure.
    """
    # Inner hash: sha256(normalized_text) — raw text never in key.
    text_hash = hashlib.sha256(normalized_text.encode("utf-8")).digest()

    outer = hashlib.sha256()
    outer.update(model_digest_hex.encode("ascii"))
    outer.update(b"\x00")
    outer.update(text_hash)
    outer.update(b"\x00")
    outer.update(voice.encode("utf-8"))
    outer.update(b"\x00")
    outer.update(str(speed_x1000).encode("ascii"))
    outer.update(b"\x00")
    outer.update(language.encode("utf-8"))
    outer.update(b"\x00")
    outer.update(adapter_version.encode("utf-8"))
    return outer.digest()


def build_raw_text_cache_key(
    model_digest_hex: str,
    raw_text: str,
    voice: str,
    speed_x1000: int,
    language: str,
    adapter_version: str,
) -> bytes:
    """Build a 32-byte key over RAW text (Codex B7 fix).

    The original ``build_frontend_cache_key`` keys on the normalized
    text — which means the caller must run the very normalization step
    the cache claims to skip before it can even ask for the cached
    value. That defeats the cache's stated purpose.

    This raw-text variant lets the caller probe the cache before
    normalizing. On a hit, the cached normalized bytes are reused and
    normalization is skipped. On a miss, the caller normalizes and
    stores under the same raw-text key plus a co-key on the normalized
    text (so two raw inputs that normalize to the same string still
    share the cached value).

    Privacy: the same hashing pattern as ``build_frontend_cache_key`` —
    raw text is sha256'd inside this function and the hash (not the
    text) enters the key.
    """
    text_hash = hashlib.sha256(raw_text.encode("utf-8")).digest()
    outer = hashlib.sha256()
    outer.update(b"raw|")
    outer.update(model_digest_hex.encode("ascii"))
    outer.update(b"\x00")
    outer.update(text_hash)
    outer.update(b"\x00")
    outer.update(voice.encode("utf-8"))
    outer.update(b"\x00")
    outer.update(str(speed_x1000).encode("ascii"))
    outer.update(b"\x00")
    outer.update(language.encode("utf-8"))
    outer.update(b"\x00")
    outer.update(adapter_version.encode("utf-8"))
    return outer.digest()


# ---------------------------------------------------------------------------
# Canonical-name metrics sink — debug-log only on the Python side.
# Authoritative emission to OCT_EVENT_METRIC is on the runtime side
# (sherpa_onnx_tts_batch_session.cpp); this Python sink lets unit
# tests assert the canonical names without booting the runtime.
# ---------------------------------------------------------------------------


def _emit_metric(name: str, value: float) -> None:
    """Best-effort canonical-name metric.  Never raises.  Never logs content."""
    try:
        logger.debug("tts_frontend_cache metric: name=%s value=%f", name, value)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# TtsFrontendCache
# ---------------------------------------------------------------------------


class TtsFrontendCache:
    """LRU cache for TTS frontend (normalized-text) results.

    Thread-safe.  All public methods acquire ``_lock``.

    Value contract: values are opaque bytes.  The cache does NOT
    inspect or log them.  Values > :data:`MAX_PHONEME_TOKEN_BYTES`
    are silently rejected (fail-closed: don't store suspicious blobs).

    Programmatic disable: pass ``enabled=False`` to the constructor
    to disable for a specific session policy override (e.g. ``private``
    sessions may opt out).
    """

    def __init__(
        self,
        cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        enabled: Optional[bool] = None,
    ) -> None:
        """Construct.

        Parameters
        ----------
        cache_max_bytes
            Maximum total value bytes to store.  Default 16 MiB.
        enabled
            If ``None`` (default), read ``OCT_TTS_FRONTEND_CACHE`` env
            (absent or ``"1"`` → enabled; ``"0"`` → disabled).
            Pass ``True`` or ``False`` to override the env gate.
        """
        if enabled is None:
            env_val = os.environ.get(_ENV_CACHE_FLAG, "1")
            self._enabled = env_val != "0"
        else:
            self._enabled = bool(enabled)

        self._cache_max_bytes = cache_max_bytes
        self._lock = threading.Lock()
        # OrderedDict used as LRU: move-to-end on hit.
        # key → bytes value
        self._store: OrderedDict[bytes, bytes] = OrderedDict()
        self._stored_bytes: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._evict_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def hit_count(self) -> int:
        with self._lock:
            return self._hit_count

    @property
    def miss_count(self) -> int:
        with self._lock:
            return self._miss_count

    @property
    def evict_count(self) -> int:
        with self._lock:
            return self._evict_count

    @property
    def stored_bytes(self) -> int:
        with self._lock:
            return self._stored_bytes

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def lookup(self, key: bytes) -> Optional[bytes]:
        """Look up ``key``.

        Returns the cached value bytes on hit, ``None`` on miss.
        Emits provisional metrics.  Never raises.
        """
        if not self._enabled:
            return None

        t0 = time.monotonic()
        result: Optional[bytes] = None
        try:
            with self._lock:
                if key in self._store:
                    self._store.move_to_end(key)
                    result = self._store[key]
                    self._hit_count += 1
                else:
                    self._miss_count += 1
        except Exception:  # noqa: BLE001
            pass

        lookup_ms = (time.monotonic() - t0) * 1000.0
        _emit_metric(
            _METRIC_HIT if result is not None else _METRIC_MISS,
            1.0,
        )
        _emit_metric(_METRIC_LOOKUP_MS, lookup_ms)
        return result

    def insert(self, key: bytes, value: bytes) -> None:
        """Insert ``key`` → ``value``.

        Silently rejected if ``value`` is empty or exceeds
        :data:`MAX_PHONEME_TOKEN_BYTES` (audio-bytes guard).
        Never raises.
        """
        if not self._enabled:
            return
        if not value or len(value) > MAX_PHONEME_TOKEN_BYTES:
            # Fail-closed: don't store suspicious blobs.
            if len(value) > MAX_PHONEME_TOKEN_BYTES:
                logger.warning(
                    "tts_frontend_cache: value rejected (size=%d > limit=%d) — possible audio bytes; NOT stored",
                    len(value),
                    MAX_PHONEME_TOKEN_BYTES,
                )
            return

        try:
            with self._lock:
                if key in self._store:
                    # Update in place + move to MRU.
                    self._stored_bytes -= len(self._store[key])
                    self._store[key] = value
                    self._stored_bytes += len(value)
                    self._store.move_to_end(key)
                    return

                # Evict LRU tail(s) until we have room.
                while self._store and self._stored_bytes + len(value) > self._cache_max_bytes:
                    _, evicted = self._store.popitem(last=False)  # LRU = first
                    self._stored_bytes -= len(evicted)
                    self._evict_count += 1

                self._store[key] = value
                self._stored_bytes += len(value)
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        """Evict all entries.  Called on model-close and runtime-close."""
        try:
            with self._lock:
                self._store.clear()
                self._stored_bytes = 0
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "TtsFrontendCache",
    "build_frontend_cache_key",
    "DEFAULT_CACHE_MAX_BYTES",
    "MAX_PHONEME_TOKEN_BYTES",
]
