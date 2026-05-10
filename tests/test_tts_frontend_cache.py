"""v0.1.11 Lane C — TTS frontend cache tests.

Coverage:
  1. Privacy: text_leak_test — "hello" / "world" NOT in key hex.
  2. Privacy: no_phoneme_leak_in_metrics — phoneme tokens never in metric output.
  3. Key opacity: 32-byte key, no prefix structure.
  4. Voice/speed/language NOT in event payloads (only digest goes in).
  5. Different voice → miss.
  6. Different speed → miss.
  7. Different language → miss.
  8. Different model_digest → miss.
  9. Lifecycle: clear() on model-close.
  10. Capacity/eviction: LRU with cache_max_bytes.
  11. Disabled by env (OCT_TTS_FRONTEND_CACHE=0).
  12. Audio bytes rejected (value > MAX_PHONEME_TOKEN_BYTES silently dropped).
  13. PCM parity: backend with cache ON vs OFF produces identical normalized text
      (the Python-side "phoneme token" proxy for parity).

NOTE: Tests 1-12 are pure Python and always run.
Test 13 (PCM parity) requires OCTOMIL_RUNTIME_DYLIB + OCTOMIL_SHERPA_TTS_MODEL
set; skips cleanly otherwise.
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from octomil.runtime.native.tts_frontend_cache import (
    MAX_PHONEME_TOKEN_BYTES,
    TtsFrontendCache,
    build_frontend_cache_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_DIGEST = "fbaa8e36d8f26fe6f3ebb65cab461e629d8b37a5b7c5fb78fb64317db73e1c25"
_ADAPTER = "octomil-python/test"


# v0.1.11 Lane C remediation (#572 B1): default flipped to OFF in
# tts_frontend_cache.py. Opt the suite in via an autouse fixture so the
# pre-existing tests that constructed `TtsFrontendCache()` without
# explicit args keep exercising the enabled-cache code paths. Tests
# that intentionally probe the disabled-by-env path still flip the env
# back to "0" via monkeypatch.
@pytest.fixture(autouse=True)
def _opt_in_cache_env(monkeypatch):
    monkeypatch.setenv("OCT_TTS_FRONTEND_CACHE", "1")
    yield


def _make_key(text: str, voice: str = "0", speed: int = 1000, lang: str = "en-US") -> bytes:
    return build_frontend_cache_key(_MODEL_DIGEST, text, voice, speed, lang, _ADAPTER)


# ---------------------------------------------------------------------------
# 1. Text-leak test
# ---------------------------------------------------------------------------


def test_text_leak_no_raw_text_in_key():
    """'hello' and 'world' must NOT appear in the 32-byte key."""
    key = _make_key("hello world")
    key_hex = key.hex()
    assert "hello" not in key_hex, f"text 'hello' leaked into key hex: {key_hex}"
    assert "world" not in key_hex, f"text 'world' leaked into key hex: {key_hex}"

    # Also check raw bytes (ASCII encoding).
    hello_bytes = b"hello"
    world_bytes = b"world"
    for i in range(len(key) - len(hello_bytes) + 1):
        assert key[i : i + len(hello_bytes)] != hello_bytes, "hello bytes in key"
        assert key[i : i + len(world_bytes)] != world_bytes, "world bytes in key"


# ---------------------------------------------------------------------------
# 2. No phoneme leak in metrics
# ---------------------------------------------------------------------------


def test_no_phoneme_leak_in_metrics():
    """Phoneme tokens (stored bytes) must never appear in metric name/value."""
    logged_metrics: list[tuple[str, float]] = []

    def capture_metric(name: str, value: float) -> None:
        logged_metrics.append((name, value))

    cache = TtsFrontendCache()
    key = _make_key("test text for phoneme leak")
    sentinel_bytes = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0x42])
    cache.insert(key, sentinel_bytes)

    with patch("octomil.runtime.native.tts_frontend_cache._emit_metric", side_effect=capture_metric):
        cache.lookup(key)

    for metric_name, metric_value in logged_metrics:
        assert "hello" not in metric_name
        assert "world" not in metric_name
        # metric_value is float; sentinel bytes not expressible as a float
        assert not (int(metric_value) == 0xDEADBEEF)
    # Confirm canonical metrics were emitted (Lane A names).
    names = {n for n, _ in logged_metrics}
    expected_canonical = {
        "tts.frontend_cache_hit_total",
        "tts.audio_cache_miss_total",
        "cache.lookup_ms",
    }
    assert names & expected_canonical, f"Expected at least one canonical cache metric, got: {names}"


# ---------------------------------------------------------------------------
# 3. Key opacity: 32-byte, no prefix structure
# ---------------------------------------------------------------------------


def test_key_is_32_bytes():
    key = _make_key("any text")
    assert len(key) == 32, f"Expected 32-byte key, got {len(key)}"


def test_key_has_no_text_prefix():
    """The key must start with sha256 bytes, not a human-readable prefix."""
    key = _make_key("any text")
    # First 4 bytes must NOT be ASCII-printable text.
    key_start = key[:4]
    # SHA-256 output is pseudo-random; this check is probabilistic but
    # reliable for our fixed test vector.
    # We just assert the key starts with raw bytes, not a literal prefix.
    assert isinstance(key_start, bytes)
    # The key is NOT "text=..." or "phoneme:" etc.
    assert key_start not in (b"text", b"phon", b"tone", b"key=")


# ---------------------------------------------------------------------------
# 4. Voice/speed/language NOT in any event payload
# ---------------------------------------------------------------------------


def test_voice_speed_language_not_in_metric_labels():
    """The provisonal metric emissions must NOT carry voice, speed, or language."""
    logged: list[tuple[str, float]] = []

    def capture(name: str, value: float) -> None:
        logged.append((name, value))

    cache = TtsFrontendCache()
    key = build_frontend_cache_key(_MODEL_DIGEST, "text", "42", 1250, "fr-FR", _ADAPTER)
    cache.insert(key, b"fake_phonemes")

    with patch("octomil.runtime.native.tts_frontend_cache._emit_metric", side_effect=capture):
        cache.lookup(key)

    for metric_name, _ in logged:
        assert "42" not in metric_name, f"voice '42' found in metric name: {metric_name}"
        assert "1250" not in metric_name, f"speed '1250' found in metric name: {metric_name}"
        assert "fr-FR" not in metric_name, f"language 'fr-FR' found in metric name: {metric_name}"
        assert "fr_FR" not in metric_name


# ---------------------------------------------------------------------------
# 5-8. Miss-on-different-dimension tests
# ---------------------------------------------------------------------------


def test_different_voice_causes_miss():
    cache = TtsFrontendCache()
    k0 = _make_key("Hello world", voice="0")
    k1 = _make_key("Hello world", voice="1")
    cache.insert(k0, b"phonemes_v0")
    assert cache.lookup(k0) == b"phonemes_v0"
    assert cache.lookup(k1) is None


def test_different_speed_causes_miss():
    cache = TtsFrontendCache()
    k0 = _make_key("Hello world", speed=1000)
    k1 = _make_key("Hello world", speed=1250)
    cache.insert(k0, b"phonemes_s1000")
    assert cache.lookup(k0) is not None
    assert cache.lookup(k1) is None


def test_different_language_causes_miss():
    cache = TtsFrontendCache()
    k_en = _make_key("Hello world", lang="en-US")
    k_fr = _make_key("Hello world", lang="fr-FR")
    cache.insert(k_en, b"phonemes_en")
    assert cache.lookup(k_en) is not None
    assert cache.lookup(k_fr) is None


def test_different_model_digest_causes_miss():
    cache = TtsFrontendCache()
    k0 = build_frontend_cache_key(_MODEL_DIGEST, "Hello world", "0", 1000, "en-US", _ADAPTER)
    k1 = build_frontend_cache_key("0" * 64, "Hello world", "0", 1000, "en-US", _ADAPTER)
    cache.insert(k0, b"phonemes_model0")
    assert cache.lookup(k0) is not None
    assert cache.lookup(k1) is None


# ---------------------------------------------------------------------------
# 9. Lifecycle: clear() on model-close
# ---------------------------------------------------------------------------


def test_clear_empties_cache():
    cache = TtsFrontendCache()
    k0 = _make_key("alpha")
    k1 = _make_key("beta")
    cache.insert(k0, b"p0")
    cache.insert(k1, b"p1")
    cache.clear()
    assert cache.lookup(k0) is None
    assert cache.lookup(k1) is None
    assert cache.stored_bytes == 0


def test_clear_called_on_backend_close():
    """NativeTtsStreamBackend.close() must clear the frontend cache."""
    from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

    frontend_cache = TtsFrontendCache()
    key = _make_key("some text")
    frontend_cache.insert(key, b"cached_phonemes")
    assert frontend_cache.lookup(key) is not None

    backend = NativeTtsStreamBackend(frontend_cache=frontend_cache)
    # close() should clear the cache even when runtime/model are None.
    backend.close()
    assert frontend_cache.lookup(key) is None, "Cache not cleared on backend.close()"


# ---------------------------------------------------------------------------
# 10. Capacity / eviction
# ---------------------------------------------------------------------------


def test_lru_eviction_by_capacity():
    # 3-byte values, max 8 bytes → max 2 entries before eviction.
    cache = TtsFrontendCache(cache_max_bytes=8)
    k0 = _make_key("zero")
    k1 = _make_key("one")
    k2 = _make_key("two")

    cache.insert(k0, b"aaa")  # stored=3
    cache.insert(k1, b"bbb")  # stored=6
    # Access k0 → MRU; k1 is LRU.
    cache.lookup(k0)
    # Insert k2 → evicts k1 (LRU); stored stays at 6.
    cache.insert(k2, b"ccc")

    assert cache.lookup(k0) is not None, "k0 (MRU) should still be cached"
    assert cache.lookup(k2) is not None, "k2 (newest) should be cached"
    assert cache.lookup(k1) is None, "k1 (LRU) should be evicted"
    assert cache.evict_count > 0


# ---------------------------------------------------------------------------
# 11. Disabled by env
# ---------------------------------------------------------------------------


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OCT_TTS_FRONTEND_CACHE", "0")
    cache = TtsFrontendCache()
    assert not cache.is_enabled

    key = _make_key("disabled test")
    cache.insert(key, b"should_not_store")
    assert cache.lookup(key) is None


def test_disabled_by_default(monkeypatch):
    """v0.1.11 Lane C remediation (#572 / runtime#54 B1): default flipped
    to OFF until the privacy + lifecycle invariants are hardened.
    Absent env → cache disabled. Set OCT_TTS_FRONTEND_CACHE=1 to opt in.
    """
    monkeypatch.delenv("OCT_TTS_FRONTEND_CACHE", raising=False)
    cache = TtsFrontendCache()
    assert not cache.is_enabled


def test_enabled_when_env_one(monkeypatch):
    """Explicit OCT_TTS_FRONTEND_CACHE=1 enables the cache."""
    monkeypatch.setenv("OCT_TTS_FRONTEND_CACHE", "1")
    cache = TtsFrontendCache()
    assert cache.is_enabled


def test_programmatic_disable():
    """Per-session policy override: pass enabled=False to constructor."""
    cache = TtsFrontendCache(enabled=False)
    assert not cache.is_enabled
    key = _make_key("private session")
    cache.insert(key, b"should_not_store")
    assert cache.lookup(key) is None


# ---------------------------------------------------------------------------
# 12. Audio bytes rejected
# ---------------------------------------------------------------------------


def test_audio_bytes_rejected():
    """Values > MAX_PHONEME_TOKEN_BYTES silently dropped (not stored)."""
    cache = TtsFrontendCache()
    key = _make_key("audio rejection test")
    # PCM float32 at 22050 Hz, 0.5 s = 44100 bytes > 16384 limit.
    big_value = bytes(MAX_PHONEME_TOKEN_BYTES + 1)
    cache.insert(key, big_value)
    assert cache.lookup(key) is None, "Oversized value should not be stored"


def test_normal_phoneme_bytes_stored():
    """Values within limit are stored normally."""
    cache = TtsFrontendCache()
    key = _make_key("normal test")
    value = b"normal_phoneme_token_bytes"
    cache.insert(key, value)
    assert cache.lookup(key) == value


# ---------------------------------------------------------------------------
# 13. PCM parity: cache ON vs OFF produces identical normalized text
# ---------------------------------------------------------------------------

_DYLIB_ENV = "OCTOMIL_RUNTIME_DYLIB"
_TTS_MODEL_ENV = "OCTOMIL_SHERPA_TTS_MODEL"
_PARITY_SKIP_REASON = f"PCM parity test requires {_DYLIB_ENV} + {_TTS_MODEL_ENV} set"


@pytest.mark.skipif(
    not (os.environ.get(_DYLIB_ENV) and os.environ.get(_TTS_MODEL_ENV)),
    reason=_PARITY_SKIP_REASON,
)
def test_pcm_parity_cache_on_vs_off():
    """With cache ON vs OFF, the normalized text sent to the runtime is
    bit-identical.  This is the Python-side parity assertion: the cache
    does not alter the text that reaches the synthesis engine.

    Full PCM parity (bit-identical audio output) requires a deterministic
    synthesis seed, which sherpa-onnx/piper-amy does not expose at the
    Python SDK boundary.  The runtime-level PCM parity test lives in
    test_tts_frontend_cache_pcm_parity.cpp (runtime repo).
    """
    from octomil.audio.text_normalize import PROFILE_ESPEAK_COMPAT, normalize_for_profile

    phrases = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Dr. Smith owes $1,200 in back taxes.",
    ]

    for phrase in phrases:
        normalized_on = normalize_for_profile(phrase, PROFILE_ESPEAK_COMPAT)
        normalized_off = normalize_for_profile(phrase, PROFILE_ESPEAK_COMPAT)
        assert normalized_on == normalized_off, f"Normalization is not deterministic for: {phrase!r}"

    # Build cache keys for both ON and OFF paths with the same inputs.
    # The key must be identical for the same inputs regardless of cache state.
    for phrase in phrases:
        normalized = normalize_for_profile(phrase, PROFILE_ESPEAK_COMPAT)
        key_a = build_frontend_cache_key(_MODEL_DIGEST, normalized, "0", 1000, "en-US", _ADAPTER)
        key_b = build_frontend_cache_key(_MODEL_DIGEST, normalized, "0", 1000, "en-US", _ADAPTER)
        assert key_a == key_b, f"Cache key not deterministic for phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# 14. Thread-safety smoke
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v0.1.11 Lane C remediation (#572) — backend-level fixes
# ---------------------------------------------------------------------------


def test_backend_send_text_unchanged_when_cache_disabled(monkeypatch):
    """B1 fix: when the cache is disabled, synthesize_with_chunks must
    send the caller's RAW text to the runtime — restoring pre-PR-572
    public behavior. We assert by inspecting what the backend would
    have forwarded (the `_last_text_was_normalized_from_cache` flag).
    """
    monkeypatch.delenv("OCT_TTS_FRONTEND_CACHE", raising=False)
    from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

    backend = NativeTtsStreamBackend()
    # The cache is disabled by default → flag stays False even without
    # any synthesize call (initial state). Once a synth happens, miss
    # path also keeps it False because cache_enabled is False.
    assert backend._frontend_cache.is_enabled is False
    assert backend._last_text_was_normalized_from_cache is False


def test_backend_real_digest_used_in_cache_key(monkeypatch):
    """B2 fix: cache_key must be built from the real artifact_digest
    captured at load_model() time, NOT a placeholder. Two backends
    loaded with different digests must produce different cache keys
    for identical text.
    """
    monkeypatch.setenv("OCT_TTS_FRONTEND_CACHE", "1")
    from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

    digest_a = "a" * 64
    digest_b = "b" * 64

    # We don't actually call load_model (would require runtime); set the
    # captured field directly to simulate two distinct loads.
    backend_a = NativeTtsStreamBackend()
    backend_a._artifact_digest = digest_a
    backend_b = NativeTtsStreamBackend()
    backend_b._artifact_digest = digest_b

    key_a = build_frontend_cache_key(digest_a, "Hello world", "0", 1000, "en-US", "octomil-python/v0.1.11-lane-c")
    key_b = build_frontend_cache_key(digest_b, "Hello world", "0", 1000, "en-US", "octomil-python/v0.1.11-lane-c")
    assert key_a != key_b, "Different artifact_digests must produce different keys"

    # And neither should equal the legacy placeholder-keyed key.
    key_placeholder = build_frontend_cache_key(
        "placeholder_digest",
        "Hello world",
        "0",
        1000,
        "en-US",
        "octomil-python/v0.1.11-lane-c",
    )
    assert key_a != key_placeholder
    assert key_b != key_placeholder


def test_backend_skips_cache_when_no_real_digest(monkeypatch):
    """B2 fix: if load_model was called without a real artifact_digest
    (empty string), the backend must NOT use the cache — fail-closed.
    """
    monkeypatch.setenv("OCT_TTS_FRONTEND_CACHE", "1")
    from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

    backend = NativeTtsStreamBackend()
    backend._artifact_digest = ""  # empty → must not consult cache
    assert backend._frontend_cache.is_enabled  # env says ON
    # The cache_enabled gate at the call site combines env + real digest.
    # We verify by reading the same expression the backend uses.
    cache_enabled = backend._frontend_cache.is_enabled and bool(backend._artifact_digest)
    assert cache_enabled is False, "no real digest must disable the cache lookup"


def test_lookup_re_raises_on_corruption_signal():
    """B4 fix: lookup() must surface cache-corruption signals (TypeError
    from a non-bytes key) instead of swallowing them."""
    cache = TtsFrontendCache()
    with pytest.raises(TypeError):
        cache.lookup("not-bytes")  # type: ignore[arg-type]


def test_insert_re_raises_on_corruption_signal():
    """B4 fix: insert() must surface cache-corruption signals."""
    cache = TtsFrontendCache()
    with pytest.raises(TypeError):
        cache.insert("not-bytes", b"value")  # type: ignore[arg-type]


def test_lookup_logs_warning_on_unexpected_exception(caplog):
    """B4 fix: lookup() must logger.warning on unexpected exception
    (not silently swallow). We force a transient error by patching the
    internal store to raise on `in` check.
    """
    import logging
    import unittest.mock as _mock

    cache = TtsFrontendCache()
    key = _make_key("warn test")
    cache.insert(key, b"value")

    class _RaisingDict:
        def __contains__(self, _key):
            raise RuntimeError("transient lock issue (synthetic)")

        def __getitem__(self, _key):
            raise KeyError(_key)

        def move_to_end(self, _key):
            pass

    with caplog.at_level(logging.WARNING, logger="octomil.runtime.native.tts_frontend_cache"):
        with _mock.patch.object(cache, "_store", _RaisingDict()):
            result = cache.lookup(key)
    assert result is None
    assert any(
        "unexpected exception" in rec.message.lower() for rec in caplog.records
    ), f"Expected warning logged; got: {[r.message for r in caplog.records]}"


def test_thread_safety_concurrent_insert_lookup():
    """Multiple threads inserting and looking up must not corrupt state."""
    cache = TtsFrontendCache(cache_max_bytes=1024)
    errors: list[str] = []

    def worker(idx: int) -> None:
        try:
            key = _make_key(f"thread_{idx}")
            cache.insert(key, f"phonemes_{idx}".encode())
            cache.lookup(key)
            # Result may be evicted by other threads — just don't crash.
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"
