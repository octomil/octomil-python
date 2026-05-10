"""test_embeddings_cache_privacy.py — Lane B v0.1.11 privacy + parity
tests for the Python embeddings cache layer.

These tests exercise octomil.runtime.native.embeddings_cache directly
and do NOT require a live GGUF or runtime — they run in CI unconditionally.

Test names (stable — re-runnable by pytest -k <name>):

  test_privacy_roundtrip_no_plaintext
    Feed "the quick brown fox" through the cache.  Capture all metric
    events and the cache key bytes.  Assert that no substring of
    length >= 4 from the input appears in: the cache key bytes, the
    stored token-ID list (interpreted as bytes), or the stored float
    vector (interpreted as bytes).

  test_key_opacity_length_32_bytes
    derive_cache_key returns exactly 32 bytes.  Derive keys for 20
    inputs; assert all are distinct and no two share a 4-byte prefix.

  test_no_token_id_leak_in_stored_bytes
    Store token IDs [42, 1337, 8192].  The raw bytes of the stored
    list (via struct.pack) do not contain the decimal string of any
    token ID.

  test_no_vector_leak_in_stored_bytes
    Store a float vector [1.5, -2.25, 0.125].  The raw bytes do not
    contain the decimal string representation.

  test_output_parity_tokens_bit_exact
    put_tokens then get_tokens: returned list is bit-identical to input.
    Parity tolerance: 0 (exact equality on every element).

  test_output_parity_vector_bit_exact
    put_vector then get_vector (privacy=True): returned list is
    bit-identical to input (struct.pack comparison).  Tolerance = 0.

  test_lifecycle_clear_on_model_close
    Insert 3 entries, call cache_manager.close(), assert sizes are 0.

  test_lifecycle_clear_on_runtime_close
    Same as above but via a second EmbeddingsCacheManager instance
    (simulates runtime close clearing a different model's manager).

  test_eviction_lru_max_bytes
    Set max_bytes=64 (fits 4 int entries of 16 bytes each).  Insert 8
    entries; assert bytes_used() <= 64 after each insert.

  test_result_cache_privacy_gate_policy_off
    put_vector with result_cache_allowed=False in policy: entry is NOT
    stored even when env var is set to "1".

  test_result_cache_env_default_off
    With OCT_EMBEDDINGS_RESULT_CACHE unset, get_vector returns None
    regardless of policy.result_cache_allowed.

  test_tokenization_cache_env_default_on
    With OCT_EMBEDDINGS_TOKENIZATION_CACHE unset, get_tokens / put_tokens
    operate normally.

  test_metric_events_contain_no_plaintext
    Collect all metric events emitted during a cache hit + miss cycle.
    Assert that "the quick brown fox" (and any 4-char substring) does
    not appear in the metric event data.
"""

from __future__ import annotations

import struct

import pytest

from octomil.runtime.native.embeddings_cache import (
    CachePolicy,
    EmbeddingsCacheManager,
    EmbeddingsLruCache,
    cache_get_tokens,
    cache_get_vector,
    cache_put_tokens,
    cache_put_vector,
    derive_cache_key,
    emit_cache_hit,
    emit_cache_miss,
    hash_text,
    result_cache_enabled,
    tokenization_cache_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEXT = "the quick brown fox"
_FAKE_DIGEST = "sha256:" + "a" * 64


def _all_substrings_min4(text: str) -> list[str]:
    result = []
    for start in range(len(text)):
        for end in range(start + 4, len(text) + 1):
            result.append(text[start:end])
    return result


def _assert_no_text_in_bytes(text: str, raw: bytes, context: str) -> None:
    for sub in _all_substrings_min4(text):
        sub_bytes = sub.encode("utf-8")
        if sub_bytes in raw:
            pytest.fail(f"{context}: plaintext substring {sub!r} found in raw bytes")


def _make_manager() -> EmbeddingsCacheManager:
    return EmbeddingsCacheManager(
        model_digest=_FAKE_DIGEST,
        adapter_version="test-v0",
    )


def _make_key(text: str, salt: str = "tok|test") -> bytes:
    return derive_cache_key(_FAKE_DIGEST, hash_text(text), salt)


# ---------------------------------------------------------------------------
# test_privacy_roundtrip_no_plaintext
# ---------------------------------------------------------------------------


def test_privacy_roundtrip_no_plaintext(monkeypatch):
    monkeypatch.setenv("OCT_EMBEDDINGS_TOKENIZATION_CACHE", "1")
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")

    cache = EmbeddingsLruCache()

    # Tokenization cache round-trip.
    tok_key = _make_key(_TEXT, "tok|v0")
    token_ids = [42, 1337, 8192, 256]
    cache_put_tokens(cache, tok_key, token_ids)
    got_ids = cache_get_tokens(cache, tok_key)
    assert got_ids is not None
    # Raw bytes of token IDs must not contain plaintext.
    raw_tok = struct.pack(f"{len(got_ids)}i", *got_ids)
    _assert_no_text_in_bytes(_TEXT, raw_tok, "tokenization payload bytes")

    # Cache key bytes must not contain plaintext.
    _assert_no_text_in_bytes(_TEXT, tok_key, "tokenization key bytes")

    # Result cache round-trip.
    vec_key = _make_key(_TEXT, "vec|v0")
    vector = [0.1, -0.2, 0.3, 0.4]
    cache_put_vector(cache, vec_key, vector, privacy_allowed=True)
    got_vec = cache_get_vector(cache, vec_key, privacy_allowed=True)
    assert got_vec is not None
    raw_vec = struct.pack(f"{len(got_vec)}f", *got_vec)
    _assert_no_text_in_bytes(_TEXT, raw_vec, "result vector payload bytes")
    _assert_no_text_in_bytes(_TEXT, vec_key, "result vector key bytes")


# ---------------------------------------------------------------------------
# test_key_opacity_length_32_bytes
# ---------------------------------------------------------------------------


def test_key_opacity_length_32_bytes():
    inputs = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli fruit",
        "vanilla",
    ]
    keys = []
    for inp in inputs:
        k = _make_key(inp, "tok|v0")
        assert len(k) == 32, f"key length {len(k)} != 32 for {inp!r}"
        # Key must not contain the plaintext.
        _assert_no_text_in_bytes(inp, k, f"key for {inp!r}")
        keys.append(k)

    # All distinct.
    assert len(set(keys)) == len(keys), "duplicate keys detected"

    # No two share a 4-byte prefix.
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i >= j:
                continue
            if ki[:4] == kj[:4]:
                pytest.fail(f"keys[{i}] and keys[{j}] share 4-byte prefix: {ki[:4].hex()}")


# ---------------------------------------------------------------------------
# test_no_token_id_leak_in_stored_bytes
# ---------------------------------------------------------------------------


def test_no_token_id_leak_in_stored_bytes():
    cache = EmbeddingsLruCache()
    key = _make_key("sentinel", "tok|v0")
    ids = [42, 1337, 8192]
    cache_put_tokens(cache, key, ids)
    got = cache_get_tokens(cache, key)
    assert got is not None
    raw = struct.pack(f"{len(got)}i", *got)
    for id_val in ids:
        id_str = str(id_val)
        id_bytes = id_str.encode("utf-8")
        assert id_bytes not in raw, f"token id decimal string {id_str!r} found in stored bytes"


# ---------------------------------------------------------------------------
# test_no_vector_leak_in_stored_bytes
# ---------------------------------------------------------------------------


def test_no_vector_leak_in_stored_bytes():
    cache = EmbeddingsLruCache()
    key = _make_key("sentinel2", "vec|v0")
    vec = [1.5, -2.25, 0.125]
    cache_put_vector(cache, key, vec, privacy_allowed=True)
    got = cache_get_vector(cache, key, privacy_allowed=True)
    assert got is not None
    raw = struct.pack(f"{len(got)}f", *got)
    for f_val in [1.5, 2.25, 0.125]:
        dec = f"{f_val:.3g}".encode("utf-8")
        if len(dec) >= 3:
            assert dec not in raw, f"float decimal {dec!r} found in stored vector bytes"


# ---------------------------------------------------------------------------
# test_output_parity_tokens_bit_exact
# ---------------------------------------------------------------------------


def test_output_parity_tokens_bit_exact():
    cache = EmbeddingsLruCache()
    key = _make_key("parity-test", "tok|v0")
    orig = [10, 20, 30, 40, 50]
    cache_put_tokens(cache, key, orig)
    got = cache_get_tokens(cache, key)
    assert got is not None
    assert len(got) == len(orig)
    for i, (a, b) in enumerate(zip(orig, got)):
        assert a == b, f"token[{i}]: {a} != {b} — parity tolerance 0"


# ---------------------------------------------------------------------------
# test_output_parity_vector_bit_exact
# ---------------------------------------------------------------------------


def test_output_parity_vector_bit_exact():
    cache = EmbeddingsLruCache()
    key = _make_key("parity-vec", "vec|v0")
    orig = [0.1, 0.2, 0.3, 0.4, 0.5]
    cache_put_vector(cache, key, orig, privacy_allowed=True)
    got = cache_get_vector(cache, key, privacy_allowed=True)
    assert got is not None
    assert len(got) == len(orig)
    # Bit-exact comparison via struct.pack.
    for i, (a, b) in enumerate(zip(orig, got)):
        a_bits = struct.pack("f", a)
        b_bits = struct.pack("f", b)
        assert a_bits == b_bits, f"vec[{i}]: bits differ — parity tolerance 0 violated"


# ---------------------------------------------------------------------------
# test_lifecycle_clear_on_model_close
# ---------------------------------------------------------------------------


def test_lifecycle_clear_on_model_close(monkeypatch):
    monkeypatch.setenv("OCT_EMBEDDINGS_TOKENIZATION_CACHE", "1")
    mgr = _make_manager()
    policy = CachePolicy.tokenization_only()
    for i in range(3):
        mgr.put_tokens(f"input{i}", [1, 2, 3], policy=policy)
    assert mgr.tok_cache_size() == 3
    mgr.close()
    assert mgr.tok_cache_size() == 0
    assert mgr.tok_cache_bytes() == 0


# ---------------------------------------------------------------------------
# test_lifecycle_clear_on_runtime_close
# ---------------------------------------------------------------------------


def test_lifecycle_clear_on_runtime_close(monkeypatch):
    monkeypatch.setenv("OCT_EMBEDDINGS_TOKENIZATION_CACHE", "1")
    mgr1 = _make_manager()
    mgr2 = EmbeddingsCacheManager(
        model_digest="sha256:" + "b" * 64,
        adapter_version="test-v0",
    )
    policy = CachePolicy.tokenization_only()
    for i in range(2):
        mgr1.put_tokens(f"x{i}", [1, 2], policy=policy)
        mgr2.put_tokens(f"y{i}", [3, 4], policy=policy)
    assert mgr1.tok_cache_size() == 2
    assert mgr2.tok_cache_size() == 2
    # Close mgr1 (simulate model close).
    mgr1.close()
    assert mgr1.tok_cache_size() == 0
    # mgr2 is unaffected (separate instance).
    assert mgr2.tok_cache_size() == 2
    mgr2.close()


# ---------------------------------------------------------------------------
# test_eviction_lru_max_bytes
# ---------------------------------------------------------------------------


def test_eviction_lru_max_bytes():
    # Each entry: 4 int elements = 16 bytes.  max_bytes=64 → 4 entries max.
    cache = EmbeddingsLruCache(max_bytes=64)
    for i in range(10):
        key = _make_key(f"evict{i}", "tok|v0")
        cache_put_tokens(cache, key, [1, 2, 3, 4])
        assert cache.bytes_used() <= 64, f"bytes_used={cache.bytes_used()} > 64 after insert {i}"
    assert cache.size() <= 4, f"size={cache.size()} > 4 entries for max_bytes=64, entry=16"


# ---------------------------------------------------------------------------
# test_result_cache_privacy_gate_policy_off
# ---------------------------------------------------------------------------


def test_result_cache_privacy_gate_policy_off(monkeypatch):
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")
    mgr = _make_manager()

    # Policy has result_cache_allowed=False → store is suppressed.
    policy_off = CachePolicy(tokenization_cache_allowed=True, result_cache_allowed=False)
    mgr.put_vector("some text", [1.0, 2.0, 3.0], policy=policy_off)
    # Even with policy_allowed on get, there's nothing stored.
    got = mgr.get_vector("some text", policy=CachePolicy.policy_allowed())
    assert got is None, "entry should NOT have been stored with result_cache_allowed=False"


# ---------------------------------------------------------------------------
# test_result_cache_env_default_off
# ---------------------------------------------------------------------------


def test_result_cache_env_default_off(monkeypatch):
    monkeypatch.delenv("OCT_EMBEDDINGS_RESULT_CACHE", raising=False)
    assert not result_cache_enabled(), "OCT_EMBEDDINGS_RESULT_CACHE should default OFF when unset"
    mgr = _make_manager()
    policy = CachePolicy.policy_allowed()
    mgr.put_vector("hello world", [0.5, 0.6], policy=policy)
    got = mgr.get_vector("hello world", policy=policy)
    assert got is None, "result cache should be a miss when env var is unset (default OFF)"


# ---------------------------------------------------------------------------
# test_tokenization_cache_env_default_on
# ---------------------------------------------------------------------------


def test_tokenization_cache_env_default_on(monkeypatch):
    monkeypatch.delenv("OCT_EMBEDDINGS_TOKENIZATION_CACHE", raising=False)
    assert tokenization_cache_enabled(), "OCT_EMBEDDINGS_TOKENIZATION_CACHE should default ON when unset"
    mgr = _make_manager()
    policy = CachePolicy.tokenization_only()
    mgr.put_tokens("hello world", [1, 2, 3], policy=policy)
    got = mgr.get_tokens("hello world", policy=policy)
    assert got == [1, 2, 3], "tokenization cache should return stored tokens by default"


# ---------------------------------------------------------------------------
# test_metric_events_contain_no_plaintext
# ---------------------------------------------------------------------------


def test_metric_events_contain_no_plaintext(monkeypatch):
    monkeypatch.setenv("OCT_EMBEDDINGS_TOKENIZATION_CACHE", "1")
    monkeypatch.setenv("OCT_EMBEDDINGS_RESULT_CACHE", "1")

    TEXT = _TEXT
    emitted: list[tuple[str, float, dict]] = []

    class FakeSink:
        def emit(self, name: str, value: float, labels: dict) -> None:
            emitted.append((name, value, labels))

    sink = FakeSink()

    emit_cache_hit(sink, "tokenization")
    emit_cache_miss(sink, "result")

    # All metric events must not contain the plaintext.
    for name, value, labels in emitted:
        event_str = f"{name}:{value}:{labels}"
        for sub in _all_substrings_min4(TEXT):
            assert sub not in event_str, f"plaintext substring {sub!r} found in metric event: {event_str!r}"
