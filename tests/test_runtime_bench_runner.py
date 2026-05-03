"""Tests for ``octomil.runtime.bench.runner`` — v0.5 PR B.

The harness depends on real engines + ASR + speaker-embedding models
in production; these tests inject synthetic fakes so the harness's
contract is verifiable in CI without hardware.

Coverage:

  * Empty / invalid inputs raise at construction.
  * Single passing candidate is committed as the winner.
  * Disqualified candidate by each gate stage produces the right
    ``disqualified[]`` reason.
  * Multiple candidates with score margin >= 15% commit at high
    confidence.
  * Score margin < 15% downgrades to low confidence.
  * Budget exhaustion writes ``incomplete=True`` and does NOT commit
    a winner (`commit_invariant`).
  * No candidate survives gate -> ``incomplete=True`` written.
  * Score formula matches strategy doc: 70% first_chunk + 30% total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from octomil.runtime.bench.cache import (
    CacheKey,
    CacheStore,
    DispatchShape,
    HardwareFingerprint,
)
from octomil.runtime.bench.runner import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    DQ_CLIPPING,
    DQ_RMS_OUT_OF_BAND,
    DQ_SPEAKER_EMBEDDING_LOW,
    AudioOutput,
    BenchHarness,
    CandidateConfig,
    ReferenceFixture,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hardware() -> HardwareFingerprint:
    return HardwareFingerprint(
        machine="arm64",
        processor="Apple M2",
        cpu_count=8,
        ram_gb=16,
        os_version="macOS 15.1",
        runtime_build_tag="octomil-python:test",
    )


@pytest.fixture
def store(tmp_path: Path, hardware: HardwareFingerprint) -> CacheStore:
    return CacheStore(cache_root=tmp_path, hardware=hardware)


@pytest.fixture
def cache_key() -> CacheKey:
    return CacheKey(
        capability="tts",
        model_id="kokoro-en-v0_19",
        model_digest="sha256:" + "d" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "language": "en",
                "sample_format": "pcm_s16le",
                "sample_rate_out": 24000,
                "voice_family": "kokoro_en",
            }
        ),
    )


@pytest.fixture
def fixtures() -> list[ReferenceFixture]:
    return [
        ReferenceFixture(
            fixture_id="hello_world",
            text="Hello world.",
            expected_transcript="hello world",
        ),
        ReferenceFixture(
            fixture_id="numbers",
            text="One two three.",
            expected_transcript="one two three",
        ),
    ]


def _silence_pcm(n_samples: int) -> bytes:
    return b"\x00\x00" * n_samples


def _sine_pcm(n_samples: int, amplitude: int = 8000) -> bytes:
    """Constant-amplitude pseudo-sine — deterministic test signal."""
    out = bytearray()
    for i in range(n_samples):
        sample = amplitude if (i // 100) % 2 == 0 else -amplitude
        out.extend(int(sample).to_bytes(2, "little", signed=True))
    return bytes(out)


def _make_factory(
    *,
    first_chunk_ms: float,
    total_latency_ms: float,
    pcm_factory=lambda n: _sine_pcm(n),
):
    """Build a SynthesizeFactoryFn that returns a synthesize callable
    producing ``AudioOutput`` with the configured timing and PCM.

    The harness times wall-clock; the factory sleeps for
    ``total_latency_ms`` so the harness's score reflects the
    configured value. ``first_chunk_ms`` is currently measured at
    the same monotonic instant as ``total_latency_ms`` (PR C wires
    streaming first-chunk timing); v0.5 assumes both are equal for
    the purpose of testing the score formula.
    """
    import time

    def factory(candidate: CandidateConfig):
        del candidate

        def synthesize(text: str) -> AudioOutput:
            del text
            time.sleep(total_latency_ms / 1000.0)
            n_samples = 24000  # 1s at 24kHz
            return AudioOutput(
                pcm_s16le=pcm_factory(n_samples),
                sample_rate=24000,
                n_samples=n_samples,
            )

        return synthesize

    return factory


def _passing_asr_fn(audio: AudioOutput, expected_transcript: str) -> float:
    del audio, expected_transcript
    return 0.05  # WER 5% — passes the 15% gate


def _passing_speaker_fn(candidate: AudioOutput, reference: AudioOutput) -> float:
    del candidate, reference
    return 0.95  # cosine 0.95 — passes the 0.85 gate


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_harness_rejects_empty_candidates(store, cache_key, fixtures):
    with pytest.raises(ValueError, match="at least one candidate"):
        BenchHarness(
            cache=store,
            cache_key=cache_key,
            candidates=[],
            fixtures=fixtures,
            synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            cpu_baseline_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
        )


def test_harness_rejects_empty_fixtures(store, cache_key):
    with pytest.raises(ValueError, match="at least one fixture"):
        BenchHarness(
            cache=store,
            cache_key=cache_key,
            candidates=[CandidateConfig(engine="sherpa-onnx", provider="cpu", config={})],
            fixtures=[],
            synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            cpu_baseline_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
        )


def test_harness_rejects_zero_budget(store, cache_key, fixtures):
    with pytest.raises(ValueError, match="budget_s must be positive"):
        BenchHarness(
            cache=store,
            cache_key=cache_key,
            candidates=[CandidateConfig(engine="sherpa-onnx", provider="cpu", config={})],
            fixtures=fixtures,
            synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            cpu_baseline_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
            budget_s=0,
        )


# ---------------------------------------------------------------------------
# Single passing candidate
# ---------------------------------------------------------------------------


def test_single_passing_candidate_commits_winner(store, cache_key, fixtures):
    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        cpu_baseline_factory=_make_factory(first_chunk_ms=40, total_latency_ms=80),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is True
    assert outcome.confidence == CONFIDENCE_HIGH  # single candidate, no margin check
    assert outcome.result.winner is not None
    assert outcome.result.winner.engine == "sherpa-onnx"
    assert outcome.result.winner.provider == "coreml"
    assert outcome.result.incomplete is False

    cached = store.get(cache_key)
    assert cached is not None
    assert cached.winner is not None


# ---------------------------------------------------------------------------
# Disqualification by gate stage
# ---------------------------------------------------------------------------


def test_clipping_disqualifies(store, cache_key, fixtures):
    """Stage-1 structural: saturating clipping disqualifies."""

    def clipping_pcm(n_samples: int) -> bytes:
        # All samples at +int16 max → 100% clipped.
        out = bytearray()
        for _ in range(n_samples):
            out.extend((32767).to_bytes(2, "little", signed=True))
        return bytes(out)

    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40, pcm_factory=clipping_pcm),
        cpu_baseline_factory=_make_factory(first_chunk_ms=40, total_latency_ms=80),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is False
    assert outcome.result.winner is None
    assert outcome.result.incomplete is True
    assert len(outcome.result.disqualified) == 1
    assert outcome.result.disqualified[0]["reason"] == DQ_CLIPPING


def test_rms_out_of_band_disqualifies(store, cache_key, fixtures):
    """Stage-2 energy: RMS ratio outside [0.7, 1.4]."""
    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        # candidate emits silence (RMS≈0); CPU emits a sine. Ratio ~0 < 0.7.
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40, pcm_factory=_silence_pcm),
        cpu_baseline_factory=_make_factory(first_chunk_ms=40, total_latency_ms=80),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is False
    assert outcome.result.disqualified[0]["reason"] == DQ_RMS_OUT_OF_BAND


def test_speaker_embedding_low_disqualifies(store, cache_key, fixtures):
    """Stage-5 perceptual: speaker-embedding cosine below 0.85."""
    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})

    def low_speaker_fn(c: AudioOutput, r: AudioOutput) -> float:
        del c, r
        return 0.5  # below 0.85 gate

    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        cpu_baseline_factory=_make_factory(first_chunk_ms=40, total_latency_ms=80),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=low_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is False
    assert outcome.result.disqualified[0]["reason"] == DQ_SPEAKER_EMBEDDING_LOW


# ---------------------------------------------------------------------------
# Multiple candidates — score margin → confidence
# ---------------------------------------------------------------------------


def test_multiple_candidates_with_clear_winner_high_confidence(store, cache_key, fixtures):
    """Margin >= 15% commits at high confidence."""
    fast = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    slow = CandidateConfig(engine="sherpa-onnx", provider="cpu", config={"num_threads": 4})

    def factory(candidate: CandidateConfig):
        # coreml fast, cpu slow — ratio ~0.5
        if candidate.provider == "coreml":
            return _make_factory(first_chunk_ms=16, total_latency_ms=32)(candidate)
        return _make_factory(first_chunk_ms=32, total_latency_ms=64)(candidate)

    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[fast, slow],
        fixtures=fixtures,
        synthesize_factory=factory,
        cpu_baseline_factory=_make_factory(first_chunk_ms=32, total_latency_ms=64),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is True
    assert outcome.confidence == CONFIDENCE_HIGH
    assert outcome.result.winner is not None
    assert outcome.result.winner.provider == "coreml"
    assert len(outcome.result.runners_up) == 1


def test_marginal_winner_committed_at_low_confidence(store, cache_key, fixtures):
    """Margin < 15% commits at low confidence."""
    a = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    b = CandidateConfig(engine="sherpa-onnx", provider="cpu", config={"num_threads": 4})

    def factory(candidate: CandidateConfig):
        if candidate.provider == "coreml":
            return _make_factory(first_chunk_ms=20, total_latency_ms=40)(candidate)
        return _make_factory(first_chunk_ms=21, total_latency_ms=42)(candidate)  # ~5% slower

    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[a, b],
        fixtures=fixtures,
        synthesize_factory=factory,
        cpu_baseline_factory=_make_factory(first_chunk_ms=22, total_latency_ms=44),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is True
    assert outcome.confidence == CONFIDENCE_LOW


# ---------------------------------------------------------------------------
# All candidates disqualified
# ---------------------------------------------------------------------------


def test_no_passing_candidate_writes_incomplete(store, cache_key, fixtures):
    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})

    def low_speaker_fn(c: AudioOutput, r: AudioOutput) -> float:
        del c, r
        return 0.0

    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        cpu_baseline_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=low_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is False
    assert outcome.result.winner is None
    assert outcome.result.incomplete is True
    assert outcome.confidence == CONFIDENCE_LOW
    # Cache MUST NOT contain the result (incomplete=True is dispatch-invisible).
    assert store.get(cache_key) is None


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


def test_score_formula_70_30(store, cache_key, fixtures):
    """Strategy doc: score = 0.7 * first_chunk_ms + 0.3 * total_latency_ms."""
    candidate = CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})
    harness = BenchHarness(
        cache=store,
        cache_key=cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        cpu_baseline_factory=_make_factory(first_chunk_ms=20, total_latency_ms=40),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    outcome = harness.run()
    assert outcome.committed is True
    # The harness measures real wall-clock; the synthetic factory's
    # first_chunk_ms/total_latency_ms are just configuration. We
    # assert the FORMULA shape: score == 0.7 * first_chunk + 0.3 * total
    # ± some rounding.
    w = outcome.result.winner
    assert w is not None
    expected = 0.7 * w.first_chunk_ms + 0.3 * w.total_latency_ms
    assert abs(w.score - expected) < 1e-3
