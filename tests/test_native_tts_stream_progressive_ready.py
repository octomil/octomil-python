"""v0.1.9 progressive TTS stream plumbing tests.

These tests pin the SDK plumbing for the live-conditional progressive
TTS stream path. They do not require a runtime artifact; they verify
the public dataclass and iterator shape that advertised runtimes use.

Coverage:

1. ``TtsAudioChunk`` dataclass shape is forward-compatible: the
   ``streaming_mode`` field is present, has a Literal[coalesced,
   progressive] type, and defaults to "progressive" for the current
   proven runtime behavior.
2. The iterator pattern works with a mocked drain that emits chunks
   at delayed intervals (simulates a future progressive runtime
   without actually requiring one).
3. Inter-chunk timing measurement works: given N chunk arrivals at
   monotonic clock times t0 < t1 < ... < t(N-1), the helper produces
   N-1 non-negative finite deltas. Lane 4's job is to verify that
   inter-chunk deltas can be measured without encoding a product
   threshold in this unit test.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Callable, Iterator, List, get_args, get_type_hints

import pytest

from octomil.runtime.native.tts_stream_backend import TtsAudioChunk

# ---------------------------------------------------------------------------
# 1. Dataclass shape is forward-compatible
# ---------------------------------------------------------------------------


class TestTtsAudioChunkForwardCompatibleShape:
    def test_streaming_mode_field_exists(self) -> None:
        field_names = {f.name for f in fields(TtsAudioChunk)}
        assert "streaming_mode" in field_names, (
            "TtsAudioChunk is missing streaming_mode field — the "
            "v0.1.9 progressive flip cannot land without breaking "
            "callers wired against the v0.1.8 dataclass shape."
        )

    def test_streaming_mode_default_is_progressive(self) -> None:
        """v0.1.9 Lane C flip: default is now 'progressive'."""
        chunk = TtsAudioChunk(
            pcm_f32=b"",
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=False,
            cumulative_duration_ms=0,
        )
        assert chunk.streaming_mode == "progressive", (
            "TtsAudioChunk default streaming_mode is no longer 'progressive'. "
            "v0.1.9 Lane C flipped this; revert requires runtime regression evidence."
        )

    def test_streaming_mode_accepts_progressive_literal(self) -> None:
        """The forward-compat surface accepts BOTH literals; only the
        default is pinned. This lets the v0.1.9 follow-up SDK PR set
        progressive on the chunk WITHOUT a dataclass-shape change."""
        chunk = TtsAudioChunk(
            pcm_f32=b"",
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=True,
            cumulative_duration_ms=1000,
            streaming_mode="progressive",
        )
        assert chunk.streaming_mode == "progressive"

    def test_streaming_mode_literal_args(self) -> None:
        """Type hint is Literal["coalesced", "progressive"] — pinned
        so an accidental string typo (e.g. "progresive") fails type
        checks at use sites. Static check is mypy's job; this test
        verifies the runtime type-hint shape is what we expect."""
        hints = get_type_hints(TtsAudioChunk, include_extras=True)
        streaming_mode_hint = hints["streaming_mode"]
        args = get_args(streaming_mode_hint)
        assert set(args) == {"coalesced", "progressive"}, (
            f"streaming_mode Literal[] args drifted: got {args}; "
            "expected exactly {coalesced, progressive}. Adding a new "
            "literal is OK long-term but requires a deliberate dataclass "
            "review — do not add silently."
        )


# ---------------------------------------------------------------------------
# 2. Iterator pattern works with delayed-emission mocked sink
# ---------------------------------------------------------------------------


def _chunk_iter(
    *,
    n_chunks: int,
    sample_rate_hz: int = 22050,
    samples_per_chunk: int = 22050,  # 1s of audio per chunk
) -> Iterator[TtsAudioChunk]:
    """Synthetic drain — yields ``n_chunks`` TtsAudioChunk objects.
    No real-time delay; consumers that want to simulate inter-arrival
    timing should drive a fake clock when consuming the iterator.
    Chunks are streaming_mode='progressive' (v0.1.9 Lane C flip)."""
    cumulative_samples = 0
    for i in range(n_chunks):
        cumulative_samples += samples_per_chunk
        yield TtsAudioChunk(
            pcm_f32=b"\x00" * (samples_per_chunk * 4),  # f32 placeholder
            sample_rate_hz=sample_rate_hz,
            chunk_index=i,
            is_final=(i == n_chunks - 1),
            cumulative_duration_ms=int(cumulative_samples * 1000 // sample_rate_hz),
            streaming_mode="progressive",
        )


class TestIteratorPatternHandlesChunkEmission:
    def test_consume_three_chunks_in_order(self) -> None:
        """Drain a 3-chunk iterator. The SDK iterator pattern hands
        chunks to consumers one at a time, in chunk_index order, with
        the last chunk flagged is_final=True."""
        chunks: List[TtsAudioChunk] = []
        for chunk in _chunk_iter(n_chunks=3):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert [c.chunk_index for c in chunks] == [0, 1, 2]
        assert [c.is_final for c in chunks] == [False, False, True]

    def test_consumed_chunks_report_progressive_streaming_mode(self) -> None:
        """v0.1.9 Lane C: chunks now report streaming_mode='progressive'
        per the flipped default. Proof artifact gated the flip."""
        chunks = list(_chunk_iter(n_chunks=2))
        assert all(c.streaming_mode == "progressive" for c in chunks), (
            "A chunk reported streaming_mode != 'progressive'. "
            "v0.1.9 Lane C flipped the default; revert requires evidence."
        )


# ---------------------------------------------------------------------------
# 3. Inter-chunk timing measurement works
#
# Lane 4's job is to verify that inter-chunk deltas CAN BE MEASURED:
# given N chunk arrivals at monotonic clock times t0 < t1 < ... < t(N-1),
# the helper returns N-1 non-negative finite floats. The progressive
# THRESHOLD (e.g. ">50 ms = progressive") is Lane 3's gate, NOT Lane 4's
# — encoding it here would pre-empt that decision.
# ---------------------------------------------------------------------------


class TestInterChunkTimingMeasurement:
    """Pins the measurement helper the v0.1.9 follow-up SDK PR will
    use to auto-detect progressive arrival. The helper itself is NOT
    public API today — these tests validate that the math + iterator
    discipline both work, using a fake clock so the test is fast and
    deterministic. The follow-up PR moves this into the backend drain
    AND picks the progressive threshold (Lane 3)."""

    @staticmethod
    def _measure_inter_chunk_delta_ms(
        chunk_iter: Iterator[TtsAudioChunk],
        clock_fn: Callable[[], float],
    ) -> List[float]:
        """Walk the iterator, recording ``clock_fn()`` (seconds) at
        each chunk arrival. Returns the list of (chunk[i+1].t -
        chunk[i].t) deltas in milliseconds. Empty list when fewer
        than 2 chunks. The follow-up PR moves this into the backend
        drain and uses ``time.monotonic`` as the production clock."""
        timestamps_ms: List[float] = []
        for _ in chunk_iter:
            timestamps_ms.append(clock_fn() * 1000.0)
        if len(timestamps_ms) < 2:
            return []
        return [timestamps_ms[i + 1] - timestamps_ms[i] for i in range(len(timestamps_ms) - 1)]

    @staticmethod
    def _fake_clock(timestamps_s: List[float]) -> Callable[[], float]:
        """Return a clock_fn that yields the given timestamps in
        order. Pre-populate with one timestamp per expected chunk."""
        idx = {"i": 0}

        def _read() -> float:
            t = timestamps_s[idx["i"]]
            idx["i"] += 1
            return t

        return _read

    def test_delta_ms_returns_n_minus_one_finite_non_negative_for_n_chunks(
        self,
    ) -> None:
        """Three chunks at monotonic times t0 < t1 < t2 produce
        exactly 2 deltas, each non-negative + finite. No threshold
        is asserted — Lane 3 picks that."""
        clock = self._fake_clock([0.000, 0.005, 0.012])
        deltas = self._measure_inter_chunk_delta_ms(_chunk_iter(n_chunks=3), clock)
        assert len(deltas) == 2
        assert all(math.isfinite(d) for d in deltas), f"non-finite delta: {deltas}"
        assert all(d >= 0.0 for d in deltas), f"negative inter-arrival delta: {deltas}"

    def test_delta_ms_handles_widely_spaced_arrivals(self) -> None:
        """Wide spacing (e.g. 70 ms gaps that a future progressive
        runtime might produce) still yields N-1 non-negative finite
        deltas. Lane 4 verifies the math; the threshold for tagging
        such deltas as 'progressive' is set in Lane 3."""
        clock = self._fake_clock([0.000, 0.070, 0.140])
        deltas = self._measure_inter_chunk_delta_ms(_chunk_iter(n_chunks=3), clock)
        assert len(deltas) == 2
        assert all(math.isfinite(d) and d >= 0.0 for d in deltas), deltas

    def test_delta_ms_returns_empty_for_single_chunk(self) -> None:
        """One chunk → no deltas to compute. The helper returns []
        rather than raising or returning a sentinel."""
        clock = self._fake_clock([0.0])
        deltas = self._measure_inter_chunk_delta_ms(_chunk_iter(n_chunks=1), clock)
        assert deltas == []

    def test_delta_ms_returns_empty_for_zero_chunks(self) -> None:
        """Empty iterator → empty deltas list, no error."""

        def _empty() -> Iterator[TtsAudioChunk]:
            return
            yield  # pragma: no cover — generator-shape marker

        clock = self._fake_clock([])
        deltas = self._measure_inter_chunk_delta_ms(_empty(), clock)
        assert deltas == []


# ---------------------------------------------------------------------------
# 4. Cross-check against backend module — method name is still
# synthesize_with_chunks (NOT 'stream').
# ---------------------------------------------------------------------------


class TestBackendPublicNamesStillIteratorShape:
    """v0.1.9 honesty pin: the public method name + module docstring
    are still iterator-of-chunks shape, NOT realtime audio."""

    def test_method_name_is_synthesize_with_chunks(self) -> None:
        from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

        assert hasattr(NativeTtsStreamBackend, "synthesize_with_chunks"), (
            "Method synthesize_with_chunks went missing — renaming this "
            "to .stream() is a public API change. Keep the iterator "
            "method name stable for current callers."
        )
        # And that the older proposed '.stream' alias does NOT exist.
        assert not hasattr(NativeTtsStreamBackend, "stream"), (
            "NativeTtsStreamBackend grew a 'stream' attribute. This "
            "looks like an API rename. Keep "
            "the public method name 'synthesize_with_chunks'."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
