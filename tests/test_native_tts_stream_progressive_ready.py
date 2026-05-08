"""v0.1.9 Lane 4 — progressive-ready scaffolding tests.

These tests pin the SDK plumbing required for the eventual
progressive-streaming flip. They MUST pass against today's
coalesced runtime (because they don't actually require progressive
behavior — just plumbing).

Coverage:

1. ``TtsAudioChunk`` dataclass shape is forward-compatible: the
   ``streaming_mode`` field is present, has a Literal[coalesced,
   progressive] type, and defaults to "coalesced" for honesty.
2. The iterator pattern works with a mocked drain that emits chunks
   at delayed intervals (simulates a future progressive runtime
   without actually requiring one).
3. Inter-chunk timing measurement works: the SDK can compute
   ``delta_ms`` between chunk i and i+1 — the metric a follow-up
   PR will use to detect progressive arrival without waiting on a
   runtime capability bump.

Honesty pin (DO NOT FLIP without the runtime release):
    The default streaming_mode is "coalesced" today. A separate
    test (test_tts_stream_no_premature_progressive_claim.py) blocks
    flipping the default to "progressive" before the runtime
    actually proves progressive arrival.

Lane 1 + Lane 2 follow-up (after runtime release lands):
    1. NativeTtsStreamBackend reads a runtime capability hint OR
       measures inter-chunk delta_ms (>50 ms gap = progressive).
    2. Each TtsAudioChunk's streaming_mode is set per-chunk based
       on that detection.
    3. The honesty header X-Octomil-Streaming-Honesty is updated
       (with a runtime version check gating the flip).
    4. ``test_tts_stream_no_premature_progressive_claim.py`` is
       updated in lockstep with that PR.
"""

from __future__ import annotations

import time
from dataclasses import fields
from typing import Iterator, List, get_args, get_type_hints

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

    def test_streaming_mode_default_is_coalesced(self) -> None:
        chunk = TtsAudioChunk(
            pcm_f32=b"",
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=False,
            cumulative_duration_ms=0,
        )
        assert chunk.streaming_mode == "coalesced", (
            "TtsAudioChunk default streaming_mode flipped away from "
            "'coalesced'. This is the v0.1.9 honesty pin — do NOT "
            "flip until the runtime release ships progressive Generate."
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


def _delayed_chunk_iter(
    *,
    n_chunks: int,
    delay_s: float,
    sample_rate_hz: int = 22050,
    samples_per_chunk: int = 22050,  # 1s of audio per chunk
) -> Iterator[TtsAudioChunk]:
    """Synthetic drain — yields ``n_chunks`` chunks separated by
    ``delay_s`` seconds. Simulates a progressive runtime delivering
    chunks during synthesis. Chunks are streaming_mode='coalesced'
    today (honesty: this test does NOT prove progressive runtime
    behavior; it proves the SDK plumbing handles the delayed-
    emission shape correctly)."""
    cumulative_samples = 0
    for i in range(n_chunks):
        cumulative_samples += samples_per_chunk
        yield TtsAudioChunk(
            pcm_f32=b"\x00" * (samples_per_chunk * 4),  # f32 placeholder
            sample_rate_hz=sample_rate_hz,
            chunk_index=i,
            is_final=(i == n_chunks - 1),
            cumulative_duration_ms=int(cumulative_samples * 1000 // sample_rate_hz),
            streaming_mode="coalesced",
        )
        if i < n_chunks - 1:
            time.sleep(delay_s)


class TestIteratorPatternHandlesDelayedEmission:
    def test_consume_three_delayed_chunks(self) -> None:
        """Drain a 3-chunk iterator with 20 ms inter-chunk delay.
        The SDK iterator pattern handles arbitrary inter-arrival
        timing; consumers see chunks one at a time, in order."""
        chunks: List[TtsAudioChunk] = []
        for chunk in _delayed_chunk_iter(n_chunks=3, delay_s=0.02):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert [c.chunk_index for c in chunks] == [0, 1, 2]
        assert [c.is_final for c in chunks] == [False, False, True]

    def test_consume_progressive_simulated_chunks_keeps_streaming_mode(self) -> None:
        """Even when chunks arrive with realistic progressive-style
        gaps (>50 ms), the dataclass field is set as per the helper
        — i.e. the v0.1.9 follow-up's detection logic isn't yet
        wired, so chunks still report 'coalesced'. This pins the
        honesty discipline: detection is a SEPARATE PR."""
        chunks = list(_delayed_chunk_iter(n_chunks=2, delay_s=0.07))
        assert all(c.streaming_mode == "coalesced" for c in chunks), (
            "A chunk reported streaming_mode='progressive' against "
            "the v0.1.8/coalesced runtime. The SDK MUST NOT auto-detect "
            "and label as progressive without the runtime release."
        )


# ---------------------------------------------------------------------------
# 3. Inter-chunk timing measurement works
# ---------------------------------------------------------------------------


class TestInterChunkTimingMeasurement:
    """Pins the measurement helper the v0.1.9 follow-up SDK PR will
    use to auto-detect progressive arrival. The helper itself is NOT
    public API today — these tests validate that the math + iterator
    discipline both work; the follow-up wires them into
    NativeTtsStreamBackend's drain."""

    @staticmethod
    def _measure_inter_chunk_delta_ms(
        chunk_iter: Iterator[TtsAudioChunk],
    ) -> List[float]:
        """Walk the iterator, recording ``time.monotonic`` at each
        chunk arrival. Returns the list of (chunk[i+1].t -
        chunk[i].t) deltas in milliseconds. Empty list when fewer
        than 2 chunks. The follow-up PR moves this into the backend
        drain."""
        timestamps_ms: List[float] = []
        for _ in chunk_iter:
            timestamps_ms.append(time.monotonic() * 1000.0)
        if len(timestamps_ms) < 2:
            return []
        return [timestamps_ms[i + 1] - timestamps_ms[i] for i in range(len(timestamps_ms) - 1)]

    def test_delta_ms_under_50_for_coalesced_arrival(self) -> None:
        """Coalesced runtime: chunks arrive together in one
        poll_event tick, so delta_ms is small (< 50). This is the
        v0.1.8 signature."""
        deltas = self._measure_inter_chunk_delta_ms(_delayed_chunk_iter(n_chunks=3, delay_s=0.001))
        assert len(deltas) == 2
        assert all(d < 50.0 for d in deltas), f"coalesced-style deltas exceeded 50 ms: {deltas}"

    def test_delta_ms_above_50_for_progressive_simulated_arrival(self) -> None:
        """Progressive simulation: 70 ms between chunks. The follow-
        up PR will use ``> 50 ms`` as the auto-detection threshold
        for setting streaming_mode='progressive'."""
        deltas = self._measure_inter_chunk_delta_ms(_delayed_chunk_iter(n_chunks=3, delay_s=0.07))
        assert len(deltas) == 2
        # Loose lower bound: 50 ms threshold.  Sleep can be slightly
        # under on busy CI runners; set the assertion at 40 ms to
        # avoid false flakes while still proving progressive shape.
        assert all(d > 40.0 for d in deltas), (
            f"progressive-simulated deltas were too small: {deltas}. The 70 ms sleep "
            "should produce > 40 ms inter-arrival deltas. If this is flaky on CI, "
            "raise the simulation delay rather than lowering the threshold."
        )

    def test_delta_ms_returns_empty_for_single_chunk(self) -> None:
        """One chunk → no deltas to compute. The helper returns []
        rather than raising or returning a sentinel."""
        deltas = self._measure_inter_chunk_delta_ms(_delayed_chunk_iter(n_chunks=1, delay_s=0.0))
        assert deltas == []

    def test_delta_ms_returns_empty_for_zero_chunks(self) -> None:
        """Empty iterator → empty deltas list, no error."""

        def _empty() -> Iterator[TtsAudioChunk]:
            return
            yield  # pragma: no cover — generator-shape marker

        deltas = self._measure_inter_chunk_delta_ms(_empty())
        assert deltas == []


# ---------------------------------------------------------------------------
# 4. Cross-check against backend module — public name still describes
# coalesced; method name is still synthesize_with_chunks (NOT 'stream').
# ---------------------------------------------------------------------------


class TestBackendPublicNamesStillCoalescedShape:
    """v0.1.9 honesty pin: the public method name + module docstring
    are still iterator-of-chunks shape, NOT realtime/progressive.
    The guard test (test_tts_stream_no_premature_progressive_claim.py)
    is the strictest version of this; this is a quick local pin."""

    def test_method_name_is_synthesize_with_chunks(self) -> None:
        from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

        assert hasattr(NativeTtsStreamBackend, "synthesize_with_chunks"), (
            "Method synthesize_with_chunks went missing — renaming this "
            "to .stream() is a public-claim change and requires the "
            "runtime release to ship first. See guard test."
        )
        # And that the v0.1.8-shape '.stream' alias does NOT exist.
        # (If it does, someone may have renamed without going through
        # the guard; the dedicated guard test is more thorough.)
        assert not hasattr(NativeTtsStreamBackend, "stream"), (
            "NativeTtsStreamBackend grew a 'stream' attribute. This "
            "looks like an early progressive-rename. Do NOT introduce "
            "until the runtime release lands; the v0.1.9 plan keeps "
            "the public method name 'synthesize_with_chunks'."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
