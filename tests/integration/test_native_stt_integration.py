"""Integration tests — NativeSttBackend against real liboctomil-runtime.

These exercise the full cffi path (oct_runtime_open → oct_model_open →
oct_session_open → oct_session_send_audio → oct_session_poll_event drain
→ oct_session_close). They require:

  * ``OCTOMIL_RUNTIME_DYLIB`` (or a fetched dev cache) pointing at a
    liboctomil-runtime built with ``OCT_ENABLE_ENGINE_WHISPER_CPP=ON``
    and ABI minor >= 9.
  * ``OCTOMIL_WHISPER_BIN`` pointing at a verified ggml-tiny.bin or
    ggml-base.bin row registered by the runtime.
  * ``research/engines/whisper.cpp/samples/jfk.wav`` on disk.

When any of these are missing the entire module skips; we never
fall back to the legacy pywhispercpp path on the product flow.

The canonical jfk.wav transcript (whitespace-tolerant) is::

    " And so my fellow Americans ask not what your country can do
      for you, ask what you can do for your country."
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode

_JFK_WAV = Path("/Users/seanb/Developer/Octomil/research/engines/whisper.cpp/samples/jfk.wav")
_CANONICAL_TRANSCRIPT = (
    " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
)


def _skip_reason() -> str | None:
    if not _JFK_WAV.is_file():
        return f"jfk.wav missing at {_JFK_WAV}"
    if not os.environ.get("OCTOMIL_WHISPER_BIN"):
        return "OCTOMIL_WHISPER_BIN not set"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _load_jfk_pcm_f32() -> np.ndarray:
    with wave.open(str(_JFK_WAV), "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000, f"jfk.wav sample rate {sr} != 16000"
        pcm = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _normalize_transcript(s: str) -> str:
    return " ".join(s.strip().split())


# ---------------------------------------------------------------------------
# Test 1 — native is selected (planner-style helper)
# ---------------------------------------------------------------------------


class TestPlannerSelection:
    """When env preconditions hold, the planner-style helper returns
    True so the SDK selects ``NativeSttBackend`` over the legacy
    pywhispercpp path."""

    def test_runtime_advertises_audio_transcription(self) -> None:
        from octomil.runtime.native.loader import NativeRuntime
        from octomil.runtime.native.stt_backend import (
            runtime_advertises_audio_transcription,
        )

        rt = NativeRuntime.open()
        try:
            assert runtime_advertises_audio_transcription(rt) is True
            caps = rt.capabilities()
            assert "audio.transcription" in caps.supported_capabilities
        finally:
            rt.close()


# ---------------------------------------------------------------------------
# Test 2 — fallback is unreachable
# ---------------------------------------------------------------------------


class TestNoFallback:
    """When the runtime stops advertising the capability mid-test
    (operator unset env, planner / SDK MUST raise typed
    ``OctomilError`` rather than silently route to pywhispercpp."""

    def test_unset_whisper_bin_raises_runtime_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        monkeypatch.delenv("OCTOMIL_WHISPER_BIN", raising=False)
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("whisper-tiny")
        # Either RUNTIME_UNAVAILABLE (capability not advertised) or
        # CHECKSUM_MISMATCH if the runtime came up partially. Both
        # are acceptable hard-stops on the product path; what is NOT
        # acceptable is a silent fallback to pywhispercpp.
        assert exc_info.value.code in {
            OctomilErrorCode.RUNTIME_UNAVAILABLE,
            OctomilErrorCode.CHECKSUM_MISMATCH,
        }


# ---------------------------------------------------------------------------
# Test 3 — transcript drained correctly + matches canonical
# ---------------------------------------------------------------------------


class TestTranscriptDrain:
    """jfk.wav -> NativeSttBackend.transcribe() must produce the
    canonical transcript with monotonic, well-ordered segments."""

    def test_transcript_matches_canonical_jfk(self) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        audio = _load_jfk_pcm_f32()
        backend = NativeSttBackend()
        backend.load_model("whisper-tiny")
        try:
            result = backend.transcribe(audio, sample_rate_hz=16000)
        finally:
            backend.close()

        assert result.text, "empty transcript"
        # Whitespace + leading-space tolerant match.
        assert _normalize_transcript(result.text) == _normalize_transcript(
            _CANONICAL_TRANSCRIPT
        ), f"canonical drift: got {result.text!r}, want {_CANONICAL_TRANSCRIPT!r}"

    def test_segments_are_well_ordered(self) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        audio = _load_jfk_pcm_f32()
        backend = NativeSttBackend()
        backend.load_model("whisper-tiny")
        try:
            result = backend.transcribe(audio, sample_rate_hz=16000)
        finally:
            backend.close()

        # At least one segment.
        assert len(result.segments) >= 1
        # Each segment well-formed.
        for seg in result.segments:
            assert seg.start_ms <= seg.end_ms, f"segment has end before start: {seg!r}"
            assert seg.text, f"segment has empty text: {seg!r}"
        # Spans non-decreasing in start_ms.
        starts = [s.start_ms for s in result.segments]
        assert starts == sorted(starts), f"segments not start-time-sorted: {starts}"
        # duration_ms reasonable (jfk.wav is ~11s).
        assert 8_000 <= result.duration_ms <= 13_000, f"unexpected duration_ms={result.duration_ms}"


# ---------------------------------------------------------------------------
# Test 4 — bad-digest path -> CHECKSUM_MISMATCH (not INFERENCE_FAILED)
# ---------------------------------------------------------------------------


class TestBadDigest:
    """Stage a tampered artifact via env override; backend must
    surface ``CHECKSUM_MISMATCH`` (or ``MODEL_NOT_FOUND`` if the
    runtime can't find the file at all). Either is a typed bounded
    code; what is NOT acceptable is a flat ``INFERENCE_FAILED`` or a
    silent fallback."""

    def test_tampered_digest_raises_typed_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        # Write a bogus "ggml-tiny.bin" that the runtime will reject
        # at digest verification time. Same filename so error messages
        # carry the recognizable "digest" substring.
        tampered = tmp_path / "ggml-tiny.bin"
        tampered.write_bytes(b"NOT_A_REAL_GGML_MODEL" * 1024)
        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", str(tampered))

        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("whisper-tiny")
        # The runtime returns OCT_STATUS_UNSUPPORTED with last_error
        # mentioning "digest"; the SDK maps that to CHECKSUM_MISMATCH.
        # If the runtime fails earlier (e.g. before digest check) we
        # still get a typed code — but explicitly NOT INFERENCE_FAILED
        # (which would suggest a runtime invariant violation rather
        # than caller-fixable misconfiguration).
        assert exc_info.value.code != OctomilErrorCode.INFERENCE_FAILED
        assert exc_info.value.code in {
            OctomilErrorCode.CHECKSUM_MISMATCH,
            OctomilErrorCode.RUNTIME_UNAVAILABLE,
            OctomilErrorCode.MODEL_NOT_FOUND,
            OctomilErrorCode.INVALID_INPUT,
        }


# ---------------------------------------------------------------------------
# Test 5 — NaN/Inf input -> INVALID_INPUT
# ---------------------------------------------------------------------------


class TestInvalidAudio:
    """The runtime rejects NaN/Inf samples with INVALID_INPUT; the SDK
    pre-flight catches it Python-side. Both paths must surface as
    ``INVALID_INPUT`` — not crash, not silent."""

    def test_nan_audio_invalid_input(self) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        backend = NativeSttBackend()
        backend.load_model("whisper-tiny")
        try:
            arr = _load_jfk_pcm_f32().copy()
            arr[100] = float("nan")
            with pytest.raises(OctomilError) as exc_info:
                backend.transcribe(arr, sample_rate_hz=16000)
            assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()

    def test_inf_audio_invalid_input(self) -> None:
        from octomil.runtime.native.stt_backend import NativeSttBackend

        backend = NativeSttBackend()
        backend.load_model("whisper-tiny")
        try:
            arr = _load_jfk_pcm_f32().copy()
            arr[200] = float("inf")
            with pytest.raises(OctomilError) as exc_info:
                backend.transcribe(arr, sample_rate_hz=16000)
            assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Test 6 — cancel mid-decode -> CANCELLED
# ---------------------------------------------------------------------------


class TestCancel:
    """Caller-driven cancel during a transcribe() call must surface as
    ``OctomilErrorCode.CANCELLED``.

    The native session is single-thread-affine for send/poll, but
    ``oct_session_cancel`` is the explicit cross-thread escape hatch
    in the ABI contract. We exercise it via a helper thread that
    fires cancel during a slow transcribe."""

    def test_cancel_during_transcribe(self) -> None:
        import threading
        import time

        from octomil.runtime.native.stt_backend import NativeSttBackend

        backend = NativeSttBackend()
        backend.load_model("whisper-tiny")
        try:
            audio = _load_jfk_pcm_f32()

            # Fire a cancel via the runtime adapter on a slight delay.
            # Strategy: wrap the runtime's open_session to register
            # the active session into a list, then cancel from a
            # sidecar thread. The runtime + caller together produce
            # CANCELLED on the next poll if the cancel lands
            # in-flight.
            #
            # Race tolerance: jfk.wav is short (~10s decoded; native
            # path is ~0.4s wall-clock) so cancel may not land
            # before SESSION_COMPLETED. We assert ONE of two
            # well-formed outcomes:
            #   1. transcribe() raised OctomilError(CANCELLED) →
            #      cancel landed.
            #   2. transcribe() returned a normal result → cancel
            #      lost the race; not a regression.
            # An INFERENCE_FAILED or any other error code IS a
            # regression and fails the test.
            sessions: list = []
            real_open_session = backend._runtime.open_session  # type: ignore[union-attr]

            def tracking_open_session(*args, **kwargs):
                sess = real_open_session(*args, **kwargs)
                sessions.append(sess)
                return sess

            backend._runtime.open_session = tracking_open_session  # type: ignore[union-attr,assignment]

            def fire_cancel() -> None:
                for _ in range(200):
                    if sessions:
                        break
                    time.sleep(0.001)
                if sessions:
                    sessions[0].cancel()

            t = threading.Thread(target=fire_cancel, daemon=True)
            t.start()
            outcome: str = ""
            err: OctomilError | None = None
            try:
                backend.transcribe(audio, sample_rate_hz=16000)
                outcome = "completed"
            except OctomilError as exc:
                outcome = "raised"
                err = exc
            finally:
                t.join(timeout=2.0)

            if outcome == "raised":
                assert err is not None
                # Cancel landed → CANCELLED. Anything else is a regression.
                assert err.code == OctomilErrorCode.CANCELLED, (
                    f"cancel landed but produced unexpected code {err.code}; "
                    f"INFERENCE_FAILED here would be a regression"
                )
            # outcome == "completed": cancel lost the race; not a
            # regression. Test passes silently.
        finally:
            backend.close()
