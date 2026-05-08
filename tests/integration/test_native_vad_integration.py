"""Integration tests — NativeVadBackend against real liboctomil-runtime.

Exercises the full cffi path:
``oct_runtime_open → oct_session_open(capability="audio.vad") →
oct_session_send_audio → oct_session_poll_event drain →
oct_session_close``.

Requires:
  * ``OCTOMIL_RUNTIME_DYLIB`` (or a fetched dev cache) pointing at a
    liboctomil-runtime built with ``OCT_ENABLE_ENGINE_SILERO_VAD=ON``
    (which transitively requires ``OCT_ENABLE_ENGINE_WHISPER_CPP=ON``)
    and ABI minor >= 9.
  * ``OCTOMIL_SILERO_VAD_MODEL`` pointing at a verified
    ``ggml-silero-v6.2.0.bin`` (SHA-256
    ``2aa269b7…fb6987``, 885 098 bytes).
  * ``research/engines/whisper.cpp/samples/jfk.wav`` on disk.

When any of those preconditions are missing the entire module skips
cleanly. The runtime PR-2F smoke produces 4 SPEECH_START + 4
SPEECH_END events on jfk.wav (segments at 0.32-2.27, 3.27-4.41,
5.38-7.68, 8.16-10.62 s); we mirror that expectation here for the
end-to-end SDK path.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode

_JFK_WAV = Path("/Users/seanb/Developer/Octomil/research/engines/whisper.cpp/samples/jfk.wav")


def _skip_reason() -> str | None:
    if not _JFK_WAV.is_file():
        return f"jfk.wav missing at {_JFK_WAV}"
    if not os.environ.get("OCTOMIL_SILERO_VAD_MODEL"):
        return "OCTOMIL_SILERO_VAD_MODEL not set"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _load_jfk_pcm_f32() -> np.ndarray:
    with wave.open(str(_JFK_WAV), "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000, f"jfk.wav sample rate {sr} != 16000"
        pcm = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Test 1 — capability is advertised
# ---------------------------------------------------------------------------


class TestPlannerSelection:
    """When env preconditions hold, the planner-style helper returns
    True so the SDK selects ``NativeVadBackend`` over a hypothetical
    legacy path. (There is no Python VAD product path at this minor
    — this confirms native is selectable, not that fallback is
    bypassed.)"""

    def test_runtime_advertises_audio_vad(self) -> None:
        from octomil.runtime.native.loader import NativeRuntime
        from octomil.runtime.native.vad_backend import runtime_advertises_audio_vad

        rt = NativeRuntime.open()
        try:
            assert runtime_advertises_audio_vad(rt) is True
            caps = rt.capabilities()
            assert "audio.vad" in caps.supported_capabilities
        finally:
            rt.close()


# ---------------------------------------------------------------------------
# Test 2 — fallback is unreachable (no Python VAD path exists)
# ---------------------------------------------------------------------------


class TestNoFallback:
    """When the runtime stops advertising the capability mid-test
    (operator unset env), the SDK MUST raise typed ``OctomilError``
    rather than silently route to a non-existent Python VAD path."""

    def test_unset_silero_vad_raises_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.vad_backend import NativeVadBackend

        monkeypatch.delenv("OCTOMIL_SILERO_VAD_MODEL", raising=False)
        backend = NativeVadBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.open()
        # Either RUNTIME_UNAVAILABLE (capability not advertised) or
        # CHECKSUM_MISMATCH if the runtime came up partially. Both
        # are typed; what is NOT acceptable is a silent fallback.
        assert exc_info.value.code in {
            OctomilErrorCode.RUNTIME_UNAVAILABLE,
            OctomilErrorCode.CHECKSUM_MISMATCH,
        }


# ---------------------------------------------------------------------------
# Test 3 — full smoke against jfk.wav
# ---------------------------------------------------------------------------


class TestFullSmoke:
    """Runtime PR-2F smoke produces 4 SPEECH_START + 4 SPEECH_END on
    jfk.wav (segments at 0.32-2.27, 3.27-4.41, 5.38-7.68, 8.16-10.62
    s). We mirror that expectation through the SDK path. Tolerance:
    >= 1 of each, alternating START / END strictly, with confidences
    in [0.0, 1.0]."""

    def test_jfk_wav_drains_alternating_transitions(self) -> None:
        from octomil.runtime.native.vad_backend import NativeVadBackend

        audio = _load_jfk_pcm_f32()
        backend = NativeVadBackend()
        backend.open()
        try:
            with backend.open_session() as session:
                session.feed_chunk(audio, sample_rate_hz=16000)
                transitions = list(session.poll_transitions(drain_until_completed=True))
        finally:
            backend.close()

        # The runtime produces strictly alternating START/END.
        kinds = [t.kind for t in transitions]
        starts = [t for t in transitions if t.kind == "speech_start"]
        ends = [t for t in transitions if t.kind == "speech_end"]
        assert len(starts) >= 1, f"expected >=1 SPEECH_START on jfk.wav; got transitions {kinds}"
        # PR-2F smoke produces exactly 4+4; allow flex in case minor
        # whisper.cpp upstream parameter drift changes counts by ±1.
        assert 3 <= len(starts) <= 5, f"unexpected SPEECH_START count {len(starts)}"
        assert len(ends) == len(starts), f"unmatched START/END pairs: {len(starts)} vs {len(ends)}; transitions={kinds}"
        # Strict alternation.
        for i, t in enumerate(transitions):
            expected = "speech_start" if i % 2 == 0 else "speech_end"
            assert t.kind == expected, f"non-alternating transition at idx {i}: got {t.kind!r}, expected {expected!r}"
        # Confidences clamped to [0.0, 1.0].
        for t in transitions:
            assert 0.0 <= t.confidence <= 1.0, f"confidence out of range: {t!r}"
        # Timestamps non-decreasing (each END after its matching START).
        ts = [t.timestamp_ms for t in transitions]
        assert ts == sorted(ts), f"timestamps not monotonic: {ts}"


# ---------------------------------------------------------------------------
# Test 4 — bad-digest path -> CHECKSUM_MISMATCH (not INFERENCE_FAILED)
# ---------------------------------------------------------------------------


class TestBadDigest:
    """Stage a tampered artifact via env override; backend must
    surface ``CHECKSUM_MISMATCH`` (or ``MODEL_NOT_FOUND`` if the
    runtime can't find the file at all). Either is a typed bounded
    code; what is NOT acceptable is ``INFERENCE_FAILED`` (which would
    suggest an internal runtime invariant violation rather than
    caller-fixable misconfiguration)."""

    def test_tampered_digest_raises_typed_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.vad_backend import NativeVadBackend

        tampered = tmp_path / "ggml-silero-v6.2.0.bin"
        # Same approximate size as the canonical 885 KB; size-based
        # gate (if any) is bypassed — we want the digest gate to fire.
        tampered.write_bytes(b"NOT_A_REAL_SILERO_MODEL" * (885_000 // 23))
        monkeypatch.setenv("OCTOMIL_SILERO_VAD_MODEL", str(tampered))

        backend = NativeVadBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.open()
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
    def test_nan_audio_invalid_input(self) -> None:
        from octomil.runtime.native.vad_backend import NativeVadBackend

        backend = NativeVadBackend()
        backend.open()
        try:
            arr = _load_jfk_pcm_f32().copy()
            arr[100] = float("nan")
            with backend.open_session() as session:
                with pytest.raises(OctomilError) as exc_info:
                    session.feed_chunk(arr, sample_rate_hz=16000)
                assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()

    def test_wrong_sample_rate_invalid_input(self) -> None:
        from octomil.runtime.native.vad_backend import NativeVadBackend

        backend = NativeVadBackend()
        backend.open()
        try:
            with pytest.raises(OctomilError) as exc_info:
                # sample-rate gate fires at open_session time.
                backend.open_session(sample_rate_hz=8000)
            assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Test 6 — cancel during VAD decode
# ---------------------------------------------------------------------------


class TestCancel:
    """Caller-driven cancel during a VAD session must surface as
    ``CANCELLED``. The native session is single-thread-affine for
    send/poll, but ``oct_session_cancel`` is the explicit cross-thread
    escape hatch. We exercise it via a sidecar thread that fires the
    cancel during the drain.

    Race tolerance: the silero VAD decode is fast on jfk.wav, so the
    cancel may not land before SESSION_COMPLETED. Two well-formed
    outcomes:
      1. poll_transitions() raised ``OctomilError(CANCELLED)``.
      2. poll_transitions() returned a normal list of transitions.
    Anything else (INFERENCE_FAILED, other code) IS a regression.
    """

    def test_cancel_during_drain(self) -> None:
        import threading
        import time as _time

        from octomil.runtime.native.vad_backend import NativeVadBackend

        backend = NativeVadBackend()
        backend.open()
        try:
            audio = _load_jfk_pcm_f32()
            outcome: str = ""
            err: OctomilError | None = None
            with backend.open_session() as session:
                session.feed_chunk(audio, sample_rate_hz=16000)

                def fire_cancel() -> None:
                    _time.sleep(0.001)
                    try:
                        # poke the underlying NativeSession.cancel().
                        if session._native_session is not None:
                            session._native_session.cancel()
                    except Exception:  # noqa: BLE001
                        pass

                t = threading.Thread(target=fire_cancel, daemon=True)
                t.start()
                try:
                    list(session.poll_transitions(drain_until_completed=True))
                    outcome = "completed"
                except OctomilError as exc:
                    outcome = "raised"
                    err = exc
                finally:
                    t.join(timeout=2.0)

            if outcome == "raised":
                assert err is not None
                assert err.code == OctomilErrorCode.CANCELLED, (
                    f"cancel landed but produced unexpected code {err.code}; "
                    "INFERENCE_FAILED here would be a regression"
                )
            # outcome == "completed": cancel lost the race; not a regression.
        finally:
            backend.close()
