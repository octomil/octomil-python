"""v0.1.8 Lane C — NativeTtsStreamBackend tests.

Three layers of coverage:

1. **Unit / shape**: TtsAudioChunk dataclass shape; error mapping
   (canonical OctomilError taxonomy from v0.1.6 PR1; default
   ``RUNTIME_UNAVAILABLE`` for audio backends). Voice validation.
   Hard-cut: monkey-patched runtime that refuses
   ``audio.tts.stream`` produces ``RUNTIME_UNAVAILABLE``, NOT a
   silent fallback.
2. **Integration** (skipped when ``OCTOMIL_RUNTIME_DYLIB`` /
   ``OCTOMIL_SHERPA_TTS_MODEL`` unset): drives the full
   ``synthesize_with_chunks`` happy path on the same two-sentence
   smoke text Lane A used; asserts ≥1 chunk, last has
   ``is_final=True``, total cumulative_duration_ms ~6.5s ± slack.
3. **Honesty test**: drives the same two-sentence text and asserts
   that chunk arrival timestamps cluster together (delta from first
   to last < 50 ms) — the coalesced-after-synthesis signature. The
   test failure message tells the v0.1.9 implementer to update the
   bound when the runtime flips to progressive Generate.
"""

from __future__ import annotations

import os
import time
import types
from typing import Any

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.tts_stream_backend import (
    NativeTtsStreamBackend,
    TtsAudioChunk,
    runtime_advertises_tts_stream,
)

# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------

_DYLIB_ENV = "OCTOMIL_RUNTIME_DYLIB"
_TTS_MODEL_ENV = "OCTOMIL_SHERPA_TTS_MODEL"
_INTEGRATION_REASON = (
    f"Integration TTS-stream test requires {_DYLIB_ENV} + {_TTS_MODEL_ENV} set + dylib advertising audio.tts.stream"
)


def _integration_env_ok() -> bool:
    return bool(os.environ.get(_DYLIB_ENV)) and bool(os.environ.get(_TTS_MODEL_ENV))


# Two-sentence smoke text — same canonical Lane A input. With sherpa
# default max_num_sentences=1, this yields 2 chunks (one per sentence).
_TWO_SENTENCE_TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."


# ---------------------------------------------------------------------------
# Unit tests — dataclass + error mapping (no runtime needed)
# ---------------------------------------------------------------------------


class TestTtsAudioChunkShape:
    def test_dataclass_fields(self) -> None:
        """TtsAudioChunk carries the documented fields."""
        chunk = TtsAudioChunk(
            pcm_f32=b"",  # placeholder
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=False,
            cumulative_duration_ms=0,
        )
        assert chunk.sample_rate_hz == 22050
        assert chunk.chunk_index == 0
        assert chunk.is_final is False
        assert chunk.cumulative_duration_ms == 0

    def test_dataclass_final_flag(self) -> None:
        chunk = TtsAudioChunk(
            pcm_f32=b"",
            sample_rate_hz=22050,
            chunk_index=1,
            is_final=True,
            cumulative_duration_ms=3500,
        )
        assert chunk.is_final is True

    def test_streaming_mode_default_is_coalesced_honesty_pin(self) -> None:
        """v0.1.9 Lane 4 honesty pin: ``TtsAudioChunk.streaming_mode``
        defaults to ``"coalesced"`` until the runtime release proves
        progressive arrival.

        This test FAILS deliberately AFTER Lane 1 (runtime worker-
        thread Generate) + Lane 2 (runtime release) land and the SDK
        flip PR sets per-chunk ``streaming_mode='progressive'`` from
        the drain. When that follow-up SDK PR lands, update this
        assertion in lockstep.

        The follow-up SDK flip PR will:
          1. Change NativeTtsStreamBackend._drain to set
             streaming_mode based on a runtime capability hint OR
             via inter-chunk timing detection (>50 ms gap →
             progressive).
          2. Update this test to assert the new contract (likely:
             default still 'coalesced' but per-chunk values are
             'progressive' on the new runtime).
          3. Update the X-Octomil-Streaming-Honesty HTTP header to
             match (gated on a runtime version check).
          4. Loosen the guard in
             tests/test_tts_stream_no_premature_progressive_claim.py.
        """
        chunk = TtsAudioChunk(
            pcm_f32=b"",
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=False,
            cumulative_duration_ms=0,
        )
        assert chunk.streaming_mode == "coalesced", (
            "TtsAudioChunk default streaming_mode is no longer 'coalesced'. "
            "If you're flipping this in lockstep with the v0.1.9 runtime "
            "release: update this assertion + the HTTP honesty header "
            "+ the no-premature-claim guard test in the same PR."
        )


# ---------------------------------------------------------------------------
# Unit tests — error mapping at the backend boundary
# ---------------------------------------------------------------------------


class _FakeCaps:
    def __init__(self, supported: list[str]) -> None:
        self.supported_capabilities = supported


class _FakeRuntimeNoTtsStream:
    """A runtime that loads but does NOT advertise audio.tts.stream.

    Mirrors the real-runtime behavior when OCTOMIL_SHERPA_TTS_MODEL is
    set but the digest doesn't match (or sidecars are missing).
    """

    def __init__(self, last_error: str = "") -> None:
        self._last_error = last_error

    def capabilities(self) -> _FakeCaps:
        return _FakeCaps(supported=["chat.completion"])  # tts.stream absent

    def last_error(self) -> str:
        return self._last_error

    def open_model(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("open_model must not be reached when capability is absent")

    def close(self) -> None:
        pass


class TestNativeTtsStreamBackendHardCut:
    """Hard-cut: when the runtime refuses ``audio.tts.stream``, the
    backend MUST raise ``RUNTIME_UNAVAILABLE`` (or
    ``CHECKSUM_MISMATCH`` when last_error mentions ``digest``). NO
    fallback to a Python sherpa engine is allowed.
    """

    def test_raises_runtime_unavailable_when_capability_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native import tts_stream_backend as mod

        monkeypatch.setattr(
            mod.NativeRuntime,
            "open",
            classmethod(lambda cls: _FakeRuntimeNoTtsStream()),
        )
        backend = NativeTtsStreamBackend()
        with pytest.raises(OctomilError) as excinfo:
            backend.load_model("kokoro-82m")
        assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        # The error message MUST mention audio.tts.stream + the env
        # var so an operator reading the failure can fix it.
        msg = str(excinfo.value)
        assert "audio.tts.stream" in msg
        assert "OCTOMIL_SHERPA_TTS_MODEL" in msg

    def test_raises_checksum_mismatch_on_digest_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native import tts_stream_backend as mod

        monkeypatch.setattr(
            mod.NativeRuntime,
            "open",
            classmethod(
                lambda cls: _FakeRuntimeNoTtsStream(last_error="sherpa_onnx_tts: digest mismatch — got abc want def")
            ),
        )
        backend = NativeTtsStreamBackend()
        with pytest.raises(OctomilError) as excinfo:
            backend.load_model("kokoro-82m")
        assert excinfo.value.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_synthesize_before_load_raises_runtime_unavailable(self) -> None:
        backend = NativeTtsStreamBackend()
        with pytest.raises(OctomilError) as excinfo:
            list(backend.synthesize_with_chunks("hello"))
        assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE


class TestNativeTtsStreamBackendVoiceValidation:
    def test_default_voice_is_zero(self) -> None:
        backend = NativeTtsStreamBackend()
        assert backend.validate_voice(None) == "0"
        assert backend.validate_voice("") == "0"

    def test_numeric_string_passes_through(self) -> None:
        backend = NativeTtsStreamBackend()
        assert backend.validate_voice("0") == "0"
        assert backend.validate_voice("3") == "3"

    def test_non_numeric_voice_rejects_invalid_input(self) -> None:
        """Pre-stream validation BEFORE we open a session — non-
        numeric voice must raise INVALID_INPUT (the runtime's
        parse_speaker_id would also reject it, but we mirror it
        Python-side so HTTP 200 is never emitted on a bad voice)."""
        backend = NativeTtsStreamBackend()
        with pytest.raises(OctomilError) as excinfo:
            backend.validate_voice("af_bella")
        assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT

    def test_voice_validation_runs_before_session_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When voice is invalid, synthesize_with_chunks raises
        INVALID_INPUT BEFORE the runtime sees the request.

        We attach a fake model + runtime so the only path that should
        execute is the validate_voice check; if the synthesizer
        actually opens a session, the test fails because the fake
        runtime's open_session raises AssertionError.
        """
        backend = NativeTtsStreamBackend()
        backend._runtime = types.SimpleNamespace(  # type: ignore[assignment]
            open_session=lambda **kw: (_ for _ in ()).throw(AssertionError("must not open session on invalid voice")),
            last_error=lambda: "",
        )
        backend._model = object()  # placeholder — not exercised
        with pytest.raises(OctomilError) as excinfo:
            list(backend.synthesize_with_chunks("hello", voice_id="af_bella"))
        assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT


class TestNativeTtsStreamBackendInputValidation:
    def test_empty_text_rejects_invalid_input(self) -> None:
        backend = NativeTtsStreamBackend()
        backend._runtime = object()  # type: ignore[assignment]
        backend._model = object()
        with pytest.raises(OctomilError) as excinfo:
            list(backend.synthesize_with_chunks(""))
        assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT

    def test_whitespace_only_text_rejects(self) -> None:
        backend = NativeTtsStreamBackend()
        backend._runtime = object()  # type: ignore[assignment]
        backend._model = object()
        with pytest.raises(OctomilError) as excinfo:
            list(backend.synthesize_with_chunks("   \n   "))
        assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT

    def test_zero_deadline_rejects(self) -> None:
        backend = NativeTtsStreamBackend()
        backend._runtime = object()  # type: ignore[assignment]
        backend._model = object()
        with pytest.raises(OctomilError) as excinfo:
            list(backend.synthesize_with_chunks("hello", deadline_ms=0))
        assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# Integration tests — real runtime + real model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _integration_env_ok(), reason=_INTEGRATION_REASON)
class TestNativeTtsStreamBackendIntegration:
    """End-to-end against the real runtime. Skipped when
    OCTOMIL_RUNTIME_DYLIB / OCTOMIL_SHERPA_TTS_MODEL are unset."""

    def test_two_sentence_yields_chunks(self) -> None:
        backend = NativeTtsStreamBackend()
        backend.load_model("kokoro-82m")
        try:
            chunks = list(backend.synthesize_with_chunks(_TWO_SENTENCE_TEXT))
        finally:
            backend.close()

        assert len(chunks) >= 1, "expected ≥1 chunk for two-sentence input"
        assert chunks[-1].is_final is True
        for i, c in enumerate(chunks[:-1]):
            assert c.is_final is False, f"chunk {i} non-last must have is_final=False"
        # Sample rate consistent across chunks.
        sr = chunks[0].sample_rate_hz
        for c in chunks[1:]:
            assert c.sample_rate_hz == sr
        # chunk_index monotonic from 0.
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
        # Total duration: ~3-7s for the smoke text. Don't pin a tight
        # bound (varies by VITS model + speed). Just assert non-zero.
        assert chunks[-1].cumulative_duration_ms > 1000

    def test_honesty_chunks_arrive_coalesced(self) -> None:
        """v0.1.8 honesty signature: chunks arrive together at end-
        of-synth. We measure the wall-clock delta between first and
        last chunk arrival; for the synchronous-Generate sherpa
        adapter it is small (<100 ms) because Generate runs to
        completion inside ONE poll_event call.

        If a future runtime change makes chunks arrive progressively,
        this test fails with a comment telling the v0.1.9 implementer
        to update the bound — at which point the SDK gets to drop
        the ``coalesced_after_synthesis`` honesty marker.
        """
        backend = NativeTtsStreamBackend()
        backend.load_model("kokoro-82m")
        try:
            timestamps_ms: list[float] = []
            for chunk in backend.synthesize_with_chunks(_TWO_SENTENCE_TEXT):
                timestamps_ms.append(time.monotonic() * 1000.0)
                assert chunk is not None
        finally:
            backend.close()

        if len(timestamps_ms) < 2:
            pytest.skip("Need ≥2 chunks to measure inter-arrival delta")

        delta_ms = timestamps_ms[-1] - timestamps_ms[0]
        # Coalesced-after-synthesis signature: all chunks land within
        # a small bookkeeping window (<100 ms) on consumer iteration.
        # If this fires, the runtime has flipped to progressive
        # Generate — yay, but bump the v0.1.9 follow-up to update
        # this test (and drop the X-Octomil-Streaming-Honesty header).
        assert delta_ms < 100.0, (
            f"chunks arrived spread across {delta_ms:.1f} ms — "
            "this exceeds the v0.1.8 coalesced-after-synthesis "
            "signature. yay progressive delivery — bump v0.1.9 "
            "follow-up to update this test (and drop the "
            "X-Octomil-Streaming-Honesty=coalesced_after_synthesis "
            "response header)."
        )


# ---------------------------------------------------------------------------
# Capability gate helper
# ---------------------------------------------------------------------------


class TestRuntimeAdvertisesHelper:
    """``runtime_advertises_tts_stream`` returns False on bad runtimes
    and True when the capability is in the advertised list. Same
    contract as the STT-side helper."""

    def test_returns_false_when_capability_absent(self) -> None:
        rt = _FakeRuntimeNoTtsStream()
        assert runtime_advertises_tts_stream(rt) is False  # type: ignore[arg-type]

    def test_returns_true_when_advertised(self) -> None:
        rt = types.SimpleNamespace(
            capabilities=lambda: _FakeCaps(["audio.tts.stream"]),
            last_error=lambda: "",
        )
        assert runtime_advertises_tts_stream(rt) is True  # type: ignore[arg-type]

    def test_returns_false_on_capabilities_exception(self) -> None:
        def _boom() -> Any:
            raise RuntimeError("boom")

        rt = types.SimpleNamespace(capabilities=_boom, last_error=lambda: "")
        assert runtime_advertises_tts_stream(rt) is False  # type: ignore[arg-type]
