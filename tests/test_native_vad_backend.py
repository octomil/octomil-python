"""Unit tests for NativeVadBackend (v0.1.5 PR-2N SDK VAD bindings).

These tests run without the dylib by mocking the runtime. The
integration tests
(``tests/integration/test_native_vad_integration.py``) exercise the
real cffi path against ``research/engines/whisper.cpp/samples/jfk.wav``.

What we lock in here:

1. Capability gate: when the runtime advertises ``audio.vad``,
   :meth:`NativeVadBackend.open` succeeds. When it does NOT advertise
   the capability, ``open()`` raises ``RUNTIME_UNAVAILABLE`` — the
   SDK does NOT silently fall back. There is no Python VAD product
   path to fall back to.
2. ``OCTOMIL_SILERO_VAD_MODEL`` digest disambiguation: when the
   runtime hides the capability with a ``"digest"`` substring in
   ``last_error``, the SDK raises ``CHECKSUM_MISMATCH`` instead of
   the generic ``RUNTIME_UNAVAILABLE``.
3. Bounded-error mapping: NOT_FOUND→MODEL_NOT_FOUND;
   INVALID_INPUT→INVALID_INPUT; UNSUPPORTED+digest→CHECKSUM_MISMATCH;
   UNSUPPORTED→RUNTIME_UNAVAILABLE; VERSION_MISMATCH→RUNTIME_UNAVAILABLE;
   CANCELLED→CANCELLED; TIMEOUT→REQUEST_TIMEOUT; default→INFERENCE_FAILED.
4. NaN/Inf samples reject ``INVALID_INPUT`` Python-side (matches
   the runtime's own rejection — saves a session round-trip).
5. Unsupported sample rate (non-16kHz) rejects ``INVALID_INPUT``.
6. Empty / wrong-shape audio buffers reject ``INVALID_INPUT``.
7. Planner-style selector helper
   :func:`runtime_advertises_audio_vad` returns False if the runtime
   fails to introspect or doesn't list the capability.
8. ``VadTransition`` dataclass shape is preserved and the unknown-
   sentinel kind is mapped to the ``"unknown"`` literal so callers
   never see a raw integer kind.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.loader import (
    OCT_STATUS_BUSY,
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    OCT_VAD_TRANSITION_SPEECH_END,
    OCT_VAD_TRANSITION_SPEECH_START,
    OCT_VAD_TRANSITION_UNKNOWN,
    NativeRuntimeError,
)
from octomil.runtime.native.vad_backend import (
    NativeVadBackend,
    VadStreamingSession,
    VadTransition,
    _kind_label,
    _runtime_status_to_sdk_error,
    _validate_chunk_pcm_f32,
    runtime_advertises_audio_vad,
)

# ---------------------------------------------------------------------------
# Bounded-error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_not_found_maps_to_model_not_found(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_NOT_FOUND, "missing")
        assert err.code == OctomilErrorCode.MODEL_NOT_FOUND

    def test_invalid_input_maps_to_invalid_input(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_INVALID_INPUT, "bad")
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_unsupported_with_digest_maps_to_checksum_mismatch(self) -> None:
        err = _runtime_status_to_sdk_error(
            OCT_STATUS_UNSUPPORTED,
            "open failed",
            last_error="ggml-silero-v6.2.0.bin digest mismatch (got abc, want xyz)",
        )
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_unsupported_without_digest_maps_to_runtime_unavailable(self) -> None:
        err = _runtime_status_to_sdk_error(
            OCT_STATUS_UNSUPPORTED,
            "open failed",
            last_error="capability 'audio.vad' not built into this runtime",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_version_mismatch_maps_to_runtime_unavailable(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_VERSION_MISMATCH, "abi skew")
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_cancelled_maps_to_cancelled(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_CANCELLED, "user")
        assert err.code == OctomilErrorCode.CANCELLED

    def test_timeout_maps_to_request_timeout(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_TIMEOUT, "deadline")
        assert err.code == OctomilErrorCode.REQUEST_TIMEOUT

    def test_busy_maps_to_server_error(self) -> None:
        err = _runtime_status_to_sdk_error(OCT_STATUS_BUSY, "saturated")
        assert err.code == OctomilErrorCode.SERVER_ERROR

    def test_unknown_terminal_maps_to_inference_failed(self) -> None:
        err = _runtime_status_to_sdk_error(99, "weird")
        assert err.code == OctomilErrorCode.INFERENCE_FAILED


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------


class TestAudioValidation:
    def test_zero_length_bytes_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(b"", sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "zero-length" in exc_info.value.error_message

    def test_zero_length_array_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(np.zeros(0, dtype=np.float32), sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_misaligned_bytes_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(b"\x00" * 7, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "multiple of 4" in exc_info.value.error_message

    def test_bad_sample_rate_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(np.zeros(160, dtype=np.float32), sample_rate_hz=8000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_rejected(self) -> None:
        arr = np.zeros(160, dtype=np.float32)
        arr[42] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_inf_rejected(self) -> None:
        arr = np.zeros(160, dtype=np.float32)
        arr[7] = float("inf")
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_2d_array_rejected(self) -> None:
        arr = np.zeros((2, 160), dtype=np.float32)
        with pytest.raises(OctomilError) as exc_info:
            _validate_chunk_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "1-D" in exc_info.value.error_message

    def test_clean_buffer_passes(self) -> None:
        arr = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
        out = _validate_chunk_pcm_f32(arr, sample_rate_hz=16000)
        assert isinstance(out, bytes)
        assert len(out) == 480 * 4


# ---------------------------------------------------------------------------
# Capability-honesty helper
# ---------------------------------------------------------------------------


class TestRuntimeAdvertises:
    def test_advertises_when_capability_present(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.vad",))
        assert runtime_advertises_audio_vad(rt) is True

    def test_does_not_advertise_when_capability_absent(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        assert runtime_advertises_audio_vad(rt) is False

    def test_returns_false_on_capabilities_failure(self) -> None:
        rt = MagicMock()
        rt.capabilities.side_effect = RuntimeError("dylib gone")
        assert runtime_advertises_audio_vad(rt) is False


# ---------------------------------------------------------------------------
# _kind_label
# ---------------------------------------------------------------------------


class TestKindLabel:
    def test_speech_start(self) -> None:
        assert _kind_label(OCT_VAD_TRANSITION_SPEECH_START) == "speech_start"

    def test_speech_end(self) -> None:
        assert _kind_label(OCT_VAD_TRANSITION_SPEECH_END) == "speech_end"

    def test_unknown_sentinel(self) -> None:
        assert _kind_label(OCT_VAD_TRANSITION_UNKNOWN) == "unknown"

    def test_future_value_falls_through_to_unknown(self) -> None:
        # The runtime reserves the right to introduce new kinds; the
        # SDK MUST surface them as "unknown" rather than crash.
        assert _kind_label(99) == "unknown"


# ---------------------------------------------------------------------------
# open() gate
# ---------------------------------------------------------------------------


class TestOpenGate:
    """The hard-cut contract: when the runtime declines, the SDK
    raises a typed error rather than falling back to a Python
    implementation."""

    def test_capability_not_advertised_raises_runtime_unavailable(self) -> None:
        backend = NativeVadBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        # Probe path: defaults to a non-digest reason so the SDK
        # routes to RUNTIME_UNAVAILABLE rather than CHECKSUM_MISMATCH.
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.vad failed",
            "audio.vad not built into this runtime build",
        )
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "audio.vad" in exc_info.value.error_message

    def test_capability_missing_with_digest_in_probe_reason_raises_checksum_mismatch(self) -> None:
        """Bad-digest disambiguation: the runtime hides the capability
        when the silero artifact's SHA-256 doesn't match the pinned
        digest. Codex R1 F-01 fix: the SDK extracts the diagnostic by
        attempting a probe ``oct_session_open(capability="audio.vad")``
        — the runtime's dispatch path is the only place that surfaces
        the silero adapter's ``cached_reason_`` (which contains the
        "digest mismatch — got X want Y" string) into thread-local
        ``last_error``. The probe call returns ``OCT_STATUS_UNSUPPORTED``
        and the SDK's ``NativeRuntimeError`` carries that text."""
        backend = NativeVadBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        # Probe path: open_session raises NativeRuntimeError with
        # last_error containing the digest substring.
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.vad failed",
            "silero_vad: OCTOMIL_SILERO_VAD_MODEL=/path digest mismatch — got abc want xyz",
        )
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
            assert "digest" in exc_info.value.error_message.lower()

    def test_capability_missing_without_digest_falls_through_to_runtime_unavailable(self) -> None:
        """When the probe returns a non-digest reason (env unset or
        engine missing), the SDK surfaces ``RUNTIME_UNAVAILABLE``
        with the diagnostic appended."""
        backend = NativeVadBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.vad failed",
            "silero_vad: OCTOMIL_SILERO_VAD_MODEL env var unset",
        )
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "OCTOMIL_SILERO_VAD_MODEL" in exc_info.value.error_message

    def test_capability_missing_probe_silent_falls_through_to_runtime_unavailable(self) -> None:
        """When the probe returns no diagnostic at all (e.g. it
        unexpectedly succeeded — shouldn't happen but is forward-
        compat), the SDK still surfaces ``RUNTIME_UNAVAILABLE``
        rather than crashing."""
        backend = NativeVadBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        # Probe SUCCEEDED — defensive path. Returns a session-like
        # mock that the helper closes silently.
        rt.open_session.return_value = MagicMock()
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_runtime_open_failure_raises_runtime_unavailable(self) -> None:
        backend = NativeVadBackend()
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            side_effect=NativeRuntimeError(OCT_STATUS_VERSION_MISMATCH, "abi skew", "v0.0.7"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_dylib_not_found_raises_runtime_unavailable(self) -> None:
        backend = NativeVadBackend()
        with patch(
            "octomil.runtime.native.vad_backend.NativeRuntime.open",
            side_effect=ImportError("liboctomil-runtime not found"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.open()
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "dylib" in exc_info.value.error_message

    def test_open_idempotent(self) -> None:
        backend = NativeVadBackend()
        backend._initialized = True
        backend._runtime = MagicMock()  # type: ignore[assignment]
        # Should be a no-op; no NativeRuntime.open call.
        with patch("octomil.runtime.native.vad_backend.NativeRuntime.open") as open_mock:
            backend.open()
            open_mock.assert_not_called()


# ---------------------------------------------------------------------------
# open_session — sample rate gate + propagation
# ---------------------------------------------------------------------------


class TestOpenSession:
    def test_unsupported_sample_rate_rejects_invalid_input(self) -> None:
        backend = NativeVadBackend()
        backend._initialized = True
        backend._runtime = MagicMock()  # type: ignore[assignment]
        with pytest.raises(OctomilError) as exc_info:
            backend.open_session(sample_rate_hz=8000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_open_session_propagates_runtime_failure(self) -> None:
        backend = NativeVadBackend()
        backend._initialized = True
        rt = MagicMock()
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "audio.vad not advertised",
            "capability not built in",
        )
        backend._runtime = rt  # type: ignore[assignment]
        with pytest.raises(OctomilError) as exc_info:
            backend.open_session()
        # No "digest" in last_error → RUNTIME_UNAVAILABLE.
        assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE


# ---------------------------------------------------------------------------
# VadTransition dataclass shape
# ---------------------------------------------------------------------------


class TestVadTransitionShape:
    def test_dataclass_basic(self) -> None:
        t = VadTransition(kind="speech_start", timestamp_ms=320, confidence=0.91)
        assert t.kind == "speech_start"
        assert t.timestamp_ms == 320
        assert t.confidence == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# VadStreamingSession — feed / poll lifecycle on a fake NativeSession
# ---------------------------------------------------------------------------


def _fake_event(
    *,
    type_: int,
    vad_transition_kind: int = 0,
    vad_timestamp_ms: int = 0,
    vad_confidence: float = 0.0,
    terminal_status: int = 0,
):
    """Build a duck-typed NativeEvent stand-in for poll_event() returns."""
    ev = MagicMock()
    ev.type = type_
    ev.vad_transition_kind = vad_transition_kind
    ev.vad_timestamp_ms = vad_timestamp_ms
    ev.vad_confidence = vad_confidence
    ev.terminal_status = terminal_status
    return ev


class TestVadStreamingSession:
    def _make_session_with_native(self, native_session: MagicMock) -> VadStreamingSession:
        rt = MagicMock()
        rt.open_session.return_value = native_session
        rt.last_error.return_value = ""
        # Build via __new__ so we don't actually call open_session
        # twice when the unit test wants tight control.
        sess = VadStreamingSession.__new__(VadStreamingSession)
        sess._runtime = rt  # type: ignore[attr-defined]
        sess._sample_rate_hz = 16000  # type: ignore[attr-defined]
        sess._native_session = native_session  # type: ignore[attr-defined]
        sess._closed = False  # type: ignore[attr-defined]
        sess._terminal_seen = False  # type: ignore[attr-defined]
        return sess

    def test_feed_chunk_validates_python_side(self) -> None:
        ns = MagicMock()
        sess = self._make_session_with_native(ns)
        arr = np.zeros(480, dtype=np.float32)
        arr[3] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            sess.feed_chunk(arr)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        ns.send_audio.assert_not_called()

    def test_feed_chunk_propagates_native_send_audio_failure(self) -> None:
        ns = MagicMock()
        ns.send_audio.side_effect = NativeRuntimeError(
            OCT_STATUS_INVALID_INPUT, "send_audio failed", "wrong sample rate"
        )
        sess = self._make_session_with_native(ns)
        arr = np.zeros(480, dtype=np.float32)
        with pytest.raises(OctomilError) as exc_info:
            sess.feed_chunk(arr)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_poll_transitions_drains_start_then_end(self) -> None:
        ns = MagicMock()
        events = [
            _fake_event(type_=1),  # SESSION_STARTED — ignored
            _fake_event(
                type_=24,
                vad_transition_kind=OCT_VAD_TRANSITION_SPEECH_START,
                vad_timestamp_ms=320,
                vad_confidence=0.95,
            ),
            _fake_event(
                type_=24,
                vad_transition_kind=OCT_VAD_TRANSITION_SPEECH_END,
                vad_timestamp_ms=2270,
                vad_confidence=0.88,
            ),
            _fake_event(type_=8, terminal_status=0),  # SESSION_COMPLETED OK
        ]
        ns.poll_event.side_effect = events
        sess = self._make_session_with_native(ns)
        out = list(sess.poll_transitions(drain_until_completed=True))
        assert len(out) == 2
        assert out[0].kind == "speech_start"
        assert out[0].timestamp_ms == 320
        assert out[0].confidence == pytest.approx(0.95)
        assert out[1].kind == "speech_end"
        assert out[1].timestamp_ms == 2270

    def test_poll_transitions_skips_unknown_kind(self) -> None:
        """Future-compat: a hypothetical newer runtime emitting a
        novel transition_kind value MUST NOT crash the SDK; the
        binding skips the event silently."""
        ns = MagicMock()
        events = [
            _fake_event(
                type_=24,
                vad_transition_kind=42,  # novel
                vad_timestamp_ms=10,
                vad_confidence=0.5,
            ),
            _fake_event(
                type_=24,
                vad_transition_kind=OCT_VAD_TRANSITION_SPEECH_START,
                vad_timestamp_ms=320,
                vad_confidence=0.9,
            ),
            _fake_event(type_=8, terminal_status=0),
        ]
        ns.poll_event.side_effect = events
        sess = self._make_session_with_native(ns)
        out = list(sess.poll_transitions(drain_until_completed=True))
        assert len(out) == 1
        assert out[0].kind == "speech_start"

    def test_poll_transitions_session_completed_non_ok_raises(self) -> None:
        ns = MagicMock()
        ns.poll_event.return_value = _fake_event(type_=8, terminal_status=OCT_STATUS_INVALID_INPUT)
        sess = self._make_session_with_native(ns)
        with pytest.raises(OctomilError) as exc_info:
            list(sess.poll_transitions(drain_until_completed=True))
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_close_idempotent(self) -> None:
        ns = MagicMock()
        sess = self._make_session_with_native(ns)
        sess.close()
        sess.close()
        ns.close.assert_called_once()

    def test_feed_chunk_after_close_raises_runtime_unavailable(self) -> None:
        ns = MagicMock()
        sess = self._make_session_with_native(ns)
        sess.close()
        with pytest.raises(OctomilError) as exc_info:
            sess.feed_chunk(np.zeros(480, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
