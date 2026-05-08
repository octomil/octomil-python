"""Unit tests for NativeSpeakerEmbeddingBackend (v0.1.5 PR-2N).

These tests run without the dylib by mocking the runtime. Integration
tests live in
``tests/integration/test_native_speaker_integration.py`` and exercise
the real cffi path against jfk.wav.

What we lock in here:

1. Capability gate: when the runtime advertises
   ``audio.speaker.embedding``, :meth:`load_model` succeeds. When it
   does NOT advertise the capability, ``load_model`` raises
   ``RUNTIME_UNAVAILABLE`` — the SDK does NOT silently fall back.
   There is no Python speaker-embedding product path.
2. ``OCTOMIL_SHERPA_SPEAKER_MODEL`` must be set; otherwise
   ``RUNTIME_UNAVAILABLE``.
3. Bounded-error mapping: NOT_FOUND → MODEL_NOT_FOUND;
   INVALID_INPUT → INVALID_INPUT; UNSUPPORTED+digest →
   CHECKSUM_MISMATCH; UNSUPPORTED → RUNTIME_UNAVAILABLE;
   VERSION_MISMATCH → RUNTIME_UNAVAILABLE; CANCELLED → CANCELLED;
   TIMEOUT → REQUEST_TIMEOUT; default → INFERENCE_FAILED.
4. NaN/Inf samples reject ``INVALID_INPUT`` Python-side.
5. Unsupported sample rate (non-16kHz) rejects ``INVALID_INPUT``.
6. Empty / wrong-shape audio rejects ``INVALID_INPUT``.
7. Model-name gate: only ``sherpa-eres2netv2-base`` is wired in
   v0.1.5; other names reject ``UNSUPPORTED_MODALITY``.
8. ``embed()`` before ``load_model()`` raises ``RUNTIME_UNAVAILABLE``.
9. Planner-style selector helper returns False if the runtime fails
   to introspect or doesn't list the capability.
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
    NativeRuntimeError,
)
from octomil.runtime.native.speaker_backend import (
    NativeSpeakerEmbeddingBackend,
    _runtime_status_to_sdk_error,
    _validate_clip_pcm_f32,
    runtime_advertises_audio_speaker_embedding,
)

_FAKE_SPEAKER_BIN = "/tmp/_pr2n_test_fake_eres2net.onnx"


@pytest.fixture(autouse=True)
def _set_speaker_bin_env(monkeypatch: pytest.MonkeyPatch):
    """Default to a fake path so the load_model gate is exercised
    deterministically; individual tests override when they need to."""
    monkeypatch.setenv("OCTOMIL_SHERPA_SPEAKER_MODEL", _FAKE_SPEAKER_BIN)
    yield


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
            last_error="ERes2NetV2 digest mismatch (got abc, want xyz)",
        )
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_unsupported_without_digest_maps_to_runtime_unavailable(self) -> None:
        err = _runtime_status_to_sdk_error(
            OCT_STATUS_UNSUPPORTED,
            "open failed",
            last_error="capability 'audio.speaker.embedding' not built into this runtime",
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
            _validate_clip_pcm_f32(b"", sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_zero_length_array_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(np.zeros(0, dtype=np.float32), sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_misaligned_bytes_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(b"\x00" * 7, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_bad_sample_rate_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(np.zeros(160, dtype=np.float32), sample_rate_hz=8000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_rejected(self) -> None:
        arr = np.zeros(16000, dtype=np.float32)
        arr[42] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_inf_rejected(self) -> None:
        arr = np.zeros(16000, dtype=np.float32)
        arr[7] = float("inf")
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_2d_array_rejected(self) -> None:
        arr = np.zeros((2, 16000), dtype=np.float32)
        with pytest.raises(OctomilError) as exc_info:
            _validate_clip_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_clean_buffer_passes(self) -> None:
        arr = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        out = _validate_clip_pcm_f32(arr, sample_rate_hz=16000)
        assert isinstance(out, bytes)
        assert len(out) == 16000 * 4


# ---------------------------------------------------------------------------
# Capability-honesty helper
# ---------------------------------------------------------------------------


class TestRuntimeAdvertises:
    def test_advertises_when_capability_present(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.speaker.embedding",))
        assert runtime_advertises_audio_speaker_embedding(rt) is True

    def test_does_not_advertise_when_capability_absent(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        assert runtime_advertises_audio_speaker_embedding(rt) is False

    def test_returns_false_on_capabilities_failure(self) -> None:
        rt = MagicMock()
        rt.capabilities.side_effect = RuntimeError("dylib gone")
        assert runtime_advertises_audio_speaker_embedding(rt) is False


# ---------------------------------------------------------------------------
# Model-name gate
# ---------------------------------------------------------------------------


class TestModelNameGate:
    """Only ``sherpa-eres2netv2-base`` is wired in v0.1.5; substituting
    a different speaker model identity is a correctness regression
    (different embedding manifold). Other names must reject
    ``UNSUPPORTED_MODALITY`` — never silently substitute the canonical
    one."""

    def test_unsupported_name_rejects(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("ecapa-tdnn-something")
        assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY
        assert "ecapa-tdnn-something" in exc_info.value.error_message

    def test_canonical_name_passes_validation(self) -> None:
        """The canonical name passes the validator gate; the test
        stops short of actually opening a runtime (which would touch
        the dylib). We confirm the load_model call goes past the
        name check by monkey-patching NativeRuntime.open to raise
        a known signal."""
        backend = NativeSpeakerEmbeddingBackend()
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            side_effect=ImportError("dylib not present in test"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "dylib" in exc_info.value.error_message


# ---------------------------------------------------------------------------
# load_model gate
# ---------------------------------------------------------------------------


class TestLoadModelGate:
    def test_missing_speaker_bin_raises_runtime_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCTOMIL_SHERPA_SPEAKER_MODEL", raising=False)
        backend = NativeSpeakerEmbeddingBackend()
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            return_value=MagicMock(
                capabilities=MagicMock(return_value=MagicMock(supported_capabilities=("audio.speaker.embedding",)))
            ),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "OCTOMIL_SHERPA_SPEAKER_MODEL" in exc_info.value.error_message

    def test_capability_not_advertised_raises_runtime_unavailable(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        # Probe path: non-digest reason routes to RUNTIME_UNAVAILABLE.
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.speaker.embedding failed",
            "audio.speaker.embedding not built into this runtime build",
        )
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "audio.speaker.embedding" in exc_info.value.error_message

    def test_capability_missing_with_digest_in_probe_raises_checksum_mismatch(self) -> None:
        """Codex R1 F-02 fix: the SDK extracts the digest diagnostic
        by probing ``oct_session_open(audio.speaker.embedding)`` —
        the dispatch path is the only place the runtime surfaces the
        sherpa adapter's ``cached_reason_`` into thread-local
        last_error. The probe call returns UNSUPPORTED and the
        ``NativeRuntimeError.last_error`` carries the digest
        substring."""
        backend = NativeSpeakerEmbeddingBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.speaker.embedding failed",
            "sherpa_onnx: OCTOMIL_SHERPA_SPEAKER_MODEL=/p digest mismatch — got abc want 1a33...",
        )
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
            assert "digest" in exc_info.value.error_message.lower()

    def test_capability_missing_long_path_digest_via_4kb_reread(self) -> None:
        """Codex R2 F-03 regression: long sherpa artifact path can
        truncate ``"digest"`` out of the loader-default 512-byte
        ``last_error`` capture. The SDK re-reads via
        ``runtime.last_error(buflen=4096)`` and surfaces
        ``CHECKSUM_MISMATCH``."""
        backend = NativeSpeakerEmbeddingBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        truncated = (
            "oct_session_open: sherpa_onnx adapter not loadable for capability audio.speaker.embedding: "
            "sherpa_onnx: OCTOMIL_SHERPA_SPEAKER_MODEL=" + ("/very/long/canonical/artifact/path" * 12)
        )
        full = (
            truncated
            + " digest mismatch — got abc want 1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
        )
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.speaker.embedding failed",
            truncated,
        )
        rt.last_error.return_value = full
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_capability_missing_without_digest_falls_through_to_runtime_unavailable(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        rt.open_session.side_effect = NativeRuntimeError(
            OCT_STATUS_UNSUPPORTED,
            "oct_session_open audio.speaker.embedding failed",
            "sherpa_onnx: OCTOMIL_SHERPA_SPEAKER_MODEL env var unset",
        )
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "OCTOMIL_SHERPA_SPEAKER_MODEL" in exc_info.value.error_message

    def test_runtime_open_failure_raises_runtime_unavailable(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        with patch(
            "octomil.runtime.native.speaker_backend.NativeRuntime.open",
            side_effect=NativeRuntimeError(OCT_STATUS_VERSION_MISMATCH, "abi skew", "v0.0.7"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("sherpa-eres2netv2-base")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_load_model_idempotent(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        backend._runtime = MagicMock()  # type: ignore[assignment]
        with patch("octomil.runtime.native.speaker_backend.NativeRuntime.open") as open_mock:
            backend.load_model("sherpa-eres2netv2-base")
            open_mock.assert_not_called()


# ---------------------------------------------------------------------------
# embed pre-flight
# ---------------------------------------------------------------------------


class TestEmbedPreflight:
    def test_embed_before_load_model_raises_runtime_unavailable(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_negative_deadline_raises_invalid_input(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        backend._runtime = MagicMock()
        backend._model = MagicMock()
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(
                np.zeros(16000, dtype=np.float32),
                sample_rate_hz=16000,
                deadline_ms=-1,
            )
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_audio_raises_invalid_input_pre_session(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        backend._runtime = MagicMock()
        backend._model = MagicMock()
        arr = np.zeros(16000, dtype=np.float32)
        arr[0] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        backend._runtime.open_session.assert_not_called()


# ---------------------------------------------------------------------------
# embed drain — duplicate / missing embedding events
# ---------------------------------------------------------------------------


def _fake_event(
    *,
    type_: int,
    values: list[float] | None = None,
    n_dim: int = 0,
    terminal_status: int = 0,
):
    ev = MagicMock()
    ev.type = type_
    ev.values = values if values is not None else []
    ev.n_dim = n_dim
    ev.terminal_status = terminal_status
    return ev


class TestEmbedDrain:
    def _wire_session(self, backend: NativeSpeakerEmbeddingBackend, events: list) -> MagicMock:
        rt = MagicMock()
        rt.last_error.return_value = ""
        sess = MagicMock()
        sess.poll_event.side_effect = events
        rt.open_session.return_value = sess
        backend._runtime = rt  # type: ignore[assignment]
        backend._model = MagicMock()
        return sess

    def test_happy_path_returns_512_dim_vector(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        vec = [0.1] * 512
        events = [
            _fake_event(type_=1),  # SESSION_STARTED
            _fake_event(type_=20, values=vec, n_dim=512),
            _fake_event(type_=8, terminal_status=0),
        ]
        sess = self._wire_session(backend, events)
        out = backend.embed(np.zeros(16000, dtype=np.float32), sample_rate_hz=16000)
        assert out.shape == (512,)
        assert out.dtype == np.float32
        sess.close.assert_called_once()

    def test_duplicate_embedding_event_raises_inference_failed(self) -> None:
        """Single-utterance contract: the runtime emits exactly one
        EMBEDDING_VECTOR per session. A duplicate is a runtime
        invariant violation; the SDK surfaces it as INFERENCE_FAILED
        rather than picking one silently."""
        backend = NativeSpeakerEmbeddingBackend()
        events = [
            _fake_event(type_=20, values=[0.1] * 512, n_dim=512),
            _fake_event(type_=20, values=[0.2] * 512, n_dim=512),
            _fake_event(type_=8, terminal_status=0),
        ]
        self._wire_session(backend, events)
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.INFERENCE_FAILED

    def test_missing_embedding_before_completion_raises_inference_failed(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        events = [
            _fake_event(type_=1),
            _fake_event(type_=8, terminal_status=0),
        ]
        self._wire_session(backend, events)
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.INFERENCE_FAILED

    def test_session_completed_non_ok_routes_to_typed_error(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        events = [
            _fake_event(type_=7),  # ERROR
            _fake_event(type_=8, terminal_status=OCT_STATUS_INVALID_INPUT),
        ]
        self._wire_session(backend, events)
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_dim_mismatch_raises_inference_failed(self) -> None:
        backend = NativeSpeakerEmbeddingBackend()
        # Runtime claims 512 dims but only delivers 256 floats — a
        # buffer-truncation runtime bug; we treat as terminal.
        events = [
            _fake_event(type_=20, values=[0.1] * 256, n_dim=512),
            _fake_event(type_=8, terminal_status=0),
        ]
        self._wire_session(backend, events)
        with pytest.raises(OctomilError) as exc_info:
            backend.embed(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.code == OctomilErrorCode.INFERENCE_FAILED
