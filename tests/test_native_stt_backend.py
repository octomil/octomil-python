"""Unit tests for NativeSttBackend (v0.1.5 PR-2B SDK STT cutover).

These tests run without the dylib by mocking the runtime. The
integration tests (tests/integration/test_native_stt_integration.py)
exercise the real cffi path against jfk.wav.

What we lock in here:
1. Capability gate: when the runtime advertises ``audio.transcription``,
   :meth:`NativeSttBackend.load_model` succeeds. When it does NOT
   advertise it, ``load_model`` raises ``RUNTIME_UNAVAILABLE`` —
   the SDK does NOT fall back to the legacy pywhispercpp path.
2. ``OCTOMIL_WHISPER_BIN`` env: must be set; otherwise
   ``RUNTIME_UNAVAILABLE``.
3. Bounded-error mapping: bad-digest UNSUPPORTED → ``CHECKSUM_MISMATCH``;
   capability-miss UNSUPPORTED → ``RUNTIME_UNAVAILABLE``;
   ``OCT_STATUS_NOT_FOUND`` → ``MODEL_NOT_FOUND``;
   ``OCT_STATUS_INVALID_INPUT`` → ``INVALID_INPUT``;
   ``OCT_STATUS_CANCELLED`` → ``CANCELLED``;
   ``OCT_STATUS_TIMEOUT`` → ``REQUEST_TIMEOUT``;
   default → ``INFERENCE_FAILED``.
4. NaN/Inf samples reject ``INVALID_INPUT`` Python-side (matches the
   runtime's own rejection — saves a session round-trip).
5. Unsupported sample rate (non-16kHz) rejects ``INVALID_INPUT``.
6. Empty / wrong-shape audio buffers reject ``INVALID_INPUT``.
7. Planner-style selector helper
   (:func:`runtime_advertises_audio_transcription`) returns False if
   the runtime fails to introspect or doesn't list the capability.
"""

from __future__ import annotations

import stat
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.loader import (
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    NativeRuntimeError,
)
from octomil.runtime.native.stt_backend import (
    _WHISPER_ARTIFACTS,
    NativeSttBackend,
    NativeTranscriptionBackend,
    Segment,
    TranscriptionResult,
    _runtime_status_to_sdk_error,
    _validate_pcm_f32,
    is_supported_native_whisper_model,
    runtime_advertises_audio_transcription,
)

_FAKE_WHISPER_BIN = "/tmp/_pr2b_test_fake_ggml_tiny.bin"


@pytest.fixture(autouse=True)
def _set_whisper_bin_env(monkeypatch: pytest.MonkeyPatch):
    """Default to a non-existent path so the load_model gate is
    exercised; individual tests override when they need to."""
    monkeypatch.setenv("OCTOMIL_WHISPER_BIN", _FAKE_WHISPER_BIN)
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
        # The cutover spec calls this "MODEL_INTEGRITY_FAILED"; the
        # SDK's bounded taxonomy uses CHECKSUM_MISMATCH which carries
        # the same semantic ("Integrity check failed after download").
        err = _runtime_status_to_sdk_error(
            OCT_STATUS_UNSUPPORTED,
            "open failed",
            last_error="ggml-tiny.bin digest mismatch (got abc, want xyz)",
        )
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_unsupported_without_digest_maps_to_runtime_unavailable(self) -> None:
        err = _runtime_status_to_sdk_error(
            OCT_STATUS_UNSUPPORTED,
            "open failed",
            last_error="capability 'audio.transcription' not built into this runtime",
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

    def test_unknown_terminal_maps_to_inference_failed(self) -> None:
        err = _runtime_status_to_sdk_error(99, "weird")
        assert err.code == OctomilErrorCode.INFERENCE_FAILED


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------


class TestAudioValidation:
    def test_zero_length_bytes_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(b"", sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "zero-length" in exc_info.value.error_message

    def test_zero_length_array_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(np.zeros(0, dtype=np.float32), sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_misaligned_bytes_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(b"\x00" * 7, sample_rate_hz=16000)  # 7 % 4 != 0
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "multiple of 4" in exc_info.value.error_message

    def test_bad_sample_rate_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(np.zeros(160, dtype=np.float32), sample_rate_hz=22050)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_rejected(self) -> None:
        arr = np.zeros(160, dtype=np.float32)
        arr[42] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "NaN" in exc_info.value.error_message or "Inf" in exc_info.value.error_message

    def test_inf_rejected(self) -> None:
        arr = np.zeros(160, dtype=np.float32)
        arr[7] = float("inf")
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_2d_array_rejected(self) -> None:
        arr = np.zeros((2, 160), dtype=np.float32)
        with pytest.raises(OctomilError) as exc_info:
            _validate_pcm_f32(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "1-D" in exc_info.value.error_message

    def test_clean_buffer_passes(self) -> None:
        arr = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        out = _validate_pcm_f32(arr, sample_rate_hz=16000)
        assert isinstance(out, bytes)
        assert len(out) == 16000 * 4


# ---------------------------------------------------------------------------
# Capability advertisement helper
# ---------------------------------------------------------------------------


class TestRuntimeAdvertises:
    def test_advertises_when_capability_present(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.transcription",))
        assert runtime_advertises_audio_transcription(rt) is True

    def test_does_not_advertise_when_capability_absent(self) -> None:
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        assert runtime_advertises_audio_transcription(rt) is False

    def test_returns_false_on_capabilities_failure(self) -> None:
        rt = MagicMock()
        rt.capabilities.side_effect = RuntimeError("dylib gone")
        assert runtime_advertises_audio_transcription(rt) is False


# ---------------------------------------------------------------------------
# load_model gate
# ---------------------------------------------------------------------------


class TestModelNameGate:
    """W1 wires tiny/base only; later sizes must reject rather than
    silently substituting a different registered artifact."""

    def test_supported_native_whisper_model_helper(self) -> None:
        assert is_supported_native_whisper_model("whisper-tiny") is True
        assert is_supported_native_whisper_model("whisper-base") is True
        assert is_supported_native_whisper_model("whisper-small") is False

    def test_whisper_base_loads_with_expected_digest(self) -> None:
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.transcription",))
        model = MagicMock()
        rt.open_model.return_value = model
        with (
            patch(
                "octomil.runtime.native.stt_backend.NativeRuntime.open",
                return_value=rt,
            ),
            patch("octomil.runtime.native.stt_backend._verify_whisper_artifact_matches_spec") as verify_mock,
        ):
            backend.load_model("whisper-base")
        verify_mock.assert_called_once_with(_FAKE_WHISPER_BIN, _WHISPER_ARTIFACTS["whisper-base"])
        rt.open_model.assert_called_once_with(
            model_uri=_FAKE_WHISPER_BIN,
            artifact_digest=_WHISPER_ARTIFACTS["whisper-base"].artifact_digest,
            engine_hint="whisper_cpp",
        )
        model.warm.assert_called_once()

    def test_whisper_small_rejects_unsupported_modality(self) -> None:
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("whisper-small")
        assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY

    def test_whisper_large_rejects_unsupported_modality(self) -> None:
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("whisper-large-v3")
        assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


class TestLoadModelGate:
    def test_missing_whisper_bin_raises_runtime_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCTOMIL_WHISPER_BIN", raising=False)
        backend = NativeSttBackend()
        # Stub the runtime to advertise so we exercise the env gate
        # specifically (the env check happens AFTER the capability
        # gate in the implementation; both must pass).
        with patch(
            "octomil.runtime.native.stt_backend.NativeRuntime.open",
            return_value=MagicMock(
                capabilities=MagicMock(return_value=MagicMock(supported_capabilities=("audio.transcription",)))
            ),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "OCTOMIL_WHISPER_BIN" in exc_info.value.error_message

    def test_capability_not_advertised_raises_runtime_unavailable(self) -> None:
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        rt.last_error.return_value = "audio.transcription not built in"
        with patch(
            "octomil.runtime.native.stt_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            # We MUST surface this rather than fall back to pywhispercpp.
            # The cutover spec is explicit: no silent product-path fallback.
            assert "audio.transcription" in exc_info.value.error_message

    def test_capability_missing_with_digest_in_last_error_raises_checksum_mismatch(self) -> None:
        """Codex R1 blocker: when the runtime hides ``audio.transcription``
        from advertisement because the artifact's SHA-256 doesn't
        match the runtime-pinned digest, surfacing
        ``RUNTIME_UNAVAILABLE`` is misleading. The runtime writes a
        diagnostic into thread-local last_error in that case; we
        disambiguate on the substring ``digest`` and surface
        ``CHECKSUM_MISMATCH`` instead."""
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("chat.completion",))
        rt.last_error.return_value = "ggml-tiny.bin digest mismatch (got abc, want xyz)"
        with patch(
            "octomil.runtime.native.stt_backend.NativeRuntime.open",
            return_value=rt,
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
            assert "digest" in exc_info.value.error_message.lower()

    def test_runtime_open_failure_raises_runtime_unavailable(self) -> None:
        backend = NativeSttBackend()
        with patch(
            "octomil.runtime.native.stt_backend.NativeRuntime.open",
            side_effect=NativeRuntimeError(OCT_STATUS_VERSION_MISMATCH, "abi skew", "v0.0.7"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_dylib_not_found_raises_runtime_unavailable(self) -> None:
        backend = NativeSttBackend()
        with patch(
            "octomil.runtime.native.stt_backend.NativeRuntime.open",
            side_effect=ImportError("liboctomil-runtime not found"),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "dylib" in exc_info.value.error_message

    def test_load_model_idempotent(self) -> None:
        backend = NativeSttBackend()
        backend._runtime = MagicMock()  # type: ignore[assignment]
        backend._model_name = "whisper-tiny"
        # Should be a no-op; no NativeRuntime.open call.
        with patch("octomil.runtime.native.stt_backend.NativeRuntime.open") as open_mock:
            backend.load_model("whisper-tiny")
            open_mock.assert_not_called()

    def test_requested_tiny_rejects_base_env_artifact_before_model_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.transcription",))
        base_spec = _WHISPER_ARTIFACTS["whisper-base"]
        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/models/whisper-base/ggml-base.bin")
        with (
            patch(
                "octomil.runtime.native.stt_backend.NativeRuntime.open",
                return_value=rt,
            ),
            patch(
                "octomil.runtime.native.stt_backend.os.stat",
                return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=base_spec.size_bytes),
            ),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
        rt.open_model.assert_not_called()

    def test_requested_base_rejects_tiny_env_artifact_before_model_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.transcription",))
        tiny_spec = _WHISPER_ARTIFACTS["whisper-tiny"]
        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/models/whisper-tiny/ggml-tiny.bin")
        with (
            patch(
                "octomil.runtime.native.stt_backend.NativeRuntime.open",
                return_value=rt,
            ),
            patch(
                "octomil.runtime.native.stt_backend.os.stat",
                return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=tiny_spec.size_bytes),
            ),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-base")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
        rt.open_model.assert_not_called()

    def test_requested_base_rejects_wrong_sha_after_size_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = NativeSttBackend()
        rt = MagicMock()
        rt.capabilities.return_value = MagicMock(supported_capabilities=("audio.transcription",))
        base_spec = _WHISPER_ARTIFACTS["whisper-base"]
        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/models/whisper-base/ggml-base.bin")
        with (
            patch(
                "octomil.runtime.native.stt_backend.NativeRuntime.open",
                return_value=rt,
            ),
            patch(
                "octomil.runtime.native.stt_backend.os.stat",
                return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=base_spec.size_bytes),
            ),
            patch(
                "octomil.runtime.native.stt_backend._sha256_file_hex",
                return_value=_WHISPER_ARTIFACTS["whisper-tiny"].sha256,
            ),
        ):
            with pytest.raises(OctomilError) as exc_info:
                backend.load_model("whisper-base")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
        rt.open_model.assert_not_called()

    def test_kernel_resolver_reraises_typed_native_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Codex R2 blocker: kernel native-first branch must re-raise
        typed OctomilError (CHECKSUM_MISMATCH / RUNTIME_UNAVAILABLE)
        rather than swallow them to None. Otherwise the public
        SDK/CLI surface loses the typed signal and routes to
        cloud or 503 generically."""
        from octomil.execution.kernel import ExecutionKernel

        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/some/fake/path.bin")
        # Stub NativeSttServeAdapter.load_model to raise the typed error.
        with patch(
            "octomil.serve.stt_serve_adapter.NativeSttServeAdapter.load_model",
            side_effect=OctomilError(
                code=OctomilErrorCode.CHECKSUM_MISMATCH,
                message="ggml-tiny.bin digest mismatch",
            ),
        ):
            kernel = ExecutionKernel.__new__(ExecutionKernel)
            object.__setattr__(kernel, "_warmed_backends", {})
            object.__setattr__(kernel, "_lookup_warmed_backend", lambda *a, **kw: None)
            with pytest.raises(OctomilError) as exc_info:
                kernel._resolve_local_transcription_backend("whisper-tiny")
            assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_kernel_resolver_returns_none_on_unexpected_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Conversely, a non-typed exception (e.g. ImportError from a
        broken cffi install) falls through to None so the caller's
        policy gate decides — preserves the legacy "no native, try
        registry, else 503" path."""
        from octomil.execution.kernel import ExecutionKernel

        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/some/fake/path.bin")
        with patch(
            "octomil.serve.stt_serve_adapter.NativeSttServeAdapter.load_model",
            side_effect=ImportError("cffi not installed"),
        ):
            kernel = ExecutionKernel.__new__(ExecutionKernel)
            object.__setattr__(kernel, "_warmed_backends", {})
            object.__setattr__(kernel, "_lookup_warmed_backend", lambda *a, **kw: None)
            # Patch out the registry-walk fallback so we exercise just
            # the native branch's None return.
            with patch("octomil.runtime.engines.get_registry") as reg_mock:
                reg_mock.return_value.detect_all.return_value = []
                result = kernel._resolve_local_transcription_backend("whisper-tiny")
            assert result is None

    def test_kernel_resolver_rejects_unregistered_whisper_with_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env is set and model is whisper-small/etc.,
        kernel returns None (caller's cloud-fallback gate decides).
        The native backend itself raises UNSUPPORTED_MODALITY but
        the kernel resolver short-circuits to None before
        constructing the adapter — log-and-skip pattern matches the
        design where the cloud-eligibility check upstream owns the
        "is this a typed error or a "no local available" signal?"
        decision."""
        from octomil.execution.kernel import ExecutionKernel

        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/some/fake/path.bin")
        kernel = ExecutionKernel.__new__(ExecutionKernel)
        object.__setattr__(kernel, "_warmed_backends", {})
        object.__setattr__(kernel, "_lookup_warmed_backend", lambda *a, **kw: None)
        with patch("octomil.runtime.engines.get_registry") as reg_mock:
            reg_mock.return_value.detect_all.return_value = []
            result = kernel._resolve_local_transcription_backend("whisper-small")
        assert result is None

    def test_kernel_resolver_accepts_whisper_base_native(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.execution.kernel import ExecutionKernel
        from octomil.serve.stt_serve_adapter import NativeSttServeAdapter

        monkeypatch.setenv("OCTOMIL_WHISPER_BIN", "/some/fake/path.bin")
        kernel = ExecutionKernel.__new__(ExecutionKernel)
        object.__setattr__(kernel, "_warmed_backends", {})
        object.__setattr__(kernel, "_lookup_warmed_backend", lambda *a, **kw: None)
        with patch(
            "octomil.serve.stt_serve_adapter.NativeSttServeAdapter.load_model",
            return_value=None,
        ) as load_mock:
            result = kernel._resolve_local_transcription_backend("whisper-base")
        assert isinstance(result, NativeSttServeAdapter)
        load_mock.assert_called_once_with("whisper-base")

    def test_already_loaded_rejects_mismatched_model_name(self) -> None:
        """When the backend is already warmed for
        whisper-tiny, a second load_model call asking for a
        different supported size must reject INVALID_INPUT rather
        than no-op."""
        backend = NativeSttBackend()
        backend._runtime = MagicMock()  # type: ignore[assignment]
        backend._model_name = "whisper-tiny"
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("whisper-base")
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# transcribe pre-flight
# ---------------------------------------------------------------------------


class TestTranscribePreflight:
    def test_transcribe_before_load_model_raises_runtime_unavailable(self) -> None:
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.transcribe(np.zeros(16000, dtype=np.float32), sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_negative_deadline_raises_invalid_input(self) -> None:
        backend = NativeSttBackend()
        # Stub past load_model so we hit the deadline gate.
        backend._runtime = MagicMock()
        backend._model = MagicMock()
        with pytest.raises(OctomilError) as exc_info:
            backend.transcribe(
                np.zeros(16000, dtype=np.float32),
                sample_rate_hz=16000,
                deadline_ms=-1,
            )
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_audio_raises_invalid_input_pre_session(self) -> None:
        # The runtime would also reject NaN, but the SDK catches it
        # Python-side to save a session round-trip. Verify the pre-
        # flight rejection happens BEFORE open_session.
        backend = NativeSttBackend()
        backend._runtime = MagicMock()
        backend._model = MagicMock()
        arr = np.zeros(16000, dtype=np.float32)
        arr[0] = float("nan")
        with pytest.raises(OctomilError) as exc_info:
            backend.transcribe(arr, sample_rate_hz=16000)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        # Session should NOT have been opened.
        backend._runtime.open_session.assert_not_called()


# ---------------------------------------------------------------------------
# TranscriptionResult / Segment shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_dataclass_defaults(self) -> None:
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.segments == []
        assert r.language == "en"
        assert r.duration_ms == 0

    def test_segment_dataclass(self) -> None:
        s = Segment(start_ms=0, end_ms=500, text="hi")
        assert s.start_ms == 0
        assert s.end_ms == 500
        assert s.text == "hi"

    def test_alias_class_is_same(self) -> None:
        assert NativeTranscriptionBackend is NativeSttBackend
