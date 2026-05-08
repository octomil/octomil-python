"""Tests for :mod:`octomil.runtime.native.error_mapping`.

v0.1.6 PR1 — SDK error typing alignment.

Covers:
- Every documented ``oct_status_t`` → expected canonical
  :class:`OctomilErrorCode`.
- Each disambiguating substring rule for ``OCT_STATUS_UNSUPPORTED``.
- Each disambiguating substring rule for ``OCT_STATUS_NOT_FOUND``.
- The catch-all paths.
- The boundary preservation: dlopen / ABI errors raised inside the
  loader are NOT routed through the mapper (those still surface as
  ``NativeRuntimeError``).
"""

from __future__ import annotations

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.error_mapping import map_oct_status
from octomil.runtime.native.loader import (
    OCT_STATUS_BUSY,
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INTERNAL,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    NativeRuntimeError,
)

# ---------------------------------------------------------------------------
# Per-status canonical mapping
# ---------------------------------------------------------------------------


class TestOctStatusToCanonicalCode:
    """Each documented ``oct_status_t`` resolves to the canonical
    :class:`OctomilErrorCode` per the v0.1.6 PR1 mapping table."""

    def test_not_found_maps_to_model_not_found(self) -> None:
        err = map_oct_status(OCT_STATUS_NOT_FOUND, "")
        assert isinstance(err, OctomilError)
        assert err.code == OctomilErrorCode.MODEL_NOT_FOUND

    def test_invalid_input_maps_to_invalid_input(self) -> None:
        err = map_oct_status(OCT_STATUS_INVALID_INPUT, "")
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_unsupported_default_audio_policy_runtime_unavailable(self) -> None:
        err = map_oct_status(OCT_STATUS_UNSUPPORTED, "")
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_unsupported_chat_policy_unsupported_modality(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "",
            default_unsupported_code=OctomilErrorCode.UNSUPPORTED_MODALITY,
        )
        assert err.code == OctomilErrorCode.UNSUPPORTED_MODALITY

    def test_version_mismatch_maps_to_runtime_unavailable(self) -> None:
        err = map_oct_status(OCT_STATUS_VERSION_MISMATCH, "")
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_cancelled_maps_to_canonical_cancelled(self) -> None:
        # Canonical taxonomy: ``CANCELLED`` (not "OPERATION_CANCELLED").
        err = map_oct_status(OCT_STATUS_CANCELLED, "")
        assert err.code == OctomilErrorCode.CANCELLED

    def test_timeout_maps_to_canonical_request_timeout(self) -> None:
        # Canonical taxonomy: ``REQUEST_TIMEOUT`` (not bare "TIMEOUT").
        err = map_oct_status(OCT_STATUS_TIMEOUT, "")
        assert err.code == OctomilErrorCode.REQUEST_TIMEOUT

    def test_busy_maps_to_server_error(self) -> None:
        err = map_oct_status(OCT_STATUS_BUSY, "")
        assert err.code == OctomilErrorCode.SERVER_ERROR

    def test_internal_maps_to_inference_failed(self) -> None:
        err = map_oct_status(OCT_STATUS_INTERNAL, "")
        assert err.code == OctomilErrorCode.INFERENCE_FAILED

    def test_unknown_status_value_maps_to_inference_failed(self) -> None:
        # Forward-compat: any future status the SDK doesn't know about
        # falls into the catch-all INFERENCE_FAILED rather than crashing.
        err = map_oct_status(999, "")
        assert err.code == OctomilErrorCode.INFERENCE_FAILED


# ---------------------------------------------------------------------------
# OCT_STATUS_UNSUPPORTED substring disambiguation
# ---------------------------------------------------------------------------


class TestUnsupportedSubstringDisambiguation:
    """Every documented ``last_error`` substring rule for
    ``OCT_STATUS_UNSUPPORTED``."""

    def test_digest_substring_maps_to_checksum_mismatch(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "ggml-tiny.bin digest mismatch (got abc, want xyz)",
        )
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_digest_takes_precedence_over_capability(self) -> None:
        # If both "digest" and a capability name appear, digest wins.
        # This matches the runtime adapter convention: when the digest
        # check fails, the capability is also de-advertised — but the
        # operator's actionable signal is "your artifact is corrupt".
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio.transcription: digest mismatch (got abc, want xyz)",
        )
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_audio_transcription_capability_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio.transcription not advertised by this runtime",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_audio_vad_capability_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio.vad not advertised by this runtime",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_audio_speaker_embedding_capability_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio.speaker.embedding not advertised by this runtime",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_octomil_whisper_bin_env_unset_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "OCTOMIL_WHISPER_BIN unset; whisper.cpp adapter cannot load",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_octomil_silero_vad_bin_env_unset_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "OCTOMIL_SILERO_VAD_BIN unset; silero adapter cannot load",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_octomil_sherpa_speaker_model_env_unset_runtime_unavailable(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "OCTOMIL_SHERPA_SPEAKER_MODEL unset; ERes2NetV2 adapter cannot load",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_changed_substring_maps_to_invalid_input(self) -> None:
        # Env drift between runtime_open and model_open. Bounded
        # INVALID_INPUT reject per v0.1.5 capability gating regime
        # (see `v0_1_5_rc1_state.md`).
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "OCTOMIL_WHISPER_BIN changed between runtime_open and model_open",
        )
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_sample_rate_substring_maps_to_invalid_input(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio sample_rate=8000 unsupported; whisper-tiny requires 16000",
        )
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_nan_substring_maps_to_invalid_input(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio buffer contains NaN samples; rejecting",
        )
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_inf_substring_maps_to_invalid_input(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio buffer contains Inf samples; rejecting",
        )
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_format_substring_maps_to_invalid_input(self) -> None:
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "audio format pcm_s16le unsupported; only float32 accepted",
        )
        assert err.code == OctomilErrorCode.INVALID_INPUT

    def test_substring_match_is_case_insensitive(self) -> None:
        err = map_oct_status(OCT_STATUS_UNSUPPORTED, "DIGEST MISMATCH")
        assert err.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_unsupported_no_marker_falls_to_default_audio(self) -> None:
        # No substring match → the default policy. Audio backends
        # default to RUNTIME_UNAVAILABLE.
        err = map_oct_status(OCT_STATUS_UNSUPPORTED, "some unparseable runtime error")
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_unsupported_no_marker_falls_to_default_chat(self) -> None:
        # No substring match + chat policy → UNSUPPORTED_MODALITY.
        err = map_oct_status(
            OCT_STATUS_UNSUPPORTED,
            "OCT_EMBED_POOLING_RANK rejected for embeddings.text",
            default_unsupported_code=OctomilErrorCode.UNSUPPORTED_MODALITY,
        )
        assert err.code == OctomilErrorCode.UNSUPPORTED_MODALITY


# ---------------------------------------------------------------------------
# OCT_STATUS_NOT_FOUND substring disambiguation
# ---------------------------------------------------------------------------


class TestNotFoundSubstringDisambiguation:
    def test_not_found_with_capability_string_maps_to_runtime_unavailable(self) -> None:
        # NOT_FOUND can also surface for "capability unloadable" — env
        # is set, file path resolves to nothing. The capability string
        # is the disambiguator.
        err = map_oct_status(
            OCT_STATUS_NOT_FOUND,
            "audio.transcription artifact path /nonexistent/whisper.bin not found",
        )
        assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    def test_not_found_no_capability_string_maps_to_model_not_found(self) -> None:
        err = map_oct_status(
            OCT_STATUS_NOT_FOUND,
            "/path/to/missing.gguf not found",
        )
        assert err.code == OctomilErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# Message composition
# ---------------------------------------------------------------------------


class TestMessageComposition:
    def test_message_only_no_last_error(self) -> None:
        err = map_oct_status(OCT_STATUS_INVALID_INPUT, "", message="bad input")
        assert err.error_message == "bad input"

    def test_last_error_only_no_message(self) -> None:
        err = map_oct_status(OCT_STATUS_INVALID_INPUT, "rt detail")
        assert err.error_message == "rt detail"

    def test_message_and_last_error_concatenated(self) -> None:
        err = map_oct_status(
            OCT_STATUS_INVALID_INPUT,
            "rt detail",
            message="STT call failed",
        )
        assert err.error_message == "STT call failed: rt detail"

    def test_no_message_no_last_error_uses_status_fallback(self) -> None:
        err = map_oct_status(OCT_STATUS_BUSY, "")
        assert "oct_status_t=" in err.error_message


# ---------------------------------------------------------------------------
# Boundary preservation — dlopen / ABI errors stay raw
# ---------------------------------------------------------------------------


class TestNativeRuntimeErrorBoundary:
    """The boundary contract: ``NativeRuntimeError`` is for the cffi
    wrapper layer. Backends catch + translate it on the product path.
    Genuine dlopen / ABI-skew failures from inside ``loader.py`` MUST
    keep raising ``NativeRuntimeError`` raw — those are not part of
    the canonical SDK taxonomy and should not be silently mapped to
    ``RUNTIME_UNAVAILABLE``.
    """

    def test_native_runtime_error_is_not_an_octomil_error(self) -> None:
        err = NativeRuntimeError(
            OCT_STATUS_VERSION_MISMATCH,
            "ABI MAJOR mismatch: dylib=2 binding=1",
        )
        assert not isinstance(err, OctomilError)

    def test_native_runtime_error_carries_raw_status_field(self) -> None:
        # Raw native errors carry the numeric status; OctomilError
        # carries the canonical code. The two are distinct.
        err = NativeRuntimeError(OCT_STATUS_VERSION_MISMATCH, "msg")
        assert err.status == OCT_STATUS_VERSION_MISMATCH
        assert not hasattr(err, "code")

    def test_loader_dlopen_path_raises_native_runtime_error_not_octomil(self) -> None:
        # Surface the boundary contract through the loader's own
        # raise pattern: when `_build_lib` detects an ABI MAJOR skew
        # at dlopen time it raises NativeRuntimeError — not
        # OctomilError. Backends that catch this on the product path
        # are expected to translate; the loader itself does not.
        # We construct the same exception type the loader would and
        # verify its taxonomy.
        err = NativeRuntimeError(
            OCT_STATUS_VERSION_MISMATCH,
            "liboctomil-runtime ABI MAJOR 2 is incompatible with binding MAJOR 1",
        )
        assert isinstance(err, RuntimeError)  # loader raises RuntimeError subclass
        assert not isinstance(err, OctomilError)
        # And the mapper, when called explicitly with the numeric
        # status the loader would carry, produces the canonical SDK
        # code — proving the translation step is intentional and
        # explicit at the backend boundary, not silent.
        translated = map_oct_status(err.status, "")
        assert isinstance(translated, OctomilError)
        assert translated.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
