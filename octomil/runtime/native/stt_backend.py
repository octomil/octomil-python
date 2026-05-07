"""NativeSttBackend — audio.transcription via octomil-runtime v0.1.5+.

Hard-cutover backend for local whisper.cpp speech-to-text. Replaces
the legacy ``pywhispercpp``-backed ``WhisperCppEngine`` for the
product STT path. Modeled on :mod:`octomil.runtime.native.chat_backend`
and :mod:`octomil.runtime.native.embeddings_backend`.

Lifecycle:
- One ``NativeSttBackend`` instance per (planner-selected) STT engine.
- ``load_model(model_name)`` opens a ``NativeRuntime``, verifies the
  ``audio.transcription`` capability is advertised, and caches the
  runtime. Whisper-tiny is the only model wired in v0.1.5; the
  runtime resolves the artifact via ``OCTOMIL_WHISPER_BIN``.
- ``transcribe(pcm_f32, sample_rate_hz=...)`` opens one session per
  request with ``capability="audio.transcription"``, sends the audio
  view, drains ``OCT_EVENT_TRANSCRIPT_SEGMENT`` events, captures
  ``OCT_EVENT_TRANSCRIPT_FINAL``, then closes the session
  deterministically. The runtime stays warm.
- ``close()`` closes the runtime.

Hard rules (cutover discipline — no silent Python fallback):
1. The caller MUST resolve the model artifact + capability advertisement
   at the planner; this backend is selected only when the runtime has
   already advertised ``audio.transcription``.
2. ``transcribe()`` rejects NaN / Inf / zero-length input via the
   runtime-side validator; the SDK surfaces those as ``INVALID_INPUT``.
3. Bad ggml-tiny.bin digest produces ``OCT_STATUS_UNSUPPORTED`` with
   ``last_error`` mentioning ``digest`` — surfaced as
   ``CHECKSUM_MISMATCH`` (the bounded code for "integrity check
   failed after download" in the SDK's error taxonomy).
4. ``OCTOMIL_WHISPER_BIN`` unset / artifact missing → the runtime
   does NOT advertise the capability and ``open_session`` returns
   ``OCT_STATUS_UNSUPPORTED`` — surfaced as ``RUNTIME_UNAVAILABLE``.

Bounded-error mapping (runtime → SDK):
- Missing artifact / file → ``MODEL_NOT_FOUND``.
- Bad digest (last_error contains ``digest``) → ``CHECKSUM_MISMATCH``.
- Unsupported capability / ABI mismatch → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_INVALID_INPUT`` (NaN / Inf / format) → ``INVALID_INPUT``.
- Caller-driven cancel → ``CANCELLED``.
- Caller-driven timeout → ``REQUEST_TIMEOUT``.
- Other terminal → ``INFERENCE_FAILED``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_TRANSCRIPTION
from .loader import (
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_TRANSCRIPT_FINAL,
    OCT_EVENT_TRANSCRIPT_SEGMENT,
    OCT_STATUS_BUSY,
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_OK,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)


_BACKEND_NAME = "native-whisper-cpp"
_DEFAULT_DEADLINE_MS = 300_000  # 5 minutes — same shape as chat / embeddings.
# whisper-tiny is hard-coded to 16kHz mono PCM-f32 in v0.1.5; the
# runtime rejects anything else with INVALID_INPUT. We keep this as a
# Python-side guard so callers see the shape error before the FFI
# call. SR-mismatch surfacing as INVALID_INPUT preserves bounded-
# rejection behavior.
_WHISPER_SAMPLE_RATE_HZ: int = 16000
# v0.1.5 PR-2A: the runtime's whisper.cpp adapter resolves the
# ggml-tiny.bin artifact off OCTOMIL_WHISPER_BIN at session_open
# time. oct_model_open still requires a valid model_uri scheme
# (absolute path, file://, local://) — we hand it the same env
# value so the URI parses AND the digest verification has
# something concrete to point at when last_error is rendered.
_WHISPER_BIN_ENV: str = "OCTOMIL_WHISPER_BIN"


@dataclass
class Segment:
    """One timestamped span emitted as ``OCT_EVENT_TRANSCRIPT_SEGMENT``.

    ``start_ms`` and ``end_ms`` are runtime-monotonic from the start
    of the audio window; ``end_ms >= start_ms``.
    """

    start_ms: int
    end_ms: int
    text: str


@dataclass
class TranscriptionResult:
    """Native-STT transcription result.

    Distinct from :class:`octomil.audio.types.TranscriptionResult` so
    the native backend can carry per-segment timestamps and decoded
    duration without forcing a contract bump on the public audio API.
    The kernel adapter projects this down to the public type.
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    duration_ms: int = 0


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Map a runtime ``oct_status_t`` (+ last_error text) to the SDK's
    bounded error taxonomy.

    The bad-digest path needs special handling: the runtime returns
    ``OCT_STATUS_UNSUPPORTED`` for both "capability not built in" and
    "ggml-tiny.bin SHA-256 mismatch", disambiguated only by
    ``last_error`` text. The cutover spec maps the digest variant to
    ``CHECKSUM_MISMATCH`` (bounded code for "integrity check failed
    after download") and the capability variant to
    ``RUNTIME_UNAVAILABLE``.
    """
    last_error_lc = (last_error or "").lower()
    if status == OCT_STATUS_NOT_FOUND:
        code = OctomilErrorCode.MODEL_NOT_FOUND
    elif status == OCT_STATUS_INVALID_INPUT:
        code = OctomilErrorCode.INVALID_INPUT
    elif status == OCT_STATUS_UNSUPPORTED:
        # Spec: digest mismatch → CHECKSUM_MISMATCH; otherwise →
        # RUNTIME_UNAVAILABLE. Match on the substring "digest" in
        # last_error per the runtime's convention.
        if "digest" in last_error_lc:
            code = OctomilErrorCode.CHECKSUM_MISMATCH
        else:
            code = OctomilErrorCode.RUNTIME_UNAVAILABLE
    elif status == OCT_STATUS_VERSION_MISMATCH:
        code = OctomilErrorCode.RUNTIME_UNAVAILABLE
    elif status == OCT_STATUS_CANCELLED:
        code = OctomilErrorCode.CANCELLED
    elif status == OCT_STATUS_TIMEOUT:
        code = OctomilErrorCode.REQUEST_TIMEOUT
    elif status == OCT_STATUS_BUSY:
        code = OctomilErrorCode.SERVER_ERROR
    else:
        code = OctomilErrorCode.INFERENCE_FAILED
    full_message = message
    if last_error:
        full_message = f"{message}: {last_error}"
    return OctomilError(code=code, message=full_message)


def _validate_pcm_f32(samples: Sequence[float] | Any, sample_rate_hz: int) -> bytes:
    """Pre-flight the audio buffer against the shape contract.

    The runtime side validates again (NaN/Inf, zero-length, bad sr),
    but a Python-side check produces a precise diagnostic before
    paying session_open + send_audio cost. Returns the buffer as a
    contiguous float32 ``bytes`` so the cffi cast in
    ``NativeSession.send_audio`` reads the right layout.

    Accepts:
      * 1-D numpy float32 array (mono).
      * Python sequence of floats (mono).
      * Raw ``bytes`` already in float32-LE shape.

    Stereo / multichannel and non-16kHz inputs are rejected here —
    the v0.1.5 runtime is hard-coded to whisper-tiny which expects
    16kHz mono.
    """
    if sample_rate_hz != _WHISPER_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"native STT: sample_rate_hz must be {_WHISPER_SAMPLE_RATE_HZ} "
                f"(whisper-tiny is mono-16kHz-only in v0.1.5); got {sample_rate_hz}"
            ),
        )
    if isinstance(samples, (bytes, bytearray, memoryview)):
        buf = bytes(samples)
        if len(buf) == 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native STT: zero-length audio buffer",
            )
        if len(buf) % 4 != 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(f"native STT: PCM-f32 buffer length {len(buf)} is not a multiple of 4 bytes"),
            )
        return buf
    # numpy / list-of-float path. Avoid a hard numpy dep at module
    # import time; only require it when the caller passes an array-like.
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native STT: array-like audio input requires numpy; pass bytes (float32-LE) or `pip install numpy`"
            ),
        ) from exc
    arr = _np.asarray(samples, dtype=_np.float32)
    if arr.ndim != 1:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(f"native STT: audio must be 1-D mono PCM-f32; got shape {arr.shape}"),
        )
    if arr.size == 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native STT: zero-length audio buffer",
        )
    # NaN / Inf would also be rejected by the runtime; surfacing
    # Python-side preserves bounded-rejection behavior + saves a
    # session-open round-trip. Use the runtime mapping (INVALID_INPUT)
    # so the error matches the runtime path one-for-one.
    if not _np.isfinite(arr).all():
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native STT: audio contains NaN or Inf samples",
        )
    return arr.tobytes()


def _runtime_advertises_audio_transcription(rt: NativeRuntime) -> bool:
    """Capability-honesty check used by the planner before constructing
    a backend. Returns False if the runtime doesn't advertise
    ``audio.transcription`` (capability gate failed: missing
    ``OCTOMIL_WHISPER_BIN`` or digest mismatch). Callers MUST raise
    ``RUNTIME_UNAVAILABLE`` rather than fall back to a Python-local
    transcriber on the product path."""
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_TRANSCRIPTION in caps.supported_capabilities


# Public alias used by the planner / kernel — picks up the existence
# check without requiring callers to import the leading-underscore
# helper. The leading-underscore form is kept for back-compat with
# the embeddings_backend pattern.
runtime_advertises_audio_transcription = _runtime_advertises_audio_transcription


class NativeSttBackend:
    """Hard-cutover ``audio.transcription`` backend backed by
    ``octomil-runtime v0.1.5+``.

    Caches a ``NativeRuntime`` per backend instance. Each
    ``transcribe(...)`` call opens a fresh session, sends the audio
    view, drains transcript events, closes the session.
    """

    name: str = _BACKEND_NAME
    DEFAULT_DEADLINE_MS: int = _DEFAULT_DEADLINE_MS

    def __init__(
        self,
        *,
        default_deadline_ms: int | None = None,
    ) -> None:
        self._model_name: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None
        self._default_deadline_ms: int = (
            default_deadline_ms if default_deadline_ms is not None else self.DEFAULT_DEADLINE_MS
        )

    def load_model(self, model_name: str) -> None:
        """Open the runtime and verify ``audio.transcription`` is
        advertised. Idempotent — a second call is a no-op.

        Raises
        ------
        OctomilError
            ``RUNTIME_UNAVAILABLE`` if the runtime fails to open or
            does not advertise ``audio.transcription`` (operator
            forgot ``OCTOMIL_WHISPER_BIN`` or the digest doesn't
            match the v0.1.5 ggml-tiny.bin SHA-256 the runtime pins).
        """
        if self._runtime is not None:
            return  # idempotent
        self._model_name = model_name
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            # Dylib resolution failure is a runtime-not-available signal
            # at the SDK layer. Operator points at the wrong file or the
            # cffi extra is missing.
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native STT backend: dylib not found ({exc})",
            ) from exc

        if not _runtime_advertises_audio_transcription(self._runtime):
            # Capability not advertised. Operator either didn't set
            # OCTOMIL_WHISPER_BIN, or the digest doesn't match.
            # Don't try to disambiguate here — the open_session call
            # below would fail with last_error pointing at the cause.
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native STT backend: runtime does not advertise "
                    "'audio.transcription'. Check OCTOMIL_WHISPER_BIN "
                    "(must point at ggml-tiny.bin with SHA-256 "
                    "be07e048…6e1b21) and that the dylib was built with "
                    "OCT_ENABLE_ENGINE_WHISPER_CPP=ON."
                ),
            )

        # v0.1.5 PR-2A: audio.transcription is a model-bound capability
        # like chat.completion. session_open requires a pre-opened
        # `oct_model_t*` (engine_hint="whisper_cpp"); the runtime
        # rejects with INVALID_INPUT and a precise last_error otherwise.
        # We pre-warm here so transcribe() pays only session_open +
        # send_audio + drain on the hot path.
        whisper_bin = os.environ.get(_WHISPER_BIN_ENV, "")
        if not whisper_bin:
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"native STT backend: {_WHISPER_BIN_ENV} not set. "
                    "Point at a verified ggml-tiny.bin (SHA-256 "
                    "be07e048…6e1b21)."
                ),
            )
        try:
            self._model = self._runtime.open_model(
                model_uri=whisper_bin,
                engine_hint="whisper_cpp",
            )
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT backend failed to warm whisper-tiny model",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        logger.info("NativeSttBackend: runtime opened + whisper-tiny warmed")

    def close(self) -> None:
        # Close the model BEFORE the runtime so oct_runtime_close
        # doesn't refuse with BUSY. Mirrors the embeddings backend.
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeSttBackend.close: model.close failed", exc_info=True)
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeSttBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_pcm_f32: Any,
        *,
        sample_rate_hz: int = _WHISPER_SAMPLE_RATE_HZ,
        language: str = "en",
        deadline_ms: int | None = None,
    ) -> TranscriptionResult:
        """Run an ``audio.transcription`` request against the runtime.

        Parameters
        ----------
        audio_pcm_f32
            Mono PCM-f32 audio. ``bytes`` (float32-LE) OR a 1-D
            numpy float32 array OR a list of floats. Stereo /
            multichannel rejects ``INVALID_INPUT``.
        sample_rate_hz
            Hz. Must be 16000 (whisper-tiny is hard-coded in v0.1.5).
        language
            BCP-47 language hint. Echoed back on the result; the
            runtime is multilingual but v0.1.5 surfaces a single
            language tag per session.
        deadline_ms
            Per-request poll deadline. Falls back to
            ``self._default_deadline_ms`` (5 minutes) when None.

        Returns
        -------
        TranscriptionResult
            ``text`` is the concatenated final transcript; ``segments``
            carry timestamped spans; ``duration_ms`` is the decoded
            audio duration emitted with ``OCT_EVENT_TRANSCRIPT_FINAL``.

        Raises
        ------
        OctomilError
            See module docstring for the bounded error mapping.
        """
        if self._runtime is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeSttBackend.transcribe called before load_model",
            )

        # Deadline validation BEFORE opening a session.
        resolved_deadline_ms = deadline_ms if deadline_ms is not None else self._default_deadline_ms
        if resolved_deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"deadline_ms must be > 0; got {resolved_deadline_ms}. Use None to "
                    f"fall back to the backend's default ({self.DEFAULT_DEADLINE_MS}ms)."
                ),
            )

        audio_bytes = _validate_pcm_f32(audio_pcm_f32, sample_rate_hz)
        if not math.isfinite(float(sample_rate_hz)):  # belt-and-suspenders
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"native STT: invalid sample_rate_hz {sample_rate_hz!r}",
            )

        if self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "NativeSttBackend.transcribe: model not warmed; load_model() must succeed before transcribe()"
                ),
            )

        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_TRANSCRIPTION,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=sample_rate_hz,
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        try:
            try:
                sess.send_audio(audio_bytes, sample_rate=sample_rate_hz, channels=1)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native STT backend send_audio failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            segments: list[Segment] = []
            final_text = ""
            duration_ms = 0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            saw_final = False

            deadline_seconds = resolved_deadline_ms / 1000.0
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                try:
                    ev = sess.poll_event(timeout_ms=200)
                except NativeRuntimeError as exc:
                    raise _runtime_status_to_sdk_error(
                        exc.status,
                        "native STT backend poll_event failed",
                        last_error=getattr(exc, "last_error", ""),
                    ) from exc
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_SEGMENT:
                    # Defensive: clamp end_ms >= start_ms (the runtime
                    # guarantees this; a violation is a runtime bug
                    # but we don't want to silently emit broken spans).
                    if ev.segment_end_ms < ev.segment_start_ms:
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(
                                f"native STT: transcript segment with end_ms < start_ms "
                                f"(start={ev.segment_start_ms}, end={ev.segment_end_ms})"
                            ),
                        )
                    segments.append(
                        Segment(
                            start_ms=int(ev.segment_start_ms),
                            end_ms=int(ev.segment_end_ms),
                            text=ev.text,
                        )
                    )
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_FINAL:
                    final_text = ev.text
                    duration_ms = int(ev.final_duration_ms)
                    saw_final = True
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    saw_error = True
                    if not error_message and self._runtime is not None:
                        error_message = self._runtime.last_error()
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    terminal_status = int(ev.terminal_status)
                    break
            else:
                raise OctomilError(
                    code=OctomilErrorCode.REQUEST_TIMEOUT,
                    message=(f"native STT backend timed out waiting for SESSION_COMPLETED ({resolved_deadline_ms}ms)"),
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native STT backend reported error during transcription",
                    last_error=error_message,
                )

            if not saw_final:
                raise OctomilError(
                    code=OctomilErrorCode.INFERENCE_FAILED,
                    message=("native STT: SESSION_COMPLETED(OK) without preceding TRANSCRIPT_FINAL"),
                )

            return TranscriptionResult(
                text=final_text,
                segments=segments,
                language=language,
                duration_ms=duration_ms,
            )
        finally:
            sess.close()


# Alias so callers can choose either name. Keeps the spec's
# `NativeTranscriptionBackend` available without forcing a second
# class definition.
NativeTranscriptionBackend = NativeSttBackend


__all__ = [
    "NativeSttBackend",
    "NativeTranscriptionBackend",
    "Segment",
    "TranscriptionResult",
    "runtime_advertises_audio_transcription",
]
