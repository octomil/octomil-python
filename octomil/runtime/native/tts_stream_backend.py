"""v0.1.8 streaming TTS — sentence-bounded chunks, coalesced delivery.

In the v0.1.8 sherpa adapter, ``OfflineTts::Generate()`` runs
synchronously inside the runtime's ``poll_event()``, so all sentence
chunks become available together at end-of-synthesis. **This is NOT
a latency win over audio.tts.batch yet.** The iterator-shaped API +
multi-chunk event sequence are forward-compatible with v0.1.9 worker-
thread Generate which will deliver real progressive first-chunk
latency.

Use ``audio.tts.batch`` when you don't need iterator semantics.
Use ``audio.tts.stream`` when your downstream consumer is iterator-
shaped (SSE response, async generator, audio playback queue). Both
run at the same total wall-clock today.

Honesty caveat (v0.1.8 sherpa adapter):
  * Chunks arrive coalesced after synthesis on the polling thread.
  * The ``cumulative_duration_ms`` we surface is a *post-synth*
    duration calculation against the cumulative PCM samples, not a
    real-time playback timeline.
  * The :class:`NativeTtsStreamBackend` API is iterator-shaped so
    callers wired against it now keep working unchanged when v0.1.9
    flips on the worker-thread adapter — at that point chunks arrive
    progressively and ``first_audio_ms`` becomes the meaningful
    latency signal.

Hard rules (cutover discipline — no silent Python sherpa fallback):

1. The caller MUST resolve the model artifact + capability
   advertisement at the planner; this backend is selected only when
   the runtime advertises ``audio.tts.stream``.
2. Bad ``OCTOMIL_SHERPA_TTS_MODEL`` (unset / digest mismatch / sidecar
   missing) → the runtime does NOT advertise the capability and
   :meth:`NativeTtsStreamBackend.load_model` raises
   ``RUNTIME_UNAVAILABLE`` (or ``CHECKSUM_MISMATCH`` when last_error
   carries a "digest" marker).
3. Voice validation runs synchronously *before* the first chunk so
   an unsupported voice raises ``INVALID_INPUT`` BEFORE the consumer
   sees any audio.
4. No silent Python sherpa fallback on the product path.

Bounded-error mapping is centralized in
:mod:`octomil.runtime.native.error_mapping` (canonical taxonomy from
v0.1.6 PR1). Audio backends use ``RUNTIME_UNAVAILABLE`` as the
``default_unsupported_code``; digest mismatches map to
``CHECKSUM_MISMATCH`` via the ``"digest"`` substring rule.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_TTS_STREAM
from .error_mapping import map_oct_status
from .loader import (
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_TTS_AUDIO_CHUNK,
    OCT_SAMPLE_FORMAT_PCM_F32LE,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_OK,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)


_BACKEND_NAME = "native-sherpa-onnx-tts-stream"
_DEFAULT_DEADLINE_MS = 300_000  # 5 minutes — same shape as native STT.
_TTS_MODEL_ENV: str = "OCTOMIL_SHERPA_TTS_MODEL"

# v0.1.9 Lane 4 — documented anchor for the progressive-streaming
# first-audio-chunk telemetry metric. A future runtime release
# (Lane 1 + Lane 2) will start emitting ``OCT_EVENT_METRIC`` with
# ``data.metric.name`` set to this string to mark first-audio-chunk
# arrival in progressive mode. The SDK does NOT introspect or filter
# on metric names at the binding boundary (RM-RT-001); this constant
# is only exposed so consumers + reviewers have a single source of
# truth to grep for. The Python-side telemetry sink registry, when
# wired by a follow-up PR, MUST forward ``OCT_EVENT_METRIC`` events
# regardless of name — no allowlist.
TTS_FIRST_AUDIO_MS_METRIC_NAME: str = "tts.first_audio_ms"


@dataclass
class TtsAudioChunk:
    """One ``OCT_EVENT_TTS_AUDIO_CHUNK`` event, parsed at the SDK boundary.

    ``pcm_f32`` is a 1-D numpy float32 array, mono. The runtime emits
    PCM in ``OCT_SAMPLE_FORMAT_PCM_F32LE`` for the sherpa-onnx adapter;
    other formats raise ``INVALID_INPUT`` at the backend boundary.

    ``chunk_index`` and ``cumulative_duration_ms`` are derived at the
    SDK layer from arrival order + cumulative sample count — they are
    NOT carried on the runtime wire today. The fields exist on the
    iterator shape so callers building progressive UIs / playback
    queues don't have to track them out-of-band; when v0.1.9 lands
    real progressive Generate, runtime-emitted timings will replace
    the SDK-side derivation without changing the public dataclass.

    Honesty caveat: in v0.1.8 sherpa, ALL chunks arrive together at
    end-of-synth on the polling thread (synchronous Generate). The
    ``cumulative_duration_ms`` is ``len(pcm_f32) / sample_rate_hz`` —
    audio-content duration, NOT real-time arrival latency. See module
    docstring.
    """

    pcm_f32: Any  # numpy.ndarray (1-D float32) — typed Any to avoid hard numpy import here.
    sample_rate_hz: int
    chunk_index: int
    is_final: bool
    cumulative_duration_ms: int
    # v0.1.9 Lane 4 — forward-compatibility marker for progressive
    # streaming. ALWAYS "coalesced" today against the v0.1.8 sherpa
    # adapter (synchronous Generate inside one poll_event). The field
    # exists so callers wiring against ``synthesize_with_chunks`` now
    # can branch on streaming_mode in their UIs WITHOUT a follow-up
    # dataclass-shape break when the v0.1.9 worker-thread runtime
    # release lands. Flip to "progressive" is gated on:
    #   * Lane 1 (runtime worker-thread Generate) shipping in
    #     octomil-runtime, AND
    #   * Lane 2 (runtime release tag bumped + dylib rebuilt + SDK
    #     binding consumes the release).
    # Until both are true, the SDK MUST emit "coalesced". A guard test
    # (tests/test_tts_stream_no_premature_progressive_claim.py) blocks
    # public-claim flips that precede the runtime release.
    streaming_mode: Literal["coalesced", "progressive"] = "coalesced"


def _runtime_advertises_tts_stream(rt: NativeRuntime) -> bool:
    """Capability-honesty check used by the planner before constructing
    a backend. Returns False if the runtime does not advertise
    ``audio.tts.stream`` (gate failed: missing
    ``OCTOMIL_SHERPA_TTS_MODEL``, digest mismatch, or sidecar files
    missing). Callers MUST raise ``RUNTIME_UNAVAILABLE`` rather than
    fall back to a Python-local TTS engine on the product path."""
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_TTS_STREAM in caps.supported_capabilities


# Public alias for planner / kernel imports — same pattern as STT.
runtime_advertises_tts_stream = _runtime_advertises_tts_stream


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Audio-policy mapper: ``OCT_STATUS_UNSUPPORTED`` → ``RUNTIME_UNAVAILABLE``
    by default, with substring rules in ``error_mapping`` promoting to
    ``CHECKSUM_MISMATCH`` (digest), ``INVALID_INPUT`` (sample shape /
    NaN), etc."""
    return map_oct_status(
        status,
        last_error,
        message=message,
        default_unsupported_code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
    )


class NativeTtsStreamBackend:
    """Hard-cut ``audio.tts.stream`` backend.

    Honesty caveat (v0.1.8 sherpa adapter): chunks arrive coalesced
    after synthesis on the polling thread; THIS IS NOT a latency win
    over audio.tts.batch yet. The iterator-shaped API is forward-
    compatible with v0.1.9 worker-thread Generate which will deliver
    real progressive first-chunk latency.

    Lifecycle mirrors :class:`NativeSttBackend`:
      * One backend instance per (planner-selected) sherpa-onnx TTS
        engine.
      * :meth:`load_model` opens a :class:`NativeRuntime`, verifies
        ``audio.tts.stream`` advertised, opens + warms an
        ``oct_model_t*`` for the sherpa engine, caches both.
      * :meth:`synthesize_with_chunks` opens one session per request,
        sends the text, drains ``OCT_EVENT_TTS_AUDIO_CHUNK`` events,
        yields :class:`TtsAudioChunk` objects to the caller as they
        arrive (in v0.1.8 they all arrive together; in v0.1.9 they
        will arrive progressively).
      * :meth:`close` shuts down model and runtime.
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self, model_name: str) -> None:
        """Open the runtime, verify ``audio.tts.stream`` advertised,
        open + warm a sherpa-onnx model handle. Idempotent on re-load
        with the same model_name.

        Raises
        ------
        OctomilError
            ``RUNTIME_UNAVAILABLE`` if the runtime fails to open or
            does not advertise ``audio.tts.stream`` (operator forgot
            ``OCTOMIL_SHERPA_TTS_MODEL`` or sidecars missing).
            ``CHECKSUM_MISMATCH`` if the artifact's digest doesn't
            match the runtime-pinned canonical SHA-256.
        """
        if self._runtime is not None and self._model_name == model_name:
            return

        # Closing first makes load_model with a different model_name
        # safe (closes the prior runtime + model handle).
        self.close()
        self._model_name = model_name

        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native TTS-stream backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native TTS-stream backend: dylib not found ({exc})",
            ) from exc

        if not _runtime_advertises_tts_stream(self._runtime):
            last_error_lc = (self._runtime.last_error() or "").lower()
            self.close()
            if "digest" in last_error_lc:
                raise OctomilError(
                    code=OctomilErrorCode.CHECKSUM_MISMATCH,
                    message=(
                        "native TTS-stream backend: sherpa-onnx TTS model "
                        "SHA-256 does not match the canonical pin. "
                        "Re-download the artifact."
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native TTS-stream backend: runtime does not advertise "
                    "'audio.tts.stream'. Check OCTOMIL_SHERPA_TTS_MODEL "
                    "(must point at the pinned VITS .onnx with sibling "
                    "tokens.txt + espeak-ng-data/) and that the dylib "
                    "was built with OCT_HAVE_SHERPA_ONNX_TTS."
                ),
            )

        tts_model_path = os.environ.get(_TTS_MODEL_ENV, "")
        if not tts_model_path:
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(f"native TTS-stream backend: {_TTS_MODEL_ENV} not set."),
            )
        try:
            self._model = self._runtime.open_model(
                model_uri=tts_model_path,
                engine_hint="sherpa_onnx",
            )
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native TTS-stream backend failed to warm sherpa-onnx model",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        logger.debug(
            "NativeTtsStreamBackend: runtime opened + sherpa-onnx warmed (%s)",
            model_name,
        )

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "NativeTtsStreamBackend.close: model.close failed",
                    exc_info=True,
                )
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "NativeTtsStreamBackend.close: runtime.close failed",
                    exc_info=True,
                )
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Pre-stream voice validation
    # ------------------------------------------------------------------
    def validate_voice(self, voice: str | None) -> str:
        """Validate the requested ``voice`` synchronously, BEFORE we
        commit to a session. Returns the resolved sid as a numeric
        string (the sherpa-onnx ABI parses ``speaker_id`` as a
        non-negative integer string; numeric voice ids are the
        identity transform).

        ``None`` / ``""`` resolve to sid=0 (model default voice). The
        runtime-side ``parse_speaker_id`` rejects non-numeric voices
        with ``INVALID_INPUT``; this method mirrors the same logic
        Python-side so the rejection lands BEFORE we open the HTTP
        200 (or yield the first chunk).
        """
        if voice is None or voice == "":
            return "0"
        v = str(voice).strip()
        if not v:
            return "0"
        if not v.isdigit():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"native TTS-stream: voice {voice!r} is not a non-negative "
                    "integer sid string. sherpa-onnx accepts numeric speaker "
                    'ids only at the runtime ABI; pass voice="0" for the '
                    "model default."
                ),
            )
        return v

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def synthesize_with_chunks(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        deadline_ms: int | None = None,
    ) -> Iterator[TtsAudioChunk]:
        """Yields sentence-bounded chunks. Chunks arrive together at
        end-of-synth in the v0.1.8 sherpa adapter. Future v0.1.9
        worker-thread Generate will yield chunks during synthesis
        without changing this method's signature.

        Method name: ``synthesize_with_chunks``, NOT ``stream`` — the
        v0.1.8 contract is iterator-shaped, not realtime / progressive.
        Callers wired against this method will keep working unchanged
        when the underlying runtime stops being synchronous.

        Parameters
        ----------
        text
            Non-empty UTF-8 string. Empty / whitespace-only rejects
            ``INVALID_INPUT`` Python-side (same shape as the runtime
            adapter's send_text guard).
        voice_id
            Numeric speaker_id string (e.g. ``"0"``). ``None`` →
            model default sid=0. See :meth:`validate_voice`.
        deadline_ms
            Per-request poll deadline. Falls back to
            ``self._default_deadline_ms`` (5 minutes) when ``None``.

        Yields
        ------
        TtsAudioChunk
            One per ``OCT_EVENT_TTS_AUDIO_CHUNK`` event. ``is_final``
            is True on the last chunk; the iterator terminates
            after the runtime emits ``OCT_EVENT_SESSION_COMPLETED(OK)``.

        Raises
        ------
        OctomilError
            See module docstring + :mod:`octomil.runtime.native.error_mapping`
            for the bounded mapping.
        """
        if self._runtime is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeTtsStreamBackend.synthesize_with_chunks called before load_model",
            )
        if self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "NativeTtsStreamBackend.synthesize_with_chunks: model not warmed; "
                    "load_model() must succeed before synthesize_with_chunks()"
                ),
            )

        if not isinstance(text, str) or not text.strip():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native TTS-stream: text must be a non-empty string",
            )

        resolved_deadline_ms = deadline_ms if deadline_ms is not None else self._default_deadline_ms
        if resolved_deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"deadline_ms must be > 0; got {resolved_deadline_ms}. Use None to "
                    f"fall back to the backend's default ({self.DEFAULT_DEADLINE_MS}ms)."
                ),
            )

        speaker_id_str = self.validate_voice(voice_id)

        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_TTS_STREAM,
                locality="on_device",
                policy_preset="private",
                speaker_id=speaker_id_str,
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native TTS-stream backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        # Defer numpy import to the first call — keeps module import
        # cheap and matches the STT backend pattern.
        try:
            import numpy as _np  # type: ignore[import-not-found]
        except ImportError as exc:
            sess.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native TTS-stream: numpy is required to surface PCM chunks as float32 arrays; pip install numpy"
                ),
            ) from exc

        # Codex r2 P2 fix: send_text MUST be synchronous (i.e. inside
        # the request scope, BEFORE any StreamingResponse begins) so
        # OctomilError(INVALID_INPUT) for malformed text / NaN bytes
        # / single-utterance double-send-text rejects surface as a
        # typed 4xx body rather than mid-stream after HTTP 200. The
        # _drain generator only handles poll_event + chunk extraction
        # post-send_text, so it can stay lazy.
        try:
            sess.send_text(text)
        except NativeRuntimeError as exc:
            sess.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native TTS-stream send_text failed",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        return self._drain(
            sess=sess,
            np_module=_np,
            resolved_deadline_ms=resolved_deadline_ms,
        )

    def _drain(
        self,
        *,
        sess: Any,
        np_module: Any,
        resolved_deadline_ms: int,
    ) -> Iterator[TtsAudioChunk]:
        """Inner generator — split out so :meth:`synthesize_with_chunks`
        can do *synchronous* validation (voice / text / deadline / model)
        AND ``send_text`` BEFORE returning the iterator. The session
        has been opened and text already submitted by the time we
        get here; from this point on, error events drain through
        the iterator as raised :class:`OctomilError`.

        The yield-on-arrival shape is the load-bearing forward-compat
        property: in v0.1.9 the runtime will block in poll_event
        between sentence chunks, so the same loop transparently
        becomes progressive without binding-side changes.
        """
        try:
            chunk_index = 0
            cumulative_samples = 0
            cumulative_sample_rate = 0
            saw_final_chunk = False
            saw_error = False
            error_message = ""
            terminal_status: int = OCT_STATUS_OK

            deadline_seconds = resolved_deadline_ms / 1000.0
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                try:
                    ev = sess.poll_event(timeout_ms=200)
                except NativeRuntimeError as exc:
                    raise _runtime_status_to_sdk_error(
                        exc.status,
                        "native TTS-stream poll_event failed",
                        last_error=getattr(exc, "last_error", ""),
                    ) from exc
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_TTS_AUDIO_CHUNK:
                    if ev.tts_sample_format != OCT_SAMPLE_FORMAT_PCM_F32LE:
                        raise OctomilError(
                            code=OctomilErrorCode.INVALID_INPUT,
                            message=(
                                "native TTS-stream: unexpected sample_format "
                                f"{ev.tts_sample_format} (expected PCM_F32LE)"
                            ),
                        )
                    if ev.tts_channels != 1:
                        raise OctomilError(
                            code=OctomilErrorCode.INVALID_INPUT,
                            message=(f"native TTS-stream: unexpected channels {ev.tts_channels} (expected mono=1)"),
                        )
                    if ev.tts_sample_rate <= 0:
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message="native TTS-stream: zero / negative sample_rate on chunk",
                        )

                    pcm_bytes = ev.tts_pcm_bytes or b""
                    pcm_f32 = np_module.frombuffer(pcm_bytes, dtype=np_module.float32).copy()
                    cumulative_samples += int(pcm_f32.size)
                    cumulative_sample_rate = int(ev.tts_sample_rate)
                    is_final = bool(ev.tts_is_final)

                    if cumulative_sample_rate > 0:
                        cumulative_duration_ms = int((cumulative_samples * 1000) // cumulative_sample_rate)
                    else:
                        cumulative_duration_ms = 0

                    chunk = TtsAudioChunk(
                        pcm_f32=pcm_f32,
                        sample_rate_hz=cumulative_sample_rate,
                        chunk_index=chunk_index,
                        is_final=is_final,
                        cumulative_duration_ms=cumulative_duration_ms,
                        # v0.1.9 Lane 4 — hardcoded "coalesced" until the
                        # runtime release (Lane 1 + Lane 2) flips to
                        # progressive Generate. A follow-up SDK PR will
                        # change this to read from a runtime capability
                        # hint, OR detect via inter-chunk wall-clock
                        # delta (>50 ms gap = progressive). Until then,
                        # the SDK honestly reports "coalesced" — see
                        # module docstring + dataclass field comment.
                        streaming_mode="coalesced",
                    )
                    chunk_index += 1
                    if is_final:
                        saw_final_chunk = True
                    yield chunk
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
                    message=(
                        f"native TTS-stream backend timed out waiting for SESSION_COMPLETED ({resolved_deadline_ms}ms)"
                    ),
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native TTS-stream backend reported error during synthesis",
                    last_error=error_message,
                )
            if not saw_final_chunk:
                raise OctomilError(
                    code=OctomilErrorCode.INFERENCE_FAILED,
                    message=(
                        "native TTS-stream: SESSION_COMPLETED(OK) without a preceding TTS_AUDIO_CHUNK with is_final=1"
                    ),
                )
        finally:
            sess.close()


__all__ = [
    "NativeTtsStreamBackend",
    "TtsAudioChunk",
    "runtime_advertises_tts_stream",
]
