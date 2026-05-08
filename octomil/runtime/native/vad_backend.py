"""NativeVadBackend — audio.vad via octomil-runtime v0.1.5+.

Hard-cutover backend for local Silero voice-activity-detection. The
runtime side (PR-2F, octomil-runtime #24, squash ``14f9c8b``) lights
up ``audio.vad`` only when:

  1. The dylib was built with ``OCT_ENABLE_ENGINE_SILERO_VAD=ON`` (which
     in turn requires ``OCT_ENABLE_ENGINE_WHISPER_CPP=ON`` because the
     adapter consumes whisper.cpp's ``whisper_vad_*`` C API rather
     than vendoring ONNX runtime).
  2. ``OCTOMIL_SILERO_VAD_MODEL`` points at a regular file.
  3. That file's SHA-256 matches
     ``2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987``
     (canonical ``ggml-silero-v6.2.0.bin`` from
     ``huggingface.co/ggml-org/whisper-vad``, 885 098 bytes).

When any of those gates fail the runtime hides the capability from
``oct_runtime_capabilities`` and ``oct_session_open`` returns
``OCT_STATUS_UNSUPPORTED``. This binding fail-closes on that signal:
NO Python implementation is invoked as a fallback. The cutover spec
is explicit — there was no prior Python VAD product path to preserve;
this is a NEW capability surface introduced in v0.1.5.

Lifecycle (model-less):
- ``audio.vad`` is the only capability in v0.1.5 that does NOT
  consume an ``oct_model_t``. The silero ``.bin`` is small (~885 KB)
  and the adapter loads it per-session via
  ``whisper_vad_init_from_file_with_params`` at ``oct_session_open``.
  The session holds a borrowed_runtime back-pointer and bumps the
  runtime's ``live_modelless_sessions`` refcount; ``oct_runtime_close``
  refuses with BUSY while VAD sessions are alive. Bindings just call
  ``oct_session_close`` to drop the refcount cleanly.
- One ``NativeVadBackend`` instance per backend slot. ``open_session()``
  returns a ``VadStreamingSession`` context manager. The runtime
  accepts arbitrary-size audio chunks (the adapter buffers and runs
  decode lazily on the first ``poll_event`` after audio is queued);
  the SDK can feed 30 ms windows OR a single full-clip ``send_audio``.

Hard rules (cutover discipline):
1. The runtime is the source of truth on whether ``audio.vad`` is
   advertised. The SDK does NOT fall through to a Python implementation
   when the runtime declines — there is no Python-local fallback to
   fall through to.
2. ``feed_chunk`` rejects NaN / Inf / zero-length / non-finite inputs
   Python-side (the runtime would also reject; we save a round-trip
   AND make the diagnostic precise).
3. Silero VAD is hard-coded to 16 kHz mono PCM-f32 at this minor.
   Other sample rates / channel counts reject ``INVALID_INPUT``.
4. Bad-digest startup paths surface as ``CHECKSUM_MISMATCH`` per the
   bounded taxonomy in :mod:`octomil._generated.error_code`.

Bounded-error mapping (runtime status → SDK ``OctomilErrorCode``):
- ``OCT_STATUS_NOT_FOUND`` → ``MODEL_NOT_FOUND``.
- ``OCT_STATUS_INVALID_INPUT`` → ``INVALID_INPUT``.
- ``OCT_STATUS_UNSUPPORTED`` with ``"digest"`` in last_error →
  ``CHECKSUM_MISMATCH``; otherwise → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_VERSION_MISMATCH`` → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_CANCELLED`` → ``CANCELLED``.
- ``OCT_STATUS_TIMEOUT`` → ``REQUEST_TIMEOUT``.
- ``OCT_STATUS_BUSY`` → ``SERVER_ERROR``.
- Any other terminal → ``INFERENCE_FAILED``.

This is the canonical low-level VAD API for the Python SDK in
v0.1.5; no prior product path is being replaced.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterator, Literal

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_VAD
from .loader import (
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_VAD_TRANSITION,
    OCT_STATUS_BUSY,
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_OK,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    OCT_VAD_TRANSITION_SPEECH_END,
    OCT_VAD_TRANSITION_SPEECH_START,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)


_BACKEND_NAME = "native-silero-vad"
# Canonical contract: silero VAD is 16 kHz mono PCM-f32. Anything else
# rejects INVALID_INPUT. v0.1.5 does not ship resampling at the runtime
# layer; callers are responsible.
_VAD_SAMPLE_RATE_HZ: int = 16000
# 5 minutes — same shape as STT / chat / embeddings backends. Caller
# can override per session.
_DEFAULT_DEADLINE_MS: int = 300_000


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Map a runtime ``oct_status_t`` (+ last_error text) onto the SDK's
    bounded error taxonomy.

    Mirrors :func:`octomil.runtime.native.stt_backend._runtime_status_to_sdk_error`
    so the VAD cutover produces identical error shapes for identical
    runtime statuses. The bad-digest disambiguation matters because
    the runtime returns ``OCT_STATUS_UNSUPPORTED`` for both "capability
    not built into this dylib" AND "ggml-silero-v6.2.0.bin SHA-256
    mismatch", separated only by ``last_error`` text.
    """
    last_error_lc = (last_error or "").lower()
    if status == OCT_STATUS_NOT_FOUND:
        code = OctomilErrorCode.MODEL_NOT_FOUND
    elif status == OCT_STATUS_INVALID_INPUT:
        code = OctomilErrorCode.INVALID_INPUT
    elif status == OCT_STATUS_UNSUPPORTED:
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


def _validate_chunk_pcm_f32(samples: Any, sample_rate_hz: int) -> bytes:
    """Pre-flight a 16 kHz mono PCM-f32 chunk for the VAD send_audio
    boundary.

    Same shape contract as :func:`octomil.runtime.native.stt_backend._validate_pcm_f32`,
    minus the multi-input batching. Rejects NaN/Inf/zero-length/wrong-
    rate inputs Python-side so the diagnostic is precise BEFORE the
    cffi call.
    """
    if sample_rate_hz != _VAD_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"native VAD: sample_rate_hz must be {_VAD_SAMPLE_RATE_HZ} "
                f"(silero VAD is mono-16kHz-only in v0.1.5); got {sample_rate_hz}"
            ),
        )
    if isinstance(samples, (bytes, bytearray, memoryview)):
        buf = bytes(samples)
        if len(buf) == 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native VAD: zero-length audio buffer",
            )
        if len(buf) % 4 != 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(f"native VAD: PCM-f32 buffer length {len(buf)} is not a multiple of 4 bytes"),
            )
        try:
            import numpy as _np  # type: ignore[import-not-found]
        except ImportError:
            return buf
        view = _np.frombuffer(buf, dtype=_np.float32)
        if not _np.isfinite(view).all():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native VAD: audio contains NaN or Inf samples",
            )
        return buf
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native VAD: array-like audio input requires numpy; pass bytes " "(float32-LE) or `pip install numpy`"
            ),
        ) from exc
    arr = _np.asarray(samples, dtype=_np.float32)
    if arr.ndim != 1:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"native VAD: audio must be 1-D mono PCM-f32; got shape {arr.shape}",
        )
    if arr.size == 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native VAD: zero-length audio buffer",
        )
    if not _np.isfinite(arr).all():
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native VAD: audio contains NaN or Inf samples",
        )
    return arr.tobytes()


def _runtime_advertises_audio_vad(rt: NativeRuntime) -> bool:
    """Capability-honesty check used by callers before constructing a
    backend. Returns False if the runtime fails introspection OR doesn't
    list ``audio.vad``. Callers MUST raise ``RUNTIME_UNAVAILABLE``
    rather than fall back to a non-existent Python VAD path."""
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_VAD in caps.supported_capabilities


runtime_advertises_audio_vad = _runtime_advertises_audio_vad


@dataclass
class VadTransition:
    """One transition edge emitted by the silero VAD adapter.

    Mirrors the runtime's
    ``oct_event_t.data.vad_transition`` payload. ``kind`` is one of
    ``"speech_start"``, ``"speech_end"``, or ``"unknown"`` (the third
    is a future-compat sentinel never emitted by v0.1.5; bindings MUST
    NOT crash on it). ``timestamp_ms`` is runtime-monotonic from the
    start of the audio window. ``confidence`` is the silero
    per-window average speech probability across the span, clamped by
    the runtime to ``[0.0, 1.0]``.
    """

    kind: Literal["speech_start", "speech_end", "unknown"]
    timestamp_ms: int
    confidence: float


def _kind_label(transition_kind: int) -> Literal["speech_start", "speech_end", "unknown"]:
    if transition_kind == OCT_VAD_TRANSITION_SPEECH_START:
        return "speech_start"
    if transition_kind == OCT_VAD_TRANSITION_SPEECH_END:
        return "speech_end"
    # Includes OCT_VAD_TRANSITION_UNKNOWN (0) and any future-added
    # sentinel value the runtime may emit. Bindings ignore unknowns
    # rather than crash, per the runtime header contract.
    return "unknown"


class VadStreamingSession:
    """Context-managed streaming wrapper over a ``audio.vad`` session.

    Single-thread-affine: ``feed_chunk`` and ``poll_transitions`` MUST
    NOT race against each other on the same instance. This mirrors the
    underlying ``NativeSession`` contract (``send_audio`` /
    ``poll_event`` / ``close`` are not safe to call concurrently
    against one handle).

    The runtime decodes lazily — buffered audio is processed on the
    first ``poll_event`` call after audio has been sent. For long
    audio: feed all chunks then drain. For short audio: feed once,
    drain once. The runtime is single-utterance per session — once
    decode runs, further ``feed_chunk`` calls reject ``INVALID_INPUT``;
    open a fresh session for the next clip.
    """

    def __init__(self, runtime: NativeRuntime, sample_rate_hz: int) -> None:
        self._runtime = runtime
        self._sample_rate_hz = sample_rate_hz
        self._native_session: Any | None = None
        self._closed: bool = False
        self._terminal_seen: bool = False
        try:
            self._native_session = runtime.open_session(
                capability=CAPABILITY_AUDIO_VAD,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=sample_rate_hz,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native VAD backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

    # ------------------------------------------------------------------
    # Context-manager surface — closes the underlying session on exit.
    # ------------------------------------------------------------------
    def __enter__(self) -> "VadStreamingSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def feed_chunk(self, audio_pcm_f32: Any, *, sample_rate_hz: int | None = None) -> None:
        """Push an audio chunk into the VAD session.

        The runtime accepts arbitrary chunk sizes — silero internally
        re-windows to 512-sample (32 ms at 16 kHz) frames, so the SDK
        can feed 30 ms / 100 ms / full-clip — whatever fits the
        caller's pipeline. For real-time pipelines, 30 ms (480
        samples) is a common choice; nothing in this binding requires
        that specific cadence.

        Raises
        ------
        OctomilError
            ``INVALID_INPUT`` for NaN / Inf / zero-length / wrong sample
            rate / non-1-D shape; or if the session has already
            decoded (single-utterance contract).
            ``RUNTIME_UNAVAILABLE`` if the session was already closed.
            See module docstring for the full bounded mapping.
        """
        if self._closed or self._native_session is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="VadStreamingSession.feed_chunk: session is closed",
            )
        sr = sample_rate_hz if sample_rate_hz is not None else self._sample_rate_hz
        audio_bytes = _validate_chunk_pcm_f32(audio_pcm_f32, sr)
        try:
            self._native_session.send_audio(audio_bytes, sample_rate=sr, channels=1)
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native VAD backend send_audio failed",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

    def poll_transitions(
        self,
        *,
        deadline_ms: int | None = None,
        drain_until_completed: bool = False,
    ) -> Iterator[VadTransition]:
        """Drain pending transitions from the session.

        Parameters
        ----------
        deadline_ms
            Wall-clock budget for the drain. None falls back to the
            module default (5 minutes).
        drain_until_completed
            When True, polls until the session emits
            ``OCT_EVENT_SESSION_COMPLETED`` (the natural end-of-stream
            for a single-utterance VAD session). When False, drains
            only currently-pending events with a short per-call
            timeout — useful for streaming readers that interleave
            ``feed_chunk`` / ``poll_transitions`` calls.

        Yields
        ------
        VadTransition
            One per ``OCT_EVENT_VAD_TRANSITION``. ``kind="unknown"``
            transitions are skipped (per the runtime header's
            future-compat sentinel rule).

        Raises
        ------
        OctomilError
            See module docstring for bounded mapping. ``REQUEST_TIMEOUT``
            on deadline expiry; runtime-side errors propagate via the
            mapper.
        """
        if self._closed or self._native_session is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="VadStreamingSession.poll_transitions: session is closed",
            )
        if self._terminal_seen:
            return

        resolved_deadline_ms = deadline_ms if deadline_ms is not None else _DEFAULT_DEADLINE_MS
        if resolved_deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(f"VadStreamingSession.poll_transitions: deadline_ms must be > 0; got {resolved_deadline_ms}"),
            )

        deadline = time.monotonic() + resolved_deadline_ms / 1000.0
        # Per-poll timeout. When draining streaming-style we keep this
        # short so the caller doesn't block; when draining-to-end we
        # use a longer slice so we don't burn CPU spinning. The
        # underlying poll_event returns OCT_EVENT_NONE on its own
        # timeout and raises only on real errors, so picking a poll
        # window is a tradeoff (responsiveness vs. CPU).
        per_poll_ms = 200 if drain_until_completed else 25
        poll_count = 0
        while time.monotonic() < deadline:
            poll_count += 1
            try:
                ev = self._native_session.poll_event(timeout_ms=per_poll_ms)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native VAD backend poll_event failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc
            if ev is None or ev.type == OCT_EVENT_NONE:
                if not drain_until_completed:
                    # Short-drain mode: one timeout slice == "no
                    # pending events right now". Return; the caller
                    # will poll again later.
                    return
                continue
            if ev.type == OCT_EVENT_SESSION_STARTED:
                continue
            if ev.type == OCT_EVENT_VAD_TRANSITION:
                kind = _kind_label(int(ev.vad_transition_kind))
                if kind == "unknown":
                    # Future-compat: never emitted by v0.1.5 but a
                    # hypothetical newer runtime might. Skip rather
                    # than crash.
                    continue
                yield VadTransition(
                    kind=kind,
                    timestamp_ms=int(ev.vad_timestamp_ms),
                    confidence=float(ev.vad_confidence),
                )
                continue
            if ev.type == OCT_EVENT_ERROR:
                # Runtime emitted an error mid-stream. The runtime
                # always follows ERROR with SESSION_COMPLETED carrying
                # the matching terminal_status; we let the next iter
                # surface the typed code. Don't raise here so the
                # caller still gets the precise terminal status.
                continue
            if ev.type == OCT_EVENT_SESSION_COMPLETED:
                self._terminal_seen = True
                terminal = int(ev.terminal_status)
                if terminal != OCT_STATUS_OK:
                    last_error = ""
                    if self._runtime is not None:
                        last_error = self._runtime.last_error()
                    raise _runtime_status_to_sdk_error(
                        terminal,
                        "native VAD backend session terminated with non-OK status",
                        last_error=last_error,
                    )
                return
            # Unknown event type — defensive ignore (forward-compat).
            continue
        # Loop fell through without seeing SESSION_COMPLETED.
        if drain_until_completed:
            raise OctomilError(
                code=OctomilErrorCode.REQUEST_TIMEOUT,
                message=(
                    f"VadStreamingSession.poll_transitions: timed out after "
                    f"{resolved_deadline_ms} ms waiting for SESSION_COMPLETED"
                ),
            )

    def close(self) -> None:
        """Close the underlying ``oct_session_t``. Idempotent."""
        if self._closed:
            return
        self._closed = True
        sess = self._native_session
        self._native_session = None
        if sess is not None:
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "VadStreamingSession.close: native session.close failed",
                    exc_info=True,
                )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class NativeVadBackend:
    """Hard-cut ``audio.vad`` backend backed by ``octomil-runtime v0.1.5+``.

    Caches a ``NativeRuntime`` per backend instance. Each
    ``open_session(...)`` call opens a model-less session (no
    ``oct_model_t`` consumed); the returned :class:`VadStreamingSession`
    is the streaming surface. Multiple sessions can be open
    concurrently against the same backend; the runtime's
    ``live_modelless_sessions`` refcount keeps ``oct_runtime_close``
    safe.

    There is no Python product path being replaced — VAD is a NEW
    capability surface in v0.1.5. The class docstring is the
    canonical low-level Python VAD entry point for the SDK.
    """

    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._runtime: NativeRuntime | None = None
        self._initialized: bool = False

    def open(self) -> None:
        """Open the underlying runtime and verify ``audio.vad`` is
        advertised. Idempotent.

        Raises
        ------
        OctomilError
            ``RUNTIME_UNAVAILABLE`` if the runtime fails to open OR
            does not advertise ``audio.vad`` (operator forgot
            ``OCTOMIL_SILERO_VAD_MODEL`` or the dylib was built
            without ``OCT_ENABLE_ENGINE_SILERO_VAD=ON``).
            ``CHECKSUM_MISMATCH`` if the artifact's digest doesn't
            match the runtime-pinned silero v6.2.0 SHA-256.
        """
        if self._initialized:
            return
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native VAD backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native VAD backend: dylib not found ({exc})",
            ) from exc

        if not _runtime_advertises_audio_vad(self._runtime):
            # Capability not advertised. Disambiguate the digest-mismatch
            # branch from the missing-env / missing-engine branch.
            #
            # Codex R1 F-01: relying on runtime.last_error() after a
            # successful capabilities() call is unreliable — the
            # capability snapshot filters out unloadable adapters
            # WITHOUT populating the runtime's thread-local last_error
            # (the silero adapter caches its diagnostic in
            # adapter-local cached_reason_; only oct_session_open's
            # dispatch block calls runtime->set_error with that
            # reason — see octomil-runtime/src/runtime.cpp:1255-1263).
            # So we extract the precise reason by attempting a probe
            # session_open against audio.vad — the dispatch will hit
            # the silero adapter, call load_status_reason(), and
            # surface the "digest mismatch — got X want Y" string via
            # NativeRuntimeError.last_error.
            probe_last_error = self._probe_audio_vad_unsupported_reason()
            self.close()
            if "digest" in probe_last_error.lower():
                raise OctomilError(
                    code=OctomilErrorCode.CHECKSUM_MISMATCH,
                    message=(
                        "native VAD backend: ggml-silero-v6.2.0.bin SHA-256 "
                        "does not match the v0.1.5 runtime-pinned digest "
                        f"(2aa269b7…fb6987). Re-download the artifact. Runtime "
                        f"diagnostic: {probe_last_error}"
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native VAD backend: runtime does not advertise "
                    "'audio.vad'. Check OCTOMIL_SILERO_VAD_MODEL "
                    "(must point at ggml-silero-v6.2.0.bin with SHA-256 "
                    "2aa269b7…fb6987) and that the dylib was built with "
                    "OCT_ENABLE_ENGINE_SILERO_VAD=ON. Runtime diagnostic: "
                    f"{probe_last_error}"
                ),
            )
        self._initialized = True
        logger.debug("NativeVadBackend: runtime opened + audio.vad advertised")

    def open_session(self, *, sample_rate_hz: int = _VAD_SAMPLE_RATE_HZ) -> VadStreamingSession:
        """Open a streaming VAD session.

        Returns a context-managed :class:`VadStreamingSession`. Use
        within a ``with`` block to guarantee the underlying
        ``oct_session_t`` is closed (which decrements the runtime's
        ``live_modelless_sessions`` refcount).

        Parameters
        ----------
        sample_rate_hz
            Hz. v0.1.5 only supports 16000.

        Raises
        ------
        OctomilError
            ``INVALID_INPUT`` for unsupported sample rates;
            ``RUNTIME_UNAVAILABLE`` if the backend hasn't been opened
            (or the runtime declined the capability — see :meth:`open`).
        """
        if sample_rate_hz != _VAD_SAMPLE_RATE_HZ:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"NativeVadBackend: sample_rate_hz must be {_VAD_SAMPLE_RATE_HZ} "
                    f"(silero VAD is mono-16kHz-only in v0.1.5); got {sample_rate_hz}"
                ),
            )
        if not self._initialized or self._runtime is None:
            self.open()
        assert self._runtime is not None  # mypy hint after open() guard
        return VadStreamingSession(self._runtime, sample_rate_hz)

    def _probe_audio_vad_unsupported_reason(self) -> str:
        """Probe ``oct_session_open(capability="audio.vad")`` to extract
        the precise dispatch-time UNSUPPORTED reason.

        Codex R1 F-01 fix: when the runtime hides ``audio.vad`` because
        the silero adapter's 5-gate check failed, the diagnostic lives
        in the adapter's ``cached_reason_`` (adapter-local), NOT in the
        runtime's thread-local ``last_error``. Only the dispatch path
        in ``oct_session_open`` propagates ``cached_reason_`` to
        ``last_error`` via ``runtime->set_error(... + load_status_reason())``
        (octomil-runtime/src/runtime.cpp:1255-1263). So we attempt a
        probe session — it WILL fail with UNSUPPORTED — and read the
        precise reason out of the runtime's last_error buffer.

        Codex R2 F-03 fix: ``NativeRuntimeError.last_error`` is captured
        by the loader at ``open_session`` failure-time using the default
        512-byte buflen (loader.py: ``self.last_error()``). The silero
        diagnostic format puts the artifact path BEFORE the word
        ``"digest"`` (silero_vad_adapter.cpp:114-122 emits
        ``"silero_vad: OCTOMIL_SILERO_VAD_MODEL=" + path + " digest mismatch — got " + got + " want " + want``),
        so a long absolute path can truncate ``"digest"`` out of the
        SDK-visible string and misclassify the integrity-failure case
        as RUNTIME_UNAVAILABLE. Mitigation: re-read the runtime
        ``last_error`` immediately after the probe failure with a
        4096-byte buffer (the runtime stores the full message
        server-side; the only loss is the SDK-side buffer slice). No
        intervening calls between the probe and this re-read, so the
        thread-local ``last_error`` is still the probe's diagnostic.

        Returns the diagnostic string (possibly empty if the probe
        couldn't extract one) so the caller can substring-match
        ``"digest"``.
        """
        if self._runtime is None:
            return ""
        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_VAD,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=_VAD_SAMPLE_RATE_HZ,
            )
        except NativeRuntimeError as exc:
            # Expected path: UNSUPPORTED with the precise reason.
            # F-03 fix: re-read with a 4 KB buffer to capture long
            # diagnostics (artifact paths can push "digest" past the
            # default 512-byte buflen).
            try:
                full = self._runtime.last_error(buflen=4096)
            except Exception:  # noqa: BLE001
                full = ""
            short = getattr(exc, "last_error", "") or ""
            # Prefer the longer of the two (defensive — the runtime's
            # last_error MAY have been overwritten by the loader's
            # close-on-error cleanup in some future change).
            return full if len(full) >= len(short) else short
        except Exception:  # noqa: BLE001
            # Any other exception (ImportError, etc.) — fall through.
            return ""
        else:
            # Unexpected: the probe SUCCEEDED. Close it and return ""
            # so the caller falls through to the generic
            # RUNTIME_UNAVAILABLE branch (this can only happen if the
            # capability was advertised between the introspection and
            # the probe; the open bumped live_modelless_sessions —
            # close it to decrement).
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass
            return ""

    def close(self) -> None:
        """Close the underlying runtime. Idempotent.

        ``oct_runtime_close`` refuses with ``BUSY`` while VAD sessions
        are alive; callers MUST close any open
        :class:`VadStreamingSession` first (the context-manager
        surface is the easy way).
        """
        self._initialized = False
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeVadBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "NativeVadBackend",
    "VadStreamingSession",
    "VadTransition",
    "runtime_advertises_audio_vad",
]
