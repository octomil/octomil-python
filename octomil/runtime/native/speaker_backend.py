"""NativeSpeakerEmbeddingBackend — audio.speaker.embedding via octomil-runtime v0.1.5+.

Hard-cutover backend for local speaker embedding extraction. The
runtime side (PR-2E, ``octomil-runtime`` squash ``c2e8d95``) lights
up ``audio.speaker.embedding`` via the sherpa-onnx wespeaker /
3D-Speaker ONNX adapter. The runtime advertises the capability only
when:

  1. The dylib was built with ``OCT_ENABLE_ENGINE_SHERPA_ONNX=ON``
     (and linked against sherpa-onnx providing
     ``OCT_HAVE_SHERPA_ONNX``).
  2. ``OCTOMIL_SHERPA_SPEAKER_MODEL`` points at a regular file.
  3. That file's SHA-256 matches
     ``1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b``
     (canonical ERes2NetV2 base, ~40 MB).

When any gate fails, the runtime hides the capability and
``oct_session_open(capability="audio.speaker.embedding")`` returns
``OCT_STATUS_UNSUPPORTED``. This binding fail-closes — there is no
prior Python product path for speaker embedding; this is a NEW
capability surface introduced in v0.1.5.

Lifecycle (model-bound, single-utterance):
- Unlike VAD, ``audio.speaker.embedding`` is a model-bound capability.
  ``oct_session_open`` requires a pre-warmed ``oct_model_t``
  (engine_hint=``"sherpa_onnx"``). The SDK opens + warms the model
  during :meth:`NativeSpeakerEmbeddingBackend.load_model` and reuses
  it across requests.
- Each ``embed(...)`` call opens a fresh session, sends the FULL
  audio clip via ``oct_session_send_audio`` (single shot — the runtime
  rejects further audio after the first send, single-utterance
  contract), drains until ``OCT_EVENT_EMBEDDING_VECTOR`` (id=20)
  followed by ``OCT_EVENT_SESSION_COMPLETED(OK)``, then closes the
  session deterministically. The model stays warm.
- Embedding dimension: 512 (canonical ERes2NetV2 output). The SDK
  surfaces the dim verbatim from the runtime event; we do NOT
  hard-code the value into the response shape so future model
  updates can extend or shrink without an SDK change.

Hard rules (cutover discipline — no silent Python fallback):
1. Caller must resolve capability advertisement at the planner; this
   backend is selected only when the runtime advertises
   ``audio.speaker.embedding``.
2. ``embed()`` rejects NaN / Inf / zero-length / wrong-shape /
   wrong-rate input via the runtime-side validator AND a Python-side
   pre-flight; both surface as ``INVALID_INPUT``.
3. Bad-digest path returns ``OCT_STATUS_UNSUPPORTED`` with
   ``last_error`` mentioning ``digest`` — surfaced as
   ``CHECKSUM_MISMATCH``.
4. Capability not advertised → ``RUNTIME_UNAVAILABLE``.

Bounded-error mapping (runtime → SDK):
- ``OCT_STATUS_NOT_FOUND`` → ``MODEL_NOT_FOUND``.
- ``OCT_STATUS_INVALID_INPUT`` → ``INVALID_INPUT``.
- ``OCT_STATUS_UNSUPPORTED`` w/ ``"digest"`` in last_error →
  ``CHECKSUM_MISMATCH``; otherwise → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_VERSION_MISMATCH`` → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_CANCELLED`` → ``CANCELLED``.
- ``OCT_STATUS_TIMEOUT`` → ``REQUEST_TIMEOUT``.
- ``OCT_STATUS_BUSY`` → ``SERVER_ERROR``.
- Any other terminal → ``INFERENCE_FAILED``.

This is the canonical low-level speaker-embedding API for the Python
SDK in v0.1.5; no prior product path is being replaced.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_SPEAKER_EMBEDDING
from .loader import (
    OCT_EVENT_EMBEDDING_VECTOR,
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
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


_BACKEND_NAME = "native-sherpa-speaker"
_DEFAULT_DEADLINE_MS = 300_000  # 5 minutes — same shape as STT / chat / embeddings.
_SPEAKER_SAMPLE_RATE_HZ: int = 16000
# Env-var the runtime adapter consults for the canonical ERes2NetV2
# .onnx artifact. Same env-driven path as silero_vad / whisper_cpp;
# bindings echo it for the model_uri so the runtime's URI parser has
# something concrete to point at when last_error is rendered.
_SHERPA_SPEAKER_BIN_ENV: str = "OCTOMIL_SHERPA_SPEAKER_MODEL"
# Canonical model identifier the SDK accepts. The runtime pins the
# specific SHA-256 — substituting a different speaker model identity
# is a correctness regression (different embedding manifold). Other
# names reject UNSUPPORTED_MODALITY.
_SUPPORTED_MODEL_NAME: str = "sherpa-eres2netv2-base"


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Map a runtime ``oct_status_t`` (+ last_error text) to the SDK's
    bounded error taxonomy. Mirrors
    :func:`octomil.runtime.native.stt_backend._runtime_status_to_sdk_error`
    so STT and speaker-embedding produce identical error shapes for
    identical runtime statuses."""
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


def _validate_clip_pcm_f32(samples: Any, sample_rate_hz: int) -> bytes:
    """Pre-flight an audio clip for the sherpa speaker send_audio
    boundary. Same shape contract as STT/VAD pre-flights — 16 kHz mono
    PCM-f32, finite samples, non-empty.
    """
    if sample_rate_hz != _SPEAKER_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"native speaker.embedding: sample_rate_hz must be "
                f"{_SPEAKER_SAMPLE_RATE_HZ} (sherpa speaker is mono-16kHz-only "
                f"in v0.1.5); got {sample_rate_hz}"
            ),
        )
    if isinstance(samples, (bytes, bytearray, memoryview)):
        buf = bytes(samples)
        if len(buf) == 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native speaker.embedding: zero-length audio buffer",
            )
        if len(buf) % 4 != 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(f"native speaker.embedding: PCM-f32 buffer length {len(buf)} " "is not a multiple of 4 bytes"),
            )
        try:
            import numpy as _np  # type: ignore[import-not-found]
        except ImportError:
            return buf
        view = _np.frombuffer(buf, dtype=_np.float32)
        if not _np.isfinite(view).all():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native speaker.embedding: audio contains NaN or Inf samples",
            )
        return buf
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native speaker.embedding: array-like audio input requires numpy; "
                "pass bytes (float32-LE) or `pip install numpy`"
            ),
        ) from exc
    arr = _np.asarray(samples, dtype=_np.float32)
    if arr.ndim != 1:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(f"native speaker.embedding: audio must be 1-D mono PCM-f32; " f"got shape {arr.shape}"),
        )
    if arr.size == 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native speaker.embedding: zero-length audio buffer",
        )
    if not _np.isfinite(arr).all():
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native speaker.embedding: audio contains NaN or Inf samples",
        )
    return arr.tobytes()


def _runtime_advertises_audio_speaker_embedding(rt: NativeRuntime) -> bool:
    """Capability-honesty check used by callers before constructing a
    backend. Returns False if the runtime fails introspection OR
    doesn't list ``audio.speaker.embedding``. Callers MUST raise
    ``RUNTIME_UNAVAILABLE`` rather than fall back to a non-existent
    Python speaker-embedding implementation."""
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_SPEAKER_EMBEDDING in caps.supported_capabilities


runtime_advertises_audio_speaker_embedding = _runtime_advertises_audio_speaker_embedding


class NativeSpeakerEmbeddingBackend:
    """Hard-cut ``audio.speaker.embedding`` backend backed by
    ``octomil-runtime v0.1.5+``.

    Caches a ``NativeRuntime`` + warmed ``NativeModel`` per backend
    instance. Each ``embed(...)`` call opens a fresh session, sends
    the audio clip, drains the embedding event, closes the session.
    """

    name: str = _BACKEND_NAME
    DEFAULT_DEADLINE_MS: int = _DEFAULT_DEADLINE_MS

    def __init__(self, *, default_deadline_ms: int | None = None) -> None:
        self._model_name: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None
        self._default_deadline_ms: int = (
            default_deadline_ms if default_deadline_ms is not None else self.DEFAULT_DEADLINE_MS
        )

    def load_model(self, model_name: str = _SUPPORTED_MODEL_NAME) -> None:
        """Open the runtime, verify ``audio.speaker.embedding`` is
        advertised, and pre-warm the sherpa model. Idempotent.

        Raises
        ------
        OctomilError
            ``UNSUPPORTED_MODALITY`` if the requested model name is not
            ``sherpa-eres2netv2-base``.
            ``RUNTIME_UNAVAILABLE`` if the runtime fails to open OR
            does not advertise ``audio.speaker.embedding`` (operator
            forgot ``OCTOMIL_SHERPA_SPEAKER_MODEL`` or the dylib was
            built without ``OCT_ENABLE_ENGINE_SHERPA_ONNX=ON``).
            ``CHECKSUM_MISMATCH`` if the artifact's SHA-256 doesn't
            match the runtime-pinned ERes2NetV2 digest.
        """
        if model_name.lower() != _SUPPORTED_MODEL_NAME:
            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message=(
                    f"native speaker.embedding backend: model {model_name!r} is not "
                    f"supported in v0.1.5. Only {_SUPPORTED_MODEL_NAME!r} is wired in "
                    "this release (the runtime pins a single ERes2NetV2 SHA-256). "
                    "Multi-model speaker embedding requires a runtime update."
                ),
            )
        if self._runtime is not None:
            return
        self._model_name = model_name
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native speaker.embedding backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native speaker.embedding backend: dylib not found ({exc})",
            ) from exc

        if not _runtime_advertises_audio_speaker_embedding(self._runtime):
            # Codex R1 F-02 fix (mirrors VAD F-01 fix): the
            # capability snapshot does NOT propagate the sherpa
            # adapter's cached_reason_ into the runtime's thread-
            # local last_error. Only the dispatch path in
            # oct_session_open does that
            # (octomil-runtime/src/runtime.cpp:1139-1147). So we
            # attempt a probe session_open against
            # audio.speaker.embedding to extract the precise
            # diagnostic — it WILL fail with UNSUPPORTED, but the
            # resulting NativeRuntimeError.last_error carries the
            # "digest mismatch — got X want Y" string when the
            # adapter declined for digest reasons.
            probe_last_error = self._probe_audio_speaker_embedding_unsupported_reason()
            self.close()
            if "digest" in probe_last_error.lower():
                raise OctomilError(
                    code=OctomilErrorCode.CHECKSUM_MISMATCH,
                    message=(
                        "native speaker.embedding backend: ERes2NetV2 SHA-256 "
                        "does not match the v0.1.5 runtime-pinned digest "
                        f"(1a331345…7a5e4b). Re-download the artifact. Runtime "
                        f"diagnostic: {probe_last_error}"
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native speaker.embedding backend: runtime does not advertise "
                    "'audio.speaker.embedding'. Check OCTOMIL_SHERPA_SPEAKER_MODEL "
                    "(must point at the ERes2NetV2 ONNX with SHA-256 "
                    "1a331345…7a5e4b) and that the dylib was built with "
                    "OCT_ENABLE_ENGINE_SHERPA_ONNX=ON. Runtime diagnostic: "
                    f"{probe_last_error}"
                ),
            )

        speaker_bin = os.environ.get(_SHERPA_SPEAKER_BIN_ENV, "")
        if not speaker_bin:
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"native speaker.embedding backend: {_SHERPA_SPEAKER_BIN_ENV} "
                    "not set. Point at a verified ERes2NetV2 .onnx (SHA-256 "
                    "1a331345…7a5e4b)."
                ),
            )
        try:
            self._model = self._runtime.open_model(
                model_uri=speaker_bin,
                engine_hint="sherpa_onnx",
            )
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native speaker.embedding backend failed to warm sherpa model",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        logger.debug("NativeSpeakerEmbeddingBackend: runtime opened + sherpa speaker model warmed")

    def _probe_audio_speaker_embedding_unsupported_reason(self) -> str:
        """Probe ``oct_session_open(capability="audio.speaker.embedding")``
        with ``model=None`` to extract the precise dispatch-time
        UNSUPPORTED reason.

        Codex R1 F-02 fix (mirrors VAD F-01): the runtime's capability
        snapshot does NOT propagate the sherpa adapter's
        ``cached_reason_`` to the runtime's thread-local last_error.
        Only oct_session_open's dispatch path does that
        (octomil-runtime/src/runtime.cpp:1139-1147). The dispatch
        order matters here: ``is_loadable_now()`` is checked BEFORE
        the model presence check (line 1149), so a probe with
        ``model=None`` returns the precise digest diagnostic when
        the adapter declined for digest reasons. If the adapter is
        loadable but the capability is hidden for some other reason,
        we'd hit the model-NULL branch and get an INVALID_INPUT
        instead — that's fine, we just won't see "digest" in the
        result and will fall through to RUNTIME_UNAVAILABLE.
        """
        if self._runtime is None:
            return ""
        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=_SPEAKER_SAMPLE_RATE_HZ,
                # model=None on purpose — we want the dispatch-time
                # adapter check to fire first.
            )
        except NativeRuntimeError as exc:
            return getattr(exc, "last_error", "") or ""
        except Exception:  # noqa: BLE001
            return ""
        else:
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass
            return ""

    def close(self) -> None:
        """Close the model BEFORE the runtime so ``oct_runtime_close``
        doesn't refuse with BUSY (mirrors the embeddings + STT backend
        ordering)."""
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "NativeSpeakerEmbeddingBackend.close: model.close failed",
                    exc_info=True,
                )
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "NativeSpeakerEmbeddingBackend.close: runtime.close failed",
                    exc_info=True,
                )
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def embed(
        self,
        audio_pcm_f32: Any,
        *,
        sample_rate_hz: int = _SPEAKER_SAMPLE_RATE_HZ,
        deadline_ms: int | None = None,
    ) -> Any:
        """Compute a speaker embedding for the given audio clip.

        Single-utterance: opens a session, sends the entire clip,
        drains to the terminal ``OCT_EVENT_EMBEDDING_VECTOR`` +
        ``OCT_EVENT_SESSION_COMPLETED(OK)``, returns the vector.

        Parameters
        ----------
        audio_pcm_f32
            Mono PCM-f32 clip. ``bytes`` (float32-LE) OR a 1-D numpy
            float32 array OR a list of floats. Stereo / multichannel
            rejects ``INVALID_INPUT``. The runtime accepts arbitrary-
            length clips; the canonical model produces a fixed
            512-dim vector regardless of input duration (the network
            includes a temporal pooling head).
        sample_rate_hz
            Hz. v0.1.5 only supports 16000.
        deadline_ms
            Per-request poll deadline. Falls back to
            ``self._default_deadline_ms`` when None.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of length ``n_dim`` (512 for the
            canonical ERes2NetV2 base). The runtime L2-normalizes
            in-engine; the SDK surfaces the normalized vector.

        Raises
        ------
        OctomilError
            See module docstring for the bounded mapping.
        """
        if self._runtime is None or self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeSpeakerEmbeddingBackend.embed called before load_model",
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

        audio_bytes = _validate_clip_pcm_f32(audio_pcm_f32, sample_rate_hz)

        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=sample_rate_hz,
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native speaker.embedding backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        try:
            try:
                sess.send_audio(audio_bytes, sample_rate=sample_rate_hz, channels=1)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native speaker.embedding backend send_audio failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            embedding_values: list[float] = []
            embedding_n_dim: int = 0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            saw_embedding = False

            deadline_seconds = resolved_deadline_ms / 1000.0
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                try:
                    ev = sess.poll_event(timeout_ms=200)
                except NativeRuntimeError as exc:
                    raise _runtime_status_to_sdk_error(
                        exc.status,
                        "native speaker.embedding backend poll_event failed",
                        last_error=getattr(exc, "last_error", ""),
                    ) from exc
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_EMBEDDING_VECTOR:
                    if saw_embedding:
                        # Single-utterance contract: runtime should emit
                        # exactly one EMBEDDING_VECTOR per session. A
                        # duplicate is a runtime invariant violation;
                        # surface as INFERENCE_FAILED rather than
                        # silently picking one.
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(
                                "native speaker.embedding: runtime emitted multiple "
                                "EMBEDDING_VECTOR events for a single-utterance session"
                            ),
                        )
                    embedding_values = list(ev.values)
                    embedding_n_dim = int(ev.n_dim)
                    if embedding_n_dim <= 0 or len(embedding_values) != embedding_n_dim:
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(
                                f"native speaker.embedding: malformed embedding event "
                                f"(n_dim={embedding_n_dim}, len(values)={len(embedding_values)})"
                            ),
                        )
                    saw_embedding = True
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
                        f"native speaker.embedding backend timed out waiting for "
                        f"SESSION_COMPLETED ({resolved_deadline_ms}ms)"
                    ),
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native speaker.embedding backend reported error during inference",
                    last_error=error_message,
                )

            if not saw_embedding:
                raise OctomilError(
                    code=OctomilErrorCode.INFERENCE_FAILED,
                    message=("native speaker.embedding: SESSION_COMPLETED(OK) without preceding EMBEDDING_VECTOR"),
                )

            try:
                import numpy as _np  # type: ignore[import-not-found]
            except ImportError as exc:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        "native speaker.embedding backend: numpy is required to "
                        "build the embedding return value; `pip install numpy`"
                    ),
                ) from exc
            return _np.asarray(embedding_values, dtype=_np.float32)
        finally:
            sess.close()


__all__ = [
    "NativeSpeakerEmbeddingBackend",
    "runtime_advertises_audio_speaker_embedding",
]
