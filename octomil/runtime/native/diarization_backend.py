"""NativeDiarizationBackend — audio.diarization via octomil-runtime.

``audio.diarization`` is LIVE_NATIVE_CONDITIONAL. The native runtime
advertises it only when the dylib was built with the sherpa-onnx
diarization subset and both ONNX artifact gates pass:

* ``OCTOMIL_DIARIZATION_SEGMENTATION_MODEL`` points at the canonical
  pyannote segmentation ``model.onnx``.
* ``OCTOMIL_SHERPA_SPEAKER_MODEL`` points at the canonical
  3D-Speaker embedding extractor ONNX.

There is no Python fallback in this module. If the native runtime does
not advertise the capability, callers get ``RUNTIME_UNAVAILABLE``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_DIARIZATION
from .error_mapping import map_oct_status
from .loader import (
    OCT_DIARIZATION_SPEAKER_UNKNOWN,
    OCT_EVENT_DIARIZATION_SEGMENT,
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_STATUS_OK,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)

_BACKEND_NAME = "native-sherpa-diarization"
_DIARIZATION_SAMPLE_RATE_HZ = 16000
_DEFAULT_DEADLINE_MS = 300_000


@dataclass(frozen=True)
class DiarizationSegment:
    """One speaker-turn segment emitted by the native runtime."""

    start_ms: int
    end_ms: int
    speaker_id: int
    speaker_label: str = ""

    @property
    def speaker_is_unknown(self) -> bool:
        return self.speaker_id == OCT_DIARIZATION_SPEAKER_UNKNOWN


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    return map_oct_status(
        status,
        last_error,
        message=message,
        default_unsupported_code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
    )


def _validate_pcm_f32(samples: Any, sample_rate_hz: int) -> bytes:
    if sample_rate_hz != _DIARIZATION_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native diarization: sample_rate_hz must be " f"{_DIARIZATION_SAMPLE_RATE_HZ}; got {sample_rate_hz}"
            ),
        )
    if isinstance(samples, (bytes, bytearray, memoryview)):
        buf = bytes(samples)
        if not buf:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native diarization: zero-length audio buffer",
            )
        if len(buf) % 4 != 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=("native diarization: PCM-f32 buffer length " f"{len(buf)} is not a multiple of 4 bytes"),
            )
        try:
            import numpy as _np  # type: ignore[import-not-found]
        except ImportError:
            return buf
        view = _np.frombuffer(buf, dtype=_np.float32)
        if not _np.isfinite(view).all():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native diarization: audio contains NaN or Inf samples",
            )
        return buf

    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native diarization: array-like audio input requires numpy; " "pass bytes (float32-LE) or install numpy"
            ),
        ) from exc
    arr = _np.asarray(samples, dtype=_np.float32)
    if arr.ndim != 1:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"native diarization: audio must be 1-D mono PCM-f32; got shape {arr.shape}",
        )
    if arr.size == 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native diarization: zero-length audio buffer",
        )
    if not _np.isfinite(arr).all():
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native diarization: audio contains NaN or Inf samples",
        )
    return arr.tobytes()


def runtime_advertises_audio_diarization(rt: NativeRuntime) -> bool:
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_DIARIZATION in caps.supported_capabilities


class NativeDiarizationBackend:
    """Low-level Python wrapper for the native audio.diarization session."""

    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._runtime: NativeRuntime | None = None
        self._initialized = False

    def open(self) -> None:
        if self._initialized:
            return
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native diarization backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native diarization backend: dylib not found ({exc})",
            ) from exc

        if not runtime_advertises_audio_diarization(self._runtime):
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native diarization backend: runtime does not advertise "
                    "'audio.diarization'. Check that the dylib was built with "
                    "OCT_ENABLE_ENGINE_DIARIZATION=ON and that "
                    "OCTOMIL_DIARIZATION_SEGMENTATION_MODEL plus "
                    "OCTOMIL_SHERPA_SPEAKER_MODEL point at canonical ONNX files."
                ),
            )
        self._initialized = True

    def diarize(
        self,
        audio_pcm_f32: Any,
        *,
        sample_rate_hz: int = _DIARIZATION_SAMPLE_RATE_HZ,
        deadline_ms: int = _DEFAULT_DEADLINE_MS,
    ) -> list[DiarizationSegment]:
        if deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"NativeDiarizationBackend.diarize: deadline_ms must be > 0; got {deadline_ms}",
            )
        if not self._initialized or self._runtime is None:
            self.open()
        assert self._runtime is not None

        audio_bytes = _validate_pcm_f32(audio_pcm_f32, sample_rate_hz)
        try:
            session = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_DIARIZATION,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=sample_rate_hz,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native diarization backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        segments: list[DiarizationSegment] = []
        try:
            try:
                session.send_audio(audio_bytes, sample_rate=sample_rate_hz, channels=1)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native diarization backend send_audio failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            deadline = time.monotonic() + deadline_ms / 1000.0
            while time.monotonic() < deadline:
                try:
                    ev = session.poll_event(timeout_ms=200)
                except NativeRuntimeError as exc:
                    raise _runtime_status_to_sdk_error(
                        exc.status,
                        "native diarization backend poll_event failed",
                        last_error=getattr(exc, "last_error", ""),
                    ) from exc
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_DIARIZATION_SEGMENT:
                    segments.append(
                        DiarizationSegment(
                            start_ms=int(ev.diarization_start_ms),
                            end_ms=int(ev.diarization_end_ms),
                            speaker_id=int(ev.diarization_speaker_id),
                            speaker_label=str(ev.diarization_speaker_label or ""),
                        )
                    )
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    terminal = int(ev.terminal_status)
                    if terminal != OCT_STATUS_OK:
                        raise _runtime_status_to_sdk_error(
                            terminal,
                            "native diarization backend session terminated with non-OK status",
                            last_error=self._runtime.last_error(),
                        )
                    return segments
            raise OctomilError(
                code=OctomilErrorCode.REQUEST_TIMEOUT,
                message=(
                    "NativeDiarizationBackend.diarize: timed out after "
                    f"{deadline_ms} ms waiting for SESSION_COMPLETED"
                ),
            )
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeDiarizationBackend: session.close failed", exc_info=True)

    def close(self) -> None:
        self._initialized = False
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeDiarizationBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "DiarizationSegment",
    "NativeDiarizationBackend",
    "runtime_advertises_audio_diarization",
]
