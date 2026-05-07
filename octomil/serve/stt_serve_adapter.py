"""Thin file-path adapter from the legacy serve API to NativeSttBackend.

The legacy `_WhisperBackend.transcribe(audio_path) -> dict` shape is
called from the FastAPI ``/v1/audio/transcriptions`` endpoint. The
v0.1.5 PR-2B cutover replaces the underlying pywhispercpp call with
:class:`octomil.runtime.native.stt_backend.NativeSttBackend`, but the
HTTP endpoint still hands the backend a temp-file path. This module
converts the file-path call shape into PCM-f32 + sample_rate_hz that
the native backend expects, then projects the rich
:class:`TranscriptionResult` back down to the legacy dict shape.

Hard rules:
1. No silent fallback. If the native backend can't open or
   ``audio.transcription`` isn't advertised, surface
   :class:`OctomilError` — do NOT route to pywhispercpp on the
   product path.
2. WAV-only for v0.1.5 cutover. Other formats (mp3, m4a, ogg)
   require an external decoder (ffmpeg / pydub). The legacy path
   accepted those because pywhispercpp pulled in libsndfile at
   build time; the native path keeps that decoder cost OUTSIDE the
   runtime ABI for now. Non-WAV inputs raise
   ``UNSUPPORTED_MODALITY`` with a clear diagnostic; the metrics +
   format-fanout work lands in a follow-up PR.
"""

from __future__ import annotations

import logging
import wave
from typing import Any

import numpy as np

from ..errors import OctomilError, OctomilErrorCode
from ..runtime.native.stt_backend import NativeSttBackend, TranscriptionResult

logger = logging.getLogger(__name__)


_WHISPER_SAMPLE_RATE_HZ: int = 16000


def _wav_to_pcm_f32(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode a 16-bit / 24-bit / 32-bit WAV into mono PCM-f32 at
    16kHz. Multichannel inputs are downmixed by averaging channels.

    Caller-side resampling is NOT implemented in v0.1.5 — non-16kHz
    WAVs reject INVALID_INPUT. This is consistent with the runtime's
    own validator.
    """
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_rate != _WHISPER_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"native STT serve: WAV sample rate {sample_rate} Hz unsupported; "
                f"v0.1.5 whisper-tiny is hard-coded to {_WHISPER_SAMPLE_RATE_HZ} Hz "
                "mono. Resample upstream (ffmpeg -ar 16000 -ac 1)."
            ),
        )

    if sample_width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        # Could be int32 PCM or float32. WAV files don't tag this in
        # the basic `wave` module — assume int32 PCM (more common).
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        # 8-bit PCM is unsigned [0, 255] with bias 128.
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=f"native STT serve: WAV sample width {sample_width} bytes unsupported",
        )

    if n_channels == 1:
        return arr, sample_rate
    if n_channels == 2:
        # Interleaved LRLR → mono via channel mean.
        stereo = arr.reshape(-1, 2)
        mono = stereo.mean(axis=1).astype(np.float32)
        return mono, sample_rate
    raise OctomilError(
        code=OctomilErrorCode.UNSUPPORTED_MODALITY,
        message=f"native STT serve: WAV with {n_channels} channels unsupported (mono / stereo only)",
    )


class NativeSttServeAdapter:
    """File-path adapter on top of :class:`NativeSttBackend`.

    Exposes the legacy ``.name``, ``.load_model(model_name)``, and
    ``.transcribe(audio_path) -> dict`` shape so the FastAPI server
    code in :mod:`octomil.serve.app` doesn't need to know about the
    native ABI.
    """

    name: str = "native-whisper-cpp"

    def __init__(self) -> None:
        self._backend = NativeSttBackend()
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        self._model_name = model_name
        self._backend.load_model(model_name)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file (WAV) and return the legacy dict
        shape::

            {"text": "...", "segments": [{"start": float, "end": float, "text": "..."}, ...]}

        ``start`` / ``end`` are seconds (float, 2-decimal rounded) to
        match the legacy pywhispercpp shape.
        """
        try:
            audio, sr = _wav_to_pcm_f32(audio_path)
        except wave.Error as exc:
            # `wave` raises wave.Error for non-WAV / malformed WAV. Map
            # to UNSUPPORTED_MODALITY so callers know to convert
            # upstream rather than retry on the same blob.
            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message=(
                    f"native STT serve: file at {audio_path!r} is not a valid WAV "
                    f"({exc}). v0.1.5 cutover requires WAV input; convert upstream."
                ),
            ) from exc

        result: TranscriptionResult = self._backend.transcribe(audio, sample_rate_hz=sr)

        segments_legacy: list[dict[str, Any]] = []
        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue
            segments_legacy.append(
                {
                    "start": round(seg.start_ms / 1000.0, 2),
                    "end": round(seg.end_ms / 1000.0, 2),
                    "text": text,
                }
            )
        return {
            "text": result.text.strip(),
            "segments": segments_legacy,
        }

    def close(self) -> None:
        self._backend.close()


__all__ = ["NativeSttServeAdapter"]
