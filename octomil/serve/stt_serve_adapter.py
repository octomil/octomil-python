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
import struct
import wave
from array import array
from typing import Any

from ..errors import OctomilError, OctomilErrorCode
from ..runtime.native.stt_backend import NativeSttBackend, TranscriptionResult

logger = logging.getLogger(__name__)


_WHISPER_SAMPLE_RATE_HZ: int = 16000


def _wav_to_pcm_f32_bytes(audio_path: str) -> tuple[bytes, int]:
    """Decode a 16-bit / 32-bit / 8-bit WAV into mono PCM-f32 bytes
    (float32 LE) at 16kHz. Multichannel inputs are downmixed by
    averaging channels. 24-bit WAVs reject UNSUPPORTED_MODALITY —
    Codex R3 nit: the previous docstring claimed 24-bit support but
    the decoder only handles widths 1, 2, 4. Adding 24-bit (3-byte)
    decode would require manual sign-extension via struct, which
    is parity-work outside the v0.1.5 cutover scope; callers that
    need 24-bit can convert upstream (ffmpeg -sample_fmt s16).

    Codex R2 blocker fix: stdlib-only implementation. The earlier
    version imported ``numpy`` unconditionally; numpy is in optional
    extras (``ml`` / ``fl`` / ``data``) and is excluded from the
    PyInstaller binary. Operators on a bare ``octomil[serve,native]``
    install would have hit an ImportError before reaching the
    backend. Using ``struct`` + ``array`` keeps the decode path
    self-contained at zero new dependency cost. The native backend
    itself only requires numpy when callers pass array-like inputs;
    bytes inputs (this adapter's path) skip that requirement.

    Caller-side resampling is NOT implemented in v0.1.5 — non-16kHz
    WAVs reject INVALID_INPUT. This matches the runtime's own
    validator.
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
                f"native whisper.cpp STT is hard-coded to {_WHISPER_SAMPLE_RATE_HZ} Hz "
                "mono. Resample upstream (ffmpeg -ar 16000 -ac 1)."
            ),
        )

    if sample_width == 2:
        # int16 LE → fp32 in [-1, 1).
        ints = array("h")  # signed 16-bit
        ints.frombytes(raw)
        # NOTE: array('h') is host-endian; WAV is LE. On every
        # supported platform (darwin-arm64, linux-amd64, etc.) the
        # host is LE, so the values are correct. Add an explicit
        # byteswap when porting to a BE host.
        floats = [v / 32768.0 for v in ints]
    elif sample_width == 4:
        # int32 LE → fp32. Same host-endian assumption as int16.
        ints32 = array("i")
        ints32.frombytes(raw)
        floats = [v / 2147483648.0 for v in ints32]
    elif sample_width == 1:
        # 8-bit PCM is unsigned [0, 255] with bias 128.
        floats = [(b - 128) / 128.0 for b in raw]
    else:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=f"native STT serve: WAV sample width {sample_width} bytes unsupported",
        )

    if n_channels == 1:
        return struct.pack(f"<{len(floats)}f", *floats), sample_rate
    if n_channels == 2:
        # Interleaved LRLR → mono via channel mean. We pair samples
        # in order; with len(floats) odd we'd have a silent
        # partial-frame at the tail (shouldn't happen for a
        # well-formed stereo WAV — wave.readframes returns frame-
        # aligned bytes).
        if len(floats) % 2 != 0:
            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message="native STT serve: stereo WAV produced odd float count",
            )
        mono = [(floats[i] + floats[i + 1]) * 0.5 for i in range(0, len(floats), 2)]
        return struct.pack(f"<{len(mono)}f", *mono), sample_rate
    raise OctomilError(
        code=OctomilErrorCode.UNSUPPORTED_MODALITY,
        message=f"native STT serve: WAV with {n_channels} channels unsupported (mono / stereo only)",
    )


def _diagnostics_to_dict(
    diagnostics: Any,
    kept_native_indices: list[int],
) -> dict[str, Any] | None:
    """Project the native ``ChunkDiagnostics`` onto a JSON-able dict.

    Returns ``None`` when chunking was off (full-buffer decode) so the
    default path stays identical to today. ``segment_owner_window`` is
    re-indexed to the RETURNED (non-empty) segments via
    ``kept_native_indices`` so it lines up with the emitted segments
    list rather than the native (pre-drop) one.
    """
    if diagnostics is None:
        return None
    owner = list(getattr(diagnostics, "segment_owner_window", []) or [])
    kept_owner = [owner[i] for i in kept_native_indices if 0 <= i < len(owner)]
    windows = [
        {
            "index": int(w.index),
            "start_ms": int(w.start_ms),
            "end_ms": int(w.end_ms),
            "off_ms": int(w.off_ms),
            "is_last": bool(w.is_last),
        }
        for w in getattr(diagnostics, "windows", []) or []
    ]
    return {
        "window_ms": int(diagnostics.window_ms),
        "overlap_ms": int(diagnostics.overlap_ms),
        "step_ms": int(diagnostics.step_ms),
        "window_count": int(diagnostics.window_count),
        "windows": windows,
        "segment_owner_window": kept_owner,
        "audio_duration_ms": int(diagnostics.audio_duration_ms),
        "final_segment_end_ms": int(diagnostics.final_segment_end_ms),
        "tail_gap_ms": int(diagnostics.tail_gap_ms),
    }


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

    def transcribe(
        self,
        audio_path: str,
        *,
        chunk_window_ms: int | None = None,
        chunk_overlap_ms: int | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file (WAV) and return the legacy dict
        shape, extended with v0.1.27 facade fields::

            {
                "text": "...",
                "segments": [
                    {
                        "start": float, "end": float, "text": "...",
                        "start_ms": int, "end_ms": int,
                        "avg_logprob": float, "no_speech_prob": float,
                    },
                    ...
                ],
                "duration_ms": int,
                "chunk_diagnostics": {...} | None,
            }

        ``start`` / ``end`` are seconds (float, 2-decimal rounded) to
        match the legacy pywhispercpp shape; the ``*_ms`` /
        ``avg_logprob`` / ``no_speech_prob`` keys are additive so legacy
        readers keep working. ``chunk_window_ms`` / ``chunk_overlap_ms``
        (when set) enable the runtime's fixed-window chunked transcribe;
        ``chunk_diagnostics`` is populated only on that path.
        """
        try:
            audio, sr = _wav_to_pcm_f32_bytes(audio_path)
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

        result: TranscriptionResult = self._backend.transcribe(
            audio,
            sample_rate_hz=sr,
            chunk_window_ms=chunk_window_ms,
            chunk_overlap_ms=chunk_overlap_ms,
        )

        # Track which RETURNED (non-empty) segment index maps to which
        # native segment index, so per-segment owner-window provenance
        # stays aligned after empty-text segments are dropped.
        segments_legacy: list[dict[str, Any]] = []
        kept_native_indices: list[int] = []
        for native_idx, seg in enumerate(result.segments):
            text = seg.text.strip()
            if not text:
                continue
            kept_native_indices.append(native_idx)
            segments_legacy.append(
                {
                    "start": round(seg.start_ms / 1000.0, 2),
                    "end": round(seg.end_ms / 1000.0, 2),
                    "text": text,
                    "start_ms": int(seg.start_ms),
                    "end_ms": int(seg.end_ms),
                    "avg_logprob": float(seg.avg_logprob),
                    "no_speech_prob": float(seg.no_speech_prob),
                }
            )
        return {
            "text": result.text.strip(),
            "segments": segments_legacy,
            "duration_ms": int(result.duration_ms),
            "chunk_diagnostics": _diagnostics_to_dict(result.chunk_diagnostics, kept_native_indices),
        }

    def close(self) -> None:
        self._backend.close()


__all__ = ["NativeSttServeAdapter"]
