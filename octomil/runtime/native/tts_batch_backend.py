"""Native ``audio.tts.batch`` backend.

The runtime already exposes batch TTS as a native capability when the
Sherpa TTS build and canonical piper-amy artifact gates pass. This
wrapper gives Python product code a batch-shaped API without falling
back to the legacy Python Sherpa engine.
"""

from __future__ import annotations

from typing import Any

from ...audio.streaming import pcm_s16le_to_wav_bytes
from .capabilities import CAPABILITY_AUDIO_TTS_BATCH
from .tts_stream_backend import NativeTtsStreamBackend, TtsAudioChunk, _runtime_advertises_tts_capability


def runtime_advertises_tts_batch(rt: Any) -> bool:
    return _runtime_advertises_tts_capability(rt, CAPABILITY_AUDIO_TTS_BATCH)


class NativeTtsBatchBackend(NativeTtsStreamBackend):
    """Batch-shaped wrapper over the runtime ``audio.tts.batch`` session."""

    name = "native-sherpa-onnx-tts-batch"
    capability_id = CAPABILITY_AUDIO_TTS_BATCH
    backend_label = "TTS-batch"
    supported_model_names = frozenset({"piper-en-amy", "piper-amy"})
    synthesize_stream = None  # type: ignore[assignment]

    def load_model(self, model_name: str, **kwargs: Any) -> None:
        if model_name.lower() not in self.supported_model_names:
            from ...errors import OctomilError, OctomilErrorCode

            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message=(
                    f"native TTS-batch supports the canonical piper-amy runtime artifact only; "
                    f"got model {model_name!r}. Use piper-en-amy or keep this capability unadvertised."
                ),
            )
        super().load_model(model_name, **kwargs)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        *,
        response_format: str = "wav",
        deadline_ms: int | None = None,
    ) -> dict[str, Any]:
        if response_format.lower() != "wav":
            from ...errors import OctomilError, OctomilErrorCode

            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native TTS-batch supports response_format='wav' only",
            )

        chunks = list(
            self.synthesize_with_chunks(
                text,
                voice_id=voice,
                speed=speed,
                deadline_ms=deadline_ms,
            )
        )
        pcm_s16 = b"".join(_chunk_to_pcm_s16le(chunk) for chunk in chunks)
        sample_rate = chunks[-1].sample_rate_hz if chunks else 22050
        duration_ms = chunks[-1].cumulative_duration_ms if chunks else 0
        return {
            "audio_bytes": pcm_s16le_to_wav_bytes(pcm_s16, sample_rate, channels=1),
            "content_type": "audio/wav",
            "format": "wav",
            "model": self._model_name,
            "voice": voice or "0",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
        }


def _chunk_to_pcm_s16le(chunk: TtsAudioChunk) -> bytes:
    import numpy as _np

    arr = _np.asarray(chunk.pcm_f32, dtype=_np.float32)
    clipped = _np.clip(arr, -1.0, 1.0)
    return bytes((clipped * 32767.0).astype(_np.int16).tobytes())


__all__ = [
    "NativeTtsBatchBackend",
    "runtime_advertises_tts_batch",
]
