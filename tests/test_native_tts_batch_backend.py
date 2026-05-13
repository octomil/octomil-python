from __future__ import annotations

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.tts_batch_backend import NativeTtsBatchBackend
from octomil.runtime.native.tts_stream_backend import TtsAudioChunk


def test_native_tts_batch_collects_runtime_chunks_into_wav(monkeypatch) -> None:
    backend = NativeTtsBatchBackend()
    backend._model_name = "piper-en-amy"  # type: ignore[attr-defined]

    def fake_chunks(*args, **kwargs):
        yield TtsAudioChunk(
            pcm_f32=np.asarray([0.0, 0.5, -0.5], dtype=np.float32),
            sample_rate_hz=22050,
            chunk_index=0,
            is_final=True,
            cumulative_duration_ms=1,
        )

    monkeypatch.setattr(backend, "synthesize_with_chunks", fake_chunks)
    result = backend.synthesize("hello", voice="0")

    assert result["content_type"] == "audio/wav"
    assert result["format"] == "wav"
    assert result["sample_rate"] == 22050
    assert result["duration_ms"] == 1
    assert result["audio_bytes"].startswith(b"RIFF")


def test_native_tts_batch_rejects_non_wav() -> None:
    backend = NativeTtsBatchBackend()
    with pytest.raises(OctomilError) as excinfo:
        backend.synthesize("hello", response_format="mp3")
    assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT
