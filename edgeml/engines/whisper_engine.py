"""Whisper.cpp engine plugin -- on-device speech-to-text via pywhispercpp.

Whisper.cpp provides fast, CPU-optimized speech-to-text inference using
OpenAI's Whisper models compiled with ggml.  This engine plugin wraps
``pywhispercpp`` to integrate with the edgeml engine registry.

Unlike LLM engines, Whisper does NOT use ``generate()`` / ``generate_stream()``.
Instead, the backend exposes a ``transcribe()`` method and the serve layer adds
an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Whisper model sizes — name to HuggingFace-style identifier.
_WHISPER_MODELS: dict[str, str] = {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3": "large-v3",
}


def _has_pywhispercpp() -> bool:
    """Check if the pywhispercpp package is importable."""
    try:
        import pywhispercpp  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_whisper_version() -> str:
    """Return pywhispercpp version string, or empty if unavailable."""
    try:
        import pywhispercpp  # type: ignore[import-untyped]

        return getattr(pywhispercpp, "__version__", "unknown")
    except ImportError:
        return ""


def is_whisper_model(model_name: str) -> bool:
    """Check if a model name refers to a Whisper speech-to-text model."""
    return model_name.lower() in _WHISPER_MODELS


class WhisperCppEngine(EnginePlugin):
    """Speech-to-text engine using whisper.cpp via pywhispercpp."""

    @property
    def name(self) -> str:
        return "whisper.cpp"

    @property
    def display_name(self) -> str:
        return "Whisper.cpp (Speech-to-Text)"

    @property
    def priority(self) -> int:
        return 35  # After onnxruntime (30), before echo (999)

    def detect(self) -> bool:
        return _has_pywhispercpp()

    def detect_info(self) -> str:
        version = _get_whisper_version()
        if not version:
            return ""
        models = ", ".join(sorted(_WHISPER_MODELS.keys()))
        return f"pywhispercpp {version}; models: {models}"

    def supports_model(self, model_name: str) -> bool:
        return is_whisper_model(model_name)

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Benchmark by transcribing a short silent audio segment.

        For Whisper, ``tokens_per_second`` is repurposed as
        ``audio_seconds_per_second`` (real-time factor).
        """
        if not _has_pywhispercpp():
            return BenchmarkResult(
                engine_name=self.name, error="pywhispercpp not available"
            )

        if not is_whisper_model(model_name):
            return BenchmarkResult(
                engine_name=self.name,
                error=f"Unsupported model: {model_name}",
            )

        try:
            from pywhispercpp.model import Model  # type: ignore[import-untyped]

            whisper_size = _WHISPER_MODELS[model_name.lower()]

            # Generate a short silent WAV file for benchmarking
            silent_path = _generate_silent_wav(duration_s=3.0)
            try:
                model = Model(whisper_size)

                start = time.monotonic()
                model.transcribe(silent_path)
                elapsed = time.monotonic() - start

                audio_duration = 3.0
                realtime_factor = audio_duration / elapsed if elapsed > 0 else 0.0

                return BenchmarkResult(
                    engine_name=self.name,
                    tokens_per_second=realtime_factor,
                    metadata={
                        "method": "transcribe",
                        "audio_seconds_per_second": realtime_factor,
                        "audio_duration_s": audio_duration,
                        "elapsed_s": round(elapsed, 3),
                        "whisper_size": whisper_size,
                    },
                )
            finally:
                os.unlink(silent_path)

        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return _WhisperBackend(model_name, **kwargs)


class _WhisperBackend:
    """Speech-to-text backend using whisper.cpp.

    Unlike LLM backends, this does NOT implement ``generate()`` or
    ``generate_stream()``.  Instead it provides ``transcribe()`` for
    audio-to-text conversion.  The serve layer adds a dedicated
    ``/v1/audio/transcriptions`` endpoint.
    """

    name = "whisper.cpp"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        self._model: Any = None

    def load_model(self, model_name: str) -> None:
        """Download (if needed) and load a Whisper model."""
        self._model_name = model_name
        whisper_size = _WHISPER_MODELS.get(model_name.lower())
        if whisper_size is None:
            raise ValueError(
                f"Unknown whisper model '{model_name}'. "
                f"Available: {', '.join(sorted(_WHISPER_MODELS))}"
            )

        from pywhispercpp.model import Model  # type: ignore[import-untyped]

        logger.info("Loading whisper model: %s (%s)", model_name, whisper_size)
        self._model = Model(whisper_size)
        logger.info("Whisper model loaded: %s", model_name)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file and return text + segments.

        Returns an OpenAI Whisper API-compatible dict::

            {
                "text": "The quick brown fox...",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "The quick brown fox"},
                    ...
                ]
            }
        """
        if self._model is None:
            self.load_model(self._model_name)

        segments_raw = self._model.transcribe(audio_path)

        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for seg in segments_raw:
            # pywhispercpp segments have t0, t1 (timestamps in ms) and text
            start = getattr(seg, "t0", 0) / 100.0
            end = getattr(seg, "t1", 0) / 100.0
            text = getattr(seg, "text", str(seg)).strip()
            if text:
                segments.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                })
                text_parts.append(text)

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "segments": segments,
        }

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


def _generate_silent_wav(
    duration_s: float = 3.0, sample_rate: int = 16000
) -> str:
    """Generate a silent WAV file for benchmarking. Returns the temp file path."""
    import struct
    import wave

    num_frames = int(sample_rate * duration_s)
    silence = struct.pack(f"<{num_frames}h", *([0] * num_frames))

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(silence)

    return path
