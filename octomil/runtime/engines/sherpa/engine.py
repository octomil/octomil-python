"""sherpa-onnx engine plugin -- on-device text-to-speech via sherpa-onnx.

sherpa-onnx (k2-fsa) ships VITS/Piper/Kokoro TTS models packaged as ONNX.
This plugin wraps the sherpa-onnx Python bindings so TTS models register
with the octomil engine registry under the canonical ``sherpa-onnx``
executor id.

Unlike LLM engines, TTS does NOT use ``generate()`` / ``generate_stream()``.
Instead, the backend exposes a ``synthesize()`` method and the serve layer
adds an OpenAI-compatible ``/v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import logging
import os
import time
import wave
from io import BytesIO
from typing import Any

from octomil.runtime.core.base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Supported TTS models -- name -> (family, default voice).
# family selects the sherpa-onnx config path:
#   "kokoro" -> OfflineTtsKokoroModelConfig (model + voices.bin + tokens + data_dir)
#   "vits"   -> OfflineTtsVitsModelConfig   (Piper-style: model.onnx + tokens + data_dir)
# Voice catalogs are model-specific; the second tuple element is the default
# voice the backend uses when the request does not specify one.
_SHERPA_TTS_MODELS: dict[str, tuple[str, str]] = {
    "kokoro-82m": ("kokoro", "af_bella"),
    "piper-en-amy": ("vits", "amy"),
    "piper-en-ryan": ("vits", "ryan"),
}


def _model_family(model_name: str) -> str:
    """Return the sherpa-onnx config family ('kokoro' or 'vits') for a model."""
    entry = _SHERPA_TTS_MODELS.get(model_name.lower())
    return entry[0] if entry else ""


def _default_voice(model_name: str) -> str:
    entry = _SHERPA_TTS_MODELS.get(model_name.lower())
    return entry[1] if entry else ""


# Kokoro v0.19+ voice catalog. Index == speaker id in the bundled voices.bin.
# Source: sherpa-onnx scripts/kokoro voice manifest. Operators can override
# this by dropping a voices.txt sidecar in the model directory.
_KOKORO_VOICES: tuple[str, ...] = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)


def _has_sherpa_onnx() -> bool:
    """Check if the sherpa_onnx package is importable."""
    try:
        import sherpa_onnx  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_sherpa_version() -> str:
    """Return sherpa_onnx version string, or empty if unavailable."""
    try:
        import sherpa_onnx  # type: ignore[import-untyped]

        return getattr(sherpa_onnx, "__version__", "unknown")
    except ImportError:
        return ""


def is_sherpa_tts_model(model_name: str) -> bool:
    """Check if a model name refers to a sherpa-onnx TTS model."""
    return model_name.lower() in _SHERPA_TTS_MODELS


class SherpaTtsEngine(EnginePlugin):
    """Text-to-speech engine using sherpa-onnx."""

    @property
    def name(self) -> str:
        return "sherpa-onnx"

    @property
    def display_name(self) -> str:
        return "sherpa-onnx (Text-to-Speech)"

    @property
    def priority(self) -> int:
        return 36  # Sits next to whisper.cpp (35).

    def detect(self) -> bool:
        return _has_sherpa_onnx()

    def detect_info(self) -> str:
        version = _get_sherpa_version()
        if not version:
            return ""
        models = ", ".join(sorted(_SHERPA_TTS_MODELS.keys()))
        return f"sherpa_onnx {version}; tts models: {models}"

    def supports_model(self, model_name: str) -> bool:
        return is_sherpa_tts_model(model_name)

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Benchmark by synthesizing a short reference utterance.

        For TTS, ``tokens_per_second`` is repurposed as
        ``audio_seconds_per_second`` (real-time factor).
        """
        if not _has_sherpa_onnx():
            return BenchmarkResult(engine_name=self.name, error="sherpa_onnx not available")

        if not is_sherpa_tts_model(model_name):
            return BenchmarkResult(
                engine_name=self.name,
                error=f"Unsupported model: {model_name}",
            )

        try:
            backend = _SherpaTtsBackend(model_name)
            backend.load_model(model_name)

            reference = "Octomil benchmark synthesis check."

            start = time.monotonic()
            result = backend.synthesize(reference)
            elapsed = time.monotonic() - start

            audio_duration_s = result["duration_ms"] / 1000.0
            realtime_factor = audio_duration_s / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=realtime_factor,
                metadata={
                    "method": "synthesize",
                    "audio_seconds_per_second": realtime_factor,
                    "audio_duration_s": round(audio_duration_s, 3),
                    "elapsed_s": round(elapsed, 3),
                    "model": model_name,
                    "sample_chars": len(reference),
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return _SherpaTtsBackend(model_name, **kwargs)


class _SherpaTtsBackend:
    """Text-to-speech backend using sherpa-onnx.

    Unlike LLM backends, this does NOT implement ``generate()`` or
    ``generate_stream()``. Instead it provides ``synthesize(text, voice, speed)``
    returning audio bytes plus metadata. The serve layer adds a dedicated
    ``/v1/audio/speech`` endpoint that mirrors OpenAI ``audio.speech.create``.
    """

    name = "sherpa-onnx"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        self._tts: Any = None
        self._sample_rate: int = 24000
        self._family: str = _model_family(model_name)
        self._default_voice: str = _default_voice(model_name)

    def load_model(self, model_name: str) -> None:
        """Load a sherpa-onnx TTS model from the configured model directory.

        Branches on model family because Kokoro and VITS/Piper expect
        different OfflineTtsModelConfig shapes:
          - kokoro: OfflineTtsKokoroModelConfig(model, voices, tokens, data_dir)
          - vits:   OfflineTtsVitsModelConfig(model, tokens, data_dir)
        """
        self._model_name = model_name
        if not is_sherpa_tts_model(model_name):
            raise ValueError(
                f"Unknown sherpa-onnx TTS model '{model_name}'. Available: {', '.join(sorted(_SHERPA_TTS_MODELS))}"
            )

        import sherpa_onnx  # type: ignore[import-untyped]

        model_dir = self._resolve_model_dir(model_name)
        family = _model_family(model_name)
        num_threads = int(self._kwargs.get("num_threads", 2))
        provider = self._kwargs.get("provider", "cpu")

        if family == "kokoro":
            inner_model_config = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                    voices=os.path.join(model_dir, "voices.bin"),
                    tokens=os.path.join(model_dir, "tokens.txt"),
                    data_dir=os.path.join(model_dir, "espeak-ng-data"),
                ),
                num_threads=num_threads,
                provider=provider,
            )
        elif family == "vits":
            inner_model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                    tokens=os.path.join(model_dir, "tokens.txt"),
                    data_dir=os.path.join(model_dir, "espeak-ng-data"),
                ),
                num_threads=num_threads,
                provider=provider,
            )
        else:
            raise ValueError(f"Unsupported sherpa-onnx TTS family '{family}' for model '{model_name}'.")

        config = sherpa_onnx.OfflineTtsConfig(model=inner_model_config)
        logger.info("Loading sherpa-onnx %s TTS: %s from %s", family, model_name, model_dir)
        self._tts = sherpa_onnx.OfflineTts(config)
        self._sample_rate = self._tts.sample_rate
        self._family = family
        self._default_voice = _default_voice(model_name)
        logger.info(
            "sherpa-onnx TTS loaded: %s (family=%s, sample_rate=%d)",
            model_name,
            family,
            self._sample_rate,
        )

    @staticmethod
    def _resolve_model_dir(model_name: str) -> str:
        """Return the on-disk directory for a sherpa-onnx model.

        Resolution order:
        1. ``OCTOMIL_SHERPA_MODELS_DIR`` env var if set.
        2. ``~/.octomil/models/sherpa/<model_name>/``.
        """
        override = os.environ.get("OCTOMIL_SHERPA_MODELS_DIR")
        if override:
            return os.path.join(override, model_name)
        return os.path.expanduser(f"~/.octomil/models/sherpa/{model_name}")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> dict[str, Any]:
        """Synthesize speech from text and return audio bytes + metadata.

        Returns::

            {
                "audio_bytes": bytes,         # WAV (PCM 16-bit mono)
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 1234,
                "voice": "af_bella",
                "model": "kokoro-82m",
            }

        ``voice`` defaults to the model's default if not provided.
        ``speed`` is a multiplier; 1.0 is default, 0.5 half-speed, 2.0 double.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")

        if self._tts is None:
            self.load_model(self._model_name)
        assert self._tts is not None

        voice_name = (voice or self._default_voice or "").strip()
        sid = self._voice_to_sid(voice_name)

        audio = self._tts.generate(text, sid=sid, speed=speed)
        samples = list(audio.samples)
        sample_rate = audio.sample_rate or self._sample_rate

        wav_bytes = _samples_to_wav(samples, sample_rate)
        duration_ms = int(round(1000 * len(samples) / sample_rate)) if sample_rate else 0

        return {
            "audio_bytes": wav_bytes,
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "voice": voice_name,
            "model": self._model_name,
        }

    def _voice_to_sid(self, voice: str) -> int:
        """Map a voice name to a sherpa-onnx speaker id.

        Resolution order:
        1. ``voices.txt`` sidecar (one voice id per line, position == sid).
           Lets the operator override the default catalog if the model bundle
           ships its own voice list.
        2. Built-in Kokoro catalog (``_KOKORO_VOICES``) for kokoro-* models.
        3. sid 0 — the model's first/default speaker.
        """
        if not voice:
            return 0
        sidecar = os.path.join(self._resolve_model_dir(self._model_name), "voices.txt")
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip().lower() == voice.lower():
                        return idx
            return 0
        if self._family == "kokoro":
            try:
                return _KOKORO_VOICES.index(voice.lower())
            except ValueError:
                return 0
        return 0

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


def _samples_to_wav(samples: list[float], sample_rate: int) -> bytes:
    """Encode float samples in [-1, 1] as a WAV byte string (PCM 16-bit mono)."""
    import struct

    pcm = bytearray()
    for s in samples:
        clipped = max(-1.0, min(1.0, s))
        pcm += struct.pack("<h", int(clipped * 32767.0))

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(pcm))
    return buf.getvalue()
