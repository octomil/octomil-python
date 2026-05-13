"""OctomilAudio — audio namespace.

Two shapes coexist:

* :class:`OctomilAudio` is the local-only namespace exposed on the legacy
  ``OctomilClient`` (``client.audio.transcriptions.create``).
* :class:`FacadeAudio` is the unified routed namespace exposed on the
  top-level :class:`octomil.Octomil` facade (``client.audio.speech.create``).
  It delegates to :class:`octomil.execution.kernel.ExecutionKernel` so a
  single code path resolves app refs and respects the routing policy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from octomil.audio.diarization import (
    DiarizationSegment,
    NativeDiarizationBackend,
    open_diarization_backend,
)
from octomil.audio.speaker_embedding import (
    NativeSpeakerEmbeddingBackend,
    open_speaker_embedding_backend,
)
from octomil.audio.speech import (
    FacadeSpeech,
    FacadeVoices,
    SpeechResponse,
    SpeechRoute,
    VoiceCatalog,
    VoiceInfo,
)
from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import (
    DiarizationResult,
    SpeakerEmbeddingResult,
    TranscriptionResult,
    TranscriptionSegment,
    VadResult,
)
from octomil.audio.vad import NativeVadBackend, VadTransition, open_vad_backend
from octomil.model_ref import ModelRef
from octomil.runtime.core.model_runtime import ModelRuntime


class OctomilAudio:
    """Namespace for audio APIs on the legacy ``OctomilClient``.

    Usage::

        result = await client.audio.transcriptions.create(audio=data)
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
    ) -> None:
        self._transcriptions = AudioTranscriptions(runtime_resolver)
        self._vad = FacadeVad()
        self._speaker_embedding = FacadeSpeakerEmbedding()
        self._diarization = FacadeDiarization()

    @property
    def transcriptions(self) -> AudioTranscriptions:
        return self._transcriptions

    @property
    def vad(self) -> "FacadeVad":
        return self._vad

    @property
    def speaker_embedding(self) -> "FacadeSpeakerEmbedding":
        return self._speaker_embedding

    @property
    def diarization(self) -> "FacadeDiarization":
        return self._diarization


class FacadeVad:
    """Native-only ``client.audio.vad`` namespace.

    This product surface is a thin route over
    :mod:`octomil.runtime.native.vad_backend`. It never dispatches to
    cloud and has no Python fallback; unavailable native runtimes raise
    bounded ``OctomilError`` values from the backend.
    """

    async def detect(
        self,
        *,
        audio: Any,
        sample_rate_hz: int = 16000,
        deadline_ms: Optional[int] = None,
    ) -> VadResult:
        """Run native VAD over mono PCM-f32 audio."""
        import asyncio

        def _run() -> VadResult:
            with open_vad_backend() as backend:
                with backend.open_session(sample_rate_hz=sample_rate_hz) as session:
                    session.feed_chunk(audio, sample_rate_hz=sample_rate_hz)
                    transitions = list(
                        session.poll_transitions(
                            deadline_ms=deadline_ms,
                            drain_until_completed=True,
                        )
                    )
            return VadResult(transitions=transitions, sample_rate_hz=sample_rate_hz)

        return await asyncio.to_thread(_run)


def _embedding_values_to_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


class FacadeSpeakerEmbedding:
    """Native-only ``client.audio.speaker_embedding`` namespace."""

    async def create(
        self,
        *,
        audio: Any,
        model: str = "sherpa-eres2netv2-base",
        sample_rate_hz: int = 16000,
        deadline_ms: Optional[int] = None,
    ) -> SpeakerEmbeddingResult:
        """Create a native speaker embedding for mono PCM-f32 audio."""
        import asyncio

        def _run() -> SpeakerEmbeddingResult:
            with open_speaker_embedding_backend(model_name=model) as backend:
                embedding = _embedding_values_to_list(
                    backend.embed(
                        audio,
                        sample_rate_hz=sample_rate_hz,
                        deadline_ms=deadline_ms,
                    )
                )
            return SpeakerEmbeddingResult(
                embedding=embedding,
                model=model,
                dimensions=len(embedding),
                sample_rate_hz=sample_rate_hz,
            )

        return await asyncio.to_thread(_run)


class FacadeDiarization:
    """Native-only ``client.audio.diarization`` namespace."""

    async def create(
        self,
        *,
        audio: Any,
        sample_rate_hz: int = 16000,
        deadline_ms: int = 300_000,
    ) -> DiarizationResult:
        """Run native speaker diarization over mono PCM-f32 audio."""
        import asyncio

        def _run() -> DiarizationResult:
            with open_diarization_backend() as backend:
                segments = backend.diarize(
                    audio,
                    sample_rate_hz=sample_rate_hz,
                    deadline_ms=deadline_ms,
                )
            return DiarizationResult(segments=segments, sample_rate_hz=sample_rate_hz)

        return await asyncio.to_thread(_run)


class FacadeTranscriptions:
    """``client.audio.transcriptions`` namespace on the unified Octomil facade.

    Mirrors :class:`FacadeSpeech`: delegates to
    :meth:`octomil.execution.kernel.ExecutionKernel.transcribe_audio`
    so a single code path handles app-ref resolution, policy
    enforcement, and locality dispatch (whisper.cpp on-device vs.
    hosted STT). Without this namespace, the unified facade had no
    public surface for transcription with ``app=`` / ``policy=`` —
    the kernel enforced the gates but no facade exposed them.
    """

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel

    async def create(
        self,
        *,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> "TranscriptionResult":
        """Transcribe audio through the unified routing kernel.

        Parameters
        ----------
        audio:
            Raw audio bytes (WAV, MP3, etc.).
        model:
            Optional model ref. Common forms: ``@app/<slug>/transcription``,
            a hosted provider model id (``whisper-1``), or a local model id
            (``whisper-tiny`` / ``whisper-base``).
        language:
            Optional BCP-47 language hint (``"en"``, ``"fr"`` …).
        response_format:
            Optional output format hint (provider-specific).
        policy:
            Optional routing policy preset override; same vocabulary as
            ``client.audio.speech.create(policy=...)``. ``"private"`` and
            ``"local_only"`` force ``cloud_available=False`` so a planner
            outage cannot leak the request to a hosted backend.
        app:
            Optional explicit app slug for ``@app/<slug>/transcription``
            resolution. When set together with a planner outage AND no
            explicit ``policy=``, the kernel raises rather than silently
            falling back to cloud (mirrors the TTS / chat / embeddings
            refusal gate).
        """
        result = await self._kernel.transcribe_audio(
            audio_data=audio,
            model=model,
            policy=policy,
            app=app,
            language=language,
        )
        return TranscriptionResult(
            text=getattr(result, "output_text", "") or "",
            language=language,
        )


class FacadeAudio:
    """Namespace for audio APIs on the top-level :class:`octomil.Octomil`.

    Wires :attr:`speech` and :attr:`transcriptions` against the
    execution kernel so app refs (``@app/<slug>/tts``,
    ``@app/<slug>/transcription``) resolve through the routing policy.

    Usage::

        client = Octomil.from_env()
        await client.initialize()
        response = await client.audio.speech.create(
            model="@app/<slug>/tts",
            input="Hello from Octomil.",
        )
        result = await client.audio.transcriptions.create(
            model="@app/<slug>/transcription",
            audio=audio_bytes,
            policy="local_only",
        )
    """

    def __init__(self, kernel: Any) -> None:
        self._speech = FacadeSpeech(kernel)
        self._transcriptions = FacadeTranscriptions(kernel)
        self._voices = FacadeVoices(kernel)
        self._vad = FacadeVad()
        self._speaker_embedding = FacadeSpeakerEmbedding()
        self._diarization = FacadeDiarization()

    @property
    def speech(self) -> FacadeSpeech:
        return self._speech

    @property
    def transcriptions(self) -> "FacadeTranscriptions":
        return self._transcriptions

    @property
    def voices(self) -> FacadeVoices:
        return self._voices

    @property
    def vad(self) -> FacadeVad:
        return self._vad

    @property
    def speaker_embedding(self) -> FacadeSpeakerEmbedding:
        return self._speaker_embedding

    @property
    def diarization(self) -> FacadeDiarization:
        return self._diarization


__all__ = [
    "OctomilAudio",
    "FacadeAudio",
    "FacadeDiarization",
    "FacadeSpeakerEmbedding",
    "FacadeSpeech",
    "FacadeTranscriptions",
    "FacadeVad",
    "FacadeVoices",
    "SpeechResponse",
    "SpeechRoute",
    "VoiceCatalog",
    "VoiceInfo",
    "AudioTranscriptions",
    "DiarizationResult",
    "DiarizationSegment",
    "NativeDiarizationBackend",
    "NativeSpeakerEmbeddingBackend",
    "NativeVadBackend",
    "open_diarization_backend",
    "open_speaker_embedding_backend",
    "open_vad_backend",
    "SpeakerEmbeddingResult",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VadResult",
    "VadTransition",
]
