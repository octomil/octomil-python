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
    ChunkDiagnostics,
    ChunkWindow,
    DiarizationResult,
    SpeakerEmbeddingResult,
    TranscriptionPartial,
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


_CLOUD_POLICIES = {"cloud", "cloud_only", "cloud_first", "performance_first"}


def _reject_cloud_policy_without_credentials(policy: Optional[str]) -> None:
    if policy and policy.lower() in _CLOUD_POLICIES:
        from octomil.errors import OctomilError, OctomilErrorCode

        raise OctomilError(
            code=OctomilErrorCode.INVALID_API_KEY,
            message=(
                f"Cloud routing policy {policy!r} requires Octomil credentials. "
                "Construct Octomil with api_key= + org_id= or a publishable key, "
                "or use a local policy such as 'local_first', 'local_only', or 'private'."
            ),
        )


def _segments_from_execution(raw_segments: Any) -> list[TranscriptionSegment]:
    """Build public segments from the kernel ``ExecutionResult.segments``.

    The native serve adapter emits dicts with additive ``start_ms`` /
    ``end_ms`` / ``avg_logprob`` / ``no_speech_prob`` keys alongside the
    legacy ``start`` / ``end`` (seconds) keys. We prefer the ``*_ms``
    keys when present and fall back to the seconds keys (legacy / cloud
    shapes) so older backends still project cleanly with logprob 0.0.
    """
    segments: list[TranscriptionSegment] = []
    if not raw_segments:
        return segments
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        if "start_ms" in seg or "end_ms" in seg:
            start_ms = int(seg.get("start_ms", 0))
            end_ms = int(seg.get("end_ms", 0))
        else:
            start_ms = int(round(float(seg.get("start", 0.0)) * 1000.0))
            end_ms = int(round(float(seg.get("end", 0.0)) * 1000.0))
        segments.append(
            TranscriptionSegment(
                text=str(seg.get("text", "")),
                start_ms=start_ms,
                end_ms=end_ms,
                avg_logprob=float(seg.get("avg_logprob", 0.0)),
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
            )
        )
    return segments


def _chunk_diagnostics_from_raw(raw: Any) -> Optional[ChunkDiagnostics]:
    """Build the public ``ChunkDiagnostics`` from the backend dict.

    Returns ``None`` when chunking was off (the backend omits or nulls
    ``chunk_diagnostics``), keeping the default path identical to today.
    """
    if not isinstance(raw, dict):
        return None
    diag = raw.get("chunk_diagnostics")
    if not isinstance(diag, dict):
        return None
    windows = [
        ChunkWindow(
            index=int(w.get("index", 0)),
            start_ms=int(w.get("start_ms", 0)),
            end_ms=int(w.get("end_ms", 0)),
            off_ms=int(w.get("off_ms", 0)),
            is_last=bool(w.get("is_last", False)),
        )
        for w in diag.get("windows", []) or []
        if isinstance(w, dict)
    ]
    return ChunkDiagnostics(
        window_ms=int(diag.get("window_ms", 0)),
        overlap_ms=int(diag.get("overlap_ms", 0)),
        step_ms=int(diag.get("step_ms", 0)),
        window_count=int(diag.get("window_count", 0)),
        windows=windows,
        segment_owner_window=[int(i) for i in diag.get("segment_owner_window", []) or []],
        audio_duration_ms=int(diag.get("audio_duration_ms", 0)),
        final_segment_end_ms=int(diag.get("final_segment_end_ms", 0)),
        tail_gap_ms=int(diag.get("tail_gap_ms", 0)),
    )


def _execution_result_to_transcription(result: Any, language: Optional[str]) -> TranscriptionResult:
    """Project a kernel ``ExecutionResult`` onto the public result.

    Segments / diagnostics are only present on the native path; cloud /
    legacy results leave ``segments`` empty and ``chunk_diagnostics``
    None, so the default surface is unchanged.
    """
    raw = getattr(result, "raw", None)
    duration_ms = 0
    if isinstance(raw, dict):
        duration_ms = int(raw.get("duration_ms", 0) or 0)
    return TranscriptionResult(
        text=getattr(result, "output_text", "") or "",
        language=language,
        segments=_segments_from_execution(getattr(result, "segments", None)),
        duration_ms=duration_ms,
        chunk_diagnostics=_chunk_diagnostics_from_raw(raw),
    )


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

    def __init__(self, kernel: Any, *, cloud_allowed: bool = True) -> None:
        self._kernel = kernel
        self._cloud_allowed = cloud_allowed

    async def create(
        self,
        *,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        chunk_window_ms: Optional[int] = None,
        chunk_overlap_ms: Optional[int] = None,
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
        chunk_window_ms:
            Optional fixed decode-window size for chunked transcription
            (native path only). ``None`` (default) runs a single
            full-buffer decode — unchanged from prior behaviour.
        chunk_overlap_ms:
            Optional overlap between consecutive decode windows. Ignored
            unless ``chunk_window_ms`` is set.
        """
        if not self._cloud_allowed:
            _reject_cloud_policy_without_credentials(policy)
        result = await self._kernel.transcribe_audio(
            audio_data=audio,
            model=model,
            policy=policy,
            app=app,
            language=language,
            chunk_window_ms=chunk_window_ms,
            chunk_overlap_ms=chunk_overlap_ms,
        )
        return _execution_result_to_transcription(result, language)


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

    def __init__(self, kernel: Any, *, cloud_allowed: bool = True, telemetry_reporter: Any | None = None) -> None:
        self._speech = FacadeSpeech(kernel, cloud_allowed=cloud_allowed, telemetry_reporter=telemetry_reporter)
        self._transcriptions = FacadeTranscriptions(kernel, cloud_allowed=cloud_allowed)
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
    "ChunkDiagnostics",
    "ChunkWindow",
    "TranscriptionPartial",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VadResult",
    "VadTransition",
]
