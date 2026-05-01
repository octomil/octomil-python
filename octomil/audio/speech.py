"""Unified ``client.audio.speech`` namespace on the top-level Octomil facade.

Single API surface for TTS that respects the app routing policy:

  >>> response = await client.audio.speech.create(
  ...     model="@app/<slug>/tts",
  ...     input="Hello from Octomil.",
  ...     voice="af_bella",      # local Kokoro voice
  ...     response_format="wav",
  ...     speed=1.0,
  ... )
  >>> response.write_to("out.wav")

Routing decisions live in :class:`octomil.execution.kernel.ExecutionKernel`.
This module is just the user-facing facade; the kernel decides whether the
call runs on-device through the sherpa-onnx engine or in cloud through the
hosted speech client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class VoiceInfo:
    """A single entry in a TTS model's voice catalog.

    Two flavours of entry the catalog can carry:

      * **Native voices** — entries pulled from the engine's own
        catalog (Kokoro voices.txt, Piper voice manifest). ``sid``
        is the sherpa-onnx speaker id; ``source="voices_txt"`` or
        ``"static_recipe"``; ``speaker`` is ``None``.
      * **Logical speakers** — entries materialized from the planner's
        ``tts_speakers`` map for an app ref. ``speaker`` is the
        logical id (``"madam_ambrose"``); ``native_voice`` is the
        underlying engine voice when the planner mapped one;
        ``requires_reference=True`` for few-shot voice-cloning
        engines (PocketTTS) where the engine needs reference audio
        rather than a sid. ``source="planner_profile"``.

    ``sid`` is the sherpa-onnx speaker id for local native artifacts;
    ``None`` for hosted/cloud catalogs and for logical speakers whose
    runtime is a voice-cloning engine. ``default`` flags the voice
    (or speaker) the backend uses when ``synthesize_speech(voice=None,
    speaker=None)`` falls back to the model default.

    The richer fields below are additive — older callers that read
    only ``id`` / ``sid`` / ``default`` keep working.
    """

    id: str
    sid: Optional[int] = None
    default: bool = False
    source: Optional[str] = None  # "voices_txt" | "static_recipe" | "planner_profile" | "hosted"
    speaker: Optional[str] = None  # logical speaker id; None for native voices
    native_voice: Optional[str] = None  # engine voice this speaker maps to (when applicable)
    requires_reference: bool = False  # True when synthesis needs reference audio (Pocket etc.)
    language: Optional[str] = None


@dataclass(frozen=True)
class VoiceCatalog:
    """Result of :meth:`FacadeVoices.list`.

    Captures *which artifact* would actually serve a synthesis
    request right now — same code path as the synthesis route, so
    UI listings and synthesis validation can never disagree about
    what voices a caller is allowed to use.

    Fields:

      - ``model`` — the canonical runtime model id the kernel
        resolved (post-app-ref, post-planner).
      - ``locality`` — ``"on_device"`` or ``"cloud"``.
      - ``source`` — provenance of the catalog: ``"voices_txt"``
        when read from the prepared artifact's sidecar,
        ``"static_recipe"`` when the static recipe's manifest was
        used (no prepared dir on disk yet), ``"hosted"`` for cloud
        provider catalogs.
      - ``artifact_id`` / ``artifact_version`` / ``digest`` —
        artifact identity. ``artifact_version`` mirrors the
        ``VERSION`` sidecar materialized from the recipe; ``digest``
        is the content SHA-256 of the source artifact when known.
      - ``default_voice`` — convenience pointer to the entry whose
        ``default=True``. ``None`` for catalogs without a flagged
        default (single-speaker bundles, hosted providers without
        a documented default).
      - ``sample_rate`` — output sample rate in Hz when known.
      - ``voices`` — ordered list of :class:`VoiceInfo`. Position
        in the list matches the speaker id for local artifacts.
    """

    model: str
    locality: str  # "on_device" | "cloud"
    source: str  # "voices_txt" | "static_recipe" | "hosted"
    voices: tuple[VoiceInfo, ...]
    artifact_id: Optional[str] = None
    artifact_version: Optional[str] = None
    digest: Optional[str] = None
    default_voice: Optional[str] = None
    sample_rate: Optional[int] = None

    @property
    def voice_ids(self) -> tuple[str, ...]:
        return tuple(v.id for v in self.voices)

    def get(self, voice_id: str) -> Optional[VoiceInfo]:
        target = voice_id.strip().lower()
        for v in self.voices:
            if v.id.lower() == target:
                return v
        return None


@dataclass
class SpeechRoute:
    """Routing metadata attached to a :class:`SpeechResponse`.

    Mirrors the ``route`` block in
    ``octomil-contracts/schemas/core/audio_speech_result.json``. No user
    content (input text, raw audio bytes, file paths, provider request ids).
    """

    locality: str  # "on_device" | "cloud"
    engine: Optional[str] = None
    policy: Optional[str] = None
    fallback_used: bool = False


@dataclass
class SpeechResponse:
    """Result of :meth:`HostedSpeech.create` and the routed facade."""

    audio_bytes: bytes
    content_type: str
    format: str
    model: str
    provider: Optional[str] = None
    voice: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_ms: Optional[int] = None
    latency_ms: float = 0.0
    route: SpeechRoute = field(default_factory=lambda: SpeechRoute(locality="on_device"))
    billed_units: Optional[int] = None
    unit_kind: Optional[str] = None

    def write_to(self, path: str) -> int:
        """Write ``audio_bytes`` to ``path``. Returns bytes written."""
        with open(path, "wb") as f:
            f.write(self.audio_bytes)
        return len(self.audio_bytes)


class FacadeSpeech:
    """``client.audio.speech`` namespace.

    Construct via :class:`FacadeAudio`; do not instantiate directly. The
    facade delegates to :meth:`octomil.execution.kernel.ExecutionKernel.synthesize_speech`
    so a single code path handles app-ref resolution, policy enforcement,
    and locality dispatch.
    """

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel

    def stream(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        speaker: Optional[str] = None,
        response_format: str = "pcm_s16le",
        speed: float = 1.0,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        priority: Any = None,
    ) -> Any:
        """Stream synthesized speech as typed events.

        Returns a :class:`octomil.audio.streaming.SpeechStream` — an async
        iterator yielding :class:`SpeechStreamStarted`, zero or more
        :class:`SpeechAudioChunk`, and a final :class:`SpeechStreamCompleted`.
        See :mod:`octomil.audio.streaming` for the full event vocabulary.

        Canonical wire format is ``pcm_s16le`` (interleaved PCM int16
        little-endian). ``wav`` is supported but produces a single final
        chunk — there is no partial-WAV streaming. ``opus`` is reserved
        and currently rejected.

        Honest cadence advertisement is on
        :attr:`SpeechStreamStarted.streaming_capability`:

          - ``final_chunk`` — one chunk at the end. No TTFB benefit.
          - ``sentence_chunk`` — one chunk per sentence. Multi-sentence
            input gets TTFB benefit; single-sentence degrades to
            ``final_chunk`` and the completion event reports
            ``capability_verified=False``.
          - ``progressive`` — sub-sentence streaming. No production
            engine in-tree advertises this today.

        Note that this method is *not* async — it returns the stream
        object directly so it can be used with ``async with`` and
        ``async for``::

            async with client.audio.speech.stream(model=..., input=...) as s:
                async for event in s:
                    ...
        """
        import time

        # Lazily build the stream so callers can use ``async with`` /
        # ``async for`` without an extra ``await``.
        #
        # ``sdk_t0`` is captured INSIDE the producer, on first iteration —
        # NOT at construction time. The stream object can be created and
        # held by a caller for arbitrary idle time before iteration
        # starts (e.g. they ``await`` something else first); only the
        # window from "iteration begins" to "completion" is honest SDK
        # work. Capturing at construction would charge that idle time to
        # ``setup_ms`` / ``e2e_first_chunk_ms`` / ``total_latency_ms``.
        async def _producer():
            sdk_t0 = time.monotonic()
            inner = await self._kernel.synthesize_speech_stream(
                model=model,
                input=input,
                voice=voice,
                speaker=speaker,
                response_format=response_format,
                speed=speed,
                policy=policy,
                app=app,
                sdk_t0=sdk_t0,
                priority=priority,
            )
            async for event in inner:
                yield event

        from octomil.audio.streaming import SpeechStream

        return SpeechStream(_producer())

    async def create(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        speaker: Optional[str] = None,
        response_format: str = "wav",
        speed: float = 1.0,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        priority: Any = None,
    ) -> SpeechResponse:
        """Synthesize speech from text.

        :param model: Octomil model ref. Common forms: ``@app/<slug>/tts``,
            a hosted provider model id (``tts-1``), or a local model id
            (``kokoro-82m``).
        :param input: Text to synthesize. Non-empty, max 4096 characters.
        :param voice: Voice id (provider-specific). Pass-through; mismatches
            with the routed locality raise ``voice_not_supported_for_locality``.
        :param response_format: Output audio format. ``"wav"`` is supported on
            every locality; other formats may require cloud routing.
        :param speed: Playback speed multiplier in ``[0.25, 4.0]``.
        :param policy: Routing policy preset override. Accepted values:
            ``"private"``, ``"local_only"``, ``"local_first"``,
            ``"cloud_first"``, ``"cloud_only"``, ``"performance_first"``.
            ``"private"`` and ``"local_only"`` force ``cloud_available=False``
            so a planner outage cannot leak the request to a hosted
            backend. PR B added this kwarg so embedded callers
            (Ren'Py games, kiosk apps) can express their privacy
            requirement directly instead of relying on the planner
            to honour an app-side policy that the SDK might not yet
            know about.
        :param app: Optional explicit app slug for ``@app/<slug>/<cap>``
            resolution. When set, the kernel uses this slug instead of
            parsing one from ``model``.
        """
        return await self._kernel.synthesize_speech(
            model=model,
            input=input,
            voice=voice,
            speaker=speaker,
            response_format=response_format,
            speed=speed,
            policy=policy,
            app=app,
            priority=priority,
        )


class FacadeVoices:
    """``client.audio.voices`` namespace.

    Construct via :class:`FacadeAudio`; do not instantiate directly.
    Delegates to
    :meth:`octomil.execution.kernel.ExecutionKernel.list_speech_voices`
    so the *same* artifact-aware resolver powers both UI listing
    and the synthesis-time voice validation. That is the closure-
    of-loop guarantee: a voice that ``voices.list`` advertises
    will never be rejected by ``speech.create`` / ``speech.stream``,
    and vice versa.
    """

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel

    async def list(
        self,
        *,
        model: str,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> VoiceCatalog:
        """Return the ordered voice catalog for ``model`` under
        the active routing policy.

        :param model: Octomil model ref. Same vocabulary as
            ``speech.create(model=...)``: ``@app/<slug>/tts``,
            a hosted provider model id (``tts-1``), or a local
            model id (``kokoro-82m``).
        :param policy: Optional routing policy preset override;
            same values as ``speech.create(policy=...)``.
        :param app: Optional explicit app slug.

        Returns a :class:`VoiceCatalog` whose ``locality`` mirrors
        what ``speech.create`` would route to right now.
        """
        return await self._kernel.list_speech_voices(
            model=model,
            policy=policy,
            app=app,
        )


__all__ = [
    "FacadeSpeech",
    "FacadeVoices",
    "SpeechResponse",
    "SpeechRoute",
    "VoiceCatalog",
    "VoiceInfo",
]
