"""AudioTranscriptions — speech-to-text API."""

from __future__ import annotations

from typing import Callable, Optional

from octomil._generated.message_role import MessageRole
from octomil._generated.model_capability import ModelCapability
from octomil.audio.types import TranscriptionResult, TranscriptionSegment
from octomil.model_ref import ModelRef, ModelRefFactory
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import GenerationConfig, RuntimeContentPart, RuntimeMessage, RuntimeRequest


class AudioTranscriptions:
    """Audio transcription API.

    Wraps the underlying audio runtime to provide speech-to-text.

    Usage::

        result = await client.audio.transcriptions.create(
            audio=audio_bytes
        )
        print(result.text)
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
    ) -> None:
        self._runtime_resolver = runtime_resolver

    async def create(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Raw audio data (WAV, MP3, etc.).
            model: Model reference. Defaults to transcription capability.
            language: Optional language hint (BCP 47 code, e.g. "en").
            response_format: Optional output format hint.

        Returns:
            TranscriptionResult with the transcribed text.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        parts = [RuntimeContentPart.audio_part(audio, "audio/wav")]
        if language:
            parts.append(RuntimeContentPart.text_part(language))
        request = RuntimeRequest(
            messages=[RuntimeMessage(role=MessageRole.USER, parts=parts)],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
        )
        response = await runtime.run(request)
        return TranscriptionResult(text=response.text, language=language)

    async def stream(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
    ) -> list[TranscriptionSegment]:
        """Stream transcription segments.

        Args:
            audio: Raw audio data.
            model: Model reference. Defaults to transcription capability.

        Returns:
            List of transcription segments.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.audio_part(audio, "audio/wav")],
                )
            ],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
        )
        segments: list[TranscriptionSegment] = []
        async for chunk in runtime.stream(request):
            if chunk.text:
                segments.append(TranscriptionSegment(text=chunk.text))
        return segments
