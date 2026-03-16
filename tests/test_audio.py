"""Tests for octomil.audio — OctomilAudio namespace."""

from __future__ import annotations

from typing import AsyncIterator, Optional

import pytest

from octomil.audio import OctomilAudio
from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import TranscriptionResult
from octomil.model_ref import ModelRef
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)


class _MockTranscriptionRuntime(ModelRuntime):
    """Mock runtime that returns fixed transcription text."""

    def __init__(self, text: str = "Hello world") -> None:
        self._text = text

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text=self._text)

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        for word in self._text.split():
            yield RuntimeChunk(text=word)


class TestAudioTranscriptions:
    @pytest.mark.asyncio
    async def test_create_transcription(self) -> None:
        mock_runtime = _MockTranscriptionRuntime("Hello world transcription")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        transcriptions = AudioTranscriptions(runtime_resolver=resolver)
        result = await transcriptions.create(audio=b"fake audio data")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world transcription"

    @pytest.mark.asyncio
    async def test_create_with_language(self) -> None:
        mock_runtime = _MockTranscriptionRuntime("Hola mundo")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        transcriptions = AudioTranscriptions(runtime_resolver=resolver)
        result = await transcriptions.create(audio=b"fake", language="es")

        assert result.text == "Hola mundo"
        assert result.language == "es"

    @pytest.mark.asyncio
    async def test_create_no_runtime_raises(self) -> None:
        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return None

        transcriptions = AudioTranscriptions(runtime_resolver=resolver)
        with pytest.raises(RuntimeError, match="No runtime"):
            await transcriptions.create(audio=b"fake")

    @pytest.mark.asyncio
    async def test_stream_transcription(self) -> None:
        mock_runtime = _MockTranscriptionRuntime("Hello world")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        transcriptions = AudioTranscriptions(runtime_resolver=resolver)
        segments = await transcriptions.stream(audio=b"fake")

        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"
        assert segments[0].start_ms == 0
        assert segments[1].start_ms == 500


class TestOctomilAudio:
    def test_transcriptions_property(self) -> None:
        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return None

        audio = OctomilAudio(runtime_resolver=resolver)
        assert isinstance(audio.transcriptions, AudioTranscriptions)

    @pytest.mark.asyncio
    async def test_end_to_end(self) -> None:
        mock_runtime = _MockTranscriptionRuntime("test transcription")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        audio = OctomilAudio(runtime_resolver=resolver)
        result = await audio.transcriptions.create(audio=b"data")
        assert result.text == "test transcription"
