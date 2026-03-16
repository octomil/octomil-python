"""Integration tests — whisper.cpp audio transcription through the SDK pipeline.

Requires:
  - whisper-cli binary built from research/engines/whisper.cpp
  - ggml-base.bin model in models/whisper-base/
  - macOS `say` command for generating test audio
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile

import pytest

from octomil.audio import OctomilAudio, TranscriptionResult
from octomil.model_ref import ModelRef
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)

# Paths
WHISPER_CLI = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "research", "engines", "whisper.cpp", "build", "bin", "whisper-cli"
    )
)
WHISPER_MODEL = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "whisper-base", "ggml-base.bin")
)

_skip_reason = None
if not os.path.isfile(WHISPER_CLI):
    _skip_reason = f"whisper-cli not found at {WHISPER_CLI}"
elif not os.path.isfile(WHISPER_MODEL):
    _skip_reason = f"whisper model not found at {WHISPER_MODEL}"

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


def _generate_speech_wav(text: str) -> bytes:
    """Generate WAV audio from text using macOS `say`."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        subprocess.run(
            ["say", "-o", tmp_path, "--data-format=LEI16@16000", text],
            check=True,
            timeout=15,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


class WhisperAudioRuntime(ModelRuntime):
    """ModelRuntime that accepts audio bytes and transcribes via whisper-cli."""

    def __init__(self, cli_path: str, model_path: str, audio_data: bytes) -> None:
        self._cli = cli_path
        self._model = model_path
        self._audio_data = audio_data

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_streaming=False)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(self._audio_data)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [self._cli, "-m", self._model, "-f", tmp_path, "--no-timestamps", "-nt"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return RuntimeResponse(text=result.stdout.strip())
        finally:
            os.unlink(tmp_path)

    async def stream(self, request: RuntimeRequest):
        response = await self.run(request)
        yield RuntimeChunk(text=response.text, finish_reason="stop")

    def close(self) -> None:
        pass


class TestWhisperTranscription:
    """End-to-end whisper transcription through the SDK pipeline."""

    def test_whisper_cli_direct(self):
        """Verify whisper-cli transcribes speech correctly."""
        audio = _generate_speech_wav("Hello world")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", tmp_path, "--no-timestamps", "-nt"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert "hello" in result.stdout.strip().lower()
        finally:
            os.unlink(tmp_path)

    def test_whisper_via_model_runtime(self):
        """Transcribe through the ModelRuntime interface."""
        audio = _generate_speech_wav("Testing the Python SDK")
        runtime = WhisperAudioRuntime(WHISPER_CLI, WHISPER_MODEL, audio)
        response = asyncio.run(runtime.run(RuntimeRequest(prompt="")))
        text = response.text.lower()
        assert "python" in text or "testing" in text

    def test_whisper_via_audio_namespace(self):
        """Transcribe through client.audio.transcriptions.create() pipeline."""
        audio = _generate_speech_wav("This is a transcription test")

        def resolver(ref: ModelRef):
            return WhisperAudioRuntime(WHISPER_CLI, WHISPER_MODEL, audio)

        audio_api = OctomilAudio(runtime_resolver=resolver)
        result = asyncio.run(audio_api.transcriptions.create(audio=audio))
        assert isinstance(result, TranscriptionResult)
        assert len(result.text) > 0
        assert "transcription" in result.text.lower() or "test" in result.text.lower()

    def test_whisper_longer_passage(self):
        """Transcribe a longer speech passage."""
        text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
        audio = _generate_speech_wav(text)

        def resolver(ref: ModelRef):
            return WhisperAudioRuntime(WHISPER_CLI, WHISPER_MODEL, audio)

        audio_api = OctomilAudio(runtime_resolver=resolver)
        result = asyncio.run(audio_api.transcriptions.create(audio=audio))
        output = result.text.lower()
        assert any(w in output for w in ["fox", "quick", "brown", "lazy", "dog", "alphabet"])
