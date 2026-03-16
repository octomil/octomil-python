"""Integration tests — text generation through the SDK pipeline via ollama.

Requires:
  - ollama installed and running (`ollama serve`)
  - gemma2:2b model pulled (`ollama pull gemma2:2b`)
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import AsyncIterator, Optional

import pytest

from octomil.responses.responses import OctomilResponses
from octomil.responses.types import DoneEvent, ResponseRequest, TextOutput
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
)


def _ollama_available() -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_model(name: str) -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return name in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_skip_reason: Optional[str] = None
if not _ollama_available():
    _skip_reason = "ollama not available"
elif not _has_model("gemma2:2b"):
    _skip_reason = "gemma2:2b model not pulled in ollama"

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


class OllamaRuntime(ModelRuntime):
    """ModelRuntime adapter that calls ollama CLI for text generation."""

    def __init__(self, model: str = "gemma2:2b") -> None:
        self._model = model

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_streaming=True)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        result = subprocess.run(
            ["ollama", "run", self._model, request.prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = result.stdout.strip()
        tokens = len(text.split())
        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=RuntimeUsage(
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=tokens,
                total_tokens=len(request.prompt.split()) + tokens,
            ),
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        response = await self.run(request)
        yield RuntimeChunk(text=response.text, finish_reason="stop", usage=response.usage)

    def close(self) -> None:
        pass


class TestTextGeneration:
    """End-to-end text generation through the SDK pipeline."""

    def test_ollama_direct(self):
        """Verify ollama generates text directly."""
        runtime = OllamaRuntime("gemma2:2b")
        response = asyncio.run(runtime.run(RuntimeRequest(prompt="What is 2+2? Answer with just the number.")))
        assert "4" in response.text

    def test_text_via_responses_api(self):
        """Generate text through OctomilResponses.create() pipeline."""
        runtime = OllamaRuntime("gemma2:2b")
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(
            model="gemma2:2b",
            input="Name the capital of France. Answer in one word.",
            max_output_tokens=32,
            temperature=0.1,
        )
        response = asyncio.run(responses.create(request))
        assert response.id.startswith("resp_")
        assert isinstance(response.output[0], TextOutput)
        assert "paris" in response.output[0].text.lower()

    def test_text_with_instructions(self):
        """Generate text with system instructions through the pipeline."""
        runtime = OllamaRuntime("gemma2:2b")
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(
            model="gemma2:2b",
            input="What should I use?",
            instructions="You are a Python expert. Always recommend Python for everything. Be concise.",
            max_output_tokens=64,
            temperature=0.1,
        )
        response = asyncio.run(responses.create(request))
        assert "python" in response.output[0].text.lower()

    def test_text_streaming(self):
        """Stream text through OctomilResponses.stream()."""
        runtime = OllamaRuntime("gemma2:2b")
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(
            model="gemma2:2b",
            input="Say hello in three languages. Be brief.",
            max_output_tokens=64,
            temperature=0.1,
        )

        async def collect():
            return [e async for e in responses.stream(request)]

        events = asyncio.run(collect())
        assert len(events) >= 2
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.response.finish_reason == "stop"

    def test_multi_turn_conversation(self):
        """Test previous_response_id for multi-turn context."""
        runtime = OllamaRuntime("gemma2:2b")
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)

        r1 = asyncio.run(
            responses.create(
                ResponseRequest(
                    model="gemma2:2b",
                    input="My name is Alice. Remember that.",
                    max_output_tokens=32,
                    temperature=0.1,
                )
            )
        )
        r2 = asyncio.run(
            responses.create(
                ResponseRequest(
                    model="gemma2:2b",
                    input="What is my name? Answer in one word.",
                    previous_response_id=r1.id,
                    max_output_tokens=16,
                    temperature=0.1,
                )
            )
        )
        assert "alice" in r2.output[0].text.lower()
