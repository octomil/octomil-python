"""Integration tests — vision/image recognition through the SDK pipeline via ollama + llava.

Requires:
  - ollama installed and running (`ollama serve`)
  - llava:7b model pulled (`ollama pull llava:7b`)
"""

from __future__ import annotations

import asyncio
import os
import struct
import subprocess
import tempfile
import zlib
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
elif not _has_model("llava:7b"):
    _skip_reason = "llava:7b model not pulled in ollama"

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


def _create_gradient_png() -> str:
    """Create a 64x64 RGB gradient PNG, return temp file path."""
    width, height = 64, 64
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)  # filter: None
        for x in range(width):
            pixels.append(int(255 * x / width))  # R
            pixels.append(int(255 * y / height))  # G
            pixels.append(int(255 * (1.0 - x / width)))  # B

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(bytes(pixels)))
    png += chunk(b"IEND", b"")

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(png)
    return path


class LlavaVisionRuntime(ModelRuntime):
    """ModelRuntime adapter that calls ollama with llava for image understanding."""

    def __init__(self, model: str = "llava:7b") -> None:
        self._model = model

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_streaming=False, supports_multimodal_input=True)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        from octomil.runtime.core.chatml_renderer import render_chatml

        prompt = render_chatml(request)
        image_path: Optional[str] = None
        if "[image:" in prompt:
            start = prompt.index("[image:") + 7
            end = prompt.index("]", start)
            image_path = prompt[start:end]
            prompt = prompt[end + 1 :].strip()

        cmd = ["ollama", "run", self._model]
        cmd.append(f"{prompt} {image_path}" if image_path else prompt)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        text = result.stdout.strip()
        tokens = len(text.split())
        prompt_tokens = len(prompt.split()) + (128 if image_path else 0)
        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=RuntimeUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens,
                total_tokens=prompt_tokens + tokens,
            ),
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        response = await self.run(request)
        yield RuntimeChunk(text=response.text, finish_reason="stop", usage=response.usage)

    def close(self) -> None:
        pass


class TestVisionRecognition:
    """End-to-end image recognition through the SDK pipeline."""

    def test_llava_describes_gradient(self):
        """Verify llava recognizes and describes a color gradient image."""
        image_path = _create_gradient_png()
        try:
            result = subprocess.run(
                ["ollama", "run", "llava:7b", f"Describe this image briefly. {image_path}"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            text = result.stdout.strip().lower()
            assert any(w in text for w in ["color", "gradient", "image", "red", "blue", "green", "purple", "spectrum"])
        finally:
            os.unlink(image_path)

    def test_vision_via_responses_api(self):
        """Run image recognition through OctomilResponses.create()."""
        image_path = _create_gradient_png()
        try:
            runtime = LlavaVisionRuntime("llava:7b")
            responses = OctomilResponses(runtime_resolver=lambda _: runtime)
            request = ResponseRequest(
                model="llava:7b",
                input=f"[image:{image_path}] Describe the colors in this image. Be concise.",
                max_output_tokens=64,
                temperature=0.1,
            )
            response = asyncio.run(responses.create(request))
            assert response.id.startswith("resp_")
            assert isinstance(response.output[0], TextOutput)
            text = response.output[0].text.lower()
            assert any(w in text for w in ["color", "gradient", "red", "blue", "green", "purple", "image", "spectrum"])
        finally:
            os.unlink(image_path)

    def test_vision_streaming(self):
        """Stream vision output through OctomilResponses.stream()."""
        image_path = _create_gradient_png()
        try:
            runtime = LlavaVisionRuntime("llava:7b")
            responses = OctomilResponses(runtime_resolver=lambda _: runtime)
            request = ResponseRequest(
                model="llava:7b",
                input=f"[image:{image_path}] What type of image is this? One sentence.",
                max_output_tokens=32,
                temperature=0.1,
            )

            async def collect():
                return [e async for e in responses.stream(request)]

            events = asyncio.run(collect())
            assert len(events) >= 2
            assert isinstance(events[-1], DoneEvent)
            assert len(events[-1].response.output) > 0
        finally:
            os.unlink(image_path)
