"""Hosted text-to-speech client surface (mirrors openai.audio.speech)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class SpeechResponse:
    """Result of a hosted speech.create() call."""

    audio_bytes: bytes
    content_type: str
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    billed_units: Optional[int] = None
    unit_kind: Optional[str] = None

    def write_to(self, path: str) -> int:
        """Write audio_bytes to ``path``. Returns bytes written."""
        with open(path, "wb") as f:
            f.write(self.audio_bytes)
        return len(self.audio_bytes)


class HostedSpeech:
    """Speech synthesis surface.

    Mirrors OpenAI's ``client.audio.speech.create(...)`` shape but returns
    a ``SpeechResponse`` (audio bytes + Octomil routing metadata) rather
    than a streaming response object — callers that want streaming should
    consume the underlying HTTP API directly until that's added here.
    """

    _SPEECH_PATH = "/audio/speech"

    def __init__(self, *, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def create(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> SpeechResponse:
        """Synthesize speech from text.

        Returns the raw audio bytes plus Octomil routing metadata
        surfaced via ``X-Octomil-*`` response headers.
        """
        if not input or not input.strip():
            raise ValueError("`input` must be a non-empty string.")

        body = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._base_url + self._SPEECH_PATH,
                json=body,
                headers=headers,
            )
        resp.raise_for_status()

        latency_raw = resp.headers.get("x-octomil-latency-ms")
        billed_raw = resp.headers.get("x-octomil-billed-units")
        try:
            latency_ms = float(latency_raw) if latency_raw is not None else None
        except ValueError:
            latency_ms = None
        try:
            billed_units = int(billed_raw) if billed_raw is not None else None
        except ValueError:
            billed_units = None

        return SpeechResponse(
            audio_bytes=resp.content,
            content_type=resp.headers.get("content-type", "application/octet-stream"),
            provider=resp.headers.get("x-octomil-provider"),
            model=resp.headers.get("x-octomil-model") or model,
            latency_ms=latency_ms,
            billed_units=billed_units,
            unit_kind=resp.headers.get("x-octomil-unit-kind"),
        )


__all__ = ["HostedSpeech", "SpeechResponse"]
