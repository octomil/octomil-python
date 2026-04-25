"""Hosted audio namespace -- groups speech (and future transcription) APIs."""

from __future__ import annotations

from .speech import HostedSpeech


class HostedAudio:
    """Audio surface on a HostedClient.

    Mirrors openai.audio shape: ``client.audio.speech.create(...)``.
    Transcription (speech-to-text) lands here when the hosted whisper
    surface ships.
    """

    def __init__(self, *, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._speech: HostedSpeech | None = None

    @property
    def speech(self) -> HostedSpeech:
        if self._speech is None:
            self._speech = HostedSpeech(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._speech


__all__ = ["HostedAudio"]
