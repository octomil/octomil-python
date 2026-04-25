"""Hosted Octomil client.

Lightweight HTTP client targeting api.octomil.com. Distinct from
``octomil.OctomilClient`` (the local-runtime facade) so hosted callers do
not pay the cost of importing the runtime planner / engine registry.
"""

from __future__ import annotations

import os
from typing import Optional

from .audio import HostedAudio

_DEFAULT_BASE_URL = "https://api.octomil.com/v1"


class HostedClient:
    """Hosted Octomil API client.

    Examples::

        from octomil.hosted import HostedClient

        client = HostedClient(api_key=os.environ["OCTOMIL_API_KEY"])
        result = client.audio.speech.create(
            model="tts-1",
            input="Hello world.",
            voice="alloy",
        )
        result.write_to("hello.mp3")
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("OCTOMIL_API_KEY")
        if not resolved_key:
            raise ValueError("HostedClient requires `api_key=` or OCTOMIL_API_KEY env var.")
        self._api_key = resolved_key
        self._base_url = (
            base_url or os.environ.get("OCTOMIL_API_BASE") or os.environ.get("OCTOMIL_API_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout
        self._audio: HostedAudio | None = None

    @property
    def audio(self) -> HostedAudio:
        if self._audio is None:
            self._audio = HostedAudio(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._audio


__all__ = ["HostedClient"]
