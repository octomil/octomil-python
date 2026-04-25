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

# Legacy control-plane bases that callers may have in their env. The hosted
# OpenAI-compatible routes are mounted at /v1, so rewrite the legacy /api/v1
# tail (and bare scheme://host) to /v1 instead of producing 404s.
_LEGACY_BASE_REWRITES: dict[str, str] = {
    "https://api.octomil.com": "https://api.octomil.com/v1",
    "https://api.octomil.com/api/v1": "https://api.octomil.com/v1",
    "http://api.octomil.com": "http://api.octomil.com/v1",
    "http://api.octomil.com/api/v1": "http://api.octomil.com/v1",
}


def _normalize_base_url(raw: str) -> str:
    """Map legacy or bare bases to the OpenAI-compatible /v1 root."""
    trimmed = raw.rstrip("/")
    return _LEGACY_BASE_REWRITES.get(trimmed, trimmed)


class HostedClient:
    """Hosted Octomil API client.

    Examples::

        from octomil.hosted import HostedClient

        client = HostedClient()  # reads OCTOMIL_SERVER_KEY (or OCTOMIL_API_KEY)
        result = client.audio.speech.create(
            model="tts-1",
            input="Hello world.",
            voice="alloy",
        )
        result.write_to("hello.mp3")

    Credential resolution order:
      1. ``api_key=`` argument
      2. ``OCTOMIL_SERVER_KEY`` env (canonical for hosted)
      3. ``OCTOMIL_API_KEY`` env (legacy compatibility)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("OCTOMIL_SERVER_KEY") or os.environ.get("OCTOMIL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "HostedClient requires `api_key=` or OCTOMIL_SERVER_KEY (or legacy OCTOMIL_API_KEY) env var."
            )
        self._api_key = resolved_key

        raw_base = (
            base_url or os.environ.get("OCTOMIL_API_BASE") or os.environ.get("OCTOMIL_API_URL") or _DEFAULT_BASE_URL
        )
        self._base_url = _normalize_base_url(raw_base)
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
