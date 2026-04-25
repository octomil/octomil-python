"""Hosted Octomil client.

Lightweight HTTP client targeting api.octomil.com. Distinct from
``octomil.OctomilClient`` (the local-runtime facade) so hosted callers do
not pay the cost of importing the runtime planner / engine registry.

v0.10.0 hosted API cutover. Fail fast on legacy control-plane conventions:
no compatibility fallback, no silent normalization. See ``HostedClient``
docstring for the canonical configuration.
"""

from __future__ import annotations

import os
from typing import Optional

from .audio import HostedAudio

DEFAULT_HOSTED_BASE_URL = "https://api.octomil.com/v1"


class HostedClient:
    """Hosted Octomil API client.

    Canonical configuration:
      * Base URL: ``https://api.octomil.com/v1`` (override via
        ``OCTOMIL_HOSTED_BASE_URL`` or the ``base_url=`` argument).
      * Credential: ``OCTOMIL_SERVER_KEY`` (or the ``api_key=`` argument).
      * Endpoint paths are relative to the hosted ``/v1`` API root, e.g.
        ``/audio/speech``.

    Explicitly NOT supported (raises ``ValueError``):
      * ``OCTOMIL_API_KEY`` as an implicit hosted credential.
      * ``OCTOMIL_API_BASE`` / ``OCTOMIL_API_URL`` as hosted base-url
        fallbacks.
      * Any base ending in ``/api/v1`` (legacy control plane).

    Examples::

        from octomil.hosted import HostedClient

        client = HostedClient()  # reads OCTOMIL_SERVER_KEY
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
        resolved_key = api_key or os.environ.get("OCTOMIL_SERVER_KEY")
        if not resolved_key:
            raise ValueError(
                "HostedClient requires `api_key=` or OCTOMIL_SERVER_KEY env var. "
                "OCTOMIL_API_KEY is not used by hosted clients in v0.10.0+."
            )
        self._api_key = resolved_key

        raw_base = base_url or os.environ.get("OCTOMIL_HOSTED_BASE_URL") or DEFAULT_HOSTED_BASE_URL
        self._base_url = _validate_hosted_base_url(raw_base)

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


def _validate_hosted_base_url(raw: str) -> str:
    """Reject legacy control-plane bases. Trim trailing slashes only.

    Raises ``ValueError`` for legacy ``/api/v1`` bases per the v0.10.0
    cutover policy.
    """
    trimmed = raw.rstrip("/")
    if trimmed.endswith("/api/v1") or trimmed.endswith("/api"):
        raise ValueError(
            f"Legacy control-plane base URLs are not supported by hosted "
            f"clients; got {raw!r}. Use https://api.octomil.com/v1 (set "
            "OCTOMIL_HOSTED_BASE_URL to override)."
        )
    return trimmed


__all__ = ["HostedClient", "DEFAULT_HOSTED_BASE_URL"]
