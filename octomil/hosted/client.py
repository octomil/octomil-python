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

        # Fail fast on legacy hosted base env vars. Silent fallback to the
        # production default would mask migration misses (an upgraded env
        # with OCTOMIL_API_BASE still set would hit prod against an
        # unintended endpoint).
        _reject_legacy_hosted_env_vars()

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


_LEGACY_HOSTED_ENV_VARS: tuple[str, ...] = ("OCTOMIL_API_BASE", "OCTOMIL_API_URL")


def _reject_legacy_hosted_env_vars() -> None:
    """Raise ValueError if any legacy hosted-base env var is set.

    OCTOMIL_API_BASE / OCTOMIL_API_URL are the legacy control-plane
    base URLs and point at .../api/v1. Silently ignoring them lets an
    upgraded environment hit the production default while the operator
    thinks they're still pointed at staging or a custom host. The
    cutover policy is fail-fast: tell the operator to migrate to
    OCTOMIL_HOSTED_BASE_URL or unset the legacy var.
    """
    set_vars = [name for name in _LEGACY_HOSTED_ENV_VARS if os.environ.get(name)]
    if set_vars:
        joined = ", ".join(set_vars)
        raise ValueError(
            f"Legacy hosted env var(s) set: {joined}. These are not used by "
            "hosted clients in v0.10.0+. Set OCTOMIL_HOSTED_BASE_URL to the "
            "canonical hosted base (e.g. https://api.octomil.com/v1) and "
            f"unset {joined} to acknowledge the migration."
        )


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
