"""Cloud URL and API key helpers.

Extracted from kernel.py -- contains helpers for resolving cloud endpoint
URLs and API keys from CloudProfile configuration.
"""

from __future__ import annotations

import os
from typing import Optional

from octomil.config.local import CloudProfile


def _openai_base_url(profile: CloudProfile) -> str:
    """Return an OpenAI-compatible hosted base URL ending in /v1."""
    base = profile.base_url.rstrip("/")
    if base.endswith("/v1") and not base.endswith("/api/v1"):
        return base
    if base.endswith("/api/v1"):
        return base[: -len("/api/v1")] + "/v1"
    return f"{base}/v1"


def _platform_api_base_url(profile: CloudProfile) -> str:
    """Return the legacy platform API base URL ending in /api/v1."""
    base = profile.base_url.rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/v1"):
        return base[: -len("/v1")] + "/api/v1"
    return f"{base}/api/v1"


def _cloud_api_key(profile: Optional[CloudProfile]) -> str:
    if profile is None:
        return ""
    return os.environ.get(profile.api_key_env, "")
