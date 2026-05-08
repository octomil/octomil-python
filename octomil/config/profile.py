"""SDK environment profile resolution — staging vs production vs dev.

A *profile* names a deployment environment of the Octomil control plane.
This module is the single source of truth in the Python SDK for:

- which base URL the SDK talks to by default,
- which cache namespace planner / capability results are stored under,
- which model artifact bucket the SDK expects presigned URLs to point at.

Profiles let the same SDK binary talk to staging or production without
risk of cross-contamination — production cached planner decisions never
leak into staging runs and vice-versa, because the cache key is
namespaced by profile.

Resolution order (first non-empty wins):

1. Explicit ``profile=`` argument passed to ``resolve_profile()``.
2. ``OCTOMIL_PROFILE`` env var (``staging``, ``production``, ``dev``).
3. Heuristic: if ``OCTOMIL_API_BASE`` / ``OCTOMIL_API_URL`` contains
   ``staging`` host, infer ``staging``. Otherwise default to
   ``production``.

The values here are duplicated from
``octomil-contracts/fixtures/core/environment_capability_manifest.json``;
once the contracts package is published as a wheel the SDK will import
the canonical loader. Until then, **any change to the profile→base_url
mapping here MUST be mirrored in the contracts manifest** or the
promotion gate will see the SDK pointing at one URL while the contract
declares another.

This module is intentionally dependency-free: no network, no disk,
pure stdlib. ``import octomil`` should not fail because a profile can't
be resolved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse


class Profile(str, Enum):
    """Named SDK environment profiles.

    String-valued so ``Profile("staging") == Profile.STAGING`` round-trips
    cleanly through env vars and JSON.
    """

    PRODUCTION = "production"
    STAGING = "staging"
    DEV = "dev"

    @classmethod
    def from_str(cls, raw: str) -> "Profile":
        """Case-insensitive lookup with helpful error.

        Accepts ``prod`` / ``stg`` aliases that operators commonly type.
        """
        if not raw:
            raise ValueError("profile name must be non-empty")
        normalized = raw.strip().lower()
        aliases = {"prod": "production", "stg": "staging", "staging-2": "staging"}
        normalized = aliases.get(normalized, normalized)
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"unknown profile {raw!r}; valid: {valid}")


# Source of truth for SDK base URLs per profile. Mirrors
# ``octomil-contracts/fixtures/core/environment_capability_manifest.json``;
# see module docstring.
# /v1 suffix is API-versioning shared across environments; the
# operational base URL the SDK uses includes it. The host-only form
# without /v1 is what e2e / health checks hit (see staging-e2e.yml).
_PROFILE_HOST_URLS: dict[Profile, str] = {
    Profile.PRODUCTION: "https://api.octomil.com",
    Profile.STAGING: "https://api.staging.octomil.com",
    Profile.DEV: "http://localhost:8000",
}

_PROFILE_BASE_URLS: dict[Profile, str] = {
    Profile.PRODUCTION: "https://api.octomil.com/v1",
    Profile.STAGING: "https://api.staging.octomil.com/v1",
    Profile.DEV: "http://localhost:8000/v1",
}

# Model-artifact buckets per profile. The server returns presigned URLs
# pointing at these buckets; the SDK uses the namespace value to verify
# (best-effort) that a presigned URL host matches the profile's
# expected bucket — a staging URL should not host prod artifacts.
_PROFILE_ARTIFACT_BUCKETS: dict[Profile, str] = {
    Profile.PRODUCTION: "octomil-models",
    Profile.STAGING: "octomil-models-staging",
    Profile.DEV: "octomil-models-dev",
}

# Exact-host markers used by ``resolve_profile`` when inferring from an
# explicit OCTOMIL_API_BASE/URL. Match is against the *parsed hostname*,
# never a substring of the raw URL — a hostile URL like
# ``https://evil.test/?next=api.staging.octomil.com`` or
# ``api.octomil.com.evil.test`` MUST NOT spoof a profile.
_HOST_INFERENCE_MARKERS: dict[Profile, frozenset[str]] = {
    Profile.STAGING: frozenset({"api.staging.octomil.com"}),
    Profile.PRODUCTION: frozenset({"api.octomil.com"}),
    Profile.DEV: frozenset({"localhost", "127.0.0.1", "0.0.0.0"}),
}


@dataclass(frozen=True)
class ProfileResolution:
    """The result of resolving a profile, with provenance.

    ``source`` tells the caller HOW the profile was picked — useful for
    logging when the SDK boots so operators can verify the right path
    was taken (vs. silently defaulting to production).
    """

    profile: Profile
    source: str  # "explicit", "env", "url_inferred", "default"


def base_url_for(profile: Profile) -> str:
    """Canonical SDK base URL (with /v1 suffix) for the given profile."""
    return _PROFILE_BASE_URLS[profile]


def host_url_for(profile: Profile) -> str:
    """Host-only base URL (no /v1 suffix). Used by clients that compose
    their own path prefix — e.g. the planner appends /api/v2/...
    paths and a /v1-suffixed base would yield /v1/api/v2/... 404s.
    """
    return _PROFILE_HOST_URLS[profile]


def artifact_bucket_for(profile: Profile) -> str:
    """Canonical R2 bucket name for model artifacts in the given profile."""
    return _PROFILE_ARTIFACT_BUCKETS[profile]


def cache_namespace_for(profile: Profile) -> str:
    """Cache key prefix for planner/capability caches.

    Including the profile in the cache key prevents cross-environment
    cache poisoning — a planner decision computed against staging
    capabilities never resolves a production request, and vice-versa.
    """
    return f"oct.{profile.value}"


def _infer_from_url(url: str) -> Optional[Profile]:
    """Parse the URL and match its hostname EXACTLY against profile
    markers. Returns None for unparseable URLs, missing hosts, or
    hosts that don't exactly match any marker.

    Substring matching would let a hostile URL like
    ``https://evil.test/?next=api.staging.octomil.com`` or
    ``api.octomil.com.evil.test`` spoof a profile. Codex post-debate
    B1 across all 7 SDKs.
    """
    if not url or not url.strip():
        return None
    try:
        parsed = urlparse(url.strip())
    except (ValueError, TypeError):
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    for profile, markers in _HOST_INFERENCE_MARKERS.items():
        if host in markers:
            return profile
    return None


def resolve_profile(
    profile: Optional[str] = None,
    *,
    env: Optional[dict[str, str]] = None,
) -> ProfileResolution:
    """Resolve the active SDK profile.

    Args:
        profile: Explicit profile name (``"production"``, ``"staging"``,
            ``"dev"``, or aliases). Wins over env / URL inference.
        env: Environment dict to read from. Defaults to ``os.environ``.
            Tests inject a custom dict to avoid global state.

    Returns:
        ProfileResolution with the picked profile and the source it
        came from. Never raises — falls back to PRODUCTION default if
        no signal is found.

    Raises:
        ValueError: only when the explicit ``profile`` argument is a
            non-empty string that doesn't name a known profile. An
            empty ``OCTOMIL_PROFILE`` env var is treated as unset.
    """
    env_dict = env if env is not None else dict(os.environ)

    # 1. Explicit argument wins.
    if profile:
        return ProfileResolution(profile=Profile.from_str(profile), source="explicit")

    # 2. OCTOMIL_PROFILE env var.
    raw_env = env_dict.get("OCTOMIL_PROFILE", "").strip()
    if raw_env:
        return ProfileResolution(profile=Profile.from_str(raw_env), source="env")

    # 3. Infer from explicit base URL if the operator pinned one
    #    without setting OCTOMIL_PROFILE — treats a staging URL as a
    #    staging profile so cache keys + artifact buckets follow.
    #    Trim BEFORE selecting so OCTOMIL_API_BASE='   ' doesn't mask
    #    a valid OCTOMIL_API_URL (codex post-debate N1).
    base_trimmed = (env_dict.get("OCTOMIL_API_BASE") or "").strip()
    url_trimmed = (env_dict.get("OCTOMIL_API_URL") or "").strip()
    explicit_url = base_trimmed or url_trimmed
    inferred = _infer_from_url(explicit_url)
    if inferred is not None:
        return ProfileResolution(profile=inferred, source="url_inferred")

    # 4. Default.
    return ProfileResolution(profile=Profile.PRODUCTION, source="default")


def resolve_base_url(
    base_url: Optional[str] = None,
    profile: Optional[str] = None,
    *,
    env: Optional[dict[str, str]] = None,
) -> str:
    """Pick the base URL the SDK should talk to.

    Resolution: explicit ``base_url`` argument > resolved profile's
    canonical URL. Env vars are read via ``resolve_profile``.

    This is the public API the SDK constructors call. Existing call
    sites that pass an explicit ``base_url`` keep working unchanged.
    """
    # Treat whitespace-only as unset so " " doesn't pin the SDK to an
    # invalid URL; fall through to profile resolution (codex
    # post-debate N1).
    if base_url and base_url.strip():
        return base_url
    resolution = resolve_profile(profile=profile, env=env)
    return base_url_for(resolution.profile)
