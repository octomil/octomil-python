"""Tests for octomil.config.profile — staging profile resolution.

The profile module is the single source of truth for SDK base URLs,
cache namespaces, and artifact buckets per environment. Pin every
decision branch — a regression here means staging cache poisoning
production or production cache leaking into staging.
"""

from __future__ import annotations

import pytest

from octomil.config.profile import (
    Profile,
    ProfileResolution,
    artifact_bucket_for,
    base_url_for,
    cache_namespace_for,
    resolve_base_url,
    resolve_profile,
)

# ── Profile enum ───────────────────────────────────────────────────────


def test_profile_enum_values_are_canonical_names() -> None:
    """The string values must match the contract manifest exactly —
    these strings appear in cache keys, telemetry tags, and the
    octomil-contracts environment_capability_manifest.json."""
    assert Profile.PRODUCTION.value == "production"
    assert Profile.STAGING.value == "staging"
    assert Profile.DEV.value == "dev"


def test_profile_from_str_canonical_names() -> None:
    assert Profile.from_str("production") is Profile.PRODUCTION
    assert Profile.from_str("staging") is Profile.STAGING
    assert Profile.from_str("dev") is Profile.DEV


def test_profile_from_str_case_insensitive() -> None:
    assert Profile.from_str("STAGING") is Profile.STAGING
    assert Profile.from_str("Staging") is Profile.STAGING


def test_profile_from_str_aliases_resolve() -> None:
    """Operators commonly type 'prod' / 'stg' — accept both."""
    assert Profile.from_str("prod") is Profile.PRODUCTION
    assert Profile.from_str("stg") is Profile.STAGING


def test_profile_from_str_rejects_unknown() -> None:
    with pytest.raises(ValueError, match=r"unknown profile"):
        Profile.from_str("preview")


def test_profile_from_str_rejects_empty() -> None:
    with pytest.raises(ValueError, match=r"non-empty"):
        Profile.from_str("")


# ── Per-profile constants ─────────────────────────────────────────────


def test_base_url_for_production_does_not_include_staging() -> None:
    """Critical safety pin — if production base URL ever drifts to a
    staging-shaped URL the SDK silently routes prod traffic to
    staging."""
    url = base_url_for(Profile.PRODUCTION)
    assert "staging" not in url
    assert url == "https://api.octomil.com/v1"


def test_base_url_for_staging_does_not_match_production() -> None:
    assert base_url_for(Profile.STAGING) != base_url_for(Profile.PRODUCTION)
    assert base_url_for(Profile.STAGING) == "https://api.staging.octomil.com/v1"


def test_base_url_for_dev_is_localhost() -> None:
    assert base_url_for(Profile.DEV).startswith("http://localhost")


def test_artifact_bucket_per_profile_is_distinct() -> None:
    """The whole point — staging artifacts MUST resolve through a
    staging bucket. A prod artifact request going to the staging
    bucket fails fast (404) instead of silently mixing data."""
    buckets = {artifact_bucket_for(p) for p in Profile}
    assert len(buckets) == 3
    assert artifact_bucket_for(Profile.PRODUCTION) == "octomil-models"
    assert artifact_bucket_for(Profile.STAGING) == "octomil-models-staging"


def test_cache_namespace_includes_profile_name() -> None:
    """Cache keys must be namespaced by profile to prevent cross-
    environment poisoning — a planner decision computed in staging
    cannot resolve a prod request and vice-versa."""
    assert cache_namespace_for(Profile.PRODUCTION) == "oct.production"
    assert cache_namespace_for(Profile.STAGING) == "oct.staging"
    assert cache_namespace_for(Profile.DEV) == "oct.dev"
    # Pairwise distinct
    namespaces = {cache_namespace_for(p) for p in Profile}
    assert len(namespaces) == 3


# ── resolve_profile — explicit argument ──────────────────────────────


def test_resolve_profile_explicit_arg_wins_over_env() -> None:
    res = resolve_profile("staging", env={"OCTOMIL_PROFILE": "production"})
    assert res.profile is Profile.STAGING
    assert res.source == "explicit"


def test_resolve_profile_explicit_arg_accepts_aliases() -> None:
    assert resolve_profile("prod", env={}).profile is Profile.PRODUCTION


def test_resolve_profile_explicit_invalid_raises() -> None:
    with pytest.raises(ValueError):
        resolve_profile("preview", env={})


# ── resolve_profile — env var ────────────────────────────────────────


def test_resolve_profile_env_picks_staging() -> None:
    res = resolve_profile(env={"OCTOMIL_PROFILE": "staging"})
    assert res.profile is Profile.STAGING
    assert res.source == "env"


def test_resolve_profile_empty_env_falls_through() -> None:
    """OCTOMIL_PROFILE='' must not be treated as a request for an
    empty profile name (would raise) — empty is just unset."""
    res = resolve_profile(env={"OCTOMIL_PROFILE": ""})
    assert res.profile is Profile.PRODUCTION
    assert res.source == "default"


def test_resolve_profile_env_case_insensitive() -> None:
    res = resolve_profile(env={"OCTOMIL_PROFILE": "STAGING"})
    assert res.profile is Profile.STAGING


# ── resolve_profile — URL inference ──────────────────────────────────


def test_resolve_profile_infers_staging_from_api_base() -> None:
    """If an operator pins OCTOMIL_API_BASE to staging without
    setting OCTOMIL_PROFILE, the SDK should infer staging — so cache
    keys and artifact buckets follow."""
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://api.staging.octomil.com/v1"})
    assert res.profile is Profile.STAGING
    assert res.source == "url_inferred"


def test_resolve_profile_infers_production_from_api_url() -> None:
    res = resolve_profile(env={"OCTOMIL_API_URL": "https://api.octomil.com/v1"})
    assert res.profile is Profile.PRODUCTION
    assert res.source == "url_inferred"


def test_resolve_profile_infers_dev_from_localhost() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "http://localhost:8000"})
    assert res.profile is Profile.DEV


def test_resolve_profile_infers_dev_from_127() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "http://127.0.0.1:8000"})
    assert res.profile is Profile.DEV


def test_resolve_profile_explicit_profile_overrides_url_inference() -> None:
    """If both OCTOMIL_PROFILE and OCTOMIL_API_BASE are set, profile
    wins — operators have explicitly named the env, URL is just the
    endpoint."""
    res = resolve_profile(
        env={
            "OCTOMIL_PROFILE": "staging",
            "OCTOMIL_API_BASE": "https://api.octomil.com/v1",
        }
    )
    assert res.profile is Profile.STAGING
    assert res.source == "env"


def test_resolve_profile_unmatched_url_falls_through_to_default() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://example.com/api"})
    assert res.profile is Profile.PRODUCTION
    assert res.source == "default"


# ── Hostile-URL inference safety (codex post-debate B1) ──────────────


def test_marker_in_query_string_does_not_spoof_profile() -> None:
    """Substring matching would let evil.test/?next=api.staging... spoof
    staging. Host parsing rejects it."""
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://evil.test/?next=api.staging.octomil.com"})
    assert res.profile is Profile.PRODUCTION
    assert res.source == "default"


def test_marker_in_path_does_not_spoof_profile() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://evil.test/api.octomil.com/v1"})
    assert res.profile is Profile.PRODUCTION


def test_marker_in_userinfo_does_not_spoof_profile() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://api.staging.octomil.com@evil.test/v1"})
    # Host is evil.test, not api.staging.octomil.com.
    assert res.profile is Profile.PRODUCTION


def test_superdomain_does_not_spoof_profile() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "https://api.octomil.com.evil.test/v1"})
    assert res.profile is Profile.PRODUCTION


def test_unparseable_url_falls_through_safely() -> None:
    res = resolve_profile(env={"OCTOMIL_API_BASE": "not a url"})
    assert res.profile is Profile.PRODUCTION


# ── Whitespace fallback (codex post-debate N1) ───────────────────────


def test_whitespace_api_base_falls_back_to_api_url() -> None:
    """OCTOMIL_API_BASE='   ' must NOT mask a valid OCTOMIL_API_URL."""
    res = resolve_profile(
        env={
            "OCTOMIL_API_BASE": "   ",
            "OCTOMIL_API_URL": "https://api.staging.octomil.com",
        }
    )
    assert res.profile is Profile.STAGING
    assert res.source == "url_inferred"


# ── resolve_profile — default ────────────────────────────────────────


def test_resolve_profile_no_signals_defaults_to_production() -> None:
    res = resolve_profile(env={})
    assert res.profile is Profile.PRODUCTION
    assert res.source == "default"


# ── resolve_base_url — public SDK helper ─────────────────────────────


def test_resolve_base_url_explicit_arg_wins() -> None:
    """An explicit base_url= passed to a constructor must win over
    profile resolution — back-compat for SDK users with custom URLs."""
    url = resolve_base_url(
        base_url="https://custom.example.com",
        env={"OCTOMIL_PROFILE": "staging"},
    )
    assert url == "https://custom.example.com"


def test_resolve_base_url_uses_profile_when_no_explicit() -> None:
    url = resolve_base_url(env={"OCTOMIL_PROFILE": "staging"})
    assert url == "https://api.staging.octomil.com/v1"


def test_resolve_base_url_default_returns_production() -> None:
    url = resolve_base_url(env={})
    assert url == "https://api.octomil.com/v1"


# ── Cross-profile isolation ──────────────────────────────────────────


def test_no_profile_shares_cache_namespace_with_another() -> None:
    """Defense in depth — even if base URLs were misconfigured, the
    cache namespaces alone must guarantee no cross-environment
    poisoning."""
    namespaces = [cache_namespace_for(p) for p in Profile]
    assert len(set(namespaces)) == len(namespaces)


def test_staging_artifact_bucket_does_not_contain_production_substring() -> None:
    """A typo regression like 'octomil-models-prod' would silently
    route staging artifacts through prod. Pin the actual values."""
    staging_bucket = artifact_bucket_for(Profile.STAGING)
    prod_bucket = artifact_bucket_for(Profile.PRODUCTION)
    assert "prod" not in staging_bucket.lower()
    # Production bucket is just 'octomil-models' (no env suffix); the
    # staging bucket extends it with -staging.
    assert staging_bucket.startswith(prod_bucket + "-") or staging_bucket != prod_bucket


# ── ProfileResolution dataclass ──────────────────────────────────────


def test_profile_resolution_is_immutable() -> None:
    """frozen=True — operators logging the resolution shouldn't be
    able to mutate it after the fact."""
    res = ProfileResolution(profile=Profile.PRODUCTION, source="default")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        res.profile = Profile.STAGING  # type: ignore[misc]
