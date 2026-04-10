"""Tests for standalone config loading and resolution."""

from __future__ import annotations

import pytest

from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CapabilityDefault,
    LoadedConfigSet,
    LocalOctomilConfig,
    RequestOverrides,
    _normalise_preset,
    _parse_config,
    resolve_capability_defaults,
)

# ---------------------------------------------------------------------------
# Preset normalisation
# ---------------------------------------------------------------------------


class TestNormalisePreset:
    def test_valid_presets(self):
        for p in ["private", "local_first", "performance_first", "cloud_first", "cloud_only"]:
            assert _normalise_preset(p) == p

    def test_quality_first_maps_to_cloud_first(self):
        assert _normalise_preset("quality_first") == "cloud_first"

    def test_quality_maps_to_cloud_first(self):
        assert _normalise_preset("quality") == "cloud_first"

    def test_strips_whitespace(self):
        assert _normalise_preset("  local_first  ") == "local_first"

    def test_case_insensitive(self):
        assert _normalise_preset("LOCAL_FIRST") == "local_first"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown policy preset"):
            _normalise_preset("turbo_mode")

    def test_auto_is_not_a_valid_standalone_preset(self):
        with pytest.raises(ValueError, match="Unknown policy preset"):
            _normalise_preset("auto")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestParseConfig:
    def test_empty_config(self):
        cfg = _parse_config({})
        assert cfg.version == 1
        assert cfg.default_profile == "local"
        assert cfg.default_policy is None
        assert cfg.capabilities == {}

    def test_full_config(self):
        data = {
            "version": 1,
            "defaults": {"profile": "local", "policy": "local_first"},
            "capabilities": {
                "chat": {"model": "gemma-1b", "policy": "local_first"},
                "embedding": {"model": "nomic-embed-text-v1.5", "policy": "private"},
                "transcription": {"model": "whisper-small"},
            },
            "cloud": {
                "profiles": {
                    "default": {
                        "base_url": "https://api.octomil.com",
                        "api_key_env": "OCTOMIL_SERVER_KEY",
                    }
                }
            },
            "app": {"slug": "my-app", "org_id": "org_123"},
        }
        cfg = _parse_config(data)
        assert cfg.version == 1
        assert cfg.default_policy == "local_first"
        assert cfg.capabilities["chat"].model == "gemma-1b"
        assert cfg.capabilities["chat"].policy == "local_first"
        assert cfg.capabilities["embedding"].policy == "private"
        assert cfg.capabilities["transcription"].model == "whisper-small"
        assert cfg.cloud_profiles["default"].base_url == "https://api.octomil.com"
        assert cfg.app is not None
        assert cfg.app.slug == "my-app"
        assert cfg.app.org_id == "org_123"

    def test_inline_policy(self):
        data = {
            "capabilities": {
                "chat": {
                    "model": "phi-mini",
                    "policy": {
                        "routing_mode": "auto",
                        "routing_preference": "local",
                        "fallback": {
                            "allow_cloud_fallback": True,
                            "allow_local_fallback": False,
                        },
                    },
                }
            }
        }
        cfg = _parse_config(data)
        cap = cfg.capabilities["chat"]
        assert cap.model == "phi-mini"
        assert cap.policy is None  # string preset is None for inline
        assert cap.inline_policy is not None
        assert cap.inline_policy.routing_mode == "auto"
        assert cap.inline_policy.routing_preference == "local"
        assert cap.inline_policy.fallback.allow_cloud_fallback is True

    def test_legacy_quality_first_normalises(self):
        data = {"defaults": {"policy": "quality_first"}}
        cfg = _parse_config(data)
        assert cfg.default_policy == "cloud_first"


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


def _make_config_set(
    project_model: str | None = None,
    project_policy: str | None = None,
    user_model: str | None = None,
    user_policy: str | None = None,
) -> LoadedConfigSet:
    project = None
    if project_model or project_policy:
        caps = {}
        if project_model or project_policy:
            caps["chat"] = CapabilityDefault(model=project_model, policy=project_policy)
        project = LocalOctomilConfig(capabilities=caps)

    user = None
    if user_model or user_policy:
        caps = {}
        if user_model or user_policy:
            caps["chat"] = CapabilityDefault(model=user_model, policy=user_policy)
        user = LocalOctomilConfig(capabilities=caps)

    return LoadedConfigSet(project=project, user=user)


class TestResolutionPrecedence:
    def test_explicit_model_wins(self):
        config_set = _make_config_set(project_model="project-model", user_model="user-model")
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(model="cli-model"),
            config_set,
        )
        assert result.model == "cli-model"
        assert result.source == "cli"

    def test_project_config_wins_over_user(self):
        config_set = _make_config_set(project_model="project-model", user_model="user-model")
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(),
            config_set,
        )
        assert result.model == "project-model"
        assert result.source == "project"

    def test_user_config_used_when_no_project(self):
        config_set = _make_config_set(user_model="user-model")
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(),
            config_set,
        )
        assert result.model == "user-model"
        assert result.source == "user"

    def test_builtin_default_when_no_config(self):
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(),
            config_set,
        )
        assert result.model == "gemma-1b"
        assert result.source == "builtin"

    def test_builtin_embedding_default(self):
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_EMBEDDING,
            RequestOverrides(),
            config_set,
        )
        assert result.model == "nomic-embed-text-v1.5"

    def test_builtin_transcription_default(self):
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_TRANSCRIPTION,
            RequestOverrides(),
            config_set,
        )
        assert result.model == "whisper-small"

    def test_explicit_policy_wins(self):
        config_set = _make_config_set(project_policy="private")
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(policy="cloud_only"),
            config_set,
        )
        assert result.policy_preset == "cloud_only"

    def test_default_policy_is_local_first(self):
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(),
            config_set,
        )
        assert result.policy_preset == "local_first"

    def test_no_credentials_required_for_local(self, monkeypatch):
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(),
            config_set,
        )
        assert result.cloud_profile is None
        assert result.app_slug is None
        assert result.org_id is None

    def test_standard_server_key_enables_builtin_cloud_profile(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config_set = LoadedConfigSet()
        result = resolve_capability_defaults(
            CAPABILITY_CHAT,
            RequestOverrides(policy="cloud_only"),
            config_set,
        )
        assert result.cloud_profile is not None
        assert result.cloud_profile.base_url == "https://api.octomil.com/v1"
