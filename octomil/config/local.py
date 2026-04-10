"""Standalone config loading from .octomil.toml and ~/.config/octomil/config.toml.

No network calls are performed here.  Config resolution is purely local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# Canonical capability keys matching server/domain vocabulary.
CAPABILITY_CHAT = "chat"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_TRANSCRIPTION = "transcription"

# Supported serving-policy preset short names.
VALID_PRESETS = frozenset({"private", "local_first", "performance_first", "cloud_first", "cloud_only"})

# Legacy presets that must normalise silently.
_LEGACY_PRESET_MAP = {"quality_first": "cloud_first", "quality": "cloud_first"}
DEFAULT_CLOUD_BASE_URL = "https://api.octomil.com/v1"


def _normalise_preset(raw: str) -> str:
    lowered = raw.strip().lower()
    mapped = _LEGACY_PRESET_MAP.get(lowered, lowered)
    if mapped not in VALID_PRESETS:
        raise ValueError(f"Unknown policy preset '{raw}'. " f"Valid presets: {', '.join(sorted(VALID_PRESETS))}")
    return mapped


@dataclass
class FallbackConfig:
    allow_cloud_fallback: bool = True
    allow_local_fallback: bool = True


@dataclass
class InlinePolicy:
    """Fully-specified inline serving policy from config."""

    routing_mode: str = "auto"
    routing_preference: Optional[str] = None
    fallback: FallbackConfig = field(default_factory=FallbackConfig)


@dataclass
class CapabilityDefault:
    """Per-capability default from config."""

    model: Optional[str] = None
    policy: Optional[str] = None  # preset short name or None
    inline_policy: Optional[InlinePolicy] = None


@dataclass
class AppBinding:
    slug: Optional[str] = None
    org_id: Optional[str] = None


@dataclass
class CloudProfile:
    name: str = "default"
    base_url: str = DEFAULT_CLOUD_BASE_URL
    api_key_env: str = "OCTOMIL_SERVER_KEY"
    org_id_env: str = "OCTOMIL_ORG_ID"


@dataclass
class LocalOctomilConfig:
    """Parsed standalone config from a single TOML file."""

    version: int = 1
    default_profile: str = "local"
    default_policy: Optional[str] = None
    capabilities: dict[str, CapabilityDefault] = field(default_factory=dict)
    policies: dict[str, InlinePolicy] = field(default_factory=dict)
    cloud_profiles: dict[str, CloudProfile] = field(default_factory=dict)
    app: Optional[AppBinding] = None
    source_path: Optional[Path] = None


@dataclass
class LoadedConfigSet:
    """Merged result of project + user config loading."""

    project: Optional[LocalOctomilConfig] = None
    user: Optional[LocalOctomilConfig] = None


@dataclass
class ResolvedExecutionDefaults:
    """Resolved defaults for a single execution request."""

    model: Optional[str] = None
    policy_preset: Optional[str] = None
    inline_policy: Optional[InlinePolicy] = None
    cloud_profile: Optional[CloudProfile] = None
    app_slug: Optional[str] = None
    org_id: Optional[str] = None
    source: Optional[str] = None  # "cli", "project", "user", "builtin"


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


def _parse_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file. Returns empty dict if missing or unparseable."""
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_fallback(raw: dict[str, Any]) -> FallbackConfig:
    return FallbackConfig(
        allow_cloud_fallback=raw.get("allow_cloud_fallback", True),
        allow_local_fallback=raw.get("allow_local_fallback", False),
    )


def _parse_inline_policy(raw: dict[str, Any]) -> InlinePolicy:
    fb_raw = raw.get("fallback", {})
    return InlinePolicy(
        routing_mode=raw.get("routing_mode", "auto"),
        routing_preference=raw.get("routing_preference"),
        fallback=_parse_fallback(fb_raw),
    )


def _parse_capability(raw: dict[str, Any]) -> CapabilityDefault:
    model = raw.get("model")
    policy_val = raw.get("policy")
    inline = None
    preset = None

    if isinstance(policy_val, dict):
        inline = _parse_inline_policy(policy_val)
    elif isinstance(policy_val, str):
        preset = _normalise_preset(policy_val)

    return CapabilityDefault(model=model, policy=preset, inline_policy=inline)


def _parse_config(data: dict[str, Any], source_path: Optional[Path] = None) -> LocalOctomilConfig:
    defaults_raw = data.get("defaults", {})
    caps_raw = data.get("capabilities", {})
    policies_raw = data.get("policies", {})
    cloud_raw = data.get("cloud", {})
    app_raw = data.get("app", {})

    capabilities: dict[str, CapabilityDefault] = {}
    for key, val in caps_raw.items():
        if isinstance(val, dict):
            capabilities[key] = _parse_capability(val)

    policies: dict[str, InlinePolicy] = {}
    for key, val in policies_raw.items():
        if isinstance(val, dict):
            policies[key] = _parse_inline_policy(val)

    cloud_profiles: dict[str, CloudProfile] = {}
    profiles_raw = cloud_raw.get("profiles", {})
    for name, prof in profiles_raw.items():
        if isinstance(prof, dict):
            cloud_profiles[name] = CloudProfile(
                name=name,
                base_url=prof.get("base_url", "https://api.octomil.com"),
                api_key_env=prof.get("api_key_env", "OCTOMIL_SERVER_KEY"),
                org_id_env=prof.get("org_id_env", "OCTOMIL_ORG_ID"),
            )

    app_binding = None
    if app_raw:
        app_binding = AppBinding(
            slug=app_raw.get("slug"),
            org_id=app_raw.get("org_id"),
        )

    default_policy = defaults_raw.get("policy")
    if default_policy:
        default_policy = _normalise_preset(default_policy)

    return LocalOctomilConfig(
        version=data.get("version", 1),
        default_profile=defaults_raw.get("profile", "local"),
        default_policy=default_policy,
        capabilities=capabilities,
        policies=policies,
        cloud_profiles=cloud_profiles,
        app=app_binding,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PROJECT_CONFIG_FILENAME = ".octomil.toml"
USER_CONFIG_DIR = Path("~/.config/octomil").expanduser()
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.toml"


def load_project_config(start_dir: Optional[Path] = None) -> Optional[LocalOctomilConfig]:
    """Walk up from *start_dir* looking for .octomil.toml."""
    search = Path(start_dir) if start_dir else Path.cwd()
    for d in [search, *search.parents]:
        candidate = d / PROJECT_CONFIG_FILENAME
        data = _parse_toml(candidate)
        if data:
            return _parse_config(data, source_path=candidate)
    return None


def load_user_config() -> Optional[LocalOctomilConfig]:
    """Load ~/.config/octomil/config.toml if it exists."""
    data = _parse_toml(USER_CONFIG_PATH)
    if data:
        return _parse_config(data, source_path=USER_CONFIG_PATH)
    return None


def load_standalone_config(start_dir: Optional[Path] = None) -> LoadedConfigSet:
    """Load both project and user configs. No network calls."""
    return LoadedConfigSet(
        project=load_project_config(start_dir),
        user=load_user_config(),
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

# Built-in hardware-aware defaults.  These are the smallest reasonable
# models that should work on most machines.
_BUILTIN_DEFAULTS: dict[str, str] = {
    CAPABILITY_CHAT: "gemma-1b",
    CAPABILITY_EMBEDDING: "nomic-embed-text-v1.5",
    CAPABILITY_TRANSCRIPTION: "whisper-small",
}


@dataclass
class RequestOverrides:
    """Explicit overrides from CLI flags or SDK arguments."""

    model: Optional[str] = None
    policy: Optional[str] = None
    app_slug: Optional[str] = None


def resolve_capability_defaults(
    capability: str,
    request: RequestOverrides,
    config_set: LoadedConfigSet,
) -> ResolvedExecutionDefaults:
    """Resolve model, policy, and cloud profile for a capability.

    Precedence (highest first):
      1. explicit request arguments
      2. project config (.octomil.toml)
      3. user config (~/.config/octomil/config.toml)
      4. built-in defaults

    No network calls.
    """
    result = ResolvedExecutionDefaults()

    # --- 1. Explicit overrides ---
    if request.model:
        result.model = request.model
        result.source = "cli"
    if request.policy:
        result.policy_preset = _normalise_preset(request.policy)
        if result.source is None:
            result.source = "cli"
    if request.app_slug:
        result.app_slug = request.app_slug
        if result.source is None:
            result.source = "cli"

    # --- 2. Project config ---
    if config_set.project:
        _merge_config_layer(result, config_set.project, capability, "project")

    # --- 3. User config ---
    if config_set.user:
        _merge_config_layer(result, config_set.user, capability, "user")

    # --- 4. Built-in defaults ---
    if result.model is None:
        builtin = _BUILTIN_DEFAULTS.get(capability)
        if builtin:
            result.model = builtin
            if result.source is None:
                result.source = "builtin"

    # Default policy is local_first for standalone (local-first by default,
    # cloud only if explicitly configured).
    if result.policy_preset is None and result.inline_policy is None:
        result.policy_preset = "local_first"

    if result.cloud_profile is None:
        result.cloud_profile = _default_cloud_profile_from_env()

    return result


def _merge_config_layer(
    result: ResolvedExecutionDefaults,
    config: LocalOctomilConfig,
    capability: str,
    source_label: str,
) -> None:
    """Merge a config layer into result, filling only unset fields."""
    cap_default = config.capabilities.get(capability)

    # Model
    if result.model is None and cap_default and cap_default.model:
        result.model = cap_default.model
        if result.source is None:
            result.source = source_label

    # Policy
    if result.policy_preset is None and result.inline_policy is None:
        if cap_default and cap_default.policy:
            result.policy_preset = cap_default.policy
            if result.source is None:
                result.source = source_label
        elif cap_default and cap_default.inline_policy:
            result.inline_policy = cap_default.inline_policy
            if result.source is None:
                result.source = source_label
        elif config.default_policy:
            result.policy_preset = config.default_policy
            if result.source is None:
                result.source = source_label

    # App binding
    if result.app_slug is None and config.app and config.app.slug:
        result.app_slug = config.app.slug
        result.org_id = config.app.org_id

    # Cloud profile (first available)
    if result.cloud_profile is None and config.cloud_profiles:
        profile_name = config.default_profile if config.default_profile != "local" else "default"
        profile = config.cloud_profiles.get(profile_name) or next(iter(config.cloud_profiles.values()), None)
        if profile:
            result.cloud_profile = profile


def _default_cloud_profile_from_env() -> Optional[CloudProfile]:
    """Return the built-in hosted profile when standard cloud auth is present."""
    if not os.environ.get("OCTOMIL_SERVER_KEY"):
        return None
    return CloudProfile(
        base_url=os.environ.get("OCTOMIL_API_BASE") or os.environ.get("OCTOMIL_API_URL") or DEFAULT_CLOUD_BASE_URL,
    )
