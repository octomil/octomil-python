"""Standalone configuration loading for Octomil.

Supports project-local (.octomil.toml) and user-local
(~/.config/octomil/config.toml) configuration without requiring
control-plane credentials or hosted API access.
"""

from octomil.config.local import (
    CapabilityDefault,
    CloudProfile,
    LoadedConfigSet,
    LocalOctomilConfig,
    ResolvedExecutionDefaults,
    load_project_config,
    load_standalone_config,
    load_user_config,
    resolve_capability_defaults,
)

__all__ = [
    "CapabilityDefault",
    "CloudProfile",
    "LoadedConfigSet",
    "LocalOctomilConfig",
    "ResolvedExecutionDefaults",
    "load_project_config",
    "load_standalone_config",
    "load_user_config",
    "resolve_capability_defaults",
]
