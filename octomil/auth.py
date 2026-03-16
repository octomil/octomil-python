"""Auth configuration types for the Octomil SDK.

Provides a structured, type-safe way to configure authentication
for OctomilClient. Replaces the old flat constructor parameters.

Usage::

    from octomil.auth import OrgApiKeyAuth, DeviceTokenAuth

    # Org API key auth
    auth = OrgApiKeyAuth(api_key="edg_...", org_id="org_123")

    # Device token auth
    auth = DeviceTokenAuth(
        device_id="dev_abc",
        bootstrap_token="jwt...",
    )

    # From environment
    auth = OrgApiKeyAuth.from_env()

    client = OctomilClient(auth=auth)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

from ._generated.auth_type import AuthType
from ._generated.principal_type import PrincipalType
from ._generated.scope import Scope
from .errors import OctomilError, OctomilErrorCode

__all__ = [
    "AuthType",
    "PrincipalType",
    "Scope",
    "OrgApiKeyAuth",
    "DeviceTokenAuth",
    "PublishableKeyAuth",
    "AuthConfig",
]

_DEFAULT_API_BASE = "https://api.octomil.com/api/v1"


@dataclass(frozen=True)
class OrgApiKeyAuth:
    """Authentication via an organization-scoped API key (edg_ prefix).

    This is the primary auth method for server-side SDKs, CLI tools,
    and CI/CD pipelines.
    """

    api_key: str
    org_id: str
    api_base: str = _DEFAULT_API_BASE
    auth_type: AuthType = AuthType.ORG_API_KEY

    @classmethod
    def from_env(
        cls,
        *,
        api_key_var: str = "OCTOMIL_API_KEY",
        org_id_var: str = "OCTOMIL_ORG_ID",
        api_base_var: str = "OCTOMIL_API_BASE",
    ) -> OrgApiKeyAuth:
        """Construct from environment variables.

        Reads ``OCTOMIL_API_KEY``, ``OCTOMIL_ORG_ID``, and optionally
        ``OCTOMIL_API_BASE`` from the environment.

        Raises:
            ValueError: If ``OCTOMIL_API_KEY`` is not set.
        """
        api_key = os.environ.get(api_key_var, "")
        if not api_key:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_API_KEY,
                message=f"Environment variable {api_key_var} is required but not set.",
            )
        org_id = os.environ.get(org_id_var, "default")
        api_base = os.environ.get(api_base_var, _DEFAULT_API_BASE)
        return cls(api_key=api_key, org_id=org_id, api_base=api_base)


@dataclass(frozen=True)
class DeviceTokenAuth:
    """Authentication via a short-lived device access token.

    Used by edge devices that go through a bootstrap/registration flow.
    The bootstrap_token is exchanged for a short-lived JWT.
    """

    device_id: str
    bootstrap_token: str
    api_base: str = _DEFAULT_API_BASE
    auth_type: AuthType = AuthType.DEVICE_TOKEN


_PUBLISHABLE_KEY_PREFIX = "oct_pub_"

# Scopes that are safe for publishable keys (client-side, no secrets).
PUBLISHABLE_KEY_SCOPES: frozenset[Scope] = frozenset(
    {
        Scope.DEVICES_REGISTER,
        Scope.DEVICES_HEARTBEAT,
        Scope.TELEMETRY_WRITE,
        Scope.MODELS_READ,
    }
)


@dataclass(frozen=True)
class PublishableKeyAuth:
    """Authentication via a publishable key (``oct_pub_`` prefix).

    Publishable keys are safe to embed in mobile apps and browser code.
    They are restricted to a fixed set of scopes:
    ``devices:register``, ``devices:heartbeat``, ``telemetry:write``,
    ``models:read``.
    """

    api_key: str
    api_base: str = _DEFAULT_API_BASE
    auth_type: AuthType = AuthType.PUBLISHABLE_KEY

    def __post_init__(self) -> None:
        if not self.api_key.startswith(_PUBLISHABLE_KEY_PREFIX):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_API_KEY,
                message=(
                    f"Publishable key must start with '{_PUBLISHABLE_KEY_PREFIX}', " f"got '{self.api_key[:16]}...'"
                ),
            )

    @property
    def scopes(self) -> frozenset[Scope]:
        """Return the fixed set of scopes allowed for publishable keys."""
        return PUBLISHABLE_KEY_SCOPES

    def headers(self) -> dict[str, str]:
        """Return HTTP headers for publishable key authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Octomil-Auth-Type": self.auth_type.value,
        }


AuthConfig = Union[OrgApiKeyAuth, DeviceTokenAuth, PublishableKeyAuth]
"""Discriminated union of supported authentication configurations."""
