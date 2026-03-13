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

__all__ = [
    "AuthType",
    "PrincipalType",
    "Scope",
    "OrgApiKeyAuth",
    "DeviceTokenAuth",
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
            raise ValueError(f"Environment variable {api_key_var} is required but not set.")
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


AuthConfig = Union[OrgApiKeyAuth, DeviceTokenAuth]
"""Discriminated union of supported authentication configurations."""
