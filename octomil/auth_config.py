"""Publishable-key auth configuration for silent device registration.

Provides typed auth config variants for SDK device registration flows.
These are separate from the existing OrgApiKeyAuth/DeviceTokenAuth in
auth.py, which target server-side and CLI use cases.

Usage::

    from octomil.auth_config import PublishableKeyAuth, BootstrapTokenAuth, AnonymousAuth

    auth = PublishableKeyAuth(key="oct_pub_live_abc123")
    auth = BootstrapTokenAuth(token="jwt...")
    auth = AnonymousAuth(app_id="my-app")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .errors import OctomilError, OctomilErrorCode

__all__ = [
    "PublishableKeyAuth",
    "BootstrapTokenAuth",
    "AnonymousAuth",
    "DeviceAuthConfig",
]

_VALID_PUBLISHABLE_PREFIXES = ("oct_pub_test_", "oct_pub_live_")


@dataclass(frozen=True)
class PublishableKeyAuth:
    """Authentication via a publishable key (client-side safe).

    Keys must start with ``oct_pub_test_`` or ``oct_pub_live_``.
    """

    key: str

    def __post_init__(self) -> None:
        if not any(self.key.startswith(p) for p in _VALID_PUBLISHABLE_PREFIXES):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_API_KEY,
                message=(
                    f"Invalid publishable key prefix. "
                    f"Key must start with one of: {', '.join(_VALID_PUBLISHABLE_PREFIXES)}"
                ),
            )


@dataclass(frozen=True)
class BootstrapTokenAuth:
    """Authentication via a bootstrap JWT token.

    Used for device-to-server registration where the server issues
    a short-lived token that the device exchanges for access credentials.
    """

    token: str


@dataclass(frozen=True)
class AnonymousAuth:
    """Anonymous authentication scoped to an app.

    Used for development, testing, or scenarios where the device
    registers without user credentials but is scoped to an app_id.
    """

    app_id: str


DeviceAuthConfig = Union[PublishableKeyAuth, BootstrapTokenAuth, AnonymousAuth]
"""Discriminated union of supported device authentication configurations."""
