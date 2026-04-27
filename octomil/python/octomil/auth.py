"""Re-export shim — canonical device-auth client lives in ``octomil.auth``.

The legacy package historically defined ``DeviceAuthClient`` and
``DeviceTokenState`` here, but tests and SDK callers expect them on
``octomil.auth`` (so ``patch('octomil.auth.keyring', ...)`` resolves
naturally). The implementation moved into the outer module; this file
keeps a thin re-export so ``octomil.python.octomil.auth.<name>`` paths
still resolve for code paths that already imported from here (e.g.
the top-level ``_LAZY_LEGACY_EXPORTS`` loader and any frozen-binary
PyInstaller resolver).
"""

from __future__ import annotations

from octomil.auth import (
    PUBLISHABLE_KEY_SCOPES,
    AuthConfig,
    DeviceAuthClient,
    DeviceTokenAuth,
    DeviceTokenState,
    OrgApiKeyAuth,
    PublishableKeyAuth,
)

__all__ = [
    "PUBLISHABLE_KEY_SCOPES",
    "AuthConfig",
    "DeviceAuthClient",
    "DeviceTokenAuth",
    "DeviceTokenState",
    "OrgApiKeyAuth",
    "PublishableKeyAuth",
]
