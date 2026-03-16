"""Device context for silent registration flow.

Tracks installation identity, registration state, and token lifecycle
for SDK device registration. The installation_id is persisted to disk;
registration state is ephemeral and revalidated on each SDK restart.

Usage::

    from octomil.device_context import DeviceContext

    ctx = DeviceContext()
    ctx.installation_id  # persisted UUID
    ctx.auth_headers()   # dict or None
"""

from __future__ import annotations

import enum
import logging
import platform
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceContext",
    "RegistrationState",
    "TokenState",
]

_INSTALLATION_DIR = Path.home() / ".octomil"
_INSTALLATION_ID_FILE = _INSTALLATION_DIR / "installation_id"


class RegistrationState(enum.Enum):
    """Device registration lifecycle state (not persisted)."""

    PENDING = "pending"
    REGISTERED = "registered"
    FAILED = "failed"


@dataclass
class TokenState:
    """Token lifecycle state.

    When ``access_token`` is ``None``, the device has no valid token.
    When set, ``expires_at`` is a Unix timestamp (seconds).
    """

    access_token: Optional[str] = None
    expires_at: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        """Return True if a token is present and not expired."""
        if self.access_token is None:
            return False
        if self.expires_at is None:
            return True
        import time

        return time.time() < self.expires_at

    @property
    def is_none(self) -> bool:
        return self.access_token is None

    @property
    def is_expired(self) -> bool:
        if self.access_token is None:
            return False
        if self.expires_at is None:
            return False
        import time

        return time.time() >= self.expires_at


@dataclass
class DeviceContext:
    """Mutable device context for the registration lifecycle.

    ``installation_id`` is loaded (or created) lazily from
    ``~/.octomil/installation_id``.

    ``registration_state`` starts as ``PENDING`` on every process start
    and is never persisted — it is revalidated from token freshness.
    """

    org_id: Optional[str] = None
    app_id: Optional[str] = None
    server_device_id: Optional[str] = None
    registration_state: RegistrationState = RegistrationState.PENDING
    token_state: TokenState = field(default_factory=TokenState)

    _installation_id: Optional[str] = field(default=None, repr=False)

    @property
    def installation_id(self) -> str:
        if self._installation_id is None:
            self._installation_id = get_or_create_installation_id()
        return self._installation_id

    def auth_headers(self) -> dict[str, str] | None:
        """Return Authorization headers if a valid token exists, else None."""
        if self.token_state.is_valid:
            return {"Authorization": f"Bearer {self.token_state.access_token}"}
        return None

    def telemetry_resource(self) -> dict[str, str]:
        """Return telemetry resource attributes for OTLP envelopes."""
        from octomil import __version__

        attrs: dict[str, str] = {
            "service.name": "octomil-sdk",
            "service.version": __version__,
            "telemetry.sdk.name": "octomil",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": __version__,
            "octomil.installation_id": self.installation_id,
            "octomil.platform": sys.platform,
            "octomil.sdk_surface": "python",
            "os.type": platform.system().lower(),
            "os.version": platform.release(),
        }
        if self.org_id:
            attrs["octomil.org_id"] = self.org_id
        if self.server_device_id:
            attrs["octomil.device_id"] = self.server_device_id
        return attrs


def get_or_create_installation_id() -> str:
    """Load or create a persistent installation UUID.

    Stored at ``~/.octomil/installation_id``. Creates the directory
    and file if they do not exist.
    """
    try:
        if _INSTALLATION_ID_FILE.exists():
            stored = _INSTALLATION_ID_FILE.read_text().strip()
            if stored:
                # Validate it looks like a UUID
                uuid.UUID(stored)
                return stored
    except (OSError, ValueError):
        logger.debug("Could not read installation_id file, generating new one")

    new_id = str(uuid.uuid4())
    try:
        _INSTALLATION_DIR.mkdir(parents=True, exist_ok=True)
        _INSTALLATION_ID_FILE.write_text(new_id + "\n")
    except OSError:
        logger.debug("Could not persist installation_id to %s", _INSTALLATION_ID_FILE)

    return new_id
