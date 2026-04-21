"""Planner routing defaults — determines when server-side plan resolution is active.

Planner routing is ON by default when API credentials/config exist.
When credentials are absent, planner routing defaults to OFF and requests
use the offline/cache heuristic (local engine selection).

Escape hatch: ``OCTOMIL_DISABLE_PLANNER=1`` env var or
``planner_routing=False`` in the Octomil facade constructor.

Privacy invariant: ``private`` and ``local_only`` routing policies NEVER
route to cloud regardless of planner state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth import AuthConfig


def resolve_planner_enabled(
    *,
    explicit_override: bool | None = None,
    auth: AuthConfig | None = None,
) -> bool:
    """Determine whether planner routing should be enabled.

    Resolution order:
    1. ``OCTOMIL_DISABLE_PLANNER=1`` env var → OFF
    2. ``explicit_override`` kwarg → use that value
    3. Auth config has server-reachable credentials → ON
    4. Otherwise → OFF
    """
    # Env var escape hatch always wins
    if os.environ.get("OCTOMIL_DISABLE_PLANNER", "").strip() in ("1", "true", "yes"):
        return False

    # Explicit kwarg override
    if explicit_override is not None:
        return explicit_override

    # Default: ON when credentials exist
    return _has_credentials(auth)


def _has_credentials(auth: AuthConfig | None) -> bool:
    """Whether the given auth config carries server-reachable credentials."""
    if auth is None:
        return False

    # Lazy import to avoid circular dependency
    from .auth import OrgApiKeyAuth, PublishableKeyAuth

    if isinstance(auth, OrgApiKeyAuth):
        return bool(auth.api_key)
    if isinstance(auth, PublishableKeyAuth):
        return bool(auth.api_key)
    # DeviceTokenAuth or other custom auth: check for token/api_key attribute
    return bool(getattr(auth, "api_key", None) or getattr(auth, "token", None))


def is_cloud_blocked(routing_policy: str | None) -> bool:
    """Whether the given routing policy MUST block cloud routing.

    ``private`` and ``local_only`` policies NEVER route to cloud,
    regardless of planner state, credentials, or server plan response.
    """
    return routing_policy in ("private", "local_only")


def default_routing_policy(planner_enabled: bool) -> str:
    """Return the default routing policy based on planner state.

    When planner is enabled, defaults to ``auto`` (server decides).
    When disabled, defaults to ``local_first`` (legacy behavior).
    """
    return "auto" if planner_enabled else "local_first"
