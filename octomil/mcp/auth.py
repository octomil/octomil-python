"""Bearer token authentication for the Octomil HTTP agent server.

- HTTP without ``OCTOMIL_MCP_API_KEY``: dev mode — warning logged, all allowed
- HTTP with ``OCTOMIL_MCP_API_KEY``: Bearer token required on protected endpoints
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

_DEV_MODE_WARNED = False


def _get_api_key() -> str | None:
    """Read the configured API key from environment."""
    return os.environ.get("OCTOMIL_MCP_API_KEY")


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str | None:
    """FastAPI dependency that validates Bearer token authentication.

    Behaviour:
    - No ``OCTOMIL_MCP_API_KEY`` set → dev mode, all requests pass through
    - Key set, valid Bearer token → returns the token
    - Key set, missing/invalid token → 401
    """
    global _DEV_MODE_WARNED
    api_key = _get_api_key()

    if not api_key:
        if not _DEV_MODE_WARNED:
            logger.warning(
                "OCTOMIL_MCP_API_KEY not set — running in dev mode. " "All requests are allowed without authentication."
            )
            _DEV_MODE_WARNED = True
        return None

    if credentials is None or credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Valid Bearer token required. Set Authorization: Bearer <your-api-key>",
            },
        )

    return credentials.credentials


def reset_dev_mode_warning() -> None:
    """Reset the dev mode warning flag (for testing)."""
    global _DEV_MODE_WARNED
    _DEV_MODE_WARNED = False
