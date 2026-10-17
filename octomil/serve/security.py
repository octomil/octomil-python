"""Inbound auth and CORS policy for the local inference server."""

from __future__ import annotations

import os
import secrets
from typing import Any, Optional

_PUBLIC_PATHS = frozenset({"/health"})
_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def install_security(app: Any, auth_token: Optional[str] = None) -> None:
    """Add CORS (loopback origins plus OCTOMIL_SERVE_CORS_ORIGINS) and optional bearer auth."""
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    token = auth_token or os.environ.get("OCTOMIL_SERVE_TOKEN") or None
    extra = [o.strip() for o in os.environ.get("OCTOMIL_SERVE_CORS_ORIGINS", "").split(",") if o.strip()]

    # Added first so CORS (added last, outermost) answers preflights before auth runs.
    if token:
        expected = token.encode()

        @app.middleware("http")
        async def bearer_auth_middleware(request: Any, call_next: Any) -> Any:
            if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or not secrets.compare_digest(auth[7:].encode(), expected):
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
            return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=extra,
        allow_origin_regex=_LOCAL_ORIGIN_REGEX,
        allow_methods=["*"],
        allow_headers=["*"],
    )
