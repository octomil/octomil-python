"""Invisible local runner server process.

Wraps the existing ``octomil serve`` app with:
- Bearer token authentication (from a file)
- Idle timeout watchdog that auto-shuts down
- ``/shutdown`` endpoint for explicit teardown
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def generate_token(token_file: Path) -> str:
    """Generate a secure bearer token and write it to a file with 0600 permissions."""
    token = secrets.token_urlsafe(48)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    return token


def _read_token(token_file: Path) -> str:
    """Read the bearer token from the token file."""
    return token_file.read_text().strip()


# ---------------------------------------------------------------------------
# Idle timeout watchdog
# ---------------------------------------------------------------------------


class _IdleWatchdog:
    """Background thread that kills the process after idle_timeout seconds."""

    def __init__(self, idle_timeout: int) -> None:
        self._idle_timeout = idle_timeout
        self._last_activity = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def touch(self) -> None:
        """Record activity (called on each request)."""
        self._last_activity = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="idle-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            idle = time.monotonic() - self._last_activity
            if idle >= self._idle_timeout:
                logger.info(
                    "Idle timeout reached (%.0fs >= %ds). Shutting down.",
                    idle,
                    self._idle_timeout,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
            # Check every 10 seconds or remaining time, whichever is shorter
            remaining = self._idle_timeout - idle
            self._stop_event.wait(timeout=min(10.0, max(remaining, 1.0)))


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def run_local_runner(
    *,
    model: str,
    engine: str | None = None,
    port: int,
    token_file: Path,
    idle_timeout: int = 1800,
) -> None:
    """Start the local runner server. Blocks until shutdown.

    This creates a FastAPI app wrapping the existing ``octomil serve``
    infrastructure, adds token auth and idle watchdog, then runs uvicorn.
    """
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from ..serve.app import create_app

    # Read the pre-generated token
    token = _read_token(token_file)

    # Create the inner serve app with minimal config
    inner_app = create_app(
        model,
        engine=engine,
        cache_enabled=True,
        max_queue_depth=0,  # no queueing for single-user runner
    )

    # Create the outer wrapper app
    app = FastAPI(title="Octomil Local Runner", version="1.0.0")

    # Idle watchdog
    watchdog = _IdleWatchdog(idle_timeout)

    # -----------------------------------------------------------------------
    # Token auth middleware
    # -----------------------------------------------------------------------

    _PUBLIC_PATHS = {"/health", "/docs", "/openapi.json"}

    @app.middleware("http")
    async def token_auth_middleware(request: Request, call_next: Any) -> Any:
        path = request.url.path

        # Public endpoints: no auth
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Check bearer token
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != token:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

        # Record activity for idle watchdog
        watchdog.touch()
        return await call_next(request)

    # -----------------------------------------------------------------------
    # Shutdown endpoint
    # -----------------------------------------------------------------------

    @app.post("/shutdown")
    async def shutdown_endpoint() -> dict[str, str]:
        """Gracefully shut down the runner."""
        logger.info("Shutdown requested via /shutdown endpoint.")

        # Schedule shutdown after response is sent
        def _deferred_shutdown() -> None:
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_deferred_shutdown, daemon=True).start()
        return {"status": "shutting_down"}

    # -----------------------------------------------------------------------
    # Health (override, no auth needed)
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": model, "engine": engine or "auto"}

    # -----------------------------------------------------------------------
    # Mount the inner serve app for inference endpoints
    # -----------------------------------------------------------------------

    app.mount("/", inner_app)

    # Start watchdog
    watchdog.start()

    logger.info(
        "Starting local runner: model=%s engine=%s port=%d idle_timeout=%ds",
        model,
        engine or "auto",
        port,
        idle_timeout,
    )

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    finally:
        watchdog.stop()
