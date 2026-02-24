"""Anonymous platform ping reporter for ``octomil serve``.

Reports aggregate inference counts to the public platform stats endpoint
so the landing page can display live usage counters.  No API key required.

All reporting is best-effort and fire-and-forget.  The reporter batches
inference counts and POSTs them every 60 seconds on a background thread.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import threading
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_PING_URL = "https://api.edgeml.io/api/v1/platform/ping"


def _generate_device_id() -> str:
    """Derive a stable anonymous device ID from hostname + MAC address."""
    hostname = platform.node()
    try:
        mac = uuid.getnode()
        raw = f"{hostname}:{mac}"
    except Exception:
        raw = hostname
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class AnonymousPingReporter:
    """Fire-and-forget reporter that pings platform stats with inference counts.

    Batches inference events and flushes every ``flush_interval`` seconds.
    Uses a daemon thread so the process exits cleanly without waiting.

    Parameters
    ----------
    ping_url:
        Full URL of the ``POST /api/v1/platform/ping`` endpoint.
    device_id:
        Anonymous device identifier.  Derived from hostname + MAC when ``None``.
    flush_interval:
        Seconds between flush attempts (default 60).
    """

    def __init__(
        self,
        ping_url: str = _DEFAULT_PING_URL,
        device_id: Optional[str] = None,
        flush_interval: int = 60,
    ) -> None:
        self._ping_url = ping_url
        self._device_id = device_id or _generate_device_id()
        self._flush_interval = flush_interval
        self._count = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._flush_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_inference(self, count: int = 1) -> None:
        """Record one or more inference completions (thread-safe)."""
        with self._lock:
            self._count += count

    def close(self) -> None:
        """Flush remaining counts and stop the background thread."""
        self._stop_event.set()
        self._flush()  # final drain
        self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Send accumulated count to the platform ping endpoint."""
        with self._lock:
            count = self._count
            self._count = 0
        if count <= 0:
            return
        try:
            httpx.post(
                self._ping_url,
                json={
                    "device_id": self._device_id,
                    "event": "inference",
                    "count": count,
                },
                timeout=5.0,
            )
        except Exception as exc:
            logger.debug("Anonymous ping failed: %s", exc)

    def _flush_loop(self) -> None:
        """Background thread: flush at regular intervals until stopped."""
        while not self._stop_event.wait(timeout=self._flush_interval):
            self._flush()
