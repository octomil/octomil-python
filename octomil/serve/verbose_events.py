"""Verbose runtime event emitter for ``octomil serve -v``.

When verbose mode is enabled, structured runtime events are:
1. Logged to the Python logger at INFO level (always)
2. Stored in an in-memory ring buffer for the local debug endpoint
3. Posted to the server as LogEntry records (when api_key is configured)

All server posting is best-effort and never blocks inference.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeEvent:
    """A single structured runtime event."""

    event_name: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerboseEventEmitter:
    """Collects and dispatches verbose runtime events.

    Parameters
    ----------
    api_key:
        Bearer token for posting events to the server.  ``None`` means
        events are only logged locally.
    api_base:
        Base URL of the Octomil API (e.g. ``https://api.octomil.com/api/v1``).
    device_id:
        Stable device identifier for the serve instance.
    model_name:
        Name of the model being served (included in log metadata).
    buffer_size:
        Maximum number of events kept in the in-memory ring buffer.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: str = "https://api.octomil.com/api/v1",
        device_id: str = "",
        model_name: str = "",
        org_id: str = "default",
        buffer_size: int = 500,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._device_id = device_id
        self._model_name = model_name
        self._org_id = org_id
        self._buffer: deque[RuntimeEvent] = deque(maxlen=buffer_size)

        # Background poster (only when api_key is configured)
        self._post_queue: queue.Queue[Optional[RuntimeEvent]] = queue.Queue(maxsize=512)
        self._poster: Optional[threading.Thread] = None
        if api_key:
            self._poster = threading.Thread(target=self._post_loop, daemon=True)
            self._poster.start()

    def emit(self, event_name: str, **metadata: Any) -> None:
        """Emit a structured runtime event.

        The event is logged locally and, if connected, queued for server posting.
        """
        event = RuntimeEvent(event_name=event_name, metadata=metadata)
        self._buffer.append(event)

        # Always log locally
        logger.info("[verbose] %s %s", event_name, metadata)

        # Queue for server posting (non-blocking, drop on full)
        if self._poster is not None:
            try:
                self._post_queue.put_nowait(event)
            except queue.Full:
                pass

    def recent_events(self) -> list[dict[str, Any]]:
        """Return recent events from the ring buffer."""
        return [e.to_dict() for e in self._buffer]

    def close(self) -> None:
        """Drain the post queue and stop the background thread."""
        if self._poster is not None:
            self._post_queue.put(None)  # sentinel
            self._poster.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Background poster
    # ------------------------------------------------------------------

    def _post_loop(self) -> None:
        """Background thread: batch-post runtime events to the server."""
        import httpx

        client = httpx.Client(timeout=5.0)
        url = f"{self._api_base}/log-entries"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            while True:
                event = self._post_queue.get()
                if event is None:
                    # Drain remaining events
                    batch = self._drain_remaining()
                    if batch:
                        self._send_batch(client, url, headers, batch)
                    break
                # Try to batch: grab more events if available
                batch = [event]
                while len(batch) < 20 and not self._post_queue.empty():
                    item = self._post_queue.get_nowait()
                    if item is None:
                        self._send_batch(client, url, headers, batch)
                        return
                    batch.append(item)
                self._send_batch(client, url, headers, batch)
        finally:
            client.close()

    def _drain_remaining(self) -> list[RuntimeEvent]:
        """Pull all remaining non-sentinel events from the queue."""
        batch: list[RuntimeEvent] = []
        while not self._post_queue.empty():
            item = self._post_queue.get_nowait()
            if item is not None:
                batch.append(item)
        return batch

    def _send_batch(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        events: list[RuntimeEvent],
    ) -> None:
        """POST a batch of events to the server — best-effort, never raises."""
        payload = {
            "device_id": self._device_id,
            "org_id": self._org_id,
            "model_id": self._model_name,
            "entries": [
                {
                    "service": f"runtime.{e.event_name}",
                    "level": "info",
                    "message": e.event_name,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for e in events
            ],
        }
        try:
            client.post(url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Verbose event post failed: %s", exc)
