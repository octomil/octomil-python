"""SDK-side telemetry agent (MVP).

Buffers sanitized telemetry events in memory, flushes them in batches to
the server's telemetry-ingest endpoint on a timer or after N events. Failure
to upload is swallowed and logged — the agent never raises into the
caller's inference path. Disk persistence is intentionally out of scope
for the MVP.

Key invariants:

- Inference must never break because of telemetry. All exceptions are
  caught and logged.
- No prompts, completions, audio, or file paths. The existing
  ``validate_telemetry_safety`` from ``telemetry.py`` runs on every
  payload before it leaves the agent.
- Disabling the agent (``OCTOMIL_TELEMETRY_DISABLED=1`` or constructor
  ``enabled=False``) makes ``record`` a no-op and prevents any background
  worker from starting.

This is the MVP — disk persistence and offline-replay queues land in a
follow-up.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional

from .telemetry import FORBIDDEN_TELEMETRY_KEYS, validate_telemetry_safety

logger = logging.getLogger(__name__)


_DEFAULT_FLUSH_INTERVAL_S = 5.0
_DEFAULT_FLUSH_THRESHOLD = 64
_DEFAULT_QUEUE_CAPACITY = 4096
_ENV_DISABLE = "OCTOMIL_TELEMETRY_DISABLED"


Sender = Callable[[list[dict[str, Any]]], Awaitable[None]]


@dataclass
class TelemetryAgentConfig:
    enabled: bool = True
    flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S
    flush_threshold: int = _DEFAULT_FLUSH_THRESHOLD
    queue_capacity: int = _DEFAULT_QUEUE_CAPACITY


class TelemetryAgent:
    """In-memory telemetry buffer with periodic flush.

    Thread-safe: ``record`` may be called from any thread / coroutine.
    The background flush is driven by an asyncio task; ``flush()`` may
    also be invoked manually from synchronous or async contexts.
    """

    def __init__(
        self,
        sender: Sender,
        *,
        config: Optional[TelemetryAgentConfig] = None,
    ) -> None:
        self._sender = sender
        cfg = config or TelemetryAgentConfig()
        if os.environ.get(_ENV_DISABLE) in {"1", "true", "True"}:
            cfg = TelemetryAgentConfig(
                enabled=False,
                flush_interval_s=cfg.flush_interval_s,
                flush_threshold=cfg.flush_threshold,
                queue_capacity=cfg.queue_capacity,
            )
        self._config = cfg
        self._queue: deque[dict[str, Any]] = deque(maxlen=cfg.queue_capacity)
        self._capacity: int = cfg.queue_capacity
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def dropped(self) -> int:
        return self._dropped

    # -- recording ----------------------------------------------------------

    def record(self, event: dict[str, Any]) -> None:
        """Enqueue a sanitized telemetry event.

        Forbidden keys cause the event to be dropped silently with a debug
        log — the caller's inference path must not depend on telemetry
        validity.
        """
        if not self._config.enabled:
            return
        try:
            sanitized = _sanitize(event)
            validate_telemetry_safety(sanitized)
        except Exception:
            self._dropped += 1
            logger.debug("dropping unsafe telemetry event", exc_info=True)
            return

        with self._lock:
            if len(self._queue) >= self._capacity:
                self._dropped += 1
            self._queue.append(sanitized)
            should_flush = len(self._queue) >= self._config.flush_threshold

        if should_flush:
            self._schedule_flush()

    # -- background loop ----------------------------------------------------

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the periodic flush task. No-op if disabled or already running."""
        if not self._config.enabled or self._task is not None:
            return
        loop = loop or asyncio.get_event_loop()
        self._stop.clear()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        """Stop the periodic flush task and flush any remaining events."""
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                logger.debug("telemetry agent task ended with exception", exc_info=True)
            self._task = None
        await self.flush()

    async def _run(self) -> None:
        interval = self._config.flush_interval_s
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            except Exception:
                logger.debug("telemetry loop wait failed", exc_info=True)
            await self.flush()

    # -- flushing -----------------------------------------------------------

    def _schedule_flush(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.flush())
        except RuntimeError:
            # No running loop in this thread — caller will flush manually.
            pass

    async def flush(self) -> None:
        """Send any pending events. Failures are swallowed."""
        if not self._config.enabled:
            return
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
        try:
            await self._sender(batch)
        except Exception:
            # Non-fatal: telemetry must never break inference. Re-queue
            # the batch only if there's room — drop oldest otherwise.
            logger.debug("telemetry flush failed", exc_info=True)
            with self._lock:
                for ev in batch:
                    if len(self._queue) >= self._capacity:
                        self._dropped += 1
                        self._queue.popleft()
                    self._queue.append(ev)

    def flush_sync(self) -> None:
        """Synchronous flush — runs the coroutine in a fresh loop if needed."""
        if not self._config.enabled:
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self.flush(), loop).result(timeout=10.0)
        else:
            loop.run_until_complete(self.flush())


# --- helpers ----------------------------------------------------------------


def _is_forbidden_key(key: Any) -> bool:
    """Case-insensitive forbidden-key check.

    Critical: matches what the server's telemetry_contract enforces, so
    the SDK never even SENDS an 'Authorization' header value that the
    server would reject. Privacy guarantee starts on the device.
    """
    return isinstance(key, str) and key.lower() in FORBIDDEN_TELEMETRY_KEYS


def _sanitize(event: dict[str, Any]) -> dict[str, Any]:
    """Drop forbidden keys at every depth before validation kicks in."""
    out: dict[str, Any] = {}
    for k, v in event.items():
        if not isinstance(k, str):
            continue
        if _is_forbidden_key(k):
            continue
        out[k] = _sanitize_value(v)
    out.setdefault("emitted_at", time.time())
    return out


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items() if isinstance(k, str) and not _is_forbidden_key(k)}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    return value


def assert_no_forbidden_keys(events: Iterable[dict[str, Any]]) -> None:
    """Test helper: raises if any event would carry a forbidden key."""
    for ev in events:
        validate_telemetry_safety(ev)
