"""Telemetry namespace -- ``client.telemetry.flush()`` / ``client.telemetry.track()``.

**Tier: Core Contract (MUST)**

Wraps the internal ``TelemetryReporter`` behind a public-facing
``client.telemetry`` sub-API::

    client.telemetry.track("user.action", {"key": "value"})
    client.telemetry.flush()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import OctomilClient

logger = logging.getLogger(__name__)


class TelemetryClient:
    """Public telemetry namespace.

    Provides ``track()`` for custom events and ``flush()`` to drain
    the internal reporter queue.
    """

    def __init__(self, client: OctomilClient) -> None:
        self._client = client

    def flush(self) -> None:
        """Drain all pending telemetry events.

        Signals the background dispatch thread to send any buffered
        events, then waits for it to complete (up to 5 seconds).
        If no reporter is initialised this is a no-op.
        """
        reporter = self._client._reporter
        if reporter is None:
            return
        try:
            reporter.close()
        except Exception:
            logger.debug("Telemetry flush failed", exc_info=True)

        # Re-create the reporter so subsequent track() calls still work.
        try:
            from .telemetry import TelemetryReporter

            self._client._reporter = TelemetryReporter(
                api_key=self._client._api_key,
                api_base=self._client._api_base,
                org_id=self._client._org_id,
            )
        except Exception:
            logger.debug("Failed to re-initialise telemetry reporter after flush", exc_info=True)
            self._client._reporter = None

    def track(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Emit a custom telemetry event.

        The event is enqueued for best-effort delivery on the
        background thread.  Never raises.

        Args:
            name: Event name (e.g. ``"user.login"``).
            attributes: Key-value pairs attached to the event.
        """
        reporter = self._client._reporter
        if reporter is None:
            return
        try:
            reporter._enqueue(name=name, attributes=attributes or {})
        except Exception:
            logger.debug("Telemetry track(%s) failed", name, exc_info=True)
