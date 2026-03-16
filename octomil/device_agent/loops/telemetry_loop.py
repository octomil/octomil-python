"""Telemetry loop — batches and uploads telemetry events.

Respects upload policy (connectivity, battery, user consent). Never
blocks inference. Delegates actual upload to TelemetryUploader and
manages storage pressure cleanup via PolicyEngine.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..policy.policy_engine import PolicyEngine
from ..telemetry.telemetry_store import TelemetryStore
from ..telemetry.telemetry_uploader import TelemetryUploader

logger = logging.getLogger(__name__)

# Storage pressure thresholds
_STORAGE_PRESSURE_BYTES = 500 * 1024 * 1024  # 500 MB free triggers cleanup
_STORAGE_CLEANUP_TARGET = 50 * 1024 * 1024  # try to free 50 MB of telemetry


class TelemetryLoop:
    """Background loop that batches and uploads telemetry events.

    Delegates to TelemetryUploader for the actual upload mechanism,
    but manages:
    - Policy-gated upload configuration (WiFi only, battery threshold)
    - Storage pressure cleanup when free space is low
    - Periodic policy refresh to adapt to changing device conditions
    """

    def __init__(
        self,
        telemetry_store: TelemetryStore,
        telemetry_uploader: TelemetryUploader,
        policy_engine: PolicyEngine,
        *,
        policy_refresh_interval: float = 60.0,
    ) -> None:
        self._store = telemetry_store
        self._uploader = telemetry_uploader
        self._policy_engine = policy_engine
        self._policy_refresh_interval = policy_refresh_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the telemetry loop and the underlying uploader."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True

        # Apply initial policy before starting upload
        self._refresh_upload_policy()

        # Start the uploader daemon thread
        self._uploader.start()

        # Start our own management thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the telemetry loop to stop and wait for it."""
        self._stop_event.set()

        # Stop the uploader first (it flushes remaining events)
        self._uploader.stop()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Periodically refreshes policy and manages storage pressure."""
        logger.info("Telemetry loop started")
        while not self._stop_event.is_set():
            try:
                self._refresh_upload_policy()
                self._check_storage_pressure()
            except Exception:
                logger.warning("Telemetry loop iteration failed", exc_info=True)

            self._stop_event.wait(timeout=self._policy_refresh_interval)
        logger.info("Telemetry loop stopped")

    def _refresh_upload_policy(self) -> None:
        """Query PolicyEngine for current telemetry upload parameters and push to uploader."""
        policy = self._policy_engine.get_telemetry_policy()
        self._uploader.set_policy(policy)

    def _check_storage_pressure(self) -> None:
        """If free storage is below threshold, aggressively drop low-priority events."""
        state = self._policy_engine.get_device_state()
        free_bytes = state.get("free_storage_bytes", float("inf"))

        if free_bytes < _STORAGE_PRESSURE_BYTES:
            dropped = self._store.storage_pressure_cleanup(_STORAGE_CLEANUP_TARGET)
            if dropped > 0:
                logger.info(
                    "Storage pressure cleanup: dropped %d events (free=%d bytes)",
                    dropped,
                    free_bytes,
                )
