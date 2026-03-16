"""Background telemetry uploader — batches and POSTs events to the platform."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Optional

import httpx

from .telemetry_store import TelemetryStore

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100
_DEFAULT_MAX_BATCH_BYTES = 262_144  # 256 KB
_DEFAULT_MAX_AGE_S = 60.0
_CELLULAR_MAX_BATCH_BYTES = 262_144  # 256 KB
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 300.0


class TelemetryUploader:
    """Daemon thread that batches unsent telemetry events and uploads them.

    Upload is best-effort: failures trigger exponential backoff but never
    raise to the caller.
    """

    def __init__(
        self,
        store: TelemetryStore,
        device_id: str,
        boot_id: str,
        api_base: str,
        api_key: str,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_age_s: float = _DEFAULT_MAX_AGE_S,
    ) -> None:
        self._store = store
        self._device_id = device_id
        self._boot_id = boot_id
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._batch_size = batch_size
        self._max_batch_bytes = max_batch_bytes
        self._max_age_s = max_age_s

        self._backoff_s = _BACKOFF_BASE_S
        self._policy_config: dict[str, Any] = {}

        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background upload daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry-uploader")
        self._thread.start()

    def stop(self) -> None:
        """Stop gracefully — flush remaining events then exit."""
        self._stop_event.set()
        self._flush_event.set()  # wake the thread
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def flush(self) -> None:
        """Signal the uploader to perform an immediate upload attempt."""
        self._flush_event.set()

    def set_policy(self, policy_config: dict[str, Any]) -> None:
        """Update upload policy from PolicyEngine.

        Accepted keys:
        - max_batch_size: int
        - min_interval: float (seconds)
        - allowed_classes: list[str]
        - network_type: str
        """
        self._policy_config = policy_config
        if "max_batch_size" in policy_config:
            self._batch_size = policy_config["max_batch_size"]
        if "min_interval" in policy_config:
            self._max_age_s = policy_config["min_interval"]
        if policy_config.get("network_type") == "cellular":
            self._max_batch_bytes = _CELLULAR_MAX_BATCH_BYTES

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop — poll store, batch, upload, back off on failure."""
        while not self._stop_event.is_set():
            try:
                self._upload_batch()
                self._backoff_s = _BACKOFF_BASE_S  # reset on success
            except Exception:
                logger.warning("Telemetry upload failed", exc_info=True)
                self._backoff_s = min(self._backoff_s * 2, _BACKOFF_MAX_S)

            # Wait for next cycle or flush signal
            self._flush_event.wait(timeout=self._max_age_s)
            self._flush_event.clear()

        # Final drain on shutdown
        try:
            self._upload_batch()
        except Exception:
            logger.warning("Final telemetry flush failed", exc_info=True)

    def _upload_batch(self) -> None:
        """Fetch unsent events, batch them, and POST to the platform."""
        events = self._store.get_unsent(batch_size=self._batch_size)
        if not events:
            return

        # Trim batch by byte size
        trimmed = self._trim_to_bytes(events)
        if not trimmed:
            return

        batch_id = uuid.uuid4().hex
        first_seq = trimmed[0]["sequence_no"]
        last_seq = trimmed[-1]["sequence_no"]

        envelope = {
            "device_id": self._device_id,
            "boot_id": self._boot_id,
            "batch_id": batch_id,
            "first_seq": first_seq,
            "last_seq": last_seq,
            "events": trimmed,
        }

        url = f"{self._api_base}/api/v1/telemetry/batches"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=envelope, headers=headers)
                resp.raise_for_status()

            body = resp.json()
            acked_seq = body.get("ackedThroughSeq")

            event_ids = [e["event_id"] for e in trimmed]
            self._store.mark_sent(event_ids, batch_id)

            if acked_seq is not None:
                self._store.mark_acked(self._boot_id, acked_seq)

            logger.debug(
                "Uploaded telemetry batch %s (%d events, seq %d..%d)",
                batch_id,
                len(trimmed),
                first_seq,
                last_seq,
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("Telemetry upload HTTP %d: %s", exc.response.status_code, exc)
            time.sleep(self._backoff_s)
            raise
        except Exception:
            time.sleep(self._backoff_s)
            raise

    def _trim_to_bytes(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim the event list so serialised size stays under max_batch_bytes."""
        result: list[dict[str, Any]] = []
        total = 0
        for event in events:
            size = len(json.dumps(event, separators=(",", ":")))
            if total + size > self._max_batch_bytes and result:
                break
            result.append(event)
            total += size
        return result
