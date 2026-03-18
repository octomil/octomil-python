"""Control plane -- device registration and heartbeat (SDK Facade Contract namespace).

**Tier: Core Contract (MUST)**

This module is part of the core SDK facade that every Octomil SDK must
implement.  See SDK_FACADE_CONTRACT.md for details.
"""

from __future__ import annotations

import logging
import platform
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ._generated import span_attributes as _span_attrs
from ._generated import span_names as _span_names
from .device_info import DeviceInfo
from .python.octomil.api_client import _ApiClient

logger = logging.getLogger(__name__)


@dataclass
class ControlSyncResult:
    """Result returned by ``control.refresh()``.

    Communicates whether the server had updated configuration or
    assignment changes since the last fetch.
    """

    updated: bool
    config_version: str
    assignments_changed: bool
    rollouts_changed: bool
    fetched_at: str  # ISO-8601 timestamp


@dataclass
class DeviceRegistration:
    """Response from device registration."""

    id: str
    device_identifier: str
    org_id: str
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatResponse:
    """Response from heartbeat."""

    status: str = "ok"
    server_time: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OctomilControl:
    """Device registration and heartbeat management.

    Matches the ``control`` namespace in SDK_FACADE_CONTRACT.md:
    - refresh() -> void
    - register(deviceId?) -> DeviceRegistration
    - heartbeat() -> HeartbeatResponse
    """

    def __init__(self, api: _ApiClient, org_id: str, telemetry: Any = None) -> None:
        self._api = api
        self._org_id = org_id
        self._device_info = DeviceInfo()
        self._server_device_id: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop: threading.Event = threading.Event()
        self._telemetry = telemetry
        self._heartbeat_sequence: int = 0

    def refresh(self) -> ControlSyncResult:
        """Fetch latest assignments and rollout state from server.

        Returns a ``ControlSyncResult`` describing whether configuration,
        assignments, or rollouts have changed since the previous fetch.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if not self._server_device_id:
            return ControlSyncResult(
                updated=False,
                config_version="",
                assignments_changed=False,
                rollouts_changed=False,
                fetched_at=now,
            )

        data = self._api.get(f"/devices/{self._server_device_id}/assignments")
        data = data if isinstance(data, dict) else {}

        return ControlSyncResult(
            updated=data.get("updated", True),
            config_version=str(data.get("config_version", "")),
            assignments_changed=data.get("assignments_changed", bool(data.get("assignments"))),
            rollouts_changed=data.get("rollouts_changed", bool(data.get("rollouts"))),
            fetched_at=now,
        )

    def register(self, device_id: Optional[str] = None) -> DeviceRegistration:
        """Register device with server. Returns DeviceRegistration."""
        effective_device_id = device_id or self._device_info.device_id

        payload = self._device_info.to_registration_dict()
        payload["device_identifier"] = effective_device_id
        payload["org_id"] = self._org_id
        payload["sdk_version"] = _get_sdk_version()

        data = self._api.post("/devices/register", payload)

        self._server_device_id = data.get("id", "")
        return DeviceRegistration(
            id=data.get("id", ""),
            device_identifier=effective_device_id,
            org_id=self._org_id,
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
        )

    def heartbeat(self) -> HeartbeatResponse:
        """Send device heartbeat to server."""
        if not self._server_device_id:
            raise RuntimeError("Device not registered. Call register() first.")

        # Emit octomil.control.heartbeat telemetry span (GAP-12)
        seq = self._heartbeat_sequence
        self._heartbeat_sequence += 1
        if self._telemetry is not None:
            self._telemetry._enqueue(
                name=_span_names.OCTOMIL_CONTROL_HEARTBEAT,
                attributes={_span_attrs.HEARTBEAT_SEQUENCE: seq},
            )

        payload: dict[str, Any] = {
            "sdk_version": _get_sdk_version(),
            "os_version": platform.platform(),
            "platform": "python",
        }

        # Merge runtime metadata (battery, network) -- best-effort
        try:
            payload["metadata"] = self._device_info.update_metadata()
        except Exception:
            pass

        data = self._api.post(
            f"/devices/{self._server_device_id}/heartbeat",
            payload,
        )

        return HeartbeatResponse(
            status=data.get("status", "ok"),
            server_time=data.get("server_time"),
            metadata=data.get("metadata", {}),
        )

    def start_heartbeat(self, interval_seconds: float = 300.0) -> None:
        """Start automatic heartbeat in background thread."""
        self.stop_heartbeat()
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_seconds,),
            daemon=True,
            name="octomil-heartbeat",
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop automatic heartbeat."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None

    def get_desired_state(self) -> list[dict[str, Any]]:
        """Return desired state as a list of per-model entries.

        Adapter method that makes ``OctomilControl`` compatible with the
        ``server_client`` interface expected by ``ArtifactLoop``.  Calls
        ``fetch_desired_state()`` and extracts the ``models`` array
        (DesiredModelEntry), flattening each entry with its nested
        ``artifactManifest`` into the flat dict format the loop expects.

        Uses the canonical contract shape only — no backwards-compat
        fallbacks for the old ``artifacts`` array.
        """
        raw = self.fetch_desired_state()
        models = raw.get("models", [])
        if not isinstance(models, list):
            return []
        result: list[dict[str, Any]] = []
        for entry in models:
            manifest = entry.get("artifactManifest", {})
            mapped: dict[str, Any] = {
                "model_id": entry["modelId"],
                "version": entry["desiredVersion"],
                "artifact_id": manifest.get("artifactId", ""),
                "manifest": manifest,
                "total_bytes": manifest.get("totalBytes", 0),
                "activation_policy": entry.get("activationPolicy", "immediate"),
            }
            result.append(mapped)
        return result

    @property
    def base_url(self) -> str:
        """Expose the API base URL so ArtifactLoop can build download URLs."""
        return getattr(self._api, "base_url", "")

    def fetch_desired_state(
        self,
        device_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """GET desired state for this device from the server.

        Conforms to the ``DesiredState`` contract.  Returns the target state
        the device should converge toward: per-model entries with delivery
        mode and activation policy, policy config, federation offers, and
        GC-eligible artifact IDs.

        Args:
            device_id: Explicit device id override.  Falls back to the
                server-assigned id from ``register()``.

        Returns:
            Desired state dict containing ``models``, ``policyConfig``,
            ``federationOffers``, ``gcEligibleArtifactIds``, etc.

        Raises:
            RuntimeError: If no device id is available (not registered).
        """
        effective_id = device_id or self._server_device_id
        if not effective_id:
            raise RuntimeError("Device not registered. Call register() first.")

        return self._api.get(f"/devices/{effective_id}/desired-state")

    def report_observed_state(
        self,
        device_id: Optional[str] = None,
        models: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """POST observed device state to the server.

        Conforms to the ``ObservedState`` contract.  Reports per-model
        observed state (installed version, active version, status, health)
        so the server can reconcile desired vs observed state.

        Args:
            device_id: Explicit device id override.  Falls back to the
                server-assigned id from ``register()``.
            models: List of per-model observed state dicts, each
                containing at minimum ``modelId`` and ``status``.

        Returns:
            Server acknowledgement dict.

        Raises:
            RuntimeError: If no device id is available (not registered).
        """
        effective_id = device_id or self._server_device_id
        if not effective_id:
            raise RuntimeError("Device not registered. Call register() first.")

        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "schemaVersion": "1.4.0",
            "deviceId": effective_id,
            "reportedAt": now,
            "models": models or [],
            "sdkVersion": _get_sdk_version(),
            "osVersion": platform.platform(),
        }

        return self._api.post(f"/devices/{effective_id}/observed-state", payload)

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(timeout=interval):
            try:
                self.heartbeat()
            except Exception:
                logger.debug("Heartbeat failed", exc_info=True)


def _get_sdk_version() -> str:
    try:
        from octomil import __version__

        return __version__
    except ImportError:
        return "0.0.0"
