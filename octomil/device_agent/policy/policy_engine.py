"""Centralized policy decision layer for the device agent.

Consulted by the updater, telemetry uploader, trainer, and federated
client to determine whether resource-consuming operations are allowed
given current device conditions.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Work classification
# ---------------------------------------------------------------------------


class WorkClass(str, enum.Enum):
    """Classification of work items for scheduling priority."""

    CRITICAL_FOREGROUND = "CRITICAL_FOREGROUND"
    BACKGROUND_IMPORTANT = "BACKGROUND_IMPORTANT"
    BACKGROUND_BEST_EFFORT = "BACKGROUND_BEST_EFFORT"


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """Tunable knobs for the policy engine.

    Defaults are conservative — suitable for a typical mobile device.
    """

    min_battery_for_background: int = 15
    min_battery_for_training: int = 50
    require_charging_for_training: bool = True
    allowed_download_networks: list[str] = field(default_factory=lambda: ["wifi", "ethernet"])
    max_cellular_download_bytes: int = 20_000_000  # 20 MB
    max_cellular_telemetry_bytes: int = 262_144  # 256 KB
    reserve_storage_bytes: int = 1_073_741_824  # 1 GB
    training_pause_on_foreground: bool = True
    federated_require_idle: bool = True
    federated_max_examples: int = 2000
    federated_max_steps: int = 100
    federated_max_update_bytes: int = 52_428_800  # 50 MB
    federated_daily_budget_eps: float = 2.0


# ---------------------------------------------------------------------------
# Internal device state snapshot
# ---------------------------------------------------------------------------


@dataclass
class _DeviceState:
    battery_pct: int = 100
    is_charging: bool = False
    network_type: str = "unknown"
    thermal_state: str = "nominal"
    free_storage_bytes: int = 10_000_000_000  # 10 GB default
    is_foreground: bool = False


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Resource-aware policy gate consulted before work is started."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()
        self._state = _DeviceState()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_device_state(
        self,
        battery_pct: int | None = None,
        is_charging: bool | None = None,
        network_type: str | None = None,
        thermal_state: str | None = None,
        free_storage_bytes: int | None = None,
        is_foreground: bool | None = None,
    ) -> None:
        """Update internal device state snapshot (thread-safe)."""
        with self._lock:
            if battery_pct is not None:
                self._state.battery_pct = battery_pct
            if is_charging is not None:
                self._state.is_charging = is_charging
            if network_type is not None:
                self._state.network_type = network_type
            if thermal_state is not None:
                self._state.thermal_state = thermal_state
            if free_storage_bytes is not None:
                self._state.free_storage_bytes = free_storage_bytes
            if is_foreground is not None:
                self._state.is_foreground = is_foreground

    def get_device_state(self) -> dict[str, Any]:
        """Return a copy of the current device state."""
        with self._lock:
            return {
                "battery_pct": self._state.battery_pct,
                "is_charging": self._state.is_charging,
                "network_type": self._state.network_type,
                "thermal_state": self._state.thermal_state,
                "free_storage_bytes": self._state.free_storage_bytes,
                "is_foreground": self._state.is_foreground,
            }

    # ------------------------------------------------------------------
    # Download policy
    # ------------------------------------------------------------------

    def should_allow_download(
        self,
        size_bytes: int,
        user_initiated: bool = False,
    ) -> tuple[bool, str]:
        """Decide whether a download of *size_bytes* is permitted."""
        with self._lock:
            s, c = self._state, self._config

            if s.free_storage_bytes < c.reserve_storage_bytes:
                return False, "storage_below_reserve"

            if s.battery_pct < c.min_battery_for_background and not s.is_charging:
                if not user_initiated:
                    return False, "battery_low"

            if s.network_type == "cellular":
                if size_bytes > c.max_cellular_download_bytes and not user_initiated:
                    return False, "cellular_size_limit"
                # Small cellular downloads are allowed — skip general network check
                return True, "ok"

            if s.network_type not in c.allowed_download_networks and not user_initiated:
                return False, "network_not_allowed"

            return True, "ok"

    # ------------------------------------------------------------------
    # Upload policy
    # ------------------------------------------------------------------

    def should_allow_upload(self, size_bytes: int) -> tuple[bool, str]:
        """Decide whether an upload of *size_bytes* is permitted."""
        with self._lock:
            s, c = self._state, self._config

            if s.battery_pct < c.min_battery_for_background and not s.is_charging:
                return False, "battery_low"

            if s.network_type == "cellular" and size_bytes > c.max_cellular_telemetry_bytes:
                return False, "cellular_size_limit"

            return True, "ok"

    # ------------------------------------------------------------------
    # Training policy
    # ------------------------------------------------------------------

    def should_allow_training(self) -> tuple[bool, str]:
        """Decide whether on-device training may proceed."""
        with self._lock:
            s, c = self._state, self._config

            if c.require_charging_for_training and not s.is_charging:
                return False, "not_charging"

            if s.battery_pct < c.min_battery_for_training:
                return False, "battery_low"

            if s.network_type not in c.allowed_download_networks:
                return False, "network_not_allowed"

            if s.thermal_state in ("serious", "critical"):
                return False, "thermal_throttle"

            if c.training_pause_on_foreground and s.is_foreground:
                return False, "foreground_active"

            if s.free_storage_bytes < c.reserve_storage_bytes:
                return False, "storage_below_reserve"

            return True, "ok"

    def should_allow_federated_training(self) -> tuple[bool, str]:
        """Decide whether federated training may proceed."""
        allowed, reason = self.should_allow_training()
        if not allowed:
            return False, reason

        with self._lock:
            c = self._config
            if c.federated_require_idle and self._state.is_foreground:
                return False, "not_idle"

        return True, "ok"

    # ------------------------------------------------------------------
    # Warmup / verification policy
    # ------------------------------------------------------------------

    def should_allow_warmup(self) -> tuple[bool, str]:
        """Decide whether model warmup / verification may proceed."""
        with self._lock:
            s, c = self._state, self._config

            if s.thermal_state in ("serious", "critical"):
                return False, "thermal_throttle"

            if s.battery_pct < c.min_battery_for_background and not s.is_charging:
                return False, "battery_low"

            return True, "ok"

    # ------------------------------------------------------------------
    # Telemetry policy
    # ------------------------------------------------------------------

    def get_telemetry_policy(self) -> dict[str, Any]:
        """Return current telemetry upload parameters."""
        with self._lock:
            s, c = self._state, self._config

            # Under low battery: only MUST_KEEP
            if s.battery_pct < c.min_battery_for_background and not s.is_charging:
                return {
                    "max_batch_size": 10,
                    "min_interval": 300.0,
                    "allowed_classes": ["MUST_KEEP"],
                    "network_type": s.network_type,
                }

            if s.network_type == "cellular":
                return {
                    "max_batch_size": 50,
                    "min_interval": 120.0,
                    "allowed_classes": ["MUST_KEEP", "IMPORTANT"],
                    "network_type": s.network_type,
                }

            return {
                "max_batch_size": 100,
                "min_interval": 60.0,
                "allowed_classes": ["MUST_KEEP", "IMPORTANT", "BEST_EFFORT"],
                "network_type": s.network_type,
            }

    # ------------------------------------------------------------------
    # Bandwidth budgets
    # ------------------------------------------------------------------

    def get_download_budget(self) -> dict[str, Any]:
        """Return download budget parameters."""
        with self._lock:
            if self._state.is_foreground:
                return {"max_mbps": 1.0}
            return {"max_mbps": 0.0}  # unlimited (0 = no cap)

    def get_upload_budget(self) -> dict[str, Any]:
        """Return upload budget parameters."""
        with self._lock:
            if self._state.is_foreground:
                return {"max_kbps": 64.0}
            return {"max_kbps": 512.0}

    # ------------------------------------------------------------------
    # Work classification
    # ------------------------------------------------------------------

    def classify_work(self, work_type: str) -> WorkClass:
        """Classify a work type for scheduling priority."""
        critical = {"inference", "user_request", "rollback", "activation"}
        important = {"download", "verification", "training", "federation"}

        if work_type in critical:
            return WorkClass.CRITICAL_FOREGROUND
        if work_type in important:
            return WorkClass.BACKGROUND_IMPORTANT
        return WorkClass.BACKGROUND_BEST_EFFORT
