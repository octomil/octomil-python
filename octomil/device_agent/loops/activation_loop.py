"""Activation loop — decides when a staged model becomes active.

Respects in-flight inference sessions by waiting for the old version's
refcount to reach zero before completing the transition.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..activation_manager import ActivationManager
from ..inference_session_manager import InferenceSessionManager
from ..model_registry import DeviceModelRegistry
from ..policy.policy_engine import PolicyEngine
from ..telemetry.telemetry_store import TelemetryStore

logger = logging.getLogger(__name__)


class ActivationLoop:
    """Background loop that monitors staged artifacts and activates them.

    Periodically checks for STAGED artifacts. When found and policy allows,
    performs warmup, activates the new version, and drains the old version
    by waiting for its refcount to reach zero.

    Respects the per-artifact ``activation_policy``:
    - ``immediate`` — activate as soon as artifact is verified/staged
    - ``next_launch`` — mark pending, activate on next DeviceAgent.start()
    - ``manual`` — stage only, wait for explicit activate() call
    - ``when_idle`` — activate when no inference sessions are active
    """

    def __init__(
        self,
        model_registry: DeviceModelRegistry,
        activation_manager: ActivationManager,
        session_manager: InferenceSessionManager,
        policy_engine: PolicyEngine,
        telemetry_store: TelemetryStore,
        *,
        check_interval: float = 30.0,
        drain_timeout: float = 300.0,
        is_startup: bool = False,
    ) -> None:
        self._model_registry = model_registry
        self._activation_manager = activation_manager
        self._session_manager = session_manager
        self._policy_engine = policy_engine
        self._telemetry_store = telemetry_store
        self._check_interval = check_interval
        self._drain_timeout = drain_timeout
        self._is_startup = is_startup
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the activation loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="activation-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the activation loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Monitors staged artifacts and manages activation."""
        logger.info("Activation loop started")
        while not self._stop_event.is_set():
            try:
                self._check_staged()
            except Exception:
                logger.warning("Activation loop iteration failed", exc_info=True)
            self._stop_event.wait(timeout=self._check_interval)
        logger.info("Activation loop stopped")

    def mark_startup_complete(self) -> None:
        """Signal that startup phase is done.

        After this call, ``next_launch`` artifacts will no longer be
        auto-activated until the next restart.
        """
        self._is_startup = False

    def _check_staged(self) -> None:
        """Find STAGED artifacts and attempt warmup + activation when policy allows.

        Queries all distinct model_ids that have STAGED artifacts, then for each
        model, picks the latest staged version and attempts activation.
        Respects the artifact's activation_policy.
        """
        db = self._model_registry._db
        rows = db.execute("SELECT DISTINCT model_id FROM model_artifacts WHERE status = 'STAGED'")

        for row in rows:
            if self._stop_event.is_set():
                break

            model_id = row["model_id"]
            staged = self._model_registry.get_staged_versions(model_id)
            if not staged:
                continue

            # Pick the latest staged version
            latest = staged[0]
            artifact_id = latest["artifact_id"]
            new_version = latest["version"]

            # Check activation policy
            if not self._should_activate(artifact_id):
                continue

            # Check warmup policy
            allowed, reason = self._policy_engine.should_allow_warmup()
            if not allowed:
                logger.debug("Warmup blocked for %s@%s: %s", model_id, new_version, reason)
                continue

            # Attempt warmup
            logger.info("Attempting warmup for %s@%s", model_id, new_version)
            warmup_ok = self._activation_manager.warmup(artifact_id)

            if not warmup_ok:
                logger.warning("Warmup failed for %s@%s, triggering rollback", model_id, new_version)
                rolled_back = self._activation_manager.auto_rollback(model_id, "warmup_failed")
                self._telemetry_store.append_auto(
                    "artifact.download.failed",
                    {
                        "artifact_id": artifact_id,
                        "error": "warmup_failed",
                        "rolled_back_to": rolled_back,
                    },
                    model_id=model_id,
                    model_version=new_version,
                )
                continue

            # Get the current active version before flipping the pointer
            current_active = self._model_registry.get_active_model(model_id)
            old_version = current_active["active_version"] if current_active else None

            # Flip the active pointer to the new version
            self._activation_manager.activate(model_id, new_version)
            logger.info("Activated %s@%s", model_id, new_version)

            self._telemetry_store.append_auto(
                "artifact.activated",
                {
                    "artifact_id": artifact_id,
                    "old_version": old_version,
                },
                model_id=model_id,
                model_version=new_version,
            )

            # Drain old version if there was one
            if old_version and old_version != new_version:
                self._drain_old_version(model_id, old_version)

    def _should_activate(self, artifact_id: str) -> bool:
        """Check whether the artifact should be activated based on its policy.

        Returns True if activation should proceed now, False to skip.
        """
        db = self._model_registry._db
        row = db.execute_one(
            "SELECT activation_policy FROM model_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if row is None:
            return True

        policy = row["activation_policy"]

        if policy == "immediate":
            return True
        elif policy == "next_launch":
            # Only activate during the startup phase
            return self._is_startup
        elif policy == "manual":
            # Never auto-activate — wait for explicit activate() call
            return False
        elif policy == "when_idle":
            # Activate only when no inference sessions are active
            active_sessions = self._session_manager.get_active_sessions()
            return len(active_sessions) == 0
        else:
            logger.warning(
                "Unknown activation_policy '%s' for artifact %s, defaulting to immediate", policy, artifact_id
            )
            return True

    def _drain_old_version(self, model_id: str, old_version: str) -> None:
        """Wait for refcount=0 on old_version via session_manager, then unload.

        Blocks up to drain_timeout seconds. If the old version cannot be
        drained in time, logs a warning but does not force-kill sessions.
        """
        logger.info(
            "Draining old version %s@%s (timeout=%ss)",
            model_id,
            old_version,
            self._drain_timeout,
        )

        drained = self._activation_manager.drain_old(
            model_id,
            old_version,
            timeout_sec=self._drain_timeout,
            refcount_fn=self._session_manager.get_refcount,
        )

        if drained:
            logger.info("Old version %s@%s fully drained", model_id, old_version)
        else:
            logger.warning(
                "Drain timeout for %s@%s, %d sessions still active",
                model_id,
                old_version,
                self._session_manager.get_refcount(model_id, old_version),
            )
