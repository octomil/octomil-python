"""Top-level DeviceAgent entrypoint.

Wires together all subsystems — database, model registry, downloaders,
verifiers, activation, inference sessions, operations, telemetry, policy
— and exposes a simple start/stop/infer surface for callers.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .activation_manager import ActivationManager
from .artifact_downloader import ArtifactDownloader
from .artifact_verifier import ArtifactVerifier
from .crash_detector import CrashDetector
from .db.local_db import LocalDB
from .inference_session_manager import InferenceSessionManager
from .loops.activation_loop import ActivationLoop
from .loops.artifact_loop import ArtifactLoop
from .loops.inference_loop import InferenceLoop, InferenceRequest
from .loops.telemetry_loop import TelemetryLoop
from .model_registry import DeviceModelRegistry
from .operation_scheduler import OperationScheduler
from .policy.policy_engine import PolicyConfig, PolicyEngine
from .telemetry.telemetry_store import TelemetryStore
from .telemetry.telemetry_uploader import TelemetryUploader

logger = logging.getLogger(__name__)


class DeviceAgent:
    """Unified device agent that manages the full on-device model lifecycle.

    Initialises all subsystems and provides a single start/stop interface.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``:memory:`` for testing.
    models_dir:
        Directory root for storing downloaded model artefacts.
    server_base_url:
        Base URL of the Octomil platform API (optional; disables server
        polling when ``None``).
    device_id:
        Unique device identifier.  Auto-generated if not supplied.
    policy_config:
        Override default policy tuning knobs.  Accepts a ``PolicyConfig``
        instance or a dict of keyword arguments for ``PolicyConfig``.
    api_key:
        API key for authenticating with the platform telemetry endpoint.
    inference_fn:
        Optional callable ``(model_id, version, model_path, prompt, **kw)``
        that runs actual inference.  When ``None``, inference returns a stub.
    server_client:
        Optional server client object (must expose ``get_desired_state()``
        returning a list and an optional ``base_url`` attribute).
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        models_dir: str | Path = "/tmp/octomil-models",
        server_base_url: Optional[str] = None,
        device_id: Optional[str] = None,
        policy_config: Optional[PolicyConfig | dict[str, Any]] = None,
        *,
        api_key: str = "",
        inference_fn: Optional[Callable[..., dict[str, Any]]] = None,
        server_client: Any = None,
        control: Any = None,
    ) -> None:
        self._device_id = device_id or uuid.uuid4().hex
        self._boot_id = uuid.uuid4().hex
        self._server_base_url = server_base_url
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # --- Core infrastructure ---
        self._db = LocalDB(db_path)
        self._apply_telemetry_schema()

        # --- Components ---
        self._model_registry = DeviceModelRegistry(self._db, models_dir=self._models_dir)
        self._downloader = ArtifactDownloader(self._db, models_dir=self._models_dir)
        self._verifier = ArtifactVerifier(self._db, models_dir=self._models_dir)
        self._activation_manager = ActivationManager(self._db, self._model_registry)
        self._session_manager = InferenceSessionManager()
        self._scheduler = OperationScheduler(self._db)

        # --- Policy ---
        if isinstance(policy_config, dict):
            self._policy_engine = PolicyEngine(PolicyConfig(**policy_config))
        elif isinstance(policy_config, PolicyConfig):
            self._policy_engine = PolicyEngine(policy_config)
        else:
            self._policy_engine = PolicyEngine()

        # --- Telemetry ---
        self._telemetry_store = TelemetryStore(
            self._db,
            device_id=self._device_id,
            boot_id=self._boot_id,
        )
        self._telemetry_uploader = TelemetryUploader(
            store=self._telemetry_store,
            device_id=self._device_id,
            boot_id=self._boot_id,
            api_base=server_base_url or "http://localhost:8000",
            api_key=api_key,
        )

        # --- Control plane ---
        self._control = control

        # Build observed-state reporter callback for the artifact loop.
        # When an OctomilControl instance is provided, the artifact loop
        # calls report_observed_state() after every reconciliation cycle
        # so the server can reconcile desired vs observed state.
        observed_state_reporter: Optional[Callable[..., Any]] = None
        if self._control is not None:
            observed_state_reporter = lambda statuses: self._control.report_observed_state(  # noqa: E731
                device_id=self._device_id,
                artifact_statuses=statuses,
            )

        # Use OctomilControl as the canonical sync source when no explicit
        # server_client was provided. OctomilControl exposes a
        # get_desired_state() method and base_url property that satisfy the
        # duck-typed server_client interface used by ArtifactLoop.
        effective_server_client = server_client if server_client is not None else self._control

        # --- Crash detector ---
        self._crash_detector = CrashDetector(self._db)

        # --- Loops ---
        self._inference_loop = InferenceLoop(
            session_manager=self._session_manager,
            model_registry=self._model_registry,
            telemetry_store=self._telemetry_store,
            inference_fn=inference_fn,
        )
        self._artifact_loop = ArtifactLoop(
            model_registry=self._model_registry,
            downloader=self._downloader,
            verifier=self._verifier,
            policy_engine=self._policy_engine,
            operation_scheduler=self._scheduler,
            telemetry_store=self._telemetry_store,
            server_client=effective_server_client,
            observed_state_reporter=observed_state_reporter,
        )
        self._activation_loop = ActivationLoop(
            model_registry=self._model_registry,
            activation_manager=self._activation_manager,
            session_manager=self._session_manager,
            policy_engine=self._policy_engine,
            telemetry_store=self._telemetry_store,
        )
        self._telemetry_loop = TelemetryLoop(
            telemetry_store=self._telemetry_store,
            telemetry_uploader=self._telemetry_uploader,
            policy_engine=self._policy_engine,
        )

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _apply_telemetry_schema(self) -> None:
        """Apply the telemetry-specific DDL to the shared database."""
        from .telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS

        for stmt in TELEMETRY_SCHEMA_STATEMENTS:
            self._db.execute(stmt)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all background loops.

        On startup, records a boot event and checks for crash loops.
        If a model is in a crash loop, triggers auto-rollback before
        starting the normal reconciliation loops.
        """
        logger.info("DeviceAgent starting (device_id=%s, boot_id=%s)", self._device_id, self._boot_id)

        # Record boot and check for crash loops
        self._check_crash_loops_on_boot()

        self._inference_loop.start()
        self._artifact_loop.start()
        self._activation_loop.start()
        self._telemetry_loop.start()
        logger.info("DeviceAgent started")

    def stop(self) -> None:
        """Stop all background loops gracefully."""
        logger.info("DeviceAgent stopping")
        self._inference_loop.stop()
        self._artifact_loop.stop()
        self._activation_loop.stop()
        self._telemetry_loop.stop()
        self._crash_detector.record_clean_shutdown(self._boot_id)
        logger.info("DeviceAgent stopped")

    def _check_crash_loops_on_boot(self) -> None:
        """Record boot event and check for crash loops on active models.

        If a model is detected as crash-looping, triggers auto-rollback
        via the activation manager before the reconciliation loops start.
        """
        # Determine current active model for crash tracking
        active_model_id: Optional[str] = None
        active_model_version: Optional[str] = None
        rows = self._db.execute("SELECT model_id, active_version FROM active_model_pointer")
        if rows:
            active_model_id = rows[0]["model_id"]
            active_model_version = rows[0]["active_version"]

        self._crash_detector.record_boot(
            boot_id=self._boot_id,
            active_model_id=active_model_id,
            active_model_version=active_model_version,
        )

        # Check all active models for crash loops
        for row in rows:
            model_id = row["model_id"]
            should_rollback, reason = self._crash_detector.should_auto_rollback(model_id)
            if should_rollback:
                logger.warning("Crash loop detected for %s: %s — triggering rollback", model_id, reason)
                rolled_back_to = self._activation_manager.auto_rollback(model_id, reason)
                if rolled_back_to:
                    logger.info("Rolled back %s to version %s", model_id, rolled_back_to)
                    self._telemetry_store.append_auto(
                        "crash_loop.rollback",
                        {
                            "model_id": model_id,
                            "rolled_back_to": rolled_back_to,
                            "reason": reason,
                        },
                    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, model_id: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run inference on the currently active model version.

        Delegates to ``InferenceLoop.process_request()``.
        """
        request = InferenceRequest(model_id=model_id, prompt=prompt, kwargs=kwargs)
        return self._inference_loop.process_request(request)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a dict summarising active models, download progress, and loop states."""
        # Active models
        active_models: dict[str, Any] = {}
        rows = self._db.execute("SELECT model_id, active_version, previous_version FROM active_model_pointer")
        for row in rows:
            model_id = row["model_id"]
            active_models[model_id] = {
                "active_version": row["active_version"],
                "previous_version": row["previous_version"],
            }

        # Download progress for in-flight artifacts
        downloads: dict[str, Any] = {}
        downloading_rows = self._db.execute(
            "SELECT artifact_id, model_id, version FROM model_artifacts WHERE status = 'DOWNLOADING'"
        )
        for row in downloading_rows:
            progress = self._downloader.get_progress(row["artifact_id"])
            downloads[row["artifact_id"]] = {
                "model_id": row["model_id"],
                "version": row["version"],
                **progress,
            }

        # Loop states
        loop_states = {
            "inference_loop": self._inference_loop.is_running,
            "artifact_loop": self._artifact_loop.is_running,
            "activation_loop": self._activation_loop.is_running,
            "telemetry_loop": self._telemetry_loop.is_running,
        }

        return {
            "device_id": self._device_id,
            "boot_id": self._boot_id,
            "active_models": active_models,
            "downloads": downloads,
            "loops": loop_states,
            "active_sessions": len(self._session_manager.get_active_sessions()),
            "device_state": self._policy_engine.get_device_state(),
        }

    # ------------------------------------------------------------------
    # Device state
    # ------------------------------------------------------------------

    def update_device_state(
        self,
        battery_pct: Optional[int] = None,
        is_charging: Optional[bool] = None,
        network_type: Optional[str] = None,
        thermal_state: Optional[str] = None,
        free_storage_bytes: Optional[int] = None,
        is_foreground: Optional[bool] = None,
    ) -> None:
        """Update the PolicyEngine with current device conditions.

        Call this from the host app whenever device state changes
        (battery, network, thermal, etc.).
        """
        self._policy_engine.update_device_state(
            battery_pct=battery_pct,
            is_charging=is_charging,
            network_type=network_type,
            thermal_state=thermal_state,
            free_storage_bytes=free_storage_bytes,
            is_foreground=is_foreground,
        )
