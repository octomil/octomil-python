"""Artifact loop — downloads, resumes, verifies, and stages artifacts.

Polls the server for desired model state, reconciles against local
registry, and drives ArtifactDownloader + ArtifactVerifier through
the artifact lifecycle. Respects PolicyEngine for download gating.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from ..artifact_downloader import ArtifactDownloader
from ..artifact_verifier import ArtifactVerifier
from ..model_registry import DeviceModelRegistry
from ..operation_scheduler import OperationScheduler
from ..policy.policy_engine import PolicyEngine
from ..telemetry.telemetry_store import TelemetryStore

logger = logging.getLogger(__name__)


class ArtifactLoop:
    """Background loop that processes artifact download and verification operations.

    Periodically polls the server for desired model versions, reconciles
    against the local registry, schedules download operations for missing
    versions, and drives artifacts through download -> verify -> staged.
    """

    def __init__(
        self,
        model_registry: DeviceModelRegistry,
        downloader: ArtifactDownloader,
        verifier: ArtifactVerifier,
        policy_engine: PolicyEngine,
        operation_scheduler: OperationScheduler,
        telemetry_store: TelemetryStore,
        *,
        server_client: Any = None,
        poll_interval: float = 300.0,
        process_interval: float = 5.0,
    ) -> None:
        self._model_registry = model_registry
        self._downloader = downloader
        self._verifier = verifier
        self._policy_engine = policy_engine
        self._scheduler = operation_scheduler
        self._telemetry_store = telemetry_store
        self._server_client = server_client
        self._poll_interval = poll_interval
        self._process_interval = process_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_poll_time: float = 0.0

    def start(self) -> None:
        """Start the artifact loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="artifact-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the artifact loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Polls for desired state and processes artifact operations."""
        logger.info("Artifact loop started")
        while not self._stop_event.is_set():
            try:
                # Recover any expired operation leases
                self._scheduler.recover_expired_leases()

                # Periodically poll server for desired model state
                now = time.monotonic()
                if now - self._last_poll_time >= self._poll_interval:
                    self._poll_desired_state()
                    self._last_poll_time = now

                # Reconcile desired vs installed
                self._reconcile()

                # Process pending download/verify operations
                self._process_pending_ops()

            except Exception:
                logger.warning("Artifact loop iteration failed", exc_info=True)

            self._stop_event.wait(timeout=self._process_interval)
        logger.info("Artifact loop stopped")

    def _poll_desired_state(self) -> None:
        """Check server for new desired model versions.

        If a server_client is configured, fetches the desired state and
        registers any new artifacts that aren't already tracked locally.
        """
        if self._server_client is None:
            return

        try:
            desired = self._server_client.get_desired_state()
        except Exception:
            logger.warning("Failed to poll desired state from server", exc_info=True)
            return

        if not isinstance(desired, list):
            return

        for entry in desired:
            model_id = entry.get("model_id")
            version = entry.get("version")
            artifact_id = entry.get("artifact_id")
            manifest = entry.get("manifest", {})
            total_bytes = entry.get("total_bytes", 0)

            if not all([model_id, version, artifact_id]):
                continue

            # Skip if already tracked
            existing = self._model_registry.get_artifact(artifact_id)
            if existing is not None:
                continue

            # Register new artifact
            manifest_json = json.dumps(manifest)
            self._model_registry.register_artifact(
                artifact_id=artifact_id,
                model_id=model_id,
                version=version,
                manifest_json=manifest_json,
                total_bytes=total_bytes,
            )

            self._telemetry_store.append_auto(
                "artifact.discovered",
                {
                    "artifact_id": artifact_id,
                    "total_bytes": total_bytes,
                },
                model_id=model_id,
                model_version=version,
            )

            logger.info(
                "Discovered artifact %s for %s@%s (%d bytes)",
                artifact_id,
                model_id,
                version,
                total_bytes,
            )

    def _reconcile(self) -> None:
        """Compare desired vs installed, schedule downloads for missing versions.

        Looks at REGISTERED artifacts (discovered but not yet downloading)
        and schedules download operations for them, subject to policy checks.
        """
        pending_ops = self._scheduler.get_pending(op_type="artifact_download")
        pending_resource_ids = {op["resource_id"] for op in pending_ops}

        # Find all REGISTERED artifacts that need downloading

        db = self._model_registry._db
        rows = db.execute(
            "SELECT artifact_id, model_id, version, total_bytes FROM model_artifacts WHERE status = 'REGISTERED'"
        )

        for row in rows:
            artifact_id = row["artifact_id"]
            if artifact_id in pending_resource_ids:
                continue  # Already scheduled

            total_bytes = row["total_bytes"]

            # Check policy before scheduling download
            allowed, reason = self._policy_engine.should_allow_download(total_bytes)
            if not allowed:
                logger.debug("Download blocked for artifact %s: %s", artifact_id, reason)
                continue

            self._scheduler.schedule(
                op_type="artifact_download",
                resource_id=artifact_id,
                payload={
                    "model_id": row["model_id"],
                    "version": row["version"],
                },
                idempotency_key=f"download:{artifact_id}",
            )

    def _process_pending_ops(self) -> None:
        """Process pending artifact download and verification operations."""
        # Process downloads
        download_ops = self._scheduler.get_pending(op_type="artifact_download")
        for op in download_ops:
            if self._stop_event.is_set():
                break
            artifact_id = op["resource_id"]
            if self._scheduler.lease(op["op_id"], owner="artifact-loop"):
                try:
                    success = self._process_artifact(artifact_id)
                    if success:
                        self._scheduler.complete(op["op_id"])
                    else:
                        self._scheduler.fail(op["op_id"], "download_incomplete")
                except Exception as exc:
                    self._scheduler.fail(op["op_id"], str(exc))

        # Process verifications
        verify_ops = self._scheduler.get_pending(op_type="artifact_verify")
        for op in verify_ops:
            if self._stop_event.is_set():
                break
            artifact_id = op["resource_id"]
            if self._scheduler.lease(op["op_id"], owner="artifact-loop"):
                try:
                    ok = self._verifier.verify_artifact(artifact_id)
                    if ok:
                        self._model_registry.update_artifact_status(artifact_id, "STAGED", staged_at="now")
                        self._telemetry_store.append_auto(
                            "artifact.staged",
                            {"artifact_id": artifact_id},
                        )
                        self._scheduler.complete(op["op_id"])
                    else:
                        self._scheduler.fail(op["op_id"], "verification_failed")
                        self._telemetry_store.append_auto(
                            "artifact.download.failed",
                            {"artifact_id": artifact_id, "error": "verification_failed"},
                        )
                except Exception as exc:
                    self._scheduler.fail(op["op_id"], str(exc))

    def _process_artifact(self, artifact_id: str) -> bool:
        """Download all chunks for an artifact, verify, and mark STAGED.

        Respects policy for download gating. Returns True if the artifact
        reached STAGED status.
        """
        artifact = self._model_registry.get_artifact(artifact_id)
        if artifact is None:
            return False

        model_id = artifact["model_id"]
        version = artifact["version"]
        manifest = json.loads(artifact["manifest_json"])
        total_bytes = artifact["total_bytes"]

        # Re-check download policy
        allowed, reason = self._policy_engine.should_allow_download(total_bytes)
        if not allowed:
            logger.info("Download blocked for %s: %s", artifact_id, reason)
            return False

        # Initialize download if needed
        status = artifact["status"]
        if status == "REGISTERED":
            base_url = ""
            if self._server_client is not None:
                base_url = getattr(self._server_client, "base_url", "")
            self._downloader.start_download(artifact_id, manifest, base_url)

            self._telemetry_store.append_auto(
                "artifact.download.started",
                {"artifact_id": artifact_id, "total_bytes": total_bytes},
                model_id=model_id,
                model_version=version,
            )

        # Resume/continue download
        start_time = time.monotonic()
        pending_chunks = self._downloader.resume_download(artifact_id)
        all_downloaded = True

        for chunk_info in pending_chunks:
            if self._stop_event.is_set():
                return False

            # Re-check policy periodically during long downloads
            allowed, reason = self._policy_engine.should_allow_download(total_bytes)
            if not allowed:
                logger.info("Download paused for %s: %s", artifact_id, reason)
                self._downloader.pause(artifact_id)
                return False

            file_path = chunk_info["file_path"]
            chunk_index = chunk_info["chunk_index"]

            # Build download URL
            base_url = ""
            if self._server_client is not None:
                base_url = getattr(self._server_client, "base_url", "")
            url = f"{base_url}/artifacts/{artifact_id}/chunks/{file_path}/{chunk_index}"

            # Find expected hash from manifest
            expected_sha = None
            for f in manifest.get("files", []):
                if f["path"] == file_path:
                    expected_sha = f.get("sha256")
                    break

            ok = self._downloader.download_chunk(artifact_id, file_path, chunk_index, url, expected_sha)
            if not ok:
                all_downloaded = False

        if not all_downloaded:
            return False

        # Assemble all files
        files_in_manifest = manifest.get("files", [])
        for file_info in files_in_manifest:
            if not self._downloader.assemble_file(artifact_id, file_info["path"]):
                return False

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Verify
        if self._verifier.verify_artifact(artifact_id):
            self._model_registry.update_artifact_status(artifact_id, "STAGED", staged_at="now")
            self._telemetry_store.append_auto(
                "artifact.download.completed",
                {
                    "artifact_id": artifact_id,
                    "bytes_downloaded": total_bytes,
                    "duration_ms": round(elapsed_ms, 2),
                },
                model_id=model_id,
                model_version=version,
            )
            return True
        else:
            self._telemetry_store.append_auto(
                "artifact.download.failed",
                {"artifact_id": artifact_id, "error": "verification_failed"},
                model_id=model_id,
                model_version=version,
            )
            return False
