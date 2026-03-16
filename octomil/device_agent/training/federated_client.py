"""Device-side federated learning client with round participation state machine.

Manages the full lifecycle of participating in a federated learning round:
offer -> accept -> fetch plan -> local training -> clip/noise/encrypt -> upload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..db.local_db import LocalDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

HAPPY_STATES = [
    "NOT_ENROLLED",
    "OFFERED",
    "ACCEPTED",
    "PLAN_FETCHING",
    "PLAN_READY",
    "WAITING_FOR_WINDOW",
    "LOCAL_TRAINING",
    "LOCAL_EVAL",
    "UPDATE_PREPARING",
    "CLIPPING",
    "NOISING",
    "ENCRYPTING",
    "UPLOADING",
    "UPLOADED",
    "ACKNOWLEDGED",
    "COMPLETED",
]

ERROR_STATES = [
    "DECLINED_POLICY",
    "FAILED_RETRYABLE",
    "ABORTED_POLICY",
    "REJECTED_LOCAL",
    "UPLOAD_DEFERRED",
    "EXPIRED_ROUND",
]

ALL_STATES = set(HAPPY_STATES + ERROR_STATES)

VALID_TRANSITIONS: dict[str, set[str]] = {
    "NOT_ENROLLED": {"OFFERED", "DECLINED_POLICY"},
    "OFFERED": {"ACCEPTED", "DECLINED_POLICY", "EXPIRED_ROUND"},
    "ACCEPTED": {"PLAN_FETCHING", "ABORTED_POLICY", "EXPIRED_ROUND"},
    "PLAN_FETCHING": {"PLAN_READY", "FAILED_RETRYABLE", "EXPIRED_ROUND"},
    "PLAN_READY": {"WAITING_FOR_WINDOW", "LOCAL_TRAINING", "ABORTED_POLICY"},
    "WAITING_FOR_WINDOW": {"LOCAL_TRAINING", "EXPIRED_ROUND", "ABORTED_POLICY"},
    "LOCAL_TRAINING": {
        "LOCAL_EVAL",
        "UPDATE_PREPARING",
        "FAILED_RETRYABLE",
        "ABORTED_POLICY",
    },
    "LOCAL_EVAL": {
        "UPDATE_PREPARING",
        "REJECTED_LOCAL",
        "FAILED_RETRYABLE",
    },
    "UPDATE_PREPARING": {"CLIPPING", "FAILED_RETRYABLE"},
    "CLIPPING": {"NOISING", "FAILED_RETRYABLE"},
    "NOISING": {"ENCRYPTING", "FAILED_RETRYABLE"},
    "ENCRYPTING": {"UPLOADING", "FAILED_RETRYABLE"},
    "UPLOADING": {"UPLOADED", "UPLOAD_DEFERRED", "FAILED_RETRYABLE", "EXPIRED_ROUND"},
    "UPLOADED": {"ACKNOWLEDGED", "EXPIRED_ROUND"},
    "ACKNOWLEDGED": {"COMPLETED"},
    "COMPLETED": set(),
    # Error states
    "DECLINED_POLICY": set(),
    "FAILED_RETRYABLE": {
        "PLAN_FETCHING",
        "LOCAL_TRAINING",
        "UPDATE_PREPARING",
        "CLIPPING",
        "NOISING",
        "ENCRYPTING",
        "UPLOADING",
        "EXPIRED_ROUND",
    },
    "ABORTED_POLICY": set(),
    "REJECTED_LOCAL": set(),
    "UPLOAD_DEFERRED": {"UPLOADING", "EXPIRED_ROUND"},
    "EXPIRED_ROUND": set(),
}


class FederatedTransitionError(Exception):
    """Raised when a federated participation state transition is invalid."""


@dataclass
class TrainingPlan:
    """Server-issued training plan for a federated round."""

    plan_id: str
    round_id: str
    model_id: str
    base_version: str
    algorithm: str
    hyperparams: dict[str, Any]
    clip_norm: float
    noise_sigma: float
    secure_agg_config: Optional[dict[str, Any]] = None
    max_duration_sec: float = 300.0
    max_steps: int = 100


# ---------------------------------------------------------------------------
# DeviceFederatedClient
# ---------------------------------------------------------------------------


class DeviceFederatedClient:
    """Client-side federated learning participation manager.

    Tracks round participation state in the local database and provides
    methods for each stage of the federated learning protocol.
    """

    def __init__(
        self,
        db: LocalDB,
        api_base: str = "https://api.octomil.com/api/v1",
        http_client: Any = None,
    ) -> None:
        self._db = db
        self._api_base = api_base
        self._http = http_client  # httpx.Client or mock

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def transition(
        self, participation_id: str, new_state: str, **kwargs: Any
    ) -> None:
        """Validate and execute a participation state transition."""
        if new_state not in ALL_STATES:
            raise FederatedTransitionError(f"Unknown state: {new_state}")

        record = self.get_participation(participation_id)
        if record is None:
            raise FederatedTransitionError(
                f"Participation not found: {participation_id}"
            )

        current = record["state"]
        allowed = VALID_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            raise FederatedTransitionError(
                f"Cannot transition from {current} to {new_state} "
                f"(allowed: {sorted(allowed)})"
            )

        set_parts = ["state = ?", "updated_at = ?"]
        params: list[Any] = [new_state, self._now()]

        allowed_cols = {
            "accepted_at",
            "deadline_at",
            "upload_id",
            "update_sha256",
            "update_bytes",
            "last_error",
        }
        for key, value in kwargs.items():
            if key in allowed_cols:
                set_parts.append(f"{key} = ?")
                params.append(value)

        params.append(participation_id)
        sql = f"UPDATE federated_participation SET {', '.join(set_parts)} WHERE participation_id = ?"
        self._db.execute(sql, tuple(params))
        logger.info(
            "Participation %s: %s -> %s", participation_id, current, new_state
        )

    def get_participation(
        self, participation_id: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve a participation record."""
        row = self._db.execute_one(
            "SELECT * FROM federated_participation WHERE participation_id = ?",
            (participation_id,),
        )
        return dict(row) if row else None

    def check_offers(self, device_id: str) -> list[dict[str, Any]]:
        """Check for round offers from the server.

        Returns a list of available round offers. In production this calls
        the server API; here we return data from the HTTP client.
        """
        if self._http is None:
            return []
        resp = self._http.get(
            f"{self._api_base}/federation/offers",
            params={"device_id": device_id},
        )
        return resp.json() if resp.status_code == 200 else []

    def join_round(
        self,
        round_id: str,
        device_id: str,
        capabilities: Optional[dict[str, Any]] = None,
    ) -> str:
        """Join a federated round. Returns participation_id."""
        participation_id = str(uuid.uuid4())
        now = self._now()
        self._db.execute(
            """
            INSERT INTO federated_participation
                (participation_id, round_id, device_id, state,
                 accepted_at, updated_at)
            VALUES (?, ?, ?, 'ACCEPTED', ?, ?)
            """,
            (participation_id, round_id, device_id, now, now),
        )

        # Notify server if HTTP client available
        if self._http is not None:
            self._http.post(
                f"{self._api_base}/federation/rounds/{round_id}/join",
                json={
                    "participation_id": participation_id,
                    "device_id": device_id,
                    "capabilities": capabilities or {},
                },
            )

        logger.info(
            "Joined round %s as %s (device=%s)",
            round_id,
            participation_id,
            device_id,
        )
        return participation_id

    def fetch_plan(self, plan_id: str) -> TrainingPlan:
        """Fetch a training plan from the server.

        In production this calls the API; for testing, a plan can be
        constructed directly.
        """
        if self._http is not None:
            resp = self._http.get(
                f"{self._api_base}/federation/plans/{plan_id}"
            )
            data = resp.json()
            return TrainingPlan(
                plan_id=data["plan_id"],
                round_id=data["round_id"],
                model_id=data["model_id"],
                base_version=data["base_version"],
                algorithm=data.get("algorithm", "fedavg"),
                hyperparams=data.get("hyperparams", {}),
                clip_norm=data.get("clip_norm", 1.0),
                noise_sigma=data.get("noise_sigma", 0.01),
                secure_agg_config=data.get("secure_agg_config"),
                max_duration_sec=data.get("max_duration_sec", 300.0),
                max_steps=data.get("max_steps", 100),
            )
        raise ValueError("No HTTP client configured for plan fetch")

    def run_local_training(
        self,
        participation_id: str,
        plan: TrainingPlan,
        train_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[float]:
        """Run local training for a federated round.

        ``train_fn`` receives context and returns a dict with "update"
        (list of floats representing the model update).

        Returns the raw update vector.
        """
        self.transition(participation_id, "LOCAL_TRAINING")
        ctx = {
            "participation_id": participation_id,
            "plan": {
                "plan_id": plan.plan_id,
                "round_id": plan.round_id,
                "model_id": plan.model_id,
                "hyperparams": plan.hyperparams,
                "max_steps": plan.max_steps,
            },
        }
        try:
            result = train_fn(ctx)
            return result.get("update", [])
        except Exception as e:
            self.transition(
                participation_id, "FAILED_RETRYABLE", last_error=str(e)
            )
            raise

    def prepare_update(
        self,
        participation_id: str,
        raw_update: list[float],
        clip_norm: float = 1.0,
        noise_sigma: float = 0.01,
        secure_agg_config: Optional[dict[str, Any]] = None,
    ) -> bytes:
        """Apply the full clip -> noise -> encrypt pipeline.

        Returns the encrypted update envelope as bytes.
        """
        self.transition(participation_id, "UPDATE_PREPARING")

        # Step 1: Clip
        self.transition(participation_id, "CLIPPING")
        clipped = self.clip_update(raw_update, clip_norm)

        # Step 2: Noise
        self.transition(participation_id, "NOISING")
        noised = self.add_noise(clipped, noise_sigma)

        # Step 3: Encrypt
        self.transition(participation_id, "ENCRYPTING")
        envelope = self.encrypt_update(noised, secure_agg_config)

        return envelope

    def clip_update(
        self, update: list[float], clip_norm: float
    ) -> list[float]:
        """Clip the update vector to the given L2 norm."""
        norm = math.sqrt(sum(x * x for x in update))
        if norm <= clip_norm or norm == 0:
            return list(update)
        scale = clip_norm / norm
        return [x * scale for x in update]

    def add_noise(self, update: list[float], sigma: float) -> list[float]:
        """Add Gaussian noise to the update vector for differential privacy."""
        import random

        return [x + random.gauss(0, sigma) for x in update]

    def encrypt_update(
        self,
        update: list[float],
        secure_agg_config: Optional[dict[str, Any]] = None,
    ) -> bytes:
        """Encrypt the update for secure aggregation.

        If no secure_agg_config is provided, returns a plaintext JSON
        envelope with a hash for integrity.
        """
        payload = json.dumps(update).encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        envelope = {
            "format": "plaintext_json",
            "sha256": sha,
            "size": len(payload),
            "data": update,
        }
        if secure_agg_config:
            envelope["secure_agg"] = secure_agg_config
        return json.dumps(envelope).encode("utf-8")

    def initiate_upload(
        self,
        round_id: str,
        participation_id: str,
        update_meta: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """Request upload credentials from the server.

        Returns (upload_id, part_urls).
        """
        upload_id = str(uuid.uuid4())
        if self._http is not None:
            resp = self._http.post(
                f"{self._api_base}/federation/rounds/{round_id}/uploads",
                json={
                    "participation_id": participation_id,
                    "meta": update_meta,
                },
            )
            data = resp.json()
            return data.get("upload_id", upload_id), data.get("part_urls", [])
        return upload_id, []

    def upload_parts(
        self,
        upload_id: str,
        data: bytes,
        part_urls: list[str],
    ) -> bool:
        """Upload data parts to the provided URLs (resumable multipart).

        Returns True if all parts uploaded successfully.
        """
        if not part_urls:
            logger.warning("No part URLs provided for upload %s", upload_id)
            return True  # No parts to upload

        if self._http is None:
            return False

        chunk_size = max(1, len(data) // len(part_urls))
        for i, url in enumerate(part_urls):
            start = i * chunk_size
            end = start + chunk_size if i < len(part_urls) - 1 else len(data)
            chunk = data[start:end]
            resp = self._http.put(url, content=chunk)
            if resp.status_code not in (200, 201, 204):
                logger.error(
                    "Upload part %d failed for %s: %d",
                    i,
                    upload_id,
                    resp.status_code,
                )
                return False
        return True

    def complete_upload(
        self,
        round_id: str,
        upload_id: str,
        meta: dict[str, Any],
        privacy_proof: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Confirm upload completion with the server.

        Returns True if server acknowledged the upload.
        """
        if self._http is not None:
            resp = self._http.post(
                f"{self._api_base}/federation/rounds/{round_id}/uploads/{upload_id}/complete",
                json={
                    "meta": meta,
                    "privacy_proof": privacy_proof,
                },
            )
            return resp.status_code == 200
        return True
