"""Local on-device trainer with state machine and checkpoint management."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..db.local_db import LocalDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

# Happy-path states
HAPPY_STATES = [
    "NEW",
    "ELIGIBLE",
    "QUEUED",
    "PREPARING_DATA",
    "WAITING_FOR_RESOURCES",
    "TRAINING",
    "CHECKPOINTING",
    "EVALUATING",
    "CANDIDATE_READY",
    "STAGED",
    "ACTIVATING",
    "ACTIVE",
    "COMPLETED",
]

# Error/terminal states
ERROR_STATES = [
    "BLOCKED_POLICY",
    "PAUSED",
    "FAILED_RETRYABLE",
    "FAILED_FATAL",
    "REJECTED",
    "ROLLBACK",
    "SUPERSEDED",
]

ALL_STATES = set(HAPPY_STATES + ERROR_STATES)

# Valid transitions: from_state -> set of allowed to_states
VALID_TRANSITIONS: dict[str, set[str]] = {
    "NEW": {"ELIGIBLE", "BLOCKED_POLICY", "FAILED_FATAL"},
    "ELIGIBLE": {"QUEUED", "BLOCKED_POLICY", "SUPERSEDED"},
    "QUEUED": {"PREPARING_DATA", "PAUSED", "BLOCKED_POLICY", "SUPERSEDED"},
    "PREPARING_DATA": {
        "WAITING_FOR_RESOURCES",
        "FAILED_RETRYABLE",
        "FAILED_FATAL",
        "PAUSED",
    },
    "WAITING_FOR_RESOURCES": {
        "TRAINING",
        "PAUSED",
        "BLOCKED_POLICY",
        "FAILED_RETRYABLE",
    },
    "TRAINING": {
        "CHECKPOINTING",
        "EVALUATING",
        "PAUSED",
        "FAILED_RETRYABLE",
        "FAILED_FATAL",
    },
    "CHECKPOINTING": {"TRAINING", "FAILED_RETRYABLE"},
    "EVALUATING": {
        "CANDIDATE_READY",
        "REJECTED",
        "FAILED_RETRYABLE",
        "FAILED_FATAL",
    },
    "CANDIDATE_READY": {"STAGED", "REJECTED", "SUPERSEDED"},
    "STAGED": {"ACTIVATING", "REJECTED", "ROLLBACK", "SUPERSEDED"},
    "ACTIVATING": {"ACTIVE", "ROLLBACK", "FAILED_RETRYABLE"},
    "ACTIVE": {"COMPLETED", "ROLLBACK", "SUPERSEDED"},
    "COMPLETED": set(),
    # Error states can sometimes recover
    "BLOCKED_POLICY": {"ELIGIBLE", "QUEUED", "FAILED_FATAL"},
    "PAUSED": {"QUEUED", "PREPARING_DATA", "TRAINING", "FAILED_FATAL"},
    "FAILED_RETRYABLE": {
        "QUEUED",
        "PREPARING_DATA",
        "TRAINING",
        "EVALUATING",
        "ACTIVATING",
        "FAILED_FATAL",
    },
    "FAILED_FATAL": set(),
    "REJECTED": set(),
    "ROLLBACK": set(),
    "SUPERSEDED": set(),
}


class InvalidTransitionError(Exception):
    """Raised when a state machine transition is not allowed."""


@dataclass
class TrainingLimits:
    """Resource limits for a training job."""

    max_steps: int = 100
    max_duration_sec: float = 300.0
    max_memory_bytes: int = 512 * 1024 * 1024
    checkpoint_every_steps: int = 10


# ---------------------------------------------------------------------------
# LocalTrainer
# ---------------------------------------------------------------------------


class LocalTrainer:
    """Runs local personalization training jobs with state machine tracking.

    All state is persisted to the local SQLite database via :class:`LocalDB`.
    Training is driven by user-provided callables (train_fn, eval_fn) to keep
    the trainer framework-agnostic.
    """

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_job(
        self,
        job_type: str,
        binding_key: str,
        base_model_id: str,
        base_version: str,
        starting_adapter_version: Optional[str] = None,
        limits: Optional[TrainingLimits] = None,
    ) -> str:
        """Create a new training job in NEW state. Returns job_id."""
        job_id = str(uuid.uuid4())
        now = self._now()
        limits_dict = {
            "max_steps": (limits or TrainingLimits()).max_steps,
            "max_duration_sec": (limits or TrainingLimits()).max_duration_sec,
            "checkpoint_every_steps": (limits or TrainingLimits()).checkpoint_every_steps,
        }
        self._db.execute(
            """
            INSERT INTO training_jobs
                (job_id, job_type, binding_key, base_model_id, base_version,
                 starting_adapter_version, state, progress_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'NEW', ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                binding_key,
                base_model_id,
                base_version,
                starting_adapter_version,
                json.dumps(limits_dict),
                now,
                now,
            ),
        )
        logger.info("Created training job %s (type=%s, binding=%s)", job_id, job_type, binding_key)
        return job_id

    def transition(self, job_id: str, new_state: str, **kwargs: Any) -> None:
        """Validate and execute a state machine transition.

        Keyword arguments are persisted as column updates (e.g. last_error,
        metrics_json, dataset_id).
        """
        if new_state not in ALL_STATES:
            raise InvalidTransitionError(f"Unknown state: {new_state}")

        job = self.get_job(job_id)
        if job is None:
            raise InvalidTransitionError(f"Job not found: {job_id}")

        current_state = job["state"]
        allowed = VALID_TRANSITIONS.get(current_state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {current_state} to {new_state} (allowed: {sorted(allowed)})"
            )

        # Build dynamic SET clause from kwargs
        set_parts = ["state = ?", "updated_at = ?"]
        params: list[Any] = [new_state, self._now()]

        allowed_cols = {
            "last_error",
            "metrics_json",
            "progress_json",
            "dataset_id",
            "round_id",
            "participation_id",
            "retry_count",
        }
        for key, value in kwargs.items():
            if key in allowed_cols:
                set_parts.append(f"{key} = ?")
                params.append(value)

        params.append(job_id)
        sql = f"UPDATE training_jobs SET {', '.join(set_parts)} WHERE job_id = ?"
        self._db.execute(sql, tuple(params))
        logger.info("Job %s: %s -> %s", job_id, current_state, new_state)

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a training job by ID."""
        row = self._db.execute_one("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
        return dict(row) if row else None

    def list_jobs(self, state: Optional[str] = None) -> list[dict[str, Any]]:
        """List training jobs, optionally filtered by state."""
        if state:
            rows = self._db.execute(
                "SELECT * FROM training_jobs WHERE state = ? ORDER BY created_at DESC",
                (state,),
            )
        else:
            rows = self._db.execute("SELECT * FROM training_jobs ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def prepare_data(
        self,
        job_id: str,
        selection_policy: dict[str, Any],
    ) -> str:
        """Transition to PREPARING_DATA and create a dataset snapshot.

        Returns the dataset_id. The actual data selection is done by
        DataPipeline — this method only manages the state transition
        and records the dataset reference.
        """
        self.transition(job_id, "PREPARING_DATA")
        dataset_id = str(uuid.uuid4())
        self.transition(
            job_id,
            "WAITING_FOR_RESOURCES",
            dataset_id=dataset_id,
            progress_json=json.dumps({"selection_policy": selection_policy}),
        )
        return dataset_id

    def train(
        self,
        job_id: str,
        train_fn: Callable[[dict[str, Any]], dict[str, Any]],
        max_steps: int = 100,
        max_duration_sec: float = 300.0,
        checkpoint_every: int = 10,
    ) -> dict[str, Any]:
        """Run training with periodic checkpointing.

        ``train_fn`` receives a context dict with keys:
          - step: current step number
          - job_id: the job ID
          - max_steps: total steps requested

        It must return a dict with at least:
          - loss: current loss value
          - done: bool indicating if training is complete
          - checkpoint_data: optional bytes for checkpoint

        Returns the final metrics dict.
        """
        self.transition(job_id, "TRAINING")
        start_time = time.monotonic()
        metrics: dict[str, Any] = {}
        step = 0

        try:
            while step < max_steps:
                elapsed = time.monotonic() - start_time
                if elapsed > max_duration_sec:
                    logger.info("Job %s: time limit reached at step %d", job_id, step)
                    break

                ctx = {"step": step, "job_id": job_id, "max_steps": max_steps}
                result = train_fn(ctx)
                step += 1
                metrics = {
                    "step": step,
                    "loss": result.get("loss", 0.0),
                    "elapsed_sec": time.monotonic() - start_time,
                }

                # Periodic checkpoint
                if step % checkpoint_every == 0 and result.get("checkpoint_data"):
                    self.transition(job_id, "CHECKPOINTING")
                    self._save_checkpoint(job_id, step, result["checkpoint_data"])
                    self.transition(job_id, "TRAINING")

                if result.get("done"):
                    break

            self._db.execute(
                "UPDATE training_jobs SET metrics_json = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(metrics), self._now(), job_id),
            )
            return metrics
        except Exception as e:
            self.transition(job_id, "FAILED_RETRYABLE", last_error=str(e))
            raise

    def evaluate(
        self,
        job_id: str,
        eval_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run evaluation comparing candidate vs current adapter.

        ``eval_fn`` receives a context dict with:
          - job_id: the job ID
          - job: full job record dict

        Returns eval result dict. Transitions to CANDIDATE_READY or REJECTED
        based on eval_fn returning {"accept": True/False}.
        """
        self.transition(job_id, "EVALUATING")
        job = self.get_job(job_id)
        ctx = {"job_id": job_id, "job": job}

        try:
            result = eval_fn(ctx)
            metrics = {
                "eval_result": result,
                "evaluated_at": self._now(),
            }
            self._db.execute(
                "UPDATE training_jobs SET metrics_json = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(metrics), self._now(), job_id),
            )

            if result.get("accept", False):
                self.transition(job_id, "CANDIDATE_READY")
            else:
                self.transition(
                    job_id,
                    "REJECTED",
                    last_error=result.get("reason", "Eval rejected candidate"),
                )
            return result
        except Exception as e:
            self.transition(job_id, "FAILED_RETRYABLE", last_error=str(e))
            raise

    def resume_from_checkpoint(self, job_id: str) -> Optional[dict[str, Any]]:
        """Resume a job from its latest checkpoint.

        Returns the checkpoint record dict, or None if no checkpoint exists.
        """
        row = self._db.execute_one(
            """
            SELECT * FROM training_checkpoints
            WHERE job_id = ? ORDER BY step DESC LIMIT 1
            """,
            (job_id,),
        )
        if row is None:
            return None
        return dict(row)

    def pause(self, job_id: str) -> None:
        """Pause a running or queued job."""
        self.transition(job_id, "PAUSED")

    def cancel(self, job_id: str) -> None:
        """Cancel a job by transitioning to FAILED_FATAL."""
        self.transition(job_id, "FAILED_FATAL", last_error="Cancelled by user")

    def _save_checkpoint(self, job_id: str, step: int, data: bytes) -> str:
        """Save a training checkpoint to the database."""
        checkpoint_id = str(uuid.uuid4())
        sha = hashlib.sha256(data).hexdigest()
        self._db.execute(
            """
            INSERT INTO training_checkpoints
                (checkpoint_id, job_id, path, step, bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                job_id,
                f"checkpoints/{job_id}/{step}",
                step,
                len(data),
                sha,
                self._now(),
            ),
        )
        logger.debug("Saved checkpoint %s at step %d for job %s", checkpoint_id, step, job_id)
        return checkpoint_id
