"""Device agent telemetry event type constants and factory functions."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Telemetry classes — control retention, upload priority, and drop order
# ---------------------------------------------------------------------------


class TelemetryClass(str, enum.Enum):
    """Priority class governing retention and upload ordering.

    Upload priority: MUST_KEEP > IMPORTANT > BEST_EFFORT.
    Drop order under storage pressure: BEST_EFFORT first, then IMPORTANT.
    MUST_KEEP events are never dropped.
    """

    MUST_KEEP = "MUST_KEEP"
    IMPORTANT = "IMPORTANT"
    BEST_EFFORT = "BEST_EFFORT"


# Numeric sort key: lower = higher priority (uploaded first, dropped last).
CLASS_PRIORITY: dict[TelemetryClass, int] = {
    TelemetryClass.MUST_KEEP: 0,
    TelemetryClass.IMPORTANT: 1,
    TelemetryClass.BEST_EFFORT: 2,
}

# ---------------------------------------------------------------------------
# Training events
# ---------------------------------------------------------------------------

TRAINING_JOB_CREATED = "training.job.created"
TRAINING_JOB_TRANSITIONED = "training.job.transitioned"
TRAINING_CHECKPOINT_SAVED = "training.checkpoint.saved"
TRAINING_EVAL_COMPLETED = "training.eval.completed"
TRAINING_CANDIDATE_REJECTED = "training.candidate.rejected"
TRAINING_ADAPTER_ACTIVATED = "training.adapter.activated"
TRAINING_ADAPTER_ROLLBACK = "training.adapter.rollback"

# ---------------------------------------------------------------------------
# Federated events
# ---------------------------------------------------------------------------

FEDERATION_OFFER_RECEIVED = "federation.offer.received"
FEDERATION_ROUND_JOINED = "federation.round.joined"
FEDERATION_PLAN_FETCHED = "federation.plan.fetched"
FEDERATION_LOCAL_TRAIN_COMPLETED = "federation.local_train.completed"
FEDERATION_UPDATE_CLIPPED = "federation.update.clipped"
FEDERATION_UPDATE_NOISED = "federation.update.noised"
FEDERATION_UPDATE_UPLOADED = "federation.update.uploaded"
FEDERATION_ROUND_EXPIRED = "federation.round.expired"

# ---------------------------------------------------------------------------
# Serving events
# ---------------------------------------------------------------------------

SERVING_BINDING_ACTIVATED = "serving.binding.activated"
SERVING_REQUEST_STARTED = "serving.request.started"
SERVING_REQUEST_COMPLETED = "serving.request.completed"

# ---------------------------------------------------------------------------
# Artifact events
# ---------------------------------------------------------------------------

ARTIFACT_DISCOVERED = "artifact.discovered"
ARTIFACT_DOWNLOAD_STARTED = "artifact.download.started"
ARTIFACT_DOWNLOAD_PROGRESS = "artifact.download.progress"
ARTIFACT_DOWNLOAD_COMPLETED = "artifact.download.completed"
ARTIFACT_DOWNLOAD_FAILED = "artifact.download.failed"
ARTIFACT_VERIFIED = "artifact.verified"
ARTIFACT_STAGED = "artifact.staged"
ARTIFACT_ACTIVATED = "artifact.activated"

# ---------------------------------------------------------------------------
# Event → default telemetry class mapping
# ---------------------------------------------------------------------------

_EVENT_CLASS_MAP: dict[str, TelemetryClass] = {
    # Training — state transitions are MUST_KEEP, evals IMPORTANT
    TRAINING_JOB_CREATED: TelemetryClass.MUST_KEEP,
    TRAINING_JOB_TRANSITIONED: TelemetryClass.MUST_KEEP,
    TRAINING_CHECKPOINT_SAVED: TelemetryClass.IMPORTANT,
    TRAINING_EVAL_COMPLETED: TelemetryClass.IMPORTANT,
    TRAINING_CANDIDATE_REJECTED: TelemetryClass.MUST_KEEP,
    TRAINING_ADAPTER_ACTIVATED: TelemetryClass.MUST_KEEP,
    TRAINING_ADAPTER_ROLLBACK: TelemetryClass.MUST_KEEP,
    # Federation
    FEDERATION_OFFER_RECEIVED: TelemetryClass.IMPORTANT,
    FEDERATION_ROUND_JOINED: TelemetryClass.MUST_KEEP,
    FEDERATION_PLAN_FETCHED: TelemetryClass.IMPORTANT,
    FEDERATION_LOCAL_TRAIN_COMPLETED: TelemetryClass.MUST_KEEP,
    FEDERATION_UPDATE_CLIPPED: TelemetryClass.IMPORTANT,
    FEDERATION_UPDATE_NOISED: TelemetryClass.IMPORTANT,
    FEDERATION_UPDATE_UPLOADED: TelemetryClass.MUST_KEEP,
    FEDERATION_ROUND_EXPIRED: TelemetryClass.IMPORTANT,
    # Serving
    SERVING_BINDING_ACTIVATED: TelemetryClass.MUST_KEEP,
    SERVING_REQUEST_STARTED: TelemetryClass.BEST_EFFORT,
    SERVING_REQUEST_COMPLETED: TelemetryClass.IMPORTANT,
    # Artifacts
    ARTIFACT_DISCOVERED: TelemetryClass.IMPORTANT,
    ARTIFACT_DOWNLOAD_STARTED: TelemetryClass.IMPORTANT,
    ARTIFACT_DOWNLOAD_PROGRESS: TelemetryClass.BEST_EFFORT,
    ARTIFACT_DOWNLOAD_COMPLETED: TelemetryClass.MUST_KEEP,
    ARTIFACT_DOWNLOAD_FAILED: TelemetryClass.MUST_KEEP,
    ARTIFACT_VERIFIED: TelemetryClass.MUST_KEEP,
    ARTIFACT_STAGED: TelemetryClass.IMPORTANT,
    ARTIFACT_ACTIVATED: TelemetryClass.MUST_KEEP,
}


def default_class_for(event_type: str) -> TelemetryClass:
    """Return the default telemetry class for an event type."""
    return _EVENT_CLASS_MAP.get(event_type, TelemetryClass.BEST_EFFORT)


# ---------------------------------------------------------------------------
# Envelope factory
# ---------------------------------------------------------------------------


def make_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    telemetry_class: TelemetryClass | None = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    occurred_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create a fully-formed telemetry event envelope."""
    return {
        "event_id": event_id or uuid.uuid4().hex,
        "device_id": device_id,
        "boot_id": boot_id,
        "session_id": session_id,
        "sequence_no": sequence_no,
        "event_type": event_type,
        "telemetry_class": (telemetry_class or default_class_for(event_type)).value,
        "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_version": model_version,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Convenience factories for common events
# ---------------------------------------------------------------------------


def training_job_created(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    job_id: str,
    model_id: str,
    model_version: str,
    config: dict[str, Any],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return make_event(
        TRAINING_JOB_CREATED,
        {"job_id": job_id, "config": config},
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        model_id=model_id,
        model_version=model_version,
        session_id=session_id,
    )


def training_job_transitioned(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    job_id: str,
    from_state: str,
    to_state: str,
    reason: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "from_state": from_state,
        "to_state": to_state,
    }
    if reason:
        payload["reason"] = reason
    return make_event(
        TRAINING_JOB_TRANSITIONED,
        payload,
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        session_id=session_id,
    )


def artifact_download_completed(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    artifact_id: str,
    model_id: str,
    model_version: str,
    bytes_downloaded: int,
    duration_ms: float,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return make_event(
        ARTIFACT_DOWNLOAD_COMPLETED,
        {
            "artifact_id": artifact_id,
            "bytes_downloaded": bytes_downloaded,
            "duration_ms": duration_ms,
        },
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        model_id=model_id,
        model_version=model_version,
        session_id=session_id,
    )


def artifact_download_failed(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    artifact_id: str,
    model_id: str,
    model_version: str,
    error: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return make_event(
        ARTIFACT_DOWNLOAD_FAILED,
        {"artifact_id": artifact_id, "error": error},
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        model_id=model_id,
        model_version=model_version,
        session_id=session_id,
    )


def serving_request_completed(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    model_id: str,
    model_version: str,
    latency_ms: float,
    tokens: int,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return make_event(
        SERVING_REQUEST_COMPLETED,
        {"latency_ms": latency_ms, "tokens": tokens},
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        model_id=model_id,
        model_version=model_version,
        session_id=session_id,
    )


def federation_update_uploaded(
    *,
    device_id: str,
    boot_id: str,
    sequence_no: int,
    round_id: str,
    bytes_uploaded: int,
    duration_ms: float,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    return make_event(
        FEDERATION_UPDATE_UPLOADED,
        {
            "round_id": round_id,
            "bytes_uploaded": bytes_uploaded,
            "duration_ms": duration_ms,
        },
        device_id=device_id,
        boot_id=boot_id,
        sequence_no=sequence_no,
        session_id=session_id,
    )
