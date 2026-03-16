"""Auto-generated span event name constants."""

FIRST_TOKEN = "first_token"
CHUNK_PRODUCED = "chunk_produced"
TOOL_CALL_EMITTED = "tool_call_emitted"
FALLBACK_TRIGGERED = "fallback_triggered"
COMPLETED = "completed"
TOOL_CALL_PARSE_SUCCEEDED = "tool_call_parse_succeeded"
TOOL_CALL_PARSE_FAILED = "tool_call_parse_failed"
DOWNLOAD_STARTED = "download_started"
DOWNLOAD_COMPLETED = "download_completed"
CHECKSUM_VERIFIED = "checksum_verified"
RUNTIME_INITIALIZED = "runtime_initialized"
CHUNK_DOWNLOAD_STARTED = "chunk_download_started"
CHUNK_DOWNLOAD_COMPLETED = "chunk_download_completed"
CHUNK_DOWNLOAD_FAILED = "chunk_download_failed"
ARTIFACT_VERIFIED = "artifact_verified"
WARMING_STARTED = "warming_started"
HEALTHCHECK_PASSED = "healthcheck_passed"
HEALTHCHECK_FAILED = "healthcheck_failed"
ACTIVATION_COMPLETE = "activation_complete"
ROLLBACK_TRIGGERED = "rollback_triggered"
PLAN_FETCHED = "plan_fetched"
LOCAL_TRAINING_STARTED = "local_training_started"
LOCAL_TRAINING_COMPLETED = "local_training_completed"
UPDATE_CLIPPED = "update_clipped"
UPDATE_NOISED = "update_noised"
UPDATE_ENCRYPTED = "update_encrypted"
UPLOAD_STARTED = "upload_started"
UPLOAD_COMPLETED = "upload_completed"
PARTICIPATION_ABORTED = "participation_aborted"
ROUND_STARTED = "round_started"
ROUND_AGGREGATED = "round_aggregated"
CANDIDATE_PUBLISHED = "candidate_published"
JOB_COMPLETED = "job_completed"
DESIRED_STATE_FETCHED = "desired_state_fetched"
OBSERVED_STATE_REPORTED = "observed_state_reported"
STATE_DRIFT_DETECTED = "state_drift_detected"
DEVICE_REGISTERED = "device.registered"

EVENT_PARENT_SPAN: dict[str, str] = {
    "first_token": "octomil.response",
    "chunk_produced": "octomil.response",
    "tool_call_emitted": "octomil.response",
    "fallback_triggered": "octomil.response",
    "completed": "octomil.response",
    "tool_call_parse_succeeded": "octomil.response",
    "tool_call_parse_failed": "octomil.response",
    "download_started": "octomil.model.load",
    "download_completed": "octomil.model.load",
    "checksum_verified": "octomil.model.load",
    "runtime_initialized": "octomil.model.load",
    "chunk_download_started": "octomil.artifact.download",
    "chunk_download_completed": "octomil.artifact.download",
    "chunk_download_failed": "octomil.artifact.download",
    "artifact_verified": "octomil.artifact.download",
    "warming_started": "octomil.artifact.activation",
    "healthcheck_passed": "octomil.artifact.activation",
    "healthcheck_failed": "octomil.artifact.activation",
    "activation_complete": "octomil.artifact.activation",
    "rollback_triggered": "octomil.artifact.activation",
    "plan_fetched": "octomil.federation.round",
    "local_training_started": "octomil.federation.round",
    "local_training_completed": "octomil.federation.round",
    "update_clipped": "octomil.federation.round",
    "update_noised": "octomil.federation.round",
    "update_encrypted": "octomil.federation.round",
    "upload_started": "octomil.federation.round",
    "upload_completed": "octomil.federation.round",
    "participation_aborted": "octomil.federation.round",
    "round_started": "octomil.training.job",
    "round_aggregated": "octomil.training.job",
    "candidate_published": "octomil.training.job",
    "job_completed": "octomil.training.job",
    "desired_state_fetched": "octomil.device.sync",
    "observed_state_reported": "octomil.device.sync",
    "state_drift_detected": "octomil.device.sync",
    "device.registered": "octomil.control.register",
}
