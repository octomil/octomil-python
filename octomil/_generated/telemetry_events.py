"""Auto-generated telemetry event names and required attributes."""

INFERENCE_STARTED = "inference.started"
INFERENCE_COMPLETED = "inference.completed"
INFERENCE_FAILED = "inference.failed"
INFERENCE_CHUNK_PRODUCED = "inference.chunk_produced"
DEPLOY_STARTED = "deploy.started"
DEPLOY_COMPLETED = "deploy.completed"

EVENT_REQUIRED_ATTRIBUTES: dict[str, list[str]] = {
    "inference.started": ["model.id"],
    "inference.completed": ["model.id", "inference.duration_ms"],
    "inference.failed": ["model.id", "error.type", "error.message"],
    "inference.chunk_produced": ["model.id", "inference.chunk_index"],
    "deploy.started": ["model.id", "model.version"],
    "deploy.completed": ["model.id", "deploy.duration_ms"],
}
