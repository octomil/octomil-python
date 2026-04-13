"""Auto-generated span attribute key constants and required lookups."""

MODEL_ID = "model.id"
MODEL_VERSION = "model.version"
RUNTIME_EXECUTOR = "runtime.executor"
REQUEST_MODE = "request.mode"
LOCALITY = "locality"
STREAMING = "streaming"
ROUTE_POLICY = "route.policy"
ROUTE_DECISION = "route.decision"
DEVICE_CLASS = "device.class"
FALLBACK_REASON = "fallback.reason"
ERROR_TYPE = "error.type"
TOOL_CALL_TIER = "tool.call_tier"
KV_CACHE_STRATEGY = "kv_cache.strategy"
KV_CACHE_QUANTIZATION_BITS = "kv_cache.quantization_bits"
KV_CACHE_COMPRESSION_RATIO = "kv_cache.compression_ratio"
MODEL_SOURCE_FORMAT = "model.source_format"
MODEL_SIZE_BYTES = "model.size_bytes"
TOOL_NAME = "tool.name"
TOOL_ROUND = "tool.round"
FALLBACK_PROVIDER = "fallback.provider"
ASSIGNMENT_COUNT = "assignment_count"
HEARTBEAT_SEQUENCE = "heartbeat.sequence"
ROLLOUT_ID = "rollout.id"
MODELS_SYNCED = "models_synced"

SPAN_REQUIRED_ATTRIBUTES: dict[str, list[str]] = {
    "octomil.response": ["model.id", "model.version", "runtime.executor", "request.mode", "locality", "streaming"],
    "octomil.model.load": ["model.id", "model.version", "runtime.executor"],
    "octomil.tool.execute": ["tool.name", "tool.round"],
    "octomil.fallback.cloud": ["model.id", "fallback.reason"],
    "octomil.control.refresh": [],
    "octomil.control.heartbeat": ["heartbeat.sequence"],
    "octomil.rollout.sync": ["rollout.id"],
}
