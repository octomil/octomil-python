"""Auto-generated span event name constants."""

FIRST_TOKEN = "first_token"
CHUNK_PRODUCED = "chunk_produced"
TOOL_CALL_EMITTED = "tool_call_emitted"
FALLBACK_TRIGGERED = "fallback_triggered"
COMPLETED = "completed"
DOWNLOAD_STARTED = "download_started"
DOWNLOAD_COMPLETED = "download_completed"
CHECKSUM_VERIFIED = "checksum_verified"
RUNTIME_INITIALIZED = "runtime_initialized"

EVENT_PARENT_SPAN: dict[str, str] = {
    "first_token": "octomil.response",
    "chunk_produced": "octomil.response",
    "tool_call_emitted": "octomil.response",
    "fallback_triggered": "octomil.response",
    "completed": "octomil.response",
    "download_started": "octomil.model.load",
    "download_completed": "octomil.model.load",
    "checksum_verified": "octomil.model.load",
    "runtime_initialized": "octomil.model.load",
}
