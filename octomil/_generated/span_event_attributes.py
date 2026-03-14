"""Auto-generated span event attribute key constants."""

OCTOMIL_TTFT_MS = "octomil.ttft_ms"
OCTOMIL_CHUNK_INDEX = "octomil.chunk.index"
OCTOMIL_CHUNK_LATENCY_MS = "octomil.chunk.latency_ms"
OCTOMIL_TOOL_NAME = "octomil.tool.name"
OCTOMIL_TOOL_ROUND = "octomil.tool.round"
OCTOMIL_FALLBACK_REASON = "octomil.fallback.reason"
OCTOMIL_FALLBACK_PROVIDER = "octomil.fallback.provider"
OCTOMIL_TOKENS_TOTAL = "octomil.tokens.total"
OCTOMIL_TOKENS_PER_SECOND = "octomil.tokens.per_second"
OCTOMIL_DURATION_MS = "octomil.duration_ms"
OCTOMIL_DOWNLOAD_URL = "octomil.download.url"
OCTOMIL_DOWNLOAD_EXPECTED_BYTES = "octomil.download.expected_bytes"
OCTOMIL_DOWNLOAD_DURATION_MS = "octomil.download.duration_ms"
OCTOMIL_DOWNLOAD_BYTES = "octomil.download.bytes"
OCTOMIL_CHECKSUM_ALGORITHM = "octomil.checksum.algorithm"
OCTOMIL_RUNTIME_EXECUTOR = "octomil.runtime.executor"
OCTOMIL_RUNTIME_INIT_MS = "octomil.runtime.init_ms"

EVENT_REQUIRED_ATTRIBUTES: dict[str, list[str]] = {
    "first_token": ["octomil.ttft_ms"],
    "chunk_produced": ["octomil.chunk.index"],
    "tool_call_emitted": ["octomil.tool.name", "octomil.tool.round"],
    "fallback_triggered": ["octomil.fallback.reason"],
    "completed": [
        "octomil.tokens.total",
        "octomil.tokens.per_second",
        "octomil.duration_ms",
    ],
    "download_started": [],
    "download_completed": ["octomil.download.duration_ms", "octomil.download.bytes"],
    "checksum_verified": [],
    "runtime_initialized": ["octomil.runtime.executor", "octomil.runtime.init_ms"],
}
