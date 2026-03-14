"""Auto-generated metric view constants."""

from typing import NamedTuple


class MetricView(NamedTuple):
    name: str
    instrument: str
    unit: str
    source_span: str


OCTOMIL_RESPONSE_DURATION = "octomil.response.duration"
OCTOMIL_RESPONSE_TTFT = "octomil.response.ttft"
OCTOMIL_RESPONSE_TOKENS_PER_SECOND = "octomil.response.tokens_per_second"
OCTOMIL_MODEL_LOAD_DURATION = "octomil.model.load.duration"
OCTOMIL_MODEL_LOAD_FAILURE_RATE = "octomil.model.load.failure_rate"
OCTOMIL_FALLBACK_RATE = "octomil.fallback.rate"
OCTOMIL_HEARTBEAT_FRESHNESS = "octomil.heartbeat.freshness"
OCTOMIL_TOOL_EXECUTE_DURATION = "octomil.tool.execute.duration"

ALL_METRIC_VIEWS = [
    MetricView("octomil.response.duration", "histogram", "ms", "octomil.response"),
    MetricView("octomil.response.ttft", "histogram", "ms", "octomil.response"),
    MetricView(
        "octomil.response.tokens_per_second",
        "histogram",
        "{tokens}/s",
        "octomil.response",
    ),
    MetricView("octomil.model.load.duration", "histogram", "ms", "octomil.model.load"),
    MetricView("octomil.model.load.failure_rate", "counter", "{failures}", "octomil.model.load"),
    MetricView("octomil.fallback.rate", "counter", "{fallbacks}", "octomil.response"),
    MetricView("octomil.heartbeat.freshness", "gauge", "s", "octomil.control.heartbeat"),
    MetricView("octomil.tool.execute.duration", "histogram", "ms", "octomil.tool.execute"),
]
