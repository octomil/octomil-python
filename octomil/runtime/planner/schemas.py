"""Runtime planner request/response schemas -- mirrors server contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class InstalledRuntime:
    """A locally-installed inference engine detected on this device."""

    engine: str
    version: str | None = None
    available: bool = True
    accelerator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceRuntimeProfile:
    """Hardware and software profile sent to the server planner endpoint."""

    sdk: Literal["python", "node", "ios", "android", "browser"]
    sdk_version: str
    platform: str
    arch: str
    os_version: str | None = None
    chip: str | None = None
    ram_total_bytes: int | None = None
    gpu_core_count: int | None = None
    accelerators: list[str] = field(default_factory=list)
    installed_runtimes: list[InstalledRuntime] = field(default_factory=list)


@dataclass
class RuntimeArtifactPlan:
    """Artifact recommendation from the server planner."""

    model_id: str
    artifact_id: str | None = None
    model_version: str | None = None
    format: str | None = None
    quantization: str | None = None
    uri: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    min_ram_bytes: int | None = None


@dataclass
class CandidateGate:
    """A planner gate the SDK must evaluate before using a candidate."""

    code: str
    required: bool
    threshold_number: float | None = None
    threshold_string: str | None = None
    window_seconds: int | None = None
    source: Literal["server", "sdk", "runtime"] = "server"


@dataclass
class RuntimeCandidatePlan:
    """A single candidate in a runtime plan (local or cloud)."""

    locality: Literal["local", "cloud"]
    priority: int
    confidence: float
    reason: str
    engine: str | None = None
    engine_version_constraint: str | None = None
    artifact: RuntimeArtifactPlan | None = None
    benchmark_required: bool = False
    gates: list[CandidateGate] = field(default_factory=list)


@dataclass
class AppResolution:
    """Server-resolved app ref details from ``@app/{slug}/{capability}`` routing."""

    app_id: str
    capability: str
    routing_policy: str
    selected_model: str
    app_slug: str | None = None
    selected_model_variant_id: str | None = None
    selected_model_version: str | None = None
    artifact_candidates: list[RuntimeArtifactPlan] = field(default_factory=list)
    preferred_engines: list[str] = field(default_factory=list)
    fallback_policy: str | None = None
    plan_ttl_seconds: int = 604800


@dataclass
class RuntimePlanResponse:
    """Full plan response from the server planner API."""

    model: str
    capability: str
    policy: str
    candidates: list[RuntimeCandidatePlan]
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_allowed: bool = True
    plan_ttl_seconds: int = 604800
    server_generated_at: str = ""
    app_resolution: AppResolution | None = None


@dataclass
class RuntimeSelection:
    """Final resolved runtime selection from the planner."""

    locality: Literal["local", "cloud"]
    engine: str | None = None
    artifact: RuntimeArtifactPlan | None = None
    benchmark_ran: bool = False
    source: str = ""  # "cache", "server_plan", "local_benchmark", "fallback"
    candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_allowed: bool = True
    reason: str = ""
    app_resolution: AppResolution | None = None
