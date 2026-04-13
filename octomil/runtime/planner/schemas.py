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


@dataclass
class RuntimePlanResponse:
    """Full plan response from the server planner API."""

    model: str
    capability: str
    policy: str
    candidates: list[RuntimeCandidatePlan]
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    plan_ttl_seconds: int = 604800
    server_generated_at: str = ""


@dataclass
class RuntimeSelection:
    """Final resolved runtime selection from the planner."""

    locality: Literal["local", "cloud"]
    engine: str | None = None
    artifact: RuntimeArtifactPlan | None = None
    benchmark_ran: bool = False
    source: str = ""  # "cache", "server_plan", "local_benchmark", "fallback"
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    reason: str = ""
