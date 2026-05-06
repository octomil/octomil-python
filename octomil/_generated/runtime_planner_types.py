"""Auto-generated from octomil-contracts runtime_planner schemas. Do not edit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model_ref_kind import ModelRefKind


@dataclass(slots=True)
class AppResolution:
    app_id: str
    capability: str
    routing_policy: str
    selected_model: str
    app_slug: str | None = None
    selected_model_variant_id: str | None = None
    selected_model_version: str | None = None
    artifact_candidates: list[RuntimeArtifactPlan] | None = None
    preferred_engines: list[str] | None = None
    fallback_policy: str | None = None
    plan_ttl_seconds: int | None = None
    public_client_allowed: bool | None = None


@dataclass(slots=True)
class CandidateGate:
    code: str
    required: bool
    source: str
    gate_class: str
    evaluation_phase: str
    fallback_eligible: bool
    threshold_number: float | None = None
    threshold_string: str | None = None
    window_seconds: int | None = None
    blocking_default: bool | None = None


@dataclass(slots=True)
class DeviceRuntimeProfile:
    sdk: str
    sdk_version: str
    platform: str
    arch: str
    os_version: str | None = None
    chip: str | None = None
    ram_total_bytes: int | None = None
    gpu_core_count: int | None = None
    accelerators: list[str] | None = None
    installed_runtimes: list[InstalledRuntime] | None = None
    supported_gate_codes: list[str] | None = None


@dataclass(slots=True)
class InstalledRuntime:
    engine: str
    version: str | None = None
    available: bool | None = None
    accelerator: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RouteAttempt:
    index: int
    locality: str
    mode: str
    status: str
    stage: str
    reason: dict[str, Any]
    engine: str | None = None
    artifact: AttemptArtifact | None = None
    gate_results: list[GateResult] | None = None


@dataclass(slots=True)
class AttemptArtifact:
    id: str | None = None
    digest: str | None = None
    cache: dict[str, Any] | None = None


@dataclass(slots=True)
class GateResult:
    code: str
    status: str
    gate_class: str
    evaluation_phase: str
    observed_number: float | None = None
    threshold_number: float | None = None
    threshold_string: str | None = None
    reason_code: str | None = None
    required: bool | None = None
    fallback_eligible: bool | None = None
    observed_string: str | None = None
    safe_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RouteEvent:
    route_id: str
    request_id: str
    fallback_used: bool
    candidate_attempts: int
    plan_id: str | None = None
    app_id: str | None = None
    app_slug: str | None = None
    deployment_id: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    capability: str | None = None
    policy: str | None = None
    planner_source: str | None = None
    model_ref: str | None = None
    model_ref_kind: ModelRefKind | None = None
    selected_locality: str | None = None
    final_locality: str | None = None
    final_mode: str | None = None
    engine: str | None = None
    artifact_id: str | None = None
    cache_status: str | None = None
    fallback_trigger_code: str | None = None
    fallback_trigger_stage: str | None = None
    attempt_details: list[RouteEventAttemptDetail] | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None


@dataclass(slots=True)
class RouteEventAttemptDetail:
    index: int
    locality: str
    mode: str
    engine: str | None
    status: str
    stage: str
    gate_summary: dict[str, Any]
    reason_code: str


@dataclass(slots=True)
class RouteMetadata:
    status: str
    model: RouteModel
    planner: PlannerInfo
    fallback: FallbackInfo
    reason: RouteReason
    execution: RouteExecution | None = None
    artifact: RouteArtifact | None = None
    attempts: list[RouteAttempt] | None = None


@dataclass(slots=True)
class RouteExecution:
    locality: str
    mode: str
    engine: str | None = None


@dataclass(slots=True)
class RouteModel:
    requested: RouteModelRequested
    resolved: RouteModelResolved | None = None


@dataclass(slots=True)
class RouteModelRequested:
    ref: str
    kind: ModelRefKind
    capability: str | None = None


@dataclass(slots=True)
class RouteModelResolved:
    id: str | None = None
    slug: str | None = None
    version_id: str | None = None
    variant_id: str | None = None


@dataclass(slots=True)
class RouteArtifact:
    id: str | None = None
    version: str | None = None
    format: str | None = None
    digest: str | None = None
    cache: ArtifactCache | None = None


@dataclass(slots=True)
class ArtifactCache:
    status: str
    managed_by: str | None = None


@dataclass(slots=True)
class PlannerInfo:
    source: str


@dataclass(slots=True)
class FallbackInfo:
    used: bool
    from_attempt: int | None = None
    to_attempt: int | None = None
    trigger: FallbackTrigger | None = None


@dataclass(slots=True)
class FallbackTrigger:
    code: str
    stage: str
    message: str
    gate_code: str | None = None
    gate_class: str | None = None
    evaluation_phase: str | None = None
    candidate_index: int | None = None
    output_visible_before_failure: bool | None = None


@dataclass(slots=True)
class RouteReason:
    code: str
    message: str


@dataclass(slots=True)
class RuntimeBenchmarkSubmission:
    model: str
    capability: str
    engine: str
    device: DeviceRuntimeProfile
    success: bool
    source: str | None = None
    model_version: str | None = None
    artifact_digest: str | None = None
    engine_version: str | None = None
    quantization: str | None = None
    benchmark_tokens: int | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    error_code: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeBenchmarkSubmissionResponse:
    id: str
    accepted: bool
    created_at: str


@dataclass(slots=True)
class RuntimeDefaultsResponse:
    default_engines: dict[str, Any]
    supported_capabilities: list[str]
    supported_policies: list[str]
    plan_ttl_seconds: int


@dataclass(slots=True)
class RuntimePlanRequest:
    model: str
    capability: str
    device: DeviceRuntimeProfile
    routing_policy: str | None = None
    app_id: str | None = None
    app_slug: str | None = None
    org_id: str | None = None
    allow_cloud_fallback: bool | None = None


@dataclass(slots=True)
class RuntimePlanResponse:
    model: str
    capability: str
    policy: str
    candidates: list[RuntimeCandidatePlan]
    server_generated_at: str
    plan_schema_version: int | None = None
    fallback_candidates: list[RuntimeCandidatePlan] | None = None
    plan_ttl_seconds: int | None = None
    fallback_allowed: bool | None = None
    public_client_allowed: bool | None = None
    plan_correlation_id: str | None = None
    app_resolution: AppResolution | None = None
    resolution: ModelResolution | None = None


@dataclass(slots=True)
class ModelResolution:
    ref_kind: ModelRefKind
    original_ref: str
    resolved_model: str
    deployment_id: str | None = None
    deployment_key: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    variant_name: str | None = None
    capability: str | None = None
    routing_policy: str | None = None


@dataclass(slots=True)
class RuntimeCandidatePlan:
    locality: str
    priority: int
    confidence: float
    reason: str
    engine: str | None = None
    engine_version_constraint: str | None = None
    artifact: RuntimeArtifactPlan | None = None
    benchmark_required: bool | None = None
    gates: list[CandidateGate] | None = None
    delivery_mode: str | None = None
    prepare_required: bool | None = None
    prepare_policy: str | None = None


@dataclass(slots=True)
class RuntimeArtifactPlan:
    model_id: str
    artifact_id: str | None = None
    model_version: str | None = None
    format: str | None = None
    quantization: str | None = None
    uri: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    min_ram_bytes: int | None = None
    required_files: list[str] | None = None
    download_urls: list[dict[str, Any]] | None = None
    manifest_uri: str | None = None
