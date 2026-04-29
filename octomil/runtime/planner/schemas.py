"""Runtime planner request/response schemas -- mirrors server contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from octomil._generated.planner_source import PlannerSource


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
    supported_gate_codes: list[str] | None = None


@dataclass
class ArtifactDownloadEndpoint:
    """One download endpoint for a planner artifact.

    Multi-URL support gives the SDK CDN/origin failover and parallel chunk
    fetches. ``headers`` carries server-issued auth (e.g. signed-URL params)
    when needed; the server never embeds long-lived bearers here.
    """

    url: str
    expires_at: str | None = None
    headers: dict[str, str] | None = None


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
    required_files: list[str] = field(default_factory=list)
    download_urls: list[ArtifactDownloadEndpoint] = field(default_factory=list)
    manifest_uri: str | None = None
    # PR C-followup: option 2 — first-class static-recipe contract.
    # When the planner selects a canonical built-in artifact, it can
    # set ``source="static_recipe"`` plus ``recipe_id`` and the SDK
    # maps that to its built-in recipe table without requiring the
    # planner to embed the URL/digest/required_files redundantly.
    # Future custom-artifact source values (``"hf_snapshot"``,
    # ``"oci_layer"``, ...) ride the same discriminator. ``None`` /
    # absent ``source`` means "use the planner-supplied artifact
    # metadata as-is" (option 1, the existing path).
    source: Literal["static_recipe"] | None = None
    recipe_id: str | None = None


@dataclass
class TtsSpeakerProfile:
    """Logical TTS speaker the planner / app config defines.

    Decouples the caller's speaker id (``"madam_ambrose"``) from the
    runtime's native voice id (``sid=4``) and from any reference-audio
    artifact a few-shot voice-cloning engine like PocketTTS needs.

    Two engines, two resolution modes:

      - Native-voice engines (Kokoro / Piper). The planner sets
        ``native_voice`` to the runtime voice label that lives in the
        artifact's ``voices.txt`` / voices.bin. The kernel hands that
        label straight to ``backend.synthesize(voice=...)``; no
        reference audio.
      - Few-shot voice-cloning engines (PocketTTS). The planner sets
        ``reference_audio`` (URL or already-prepared local path) plus
        ``reference_sample_rate`` so the engine has a sample to clone
        from. ``native_voice`` is typically absent.

    A planner that omits a profile entirely is allowed; the kernel
    then falls back to native-voice resolution against the model's
    own catalog (existing behaviour for non-app refs and for app refs
    that haven't been populated yet).
    """

    speaker_id: str
    native_voice: str | None = None
    reference_audio: str | None = None
    reference_sample_rate: int | None = None
    language: str | None = None
    style: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateGate:
    """A planner gate the SDK must evaluate before using a candidate."""

    code: str
    required: bool
    threshold_number: float | None = None
    threshold_string: str | None = None
    window_seconds: int | None = None
    source: Literal["server", "sdk", "runtime"] = "server"
    gate_class: str | None = None  # "readiness" | "performance" | "output_quality"
    evaluation_phase: str | None = None  # "pre_inference" | "during_inference" | "post_inference"
    fallback_eligible: bool | None = None
    blocking_default: bool | None = None


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
    delivery_mode: Literal["hosted_gateway", "sdk_runtime", "external_endpoint"] | None = None
    prepare_required: bool = False
    prepare_policy: Literal["lazy", "explicit_only", "disabled"] = "lazy"
    # Per-candidate TTS speaker profiles. Keyed by ``speaker_id``.
    # Overrides the app-level ``tts_speakers`` map when both are set —
    # lets a planner attach speaker profiles to a specific candidate
    # (e.g. only the PocketTTS candidate carries reference-audio
    # entries; the Kokoro candidate carries native-voice entries).
    tts_speakers: dict[str, TtsSpeakerProfile] = field(default_factory=dict)


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
    public_client_allowed: bool = False
    # App-scoped TTS speaker profiles. Keyed by ``speaker_id``. When
    # ``client.audio.speech.stream(model="@app/<slug>/tts",
    # speaker="madam_ambrose", ...)`` lands, the kernel looks up the
    # speaker here first (or in the per-candidate ``tts_speakers``)
    # and the resolved profile feeds the engine — caller never names
    # the artifact, the policy, or the reference audio.
    tts_speakers: dict[str, TtsSpeakerProfile] = field(default_factory=dict)


@dataclass
class ModelResolution:
    """Generalized resolution metadata for non-app model ref types.

    Returned by the server when the model ref resolves through a deployment,
    experiment, capability default, or plain model lookup. Carries the
    deployment_id, experiment_id, and variant_id needed for SDK route
    telemetry correlation.
    """

    ref_kind: str
    original_ref: str
    resolved_model: str
    deployment_id: str | None = None
    deployment_key: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    variant_name: str | None = None
    capability: str | None = None
    routing_policy: str | None = None


@dataclass
class RuntimePlanResponse:
    """Full plan response from the server planner API."""

    model: str
    capability: str
    policy: str
    candidates: list[RuntimeCandidatePlan]
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_allowed: bool = True
    public_client_allowed: bool = False
    plan_ttl_seconds: int = 604800
    server_generated_at: str = ""
    app_resolution: AppResolution | None = None
    resolution: ModelResolution | None = None


# ---------------------------------------------------------------------------
# Planner source normalization
# ---------------------------------------------------------------------------

CANONICAL_PLANNER_SOURCES: frozenset[str] = frozenset(source.value for source in PlannerSource)

_PLANNER_SOURCE_ALIASES: dict[str, str] = {
    "local_default": PlannerSource.OFFLINE.value,
    "server_plan": PlannerSource.SERVER.value,
    "cached": PlannerSource.CACHE.value,
    "fallback": PlannerSource.OFFLINE.value,
    "none": PlannerSource.OFFLINE.value,
    "local_benchmark": PlannerSource.OFFLINE.value,
}


def normalize_planner_source(source: str) -> str:
    """Normalize a planner source string to a canonical value.

    Canonical values: ``"server"``, ``"cache"``, ``"offline"``.
    Deprecated aliases are mapped to their canonical equivalent.
    Unknown or blank values collapse to ``"offline"`` so SDK output
    boundaries never emit a contract-invalid planner source.
    """
    if source in CANONICAL_PLANNER_SOURCES:
        return source
    return _PLANNER_SOURCE_ALIASES.get(source, PlannerSource.OFFLINE.value)


@dataclass
class RuntimeSelection:
    """Final resolved runtime selection from the planner."""

    locality: Literal["local", "cloud"]
    engine: str | None = None
    artifact: RuntimeArtifactPlan | None = None
    benchmark_ran: bool = False
    source: str = ""  # Internal value; normalized via normalize_planner_source()
    candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    fallback_allowed: bool = True
    reason: str = ""
    app_resolution: AppResolution | None = None
    resolution: ModelResolution | None = None
