"""HTTP client for server runtime planner API."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from .schemas import (
    AppResolution,
    ArtifactDownloadEndpoint,
    CandidateGate,
    DeviceRuntimeProfile,
    ModelResolution,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_PLAN_PATH = "/api/v2/runtime/plan"
_DEFAULT_BENCHMARK_PATH = "/api/v2/runtime/benchmarks"
_DEFAULT_DEFAULTS_PATH = "/api/v2/runtime/defaults"


def _parse_artifact(artifact_data: dict[str, Any]) -> RuntimeArtifactPlan:
    """Parse an artifact dict from the server into a RuntimeArtifactPlan.

    Carries the prepare-lifecycle fields added in PR 1: required_files,
    download_urls (multi-URL with optional headers/expiry), and manifest_uri.
    """
    download_urls = []
    for ep in artifact_data.get("download_urls", []) or []:
        if isinstance(ep, dict) and ep.get("url"):
            download_urls.append(
                ArtifactDownloadEndpoint(
                    url=ep["url"],
                    expires_at=ep.get("expires_at"),
                    headers=ep.get("headers"),
                )
            )
    return RuntimeArtifactPlan(
        model_id=artifact_data.get("model_id", ""),
        artifact_id=artifact_data.get("artifact_id"),
        model_version=artifact_data.get("model_version"),
        format=artifact_data.get("format"),
        quantization=artifact_data.get("quantization"),
        uri=artifact_data.get("uri"),
        digest=artifact_data.get("digest"),
        size_bytes=artifact_data.get("size_bytes"),
        min_ram_bytes=artifact_data.get("min_ram_bytes"),
        required_files=list(artifact_data.get("required_files", []) or []),
        download_urls=download_urls,
        manifest_uri=artifact_data.get("manifest_uri"),
    )


def _parse_candidate(data: dict[str, Any]) -> RuntimeCandidatePlan:
    """Parse a candidate dict from the server into a RuntimeCandidatePlan."""
    artifact_data = data.get("artifact")
    artifact = None
    if artifact_data and isinstance(artifact_data, dict):
        artifact = _parse_artifact(artifact_data)
    gates = []
    for gate_data in data.get("gates", []):
        if isinstance(gate_data, dict):
            gates.append(
                CandidateGate(
                    code=gate_data.get("code", ""),
                    required=gate_data.get("required", True),
                    threshold_number=gate_data.get("threshold_number"),
                    threshold_string=gate_data.get("threshold_string"),
                    window_seconds=gate_data.get("window_seconds"),
                    source=gate_data.get("source", "server"),
                )
            )

    return RuntimeCandidatePlan(
        locality=data.get("locality", "local"),
        priority=data.get("priority", 0),
        confidence=data.get("confidence", 0.0),
        reason=data.get("reason", ""),
        engine=data.get("engine"),
        engine_version_constraint=data.get("engine_version_constraint"),
        artifact=artifact,
        benchmark_required=data.get("benchmark_required", False),
        gates=gates,
        delivery_mode=data.get("delivery_mode"),
        prepare_required=data.get("prepare_required", False),
        prepare_policy=data.get("prepare_policy", "lazy"),
    )


def _parse_app_resolution(data: dict[str, Any]) -> AppResolution:
    """Parse an app_resolution dict from the server response."""
    artifact_candidates = []
    for ac in data.get("artifact_candidates", []):
        if isinstance(ac, dict):
            artifact_candidates.append(_parse_artifact(ac))
    return AppResolution(
        app_id=data.get("app_id", ""),
        capability=data.get("capability", ""),
        routing_policy=data.get("routing_policy", ""),
        selected_model=data.get("selected_model", ""),
        app_slug=data.get("app_slug"),
        selected_model_variant_id=data.get("selected_model_variant_id"),
        selected_model_version=data.get("selected_model_version"),
        artifact_candidates=artifact_candidates,
        preferred_engines=data.get("preferred_engines", []),
        fallback_policy=data.get("fallback_policy"),
        plan_ttl_seconds=data.get("plan_ttl_seconds", 604800),
        public_client_allowed=data.get("public_client_allowed", False),
    )


def _parse_model_resolution(data: dict[str, Any]) -> ModelResolution:
    """Parse a resolution dict from the server response."""
    return ModelResolution(
        ref_kind=data.get("ref_kind", ""),
        original_ref=data.get("original_ref", ""),
        resolved_model=data.get("resolved_model", ""),
        deployment_id=data.get("deployment_id"),
        deployment_key=data.get("deployment_key"),
        experiment_id=data.get("experiment_id"),
        variant_id=data.get("variant_id"),
        variant_name=data.get("variant_name"),
        capability=data.get("capability"),
        routing_policy=data.get("routing_policy"),
    )


def _parse_plan_response(data: dict[str, Any]) -> RuntimePlanResponse:
    """Parse a full plan response dict into a RuntimePlanResponse."""
    candidates = [_parse_candidate(c) for c in data.get("candidates", [])]
    fallback = [_parse_candidate(c) for c in data.get("fallback_candidates", [])]

    app_resolution: AppResolution | None = None
    ar_data = data.get("app_resolution")
    if ar_data and isinstance(ar_data, dict):
        app_resolution = _parse_app_resolution(ar_data)

    resolution: ModelResolution | None = None
    res_data = data.get("resolution")
    if res_data and isinstance(res_data, dict):
        resolution = _parse_model_resolution(res_data)

    return RuntimePlanResponse(
        model=data.get("model", ""),
        capability=data.get("capability", ""),
        policy=data.get("policy", ""),
        candidates=candidates,
        fallback_candidates=fallback,
        fallback_allowed=data.get("fallback_allowed", True),
        public_client_allowed=data.get("public_client_allowed", False),
        plan_ttl_seconds=data.get("plan_ttl_seconds", 604800),
        server_generated_at=data.get("server_generated_at", ""),
        app_resolution=app_resolution,
        resolution=resolution,
    )


class RuntimePlannerClient:
    """HTTP client for the server-side runtime planner API.

    Uses httpx for HTTP requests. If httpx is not available (should not happen
    since it's a core dependency), all methods degrade gracefully.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or "https://api.octomil.com").rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def fetch_plan(
        self,
        *,
        model: str,
        capability: str,
        routing_policy: str | None = None,
        device: DeviceRuntimeProfile,
        allow_cloud_fallback: bool | None = None,
        app_slug: str | None = None,
    ) -> RuntimePlanResponse | None:
        """Fetch runtime plan from server. Returns None on any failure."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — cannot fetch plan")
            return None

        payload: dict[str, Any] = {
            "model": model,
            "capability": capability,
            "device": asdict(device),
        }
        if routing_policy is not None:
            payload["routing_policy"] = routing_policy
        if allow_cloud_fallback is not None:
            payload["allow_cloud_fallback"] = allow_cloud_fallback
        if app_slug is not None:
            payload["app_slug"] = app_slug

        url = f"{self._base_url}{_DEFAULT_PLAN_PATH}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
                return _parse_plan_response(data)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Failed to fetch runtime plan from %s: %s", url, exc)
            return None

    def fetch_defaults(self) -> dict[str, Any] | None:
        """Fetch server runtime defaults. Returns None on any failure."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — cannot fetch defaults")
            return None

        url = f"{self._base_url}{_DEFAULT_DEFAULTS_PATH}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Failed to fetch runtime defaults from %s: %s", url, exc)
            return None

    def upload_benchmark(self, payload: dict[str, Any]) -> bool:
        """Upload benchmark telemetry. Returns True on success."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — cannot upload benchmark")
            return False

        url = f"{self._base_url}{_DEFAULT_BENCHMARK_PATH}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=self._headers(), json=payload)
                resp.raise_for_status()
                return True
        except (httpx.HTTPError, Exception) as exc:
            logger.debug("Failed to upload benchmark to %s: %s", url, exc)
            return False
