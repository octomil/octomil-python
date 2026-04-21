"""HTTP client for server runtime planner API."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from .schemas import (
    AppResolution,
    DeviceRuntimeProfile,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_PLAN_PATH = "/api/v2/runtime/plan"
_DEFAULT_BENCHMARK_PATH = "/api/v2/runtime/benchmarks"
_DEFAULT_DEFAULTS_PATH = "/api/v2/runtime/defaults"


def _parse_candidate(data: dict[str, Any]) -> RuntimeCandidatePlan:
    """Parse a candidate dict from the server into a RuntimeCandidatePlan."""
    artifact_data = data.get("artifact")
    artifact = None
    if artifact_data and isinstance(artifact_data, dict):
        artifact = RuntimeArtifactPlan(
            model_id=artifact_data.get("model_id", ""),
            artifact_id=artifact_data.get("artifact_id"),
            model_version=artifact_data.get("model_version"),
            format=artifact_data.get("format"),
            quantization=artifact_data.get("quantization"),
            uri=artifact_data.get("uri"),
            digest=artifact_data.get("digest"),
            size_bytes=artifact_data.get("size_bytes"),
            min_ram_bytes=artifact_data.get("min_ram_bytes"),
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
    )


def _parse_app_resolution(data: dict[str, Any]) -> AppResolution:
    """Parse an app_resolution dict from the server response."""
    artifact_candidates = []
    for ac in data.get("artifact_candidates", []):
        if isinstance(ac, dict):
            artifact_candidates.append(
                RuntimeArtifactPlan(
                    model_id=ac.get("model_id", ""),
                    artifact_id=ac.get("artifact_id"),
                    model_version=ac.get("model_version"),
                    format=ac.get("format"),
                    quantization=ac.get("quantization"),
                    uri=ac.get("uri"),
                    digest=ac.get("digest"),
                    size_bytes=ac.get("size_bytes"),
                    min_ram_bytes=ac.get("min_ram_bytes"),
                )
            )
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
    )


def _parse_plan_response(data: dict[str, Any]) -> RuntimePlanResponse:
    """Parse a full plan response dict into a RuntimePlanResponse."""
    candidates = [_parse_candidate(c) for c in data.get("candidates", [])]
    fallback = [_parse_candidate(c) for c in data.get("fallback_candidates", [])]

    app_resolution: AppResolution | None = None
    ar_data = data.get("app_resolution")
    if ar_data and isinstance(ar_data, dict):
        app_resolution = _parse_app_resolution(ar_data)

    return RuntimePlanResponse(
        model=data.get("model", ""),
        capability=data.get("capability", ""),
        policy=data.get("policy", ""),
        candidates=candidates,
        fallback_candidates=fallback,
        plan_ttl_seconds=data.get("plan_ttl_seconds", 604800),
        server_generated_at=data.get("server_generated_at", ""),
        app_resolution=app_resolution,
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
