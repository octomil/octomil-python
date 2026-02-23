from __future__ import annotations

from typing import Any, List, Optional

from .api_client import _ApiClient


class RolloutsAPI:
    """Control-plane rollout management API."""

    def __init__(self, api: _ApiClient):
        self.api = api

    def create(
        self,
        model_id: str,
        version: str,
        rollout_percentage: float = 10.0,
        target_percentage: float = 100.0,
        increment_step: float = 10.0,
        start_immediately: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "version": version,
            "rollout_percentage": rollout_percentage,
            "target_percentage": target_percentage,
            "increment_step": increment_step,
            "start_immediately": start_immediately,
        }
        return self.api.post(f"/models/{model_id}/rollouts", payload)

    def list(self, model_id: str, status_filter: Optional[str] = None) -> List[dict[str, Any]]:
        params = {"status_filter": status_filter} if status_filter else None
        return self.api.get(f"/models/{model_id}/rollouts", params=params)

    def list_active(self, model_id: str) -> List[dict[str, Any]]:
        return self.api.get(f"/models/{model_id}/rollouts/active")

    def get(self, model_id: str, rollout_id: int) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}/rollouts/{rollout_id}")

    def start(self, model_id: str, rollout_id: int) -> dict[str, Any]:
        return self.api.post(f"/models/{model_id}/rollouts/{rollout_id}/start", {})

    def pause(self, model_id: str, rollout_id: int) -> dict[str, Any]:
        return self.api.post(f"/models/{model_id}/rollouts/{rollout_id}/pause", {})

    def resume(self, model_id: str, rollout_id: int) -> dict[str, Any]:
        return self.api.post(f"/models/{model_id}/rollouts/{rollout_id}/resume", {})

    def advance(
        self,
        model_id: str,
        rollout_id: int,
        custom_increment: Optional[float] = None,
    ) -> dict[str, Any]:
        payload = {}
        if custom_increment is not None:
            payload["custom_increment"] = custom_increment
        return self.api.post(f"/models/{model_id}/rollouts/{rollout_id}/advance", payload)

    def update_percentage(
        self,
        model_id: str,
        rollout_id: int,
        percentage: float,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"percentage": percentage}
        if reason is not None:
            payload["reason"] = reason
        return self.api.post(f"/models/{model_id}/rollouts/{rollout_id}/update-percentage", payload)

    def get_status_history(self, model_id: str, rollout_id: int) -> List[dict[str, Any]]:
        return self.api.get(f"/models/{model_id}/rollouts/{rollout_id}/status-history")

    def get_affected_devices(self, model_id: str, rollout_id: int) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}/rollouts/{rollout_id}/affected-devices")

    def delete(self, model_id: str, rollout_id: int, force: bool = False) -> dict[str, Any]:
        params = {"force": str(force).lower()}
        return self.api.delete(f"/models/{model_id}/rollouts/{rollout_id}", params=params)


class FederatedAnalyticsAPI:
    """Federated analytics API for cross-site statistical analysis."""

    def __init__(self, api: _ApiClient, federation_id: str):
        self.api = api
        self.federation_id = federation_id
        self._base = f"/federations/{federation_id}/analytics"

    def descriptive(
        self,
        variable: str,
        group_by: str = "device_group",
        group_ids: Optional[list[str]] = None,
        include_percentiles: bool = True,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variable": variable,
            "group_by": group_by,
            "include_percentiles": include_percentiles,
        }
        if group_ids is not None:
            payload["group_ids"] = group_ids
        if filters is not None:
            payload["filters"] = filters
        return self.api.post(f"{self._base}/descriptive", payload)

    def t_test(
        self,
        variable: str,
        group_a: str,
        group_b: str,
        confidence_level: float = 0.95,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variable": variable,
            "group_a": group_a,
            "group_b": group_b,
            "confidence_level": confidence_level,
        }
        if filters is not None:
            payload["filters"] = filters
        return self.api.post(f"{self._base}/t-test", payload)

    def chi_square(
        self,
        variable_1: str,
        variable_2: str,
        group_ids: Optional[list[str]] = None,
        confidence_level: float = 0.95,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variable_1": variable_1,
            "variable_2": variable_2,
            "confidence_level": confidence_level,
        }
        if group_ids is not None:
            payload["group_ids"] = group_ids
        if filters is not None:
            payload["filters"] = filters
        return self.api.post(f"{self._base}/chi-square", payload)

    def anova(
        self,
        variable: str,
        group_by: str = "device_group",
        group_ids: Optional[list[str]] = None,
        confidence_level: float = 0.95,
        post_hoc: bool = True,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variable": variable,
            "group_by": group_by,
            "confidence_level": confidence_level,
            "post_hoc": post_hoc,
        }
        if group_ids is not None:
            payload["group_ids"] = group_ids
        if filters is not None:
            payload["filters"] = filters
        return self.api.post(f"{self._base}/anova", payload)

    def list_queries(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self.api.get(f"{self._base}/queries", params=params)

    def get_query(self, query_id: str) -> dict[str, Any]:
        return self.api.get(f"{self._base}/queries/{query_id}")


class ExperimentsAPI:
    """Control-plane experiment management and analytics API."""

    def __init__(self, api: _ApiClient, org_id: str):
        self.api = api
        self.org_id = org_id

    def create(
        self,
        *,
        name: str,
        model_id: str,
        control_version: str,
        treatment_version: str,
        control_allocation: float = 50.0,
        treatment_allocation: float = 50.0,
        description: Optional[str] = None,
        traffic_percentage: float = 100.0,
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        primary_metric: str = "conversion",
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "model_id": model_id,
            "description": description,
            "traffic_percentage": traffic_percentage,
            "min_sample_size": min_sample_size,
            "confidence_level": confidence_level,
            "primary_metric": primary_metric,
            "variants": [
                {
                    "name": "control",
                    "model_version": control_version,
                    "traffic_allocation": control_allocation,
                    "is_control": True,
                },
                {
                    "name": "treatment",
                    "model_version": treatment_version,
                    "traffic_allocation": treatment_allocation,
                    "is_control": False,
                },
            ],
        }
        return self.api.post("/experiments", payload)

    def list(
        self,
        model_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        params: dict[str, Any] = {
            "org_id": self.org_id,
            "limit": limit,
            "offset": offset,
        }
        if model_id:
            params["model_id"] = model_id
        if status_filter:
            params["status"] = status_filter
        return self.api.get("/experiments", params=params)

    def get(self, experiment_id: str) -> dict[str, Any]:
        return self.api.get(f"/experiments/{experiment_id}")

    def start(self, experiment_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/start", {})

    def pause(self, experiment_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/pause", {})

    def resume(self, experiment_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/resume", {})

    def complete(self, experiment_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/complete", {})

    def cancel(self, experiment_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/cancel", {})

    def update_allocations(
        self, experiment_id: str, variants: List[dict[str, Any]]
    ) -> dict[str, Any]:
        return self.api.patch(f"/experiments/{experiment_id}/allocations", {"variants": variants})

    def get_target_groups(self, experiment_id: str) -> dict[str, Any]:
        return self.api.get(f"/experiments/{experiment_id}/target-groups")

    def set_target_groups(self, experiment_id: str, group_ids: List[str]) -> dict[str, Any]:
        return self.api.put(f"/experiments/{experiment_id}/target-groups", {"group_ids": group_ids})

    def add_target_group(self, experiment_id: str, group_id: str) -> dict[str, Any]:
        return self.api.post(f"/experiments/{experiment_id}/target-groups/{group_id}", {})

    def remove_target_group(self, experiment_id: str, group_id: str) -> dict[str, Any]:
        return self.api.delete(f"/experiments/{experiment_id}/target-groups/{group_id}")

    def get_analytics(self, experiment_id: str) -> dict[str, Any]:
        return self.api.get(f"/experiments/{experiment_id}/analytics")

    def get_analytics_sample_size(self, experiment_id: str) -> dict[str, Any]:
        return self.api.get(f"/experiments/{experiment_id}/analytics/sample-size")

    def get_analytics_timeseries(self, experiment_id: str) -> dict[str, Any]:
        return self.api.get(f"/experiments/{experiment_id}/analytics/timeseries")
