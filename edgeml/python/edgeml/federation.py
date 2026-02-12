from __future__ import annotations

from typing import Any, Iterable, Optional

from .api_client import EdgeMLClientError, _ApiClient
from .control_plane import ExperimentsAPI, FederatedAnalyticsAPI, RolloutsAPI


class Federation:
    def __init__(
        self,
        auth_token_provider,
        name: str | None = None,
        org_id: str = "default",
        api_base: str = "https://api.edgeml.io/api/v1",
    ):
        self.api = _ApiClient(
            auth_token_provider=auth_token_provider,
            api_base=api_base,
        )
        self.org_id = org_id
        self.name = name or "default"
        self.last_model_id: Optional[str] = None
        self.last_version: Optional[str] = None
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=self.org_id)
        self.federation_id = self._resolve_or_create_federation()
        self.analytics = FederatedAnalyticsAPI(self.api, self.federation_id)

    def _resolve_or_create_federation(self) -> str:
        existing = self.api.get(
            "/federations",
            params={"org_id": self.org_id, "name": self.name},
        )
        if existing:
            return existing[0]["id"]
        created = self.api.post(
            "/federations",
            {"org_id": self.org_id, "name": self.name},
        )
        return created["id"]

    def invite(self, org_ids: Iterable[str]) -> list[dict[str, Any]]:
        payload = {"org_ids": list(org_ids)}
        return self.api.post(f"/federations/{self.federation_id}/invite", payload)

    def _resolve_model_id(self, model: str) -> str:
        data = self.api.get("/models", params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == model:
                return item["id"]
        return model

    def train(
        self,
        model: str,
        algorithm: str = "fedavg",
        rounds: int = 1,
        min_updates: int = 1,
        base_version: Optional[str] = None,
        new_version: Optional[str] = None,
        publish: bool = True,
        strategy: str = "metrics",
        update_format: str = "delta",
        architecture: Optional[str] = None,
        input_dim: int = 16,
        hidden_dim: int = 8,
        output_dim: int = 4,
    ) -> dict[str, Any]:
        if algorithm.lower() != "fedavg":
            raise EdgeMLClientError(f"Unsupported algorithm: {algorithm}")

        model_id = self._resolve_model_id(model)
        self.last_model_id = model_id
        result: Optional[dict[str, Any]] = None
        current_base = base_version

        for _ in range(rounds):
            payload = {
                "model_id": model_id,
                "base_version": current_base,
                "new_version": new_version,
                "min_updates": min_updates,
                "publish": publish,
                "strategy": strategy,
                "update_format": update_format,
                "architecture": architecture,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
            }
            result = self.api.post("/training/aggregate", payload)
            current_base = result.get("new_version")
            self.last_version = current_base
            new_version = None

        return result or {}

    def deploy(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        rollout_percentage: int = 10,
        target_percentage: int = 100,
        increment_step: int = 10,
        start_immediately: bool = True,
    ) -> dict[str, Any]:
        model_id = model_id or self.last_model_id
        if not model_id:
            raise EdgeMLClientError("model_id is required for deploy()")

        if not version:
            if self.last_version:
                version = self.last_version
            else:
                latest = self.api.get(f"/models/{model_id}/versions/latest")
                version = latest.get("version")
        if not version:
            raise EdgeMLClientError("version is required for deploy()")

        return self.rollouts.create(
            model_id=model_id,
            version=version,
            rollout_percentage=float(rollout_percentage),
            target_percentage=float(target_percentage),
            increment_step=float(increment_step),
            start_immediately=start_immediately,
        )

