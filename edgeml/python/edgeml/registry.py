from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional

import httpx

from .api_client import EdgeMLClientError, _ApiClient
from .control_plane import ExperimentsAPI, RolloutsAPI

_MODELS_PATH = "/models"


class ModelRegistry:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        org_id: str = "default",
        api_base: str = "https://api.edgeml.io/api/v1",
        timeout: float = 60.0,
    ):
        self.api = _ApiClient(
            auth_token_provider=auth_token_provider,
            api_base=api_base,
            timeout=timeout,
        )
        self.org_id = org_id
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=self.org_id)

    def resolve_model_id(self, model: str) -> str:
        data = self.api.get(_MODELS_PATH, params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == model:
                return item["id"]
        return model

    def get_latest_version_info(self, model_id: str) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}/versions/latest")

    def list_models(
        self,
        framework: Optional[str] = None,
        use_case: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"org_id": self.org_id, "limit": limit, "offset": offset}
        if framework:
            params["framework"] = framework
        if use_case:
            params["use_case"] = use_case
        return self.api.get(_MODELS_PATH, params=params)

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}")

    def update_model(self, model_id: str, **fields: Any) -> dict[str, Any]:
        return self.api.patch(f"/models/{model_id}", fields)

    def delete_model(self, model_id: str) -> dict[str, Any]:
        return self.api.delete(f"/models/{model_id}")

    def get_latest_version(self, model_id: str) -> str:
        info = self.get_latest_version_info(model_id)
        version = info.get("version")
        if not version:
            raise EdgeMLClientError("Latest version not found")
        return str(version)

    def list_versions(
        self,
        model_id: str,
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status_filter:
            params["status"] = status_filter
        return self.api.get(f"/models/{model_id}/versions", params=params)

    def get_version(self, model_id: str, version: str) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}/versions/{version}")

    def create_version(
        self,
        model_id: str,
        version: str,
        storage_path: str,
        fmt: str,
        checksum: str,
        size_bytes: int,
        description: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": version,
            "storage_path": storage_path,
            "format": fmt,
            "checksum": checksum,
            "size_bytes": size_bytes,
        }
        if description is not None:
            payload["description"] = description
        if metrics is not None:
            payload["metrics"] = metrics
        return self.api.post(f"/models/{model_id}/versions", payload)

    def get_download_url(self, model_id: str, version: str, fmt: str = "onnx") -> dict[str, Any]:
        return self.api.get(
            f"/models/{model_id}/versions/{version}/download-url",
            params={"format": fmt},
        )

    def download_version(
        self,
        model_id: str,
        version: str,
        fmt: str,
        destination: str,
    ) -> str:
        payload = self.get_download_url(model_id, version, fmt=fmt)
        url = payload.get("url")
        if not url:
            raise EdgeMLClientError("Download URL missing from response")
        with httpx.Client(timeout=self.api.timeout) as client:
            res = client.get(url)
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        with open(destination, "wb") as handle:
            handle.write(res.content)
        return destination

    def ensure_model(
        self,
        name: str,
        framework: str,
        use_case: str,
        description: Optional[str] = None,
        model_contract: Optional[dict[str, Any]] = None,
        data_contract: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        data = self.api.get(_MODELS_PATH, params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == name:
                return item
        payload = {
            "name": name,
            "description": description or "",
            "framework": framework,
            "use_case": use_case,
            "org_id": self.org_id,
        }
        if model_contract:
            payload["model_contract"] = model_contract
        if data_contract:
            payload["data_contract"] = data_contract
        return self.api.post(_MODELS_PATH, payload)

    def upload_version_from_path(
        self,
        model_id: str,
        file_path: str,
        version: str,
        description: Optional[str] = None,
        formats: Optional[str] = None,
        onnx_data_path: Optional[str] = None,
        architecture: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"version": version}
        if description:
            data["description"] = description
        if formats:
            data["formats"] = formats
        if architecture:
            data["architecture"] = architecture
        if input_dim is not None:
            data["input_dim"] = str(input_dim)
        if hidden_dim is not None:
            data["hidden_dim"] = str(hidden_dim)
        if output_dim is not None:
            data["output_dim"] = str(output_dim)

        with contextlib.ExitStack() as stack:
            files: dict[str, Any] = {"file": stack.enter_context(open(file_path, "rb"))}
            if onnx_data_path:
                files["onnx_data"] = stack.enter_context(open(onnx_data_path, "rb"))
            with httpx.Client(timeout=self.api.timeout) as client:
                res = client.post(
                    f"{self.api.api_base}/models/{model_id}/versions/upload",
                    data=data,
                    files=files,
                    headers=self.api._headers(),
                )
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def publish_version(self, model_id: str, version: str) -> dict[str, Any]:
        return self.api.post(f"/models/{model_id}/versions/{version}/publish", {})

    def deprecate_version(self, model_id: str, version: str) -> dict[str, Any]:
        return self.api.post(f"/models/{model_id}/versions/{version}/deprecate", {})

    def update_version_metrics(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return self.api.patch(f"/models/{model_id}/versions/{version}/metrics", {"metrics": metrics})

    def delete_version(self, model_id: str, version: str) -> dict[str, Any]:
        return self.api.delete(f"/models/{model_id}/versions/{version}")

    def create_rollout(
        self,
        model_id: str,
        version: str,
        rollout_percentage: int = 10,
        target_percentage: int = 100,
        increment_step: int = 10,
        start_immediately: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "version": version,
            "rollout_percentage": rollout_percentage,
            "target_percentage": target_percentage,
            "increment_step": increment_step,
            "start_immediately": start_immediately,
        }
        return self.rollouts.create(
            model_id=model_id,
            version=payload["version"],
            rollout_percentage=float(payload["rollout_percentage"]),
            target_percentage=float(payload["target_percentage"]),
            increment_step=float(payload["increment_step"]),
            start_immediately=bool(payload["start_immediately"]),
        )

    def deploy_version(
        self,
        model_id: str,
        version: str,
        rollout_percentage: int = 10,
        target_percentage: int = 100,
        increment_step: int = 10,
        start_immediately: bool = True,
    ) -> dict[str, Any]:
        return self.create_rollout(
            model_id=model_id,
            version=version,
            rollout_percentage=rollout_percentage,
            target_percentage=target_percentage,
            increment_step=increment_step,
            start_immediately=start_immediately,
        )
