from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional

import httpx

from .api_client import EdgeMLClientError, _ApiClient
from .control_plane import ExperimentsAPI, RolloutsAPI

_MODELS_PATH = "/models"


def _detect_device_type() -> Optional[str]:
    """Auto-detect device profile from the runtime environment.

    On mobile SDKs (iOS/Android), this maps hardware info to a server
    device profile key.  The Python SDK runs on servers/desktops where
    auto-detection is not meaningful, so this returns ``None``.

    Override by setting the ``EDGEML_DEVICE_TYPE`` environment variable.
    """
    import os

    return os.environ.get("EDGEML_DEVICE_TYPE")


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
        # Check own org's models first
        data = self.api.get(_MODELS_PATH, params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == model:
                return item["id"]

        # Fallback: check models shared via federations the org belongs to
        try:
            federations = self.api.get("/federations", params={"org_id": self.org_id})
            if isinstance(federations, list):
                for fed in federations:
                    fed_id = fed.get("id")
                    if not fed_id:
                        continue
                    fed_models = self.api.get(f"/federations/{fed_id}/models")
                    if isinstance(fed_models, list):
                        for fm in fed_models:
                            if fm.get("name") == model:
                                return fm["id"]
        except Exception:
            pass  # Federation lookup is best-effort

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

    def download(
        self,
        model_id: str,
        version: str,
        destination: str,
        *,
        format: Optional[str] = None,
        device_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Download a model, optionally resolved for a specific device.

        Pass either ``format`` for an explicit format download, or
        ``device_type`` to auto-resolve the right format and fetch the
        device-specific MNN runtime config.  If neither is provided,
        checks the ``EDGEML_DEVICE_TYPE`` environment variable for
        auto-detection; otherwise defaults to ``onnx``.

        Args:
            model_id: Model identifier.
            version: Model version string.
            destination: Local directory to save files into.
            format: Explicit format (``onnx``, ``coreml``, ``tflite``).
            device_type: Device profile key (e.g. ``iphone_15_pro``).
                Mutually exclusive with *format*.

        Returns:
            Dict with ``model_path`` and, when a device-specific MNN
            config exists, ``mnn_config`` and ``config_path``.
        """
        import json
        import os

        os.makedirs(destination, exist_ok=True)

        ios_devices = {"iphone_15_pro", "iphone_14"}

        # Auto-detect device type from environment when not specified
        if device_type is None and format is None:
            device_type = _detect_device_type()

        if device_type is not None and format is None:
            fmt = "coreml" if device_type in ios_devices else "tflite"
        else:
            fmt = format or "onnx"

        # Download model file
        model_dest = os.path.join(destination, f"model.{fmt}")
        try:
            self._download_file(model_id, version, fmt, model_dest)
        except EdgeMLClientError:
            if device_type is not None:
                # Fall back to ONNX when target format isn't available
                fmt = "onnx"
                model_dest = os.path.join(destination, "model.onnx")
                self._download_file(model_id, version, fmt, model_dest)
            else:
                raise

        result: dict[str, Any] = {"model_path": model_dest, "format": fmt}

        # Fetch device-specific MNN config when downloading for a device
        if device_type is not None:
            try:
                config = self.get_device_config(model_id, device_type)
                config_path = os.path.join(destination, "mnn_config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                result["mnn_config"] = config
                result["config_path"] = config_path
            except EdgeMLClientError:
                pass  # No MNN config — standard runtime

        return result

    def _download_file(
        self,
        model_id: str,
        version: str,
        fmt: str,
        destination: str,
    ) -> None:
        """Download a single model file by format."""
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

    def download_version(
        self,
        model_id: str,
        version: str,
        fmt: str,
        destination: str,
    ) -> str:
        """Download a model file by explicit format.

        Prefer :meth:`download` which also supports device-aware downloads.
        """
        self._download_file(model_id, version, fmt, destination)
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
        payload: dict[str, Any] = {
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
        return self.api.patch(
            f"/models/{model_id}/versions/{version}/metrics", {"metrics": metrics}
        )

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
        return self.rollouts.create(
            model_id=model_id,
            version=version,
            rollout_percentage=float(rollout_percentage),
            target_percentage=float(target_percentage),
            increment_step=float(increment_step),
            start_immediately=start_immediately,
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

    def optimize(
        self,
        model_id: str,
        target_devices: Optional[list[str]] = None,
        accuracy_threshold: float = 0.95,
        size_budget_mb: Optional[float] = None,
    ) -> dict[str, Any]:
        """Optimize a model for on-device deployment.

        Runs the full optimization pipeline: feasibility check, pruning,
        quantization, format conversion, and validation.

        Args:
            model_id: Model identifier.
            target_devices: Target device profiles (e.g. ["iphone_15_pro", "pixel_8"]).
                Defaults to ["mid_range_android"] on the server if not specified.
            accuracy_threshold: Minimum accuracy retention ratio (0.0-1.0).
            size_budget_mb: Maximum model size in MB (optional).

        Returns:
            Dict with keys: feasible, original_size_mb, optimized_size_mb,
            compression_ratio, optimizations_applied, converted_paths,
            report, runtime_recommendation.
        """
        payload: dict[str, Any] = {"accuracy_threshold": accuracy_threshold}
        if target_devices is not None:
            payload["target_devices"] = target_devices
        if size_budget_mb is not None:
            payload["size_budget_mb"] = size_budget_mb
        return self.api.post(f"/models/{model_id}/optimize", payload)

    def get_device_config(
        self,
        model_id: str,
        device_type: str,
    ) -> dict[str, Any]:
        """Get the MNN runtime config optimized for a specific device type.

        Mobile SDKs use this to fetch the right runtime configuration
        (backend, threads, quantization, speculative decoding) for their device.

        Args:
            model_id: Model identifier.
            device_type: Device profile key (e.g. "iphone_15_pro", "pixel_8").

        Returns:
            MNN runtime config dict for the device.
        """
        return self.api.get(f"/models/{model_id}/optimized-config/{device_type}")

    def download_optimized(
        self,
        model_id: str,
        version: str,
        device_type: str,
        destination: str,
    ) -> dict[str, Any]:
        """Download the optimized model for a device.

        Convenience alias for ``download(device_type=...)``.
        Prefer :meth:`download` directly.
        """
        return self.download(
            model_id,
            version,
            destination,
            device_type=device_type,
        )

    def check_compatibility(
        self,
        model_id: str,
        target_devices: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Check model compatibility with target edge devices.

        Lightweight read-only check that classifies the model by size and
        reports per-device compatibility without running optimization.

        Args:
            model_id: Model identifier.
            target_devices: Device profile keys to check. None checks all.

        Returns:
            Dict with keys: model_name, model_params, model_size_mb,
            feasibility_category, compatible_devices, incompatible_devices,
            summary, recommendations.
        """
        params: dict[str, Any] = {}
        if target_devices is not None:
            params["devices"] = ",".join(target_devices)
        return self.api.get(f"/models/{model_id}/compatibility", params=params)
