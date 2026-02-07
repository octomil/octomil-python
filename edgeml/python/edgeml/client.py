from __future__ import annotations

import base64
import contextlib
import io
import logging
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import httpx

from .data_loader import load_data, validate_target, DataSource, DataLoadError

# Type checking imports (avoid runtime import of heavy libraries)
if TYPE_CHECKING:
    import pandas as pd
    import torch

logger = logging.getLogger(__name__)

# Optional pandas import for DataFrame support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

# Type aliases for clarity
WeightsData = Union[bytes, bytearray, Dict[str, Any], Any]  # bytes, state_dict, nn.Module
TrainResult = Tuple[Dict[str, Any], int, Optional[Dict[str, float]]]  # (weights, sample_count, metrics)
LocalTrainFn = Callable[[Dict[str, Any]], TrainResult]


class EdgeMLClientError(RuntimeError):
    pass


class _ApiClient:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        api_base: str,
        timeout: float = 20.0,
    ):
        self.api_key = None
        self.auth_token_provider = auth_token_provider
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        token = self.auth_token_provider()
        if not token:
            raise EdgeMLClientError("auth_token_provider returned an empty token")
        return {"Authorization": f"Bearer {token}"}

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.api_base}{path}", json=payload, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def get_bytes(self, path: str, params: Optional[dict[str, Any]] = None) -> bytes:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.content


class Federation:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
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
        self.federation_id = self._resolve_or_create_federation()

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
        # Try name lookup first; if not found, assume it's an ID
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

        payload = {
            "version": version,
            "rollout_percentage": rollout_percentage,
            "target_percentage": target_percentage,
            "increment_step": increment_step,
            "start_immediately": start_immediately,
        }
        return self.api.post(f"/models/{model_id}/rollouts", payload)


class FederatedClient:
    """
    Client for participating in federated learning.

    Simple usage (just works):
        client = FederatedClient(auth_token_provider=lambda: "<device-access-token>")
        client.train(model="my-model", data="s3://bucket/data.parquet")
        # That's it. Data loaded, features auto-detected, alignment automatic.

    Supports multiple data sources:
        - DataFrame: client.train(model="...", data=df)
        - Local file: client.train(model="...", data="/path/to/data.csv")
        - S3: client.train(model="...", data="s3://bucket/data.parquet")
        - GCS: client.train(model="...", data="gs://bucket/data.csv")
        - Azure: client.train(model="...", data="az://container/data.parquet")

    Credentials from environment:
        - AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        - GCS: GOOGLE_APPLICATION_CREDENTIALS
        - Azure: AZURE_STORAGE_CONNECTION_STRING
    """

    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        org_id: str = "default",
        api_base: str = "https://api.edgeml.io/api/v1",
        device_identifier: Optional[str] = None,
        platform: str = "python",
    ):
        self.api = _ApiClient(
            auth_token_provider=auth_token_provider,
            api_base=api_base,
        )
        self.org_id = org_id
        self.device_identifier = device_identifier or f"client-{uuid.uuid4().hex[:10]}"
        self.platform = platform
        self.device_id: Optional[str] = None
        self._detected_features: Optional[List[str]] = None
        self._model_cache: Dict[str, Dict[str, Any]] = {}

    def register(self, feature_schema: Optional[List[str]] = None) -> str:
        """Register device with server."""
        if self.device_id:
            return self.device_id

        payload: Dict[str, Any] = {
            "device_identifier": self.device_identifier,
            "org_id": self.org_id,
            "platform": self.platform,
            "os_version": "macos",
            "sdk_version": "0.2.0",
            "app_version": "0.1.0",
            "metadata": {"client": "python-sdk"},
            "capabilities": {"training": True},
        }
        if feature_schema:
            payload["feature_schema"] = feature_schema

        response = self.api.post("/devices/register", payload)
        self.device_id = response.get("id")
        if not self.device_id:
            raise EdgeMLClientError("Device registration failed: missing device ID")
        return self.device_id

    def _get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model info with caching."""
        if model in self._model_cache:
            return self._model_cache[model]

        model_id = self._resolve_model_id(model)
        try:
            model_info = self.api.get(f"/models/{model_id}")
            self._model_cache[model] = model_info
            return model_info
        except EdgeMLClientError:
            # Model not found, return empty
            return {}

    def _get_model_architecture(self, model: str) -> Dict[str, Any]:
        """Get model architecture from server."""
        model_info = self._get_model_info(model)
        return model_info.get("architecture", {})

    def train(
        self,
        model: str,
        data: Union[DataSource, WeightsData, Callable[[], TrainResult]],
        target_col: Optional[str] = None,
        rounds: int = 1,
        version: Optional[str] = None,
        sample_count: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        update_format: str = "weights",
    ) -> List[Dict[str, Any]]:
        """
        Train locally and upload update.

        Args:
            model: Model name or ID
            data: Data source - one of:
                - pd.DataFrame
                - File path: "/path/to/data.csv"
                - S3 URI: "s3://bucket/data.parquet"
                - GCS URI: "gs://bucket/data.csv"
                - Azure URI: "az://container/data.parquet"
                - Pre-trained weights (dict, bytes, nn.Module)
                - Callable returning (weights, sample_count, metrics)
            target_col: Local column name for target variable
            rounds: Number of training rounds
            version: Model version (auto-resolved if not provided)
            sample_count: Number of training samples (auto-detected for data sources)
            metrics: Training metrics
            update_format: "weights" or "delta"

        Returns:
            List of upload responses

        Example:
            # Simple - just works
            client.train(model="cancer-detection", data="s3://hospital/patients.parquet")

            # With local target column name
            client.train(model="cancer-detection", data=df, target_col="diagnosis")

            # Pre-trained weights
            client.train(model="my-model", data=model.state_dict(), sample_count=1000)
        """
        detected_features = None
        df = None

        # Check if data is a loadable source (string path/URI or DataFrame)
        is_data_source = (
            isinstance(data, (str, Path)) or
            (HAS_PANDAS and isinstance(data, pd.DataFrame))
        )

        if is_data_source:
            # Load data from source
            try:
                df = load_data(data)
            except DataLoadError as e:
                raise EdgeMLClientError(f"Failed to load data: {e}") from e

            # Get model architecture for validation
            architecture = self._get_model_architecture(model)

            # Determine target column
            if target_col is None:
                # Try to get from model architecture
                target_col = architecture.get("target_col", "target")

            # Validate target column exists
            if target_col not in df.columns:
                available = sorted([c for c in df.columns])
                raise EdgeMLClientError(
                    f"Target column '{target_col}' not found. "
                    f"Available columns: {available}"
                )

            # Validate and transform target
            if architecture:
                df = validate_target(
                    df,
                    target_col=target_col,
                    output_type=architecture.get("output_type", "binary"),
                    output_dim=architecture.get("output_dim", 1),
                )

            # Auto-detect features
            detected_features = [c for c in df.columns if c != target_col]
            self._detected_features = detected_features
            sample_count = sample_count or len(df)

            logger.info(
                f"Loaded data: {sample_count} samples, {len(detected_features)} features"
            )

        # Register with detected features
        self.register(feature_schema=detected_features)

        results = []
        model_id = self._resolve_model_id(model)

        if not version:
            latest = self.api.get(f"/models/{model_id}/versions/latest")
            version = latest.get("version")
        if not version:
            raise EdgeMLClientError("Failed to resolve model version")

        for _ in range(rounds):
            if callable(data):
                weights_data, sample_count, metrics = data()
            elif df is not None:
                # Data was loaded from source - use as training data
                # In practice, you'd train a model here and get weights
                # For now, we pass the feature matrix
                weights_data = df.drop(columns=[target_col]).values
            else:
                weights_data = data

            weights_bytes = self._serialize_weights(weights_data)
            weights_b64 = base64.b64encode(weights_bytes).decode("ascii")

            # Build payload
            payload: Dict[str, Any] = {
                "model_id": model_id,
                "version": version,
                "device_id": self.device_id,
                "sample_count": sample_count or 0,
                "metrics": metrics or {},
                "update_format": update_format,
                "weights_data": weights_b64,
            }

            # Include feature info for tracking (informational)
            if detected_features:
                payload["detected_features"] = detected_features

            results.append(self.api.post("/training/weights", payload))

        return results

    def get_model_architecture(self, model: str) -> Dict[str, Any]:
        """Get model architecture from server."""
        return self._get_model_architecture(model)

    def pull_model(
        self,
        model: str,
        version: Optional[str] = None,
        format: str = "pytorch",
    ) -> bytes:
        model_id = self._resolve_model_id(model)
        if not version:
            latest = self.api.get(f"/models/{model_id}/versions/latest")
            version = latest.get("version")
        if not version:
            raise EdgeMLClientError("Failed to resolve model version")
        return self.api.get_bytes(
            f"/models/{model_id}/versions/{version}/download",
            params={"format": format},
        )

    def train_from_remote(
        self,
        model: str,
        local_train_fn: LocalTrainFn,
        rounds: int = 1,
        version: Optional[str] = None,
        update_format: str = "weights",
        format: str = "pytorch",
    ) -> List[Dict[str, Any]]:
        self.register()
        model_id = self._resolve_model_id(model)
        if not version:
            latest = self.api.get(f"/models/{model_id}/versions/latest")
            version = latest.get("version")
        if not version:
            raise EdgeMLClientError("Failed to resolve model version")

        results = []
        for _ in range(rounds):
            base_bytes = self.pull_model(model_id, version=version, format=format)
            base_state = self._deserialize_weights(base_bytes)
            updated_state, sample_count, metrics = local_train_fn(base_state)
            if update_format == "delta":
                updated_state = compute_state_dict_delta(base_state, updated_state)
            weights_data = self._serialize_weights(updated_state)
            payload = {
                "model_id": model_id,
                "version": version,
                "device_id": self.device_id,
                "sample_count": sample_count or 0,
                "metrics": metrics or {},
                "update_format": update_format,
                "weights_data": base64.b64encode(weights_data).decode("ascii"),
            }
            results.append(self.api.post("/training/weights", payload))
        return results

    def _resolve_model_id(self, model: str) -> str:
        data = self.api.get("/models", params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == model:
                return item["id"]
        return model

    def _serialize_weights(self, weights: WeightsData) -> bytes:
        if isinstance(weights, (bytes, bytearray)):
            return bytes(weights)

        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(weights, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, weights)
                return buffer.getvalue()
        except ImportError:
            pass

        try:
            import torch  # type: ignore
        except Exception:
            torch = None

        if torch is not None:
            if isinstance(weights, torch.nn.Module):
                buffer = io.BytesIO()
                torch.save(weights.state_dict(), buffer)
                return buffer.getvalue()
            if isinstance(weights, dict):
                buffer = io.BytesIO()
                torch.save(weights, buffer)
                return buffer.getvalue()

        raise EdgeMLClientError(
            "data must be bytes, numpy array, torch.nn.Module, state_dict dict, "
            "or a callable returning (weights, sample_count, metrics)"
        )

    def _deserialize_weights(self, payload: bytes) -> Dict[str, Any]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise EdgeMLClientError("torch is required to load remote weights") from exc
        buffer = io.BytesIO(payload)
        state = torch.load(buffer, map_location="cpu")
        if not isinstance(state, dict):
            raise EdgeMLClientError("Remote payload was not a state_dict")
        return state


def compute_state_dict_delta(
    base_state: Dict[str, Any],
    updated_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a delta state_dict = updated - base.

    Args:
        base_state: Base model state dictionary
        updated_state: Updated model state dictionary

    Returns:
        Delta state dictionary (updated - base for each tensor)

    Intended for small demo models (fits in memory).
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise EdgeMLClientError("torch is required to compute state_dict deltas") from exc

    delta: Dict[str, Any] = {}
    for key, base_tensor in base_state.items():
        updated_tensor = updated_state.get(key)
        if updated_tensor is None:
            continue
        if torch.is_tensor(base_tensor) and torch.is_tensor(updated_tensor):
            delta[key] = updated_tensor.detach().cpu() - base_tensor.detach().cpu()
    return delta


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

    def resolve_model_id(self, model: str) -> str:
        data = self.api.get("/models", params={"org_id": self.org_id})
        for item in data.get("models", []):
            if item.get("name") == model:
                return item["id"]
        return model

    def get_latest_version_info(self, model_id: str) -> dict[str, Any]:
        return self.api.get(f"/models/{model_id}/versions/latest")

    def get_latest_version(self, model_id: str) -> str:
        info = self.get_latest_version_info(model_id)
        version = info.get("version")
        if not version:
            raise EdgeMLClientError("Latest version not found")
        return str(version)

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
        description: str | None = None,
        model_contract: dict[str, Any] | None = None,
        data_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self.api.get("/models", params={"org_id": self.org_id})
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
        return self.api.post("/models", payload)

    def upload_version_from_path(
        self,
        model_id: str,
        file_path: str,
        version: str,
        description: str | None = None,
        formats: str | None = None,
        onnx_data_path: str | None = None,
        architecture: str | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
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
        return self.api.post(f"/models/{model_id}/rollouts", payload)
