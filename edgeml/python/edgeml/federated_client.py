from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from .api_client import _ApiClient, EdgeMLClientError
from .control_plane import ExperimentsAPI, RolloutsAPI
from .data_loader import load_data, validate_target, DataSource, DataLoadError

if TYPE_CHECKING:
    import pandas as pd
    import torch

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

WeightsData = Union[bytes, bytearray, Dict[str, Any], Any]
TrainResult = Tuple[Dict[str, Any], int, Optional[Dict[str, float]]]
LocalTrainFn = Callable[[Dict[str, Any]], TrainResult]


class FederatedClient:
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
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=self.org_id)

    def register(self, feature_schema: Optional[List[str]] = None) -> str:
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
        if model in self._model_cache:
            return self._model_cache[model]

        model_id = self._resolve_model_id(model)
        try:
            model_info = self.api.get(f"/models/{model_id}")
            self._model_cache[model] = model_info
            return model_info
        except EdgeMLClientError:
            return {}

    def _get_model_architecture(self, model: str) -> Dict[str, Any]:
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
        detected_features = None
        df = None

        is_data_source = (
            isinstance(data, (str, Path)) or
            (HAS_PANDAS and isinstance(data, pd.DataFrame))
        )

        if is_data_source:
            try:
                df = load_data(data)
            except DataLoadError as e:
                raise EdgeMLClientError(f"Failed to load data: {e}") from e

            architecture = self._get_model_architecture(model)
            if target_col is None:
                target_col = architecture.get("target_col", "target")

            if target_col not in df.columns:
                available = sorted([c for c in df.columns])
                raise EdgeMLClientError(
                    f"Target column '{target_col}' not found. "
                    f"Available columns: {available}"
                )

            if architecture:
                df = validate_target(
                    df,
                    target_col=target_col,
                    output_type=architecture.get("output_type", "binary"),
                    output_dim=architecture.get("output_dim", 1),
                )

            detected_features = [c for c in df.columns if c != target_col]
            self._detected_features = detected_features
            sample_count = sample_count or len(df)

            logger.info(
                f"Loaded data: {sample_count} samples, {len(detected_features)} features"
            )

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
                weights_data = df.drop(columns=[target_col]).values
            else:
                weights_data = data

            weights_bytes = self._serialize_weights(weights_data)
            weights_b64 = base64.b64encode(weights_bytes).decode("ascii")

            payload: Dict[str, Any] = {
                "model_id": model_id,
                "version": version,
                "device_id": self.device_id,
                "sample_count": sample_count or 0,
                "metrics": metrics or {},
                "update_format": update_format,
                "weights_data": weights_b64,
            }

            if detected_features:
                payload["detected_features"] = detected_features

            results.append(self.api.post("/training/weights", payload))

        return results

    def get_model_architecture(self, model: str) -> Dict[str, Any]:
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

