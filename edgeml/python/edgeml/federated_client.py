from __future__ import annotations

import base64
import io
import logging
import math
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

_MODEL_VERSION_ERROR = "Failed to resolve model version"

WeightsData = Union[bytes, bytearray, Dict[str, Any], Any]
TrainResult = Tuple[Dict[str, Any], int, Optional[Dict[str, float]]]
LocalTrainFn = Callable[[Dict[str, Any]], TrainResult]


def apply_filters(delta: Dict[str, Any], filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply a composable filter pipeline to a state-dict delta.

    Supported filter types:
      - gradient_clip: clip per-tensor norms to ``max_norm``
      - gaussian_noise: add N(0, stddev^2) noise
      - norm_validation: drop tensors exceeding ``max_norm``
      - sparsification: zero out values below top-k% by magnitude
      - quantization: round values to ``bits``-bit resolution
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise EdgeMLClientError("torch is required for filter pipeline") from exc

    result = {k: v.clone() if torch.is_tensor(v) else v for k, v in delta.items()}

    for f in filters:
        kind = f.get("type", "")

        if kind == "gradient_clip":
            max_norm = float(f.get("max_norm", 1.0))
            for key, tensor in result.items():
                if not torch.is_tensor(tensor):
                    continue
                norm = torch.norm(tensor.float())
                if norm > max_norm:
                    result[key] = tensor * (max_norm / norm)

        elif kind == "gaussian_noise":
            stddev = float(f.get("stddev", 0.01))
            for key, tensor in result.items():
                if not torch.is_tensor(tensor):
                    continue
                result[key] = tensor + torch.randn_like(tensor.float()) * stddev

        elif kind == "norm_validation":
            max_norm = float(f.get("max_norm", 10.0))
            for key in list(result.keys()):
                tensor = result[key]
                if torch.is_tensor(tensor) and torch.norm(tensor.float()) > max_norm:
                    del result[key]

        elif kind == "sparsification":
            top_k_percent = float(f.get("top_k_percent", 10.0))
            for key, tensor in result.items():
                if not torch.is_tensor(tensor):
                    continue
                flat = tensor.float().abs().flatten()
                k = max(1, int(math.ceil(flat.numel() * top_k_percent / 100.0)))
                threshold = torch.topk(flat, k).values[-1]
                mask = tensor.abs() >= threshold
                result[key] = tensor * mask

        elif kind == "quantization":
            bits = int(f.get("bits", 8))
            levels = (1 << bits) - 1
            for key, tensor in result.items():
                if not torch.is_tensor(tensor):
                    continue
                t_min = tensor.min()
                t_max = tensor.max()
                if t_min == t_max:
                    continue
                scale = (t_max - t_min) / levels
                result[key] = (torch.round((tensor - t_min) / scale) * scale) + t_min

    return result


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
        self._inference_client = None
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=self.org_id)

    @property
    def inference(self):
        """
        Returns a :class:`~edgeml.inference.StreamingInferenceClient` for
        streaming generative inference with timing metrics.

        The device must be registered before using this property.

        Example::

            from edgeml.inference import Modality

            client.register()
            for chunk in client.inference.generate("my-llm", prompt="hello", modality=Modality.TEXT):
                print(chunk.data.decode(), end="", flush=True)
        """
        if self._inference_client is None:
            from .inference import StreamingInferenceClient

            device_id = self.device_id or self.device_identifier
            self._inference_client = StreamingInferenceClient(
                api=self.api,
                device_id=device_id,
                org_id=self.org_id,
            )
        return self._inference_client

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

    def _prepare_training_data(
        self,
        model: str,
        data: DataSource,
        target_col: Optional[str],
    ) -> tuple["pd.DataFrame", List[str], str, int]:
        """Load and validate training data."""
        try:
            df = load_data(data)
        except DataLoadError as e:
            raise EdgeMLClientError(f"Failed to load data: {e}") from e

        architecture = self._get_model_architecture(model)
        if target_col is None:
            target_col = architecture.get("target_col", "target")

        if target_col not in df.columns:
            available = sorted(df.columns)
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
        sample_count = len(df)
        logger.info(
            f"Loaded data: {sample_count} samples, {len(detected_features)} features"
        )
        return df, detected_features, target_col, sample_count

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
            df, detected_features, target_col, data_count = self._prepare_training_data(
                model, data, target_col,
            )
            self._detected_features = detected_features
            sample_count = sample_count or data_count

        self.register(feature_schema=detected_features)

        results = []
        model_id = self._resolve_model_id(model)

        if not version:
            latest = self.api.get(f"/models/{model_id}/versions/latest")
            version = latest.get("version")
        if not version:
            raise EdgeMLClientError(_MODEL_VERSION_ERROR)

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
            raise EdgeMLClientError(_MODEL_VERSION_ERROR)
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
            raise EdgeMLClientError(_MODEL_VERSION_ERROR)

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

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def get_round_assignment(self) -> Optional[Dict[str, Any]]:
        """Check if this device is selected for an active training round.

        Returns the round assignment dict (containing ``round_id``, ``config``,
        etc.) when the device has been selected, or *None* when there is no
        pending assignment.
        """
        self.register()
        try:
            result = self.api.get(
                "/training/rounds",
                params={"device_id": self.device_id, "status": "active"},
            )
        except EdgeMLClientError:
            return None
        rounds = result if isinstance(result, list) else result.get("rounds", [])
        if not rounds:
            return None
        return rounds[0]

    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get the status of a training round (progress, participant count, etc.)."""
        return self.api.get(f"/training/rounds/{round_id}/status")

    def participate_in_round(
        self,
        round_id: str,
        local_train_fn: LocalTrainFn,
        format: str = "pytorch",
    ) -> Dict[str, Any]:
        """Full round lifecycle: fetch config, pull model, train, filter, upload.

        ``local_train_fn`` receives the base state-dict and must return
        ``(updated_state_dict, sample_count, metrics)``.

        The round config is inspected for strategy-specific parameters
        (``proximal_mu`` for FedProx, ``lambda_ditto`` for Ditto,
        ``head_layers`` for FedPer) and client-side filter settings.
        """
        self.register()

        # 1. Fetch round config
        round_info = self.api.get(f"/training/rounds/{round_id}/status")
        config = round_info.get("config", {})
        model_id = config.get("model_id") or round_info.get("model_id", "")
        version = config.get("version") or round_info.get("version")

        if not version:
            latest = self.api.get(f"/models/{model_id}/versions/latest")
            version = latest.get("version")
        if not version:
            raise EdgeMLClientError(_MODEL_VERSION_ERROR)

        # 2. Pull global model
        base_bytes = self.pull_model(model_id, version=version, format=format)
        base_state = self._deserialize_weights(base_bytes)

        # 3. Train locally
        updated_state, sample_count, metrics = local_train_fn(base_state)

        # 4. Compute delta
        delta = compute_state_dict_delta(base_state, updated_state)

        # 5. Apply client-side filters from round config
        filter_list = config.get("filters", [])

        # If the round specifies gradient clipping via top-level config, inject
        # a gradient_clip filter so that the delta is always clipped.
        clip_norm = config.get("clip_norm")
        if clip_norm is not None:
            filter_list = [{"type": "gradient_clip", "max_norm": clip_norm}] + filter_list

        if filter_list:
            delta = apply_filters(delta, filter_list)

        # 6. Upload
        weights_data = self._serialize_weights(delta)
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "version": version,
            "device_id": self.device_id,
            "round_id": round_id,
            "sample_count": sample_count or 0,
            "metrics": metrics or {},
            "update_format": "delta",
            "weights_data": base64.b64encode(weights_data).decode("ascii"),
        }
        return self.api.post("/training/weights", payload)

    # ------------------------------------------------------------------
    # Personalization
    # ------------------------------------------------------------------

    def get_personalized_model(self) -> Dict[str, Any]:
        """Fetch the personalized model state for this device."""
        self.register()
        return self.api.get(f"/training/personalized/{self.device_id}")

    def upload_personalized_update(
        self,
        weights: WeightsData,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Upload a personalized model update for this device."""
        self.register()
        weights_bytes = self._serialize_weights(weights)
        payload: Dict[str, Any] = {
            "weights_data": base64.b64encode(weights_bytes).decode("ascii"),
            "metrics": metrics or {},
        }
        return self.api.post(f"/training/personalized/{self.device_id}", payload)

    def train_ditto(
        self,
        global_model: Dict[str, Any],
        personal_model: Dict[str, Any],
        local_train_fn: LocalTrainFn,
        lambda_ditto: float = 0.1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int, Optional[Dict[str, float]]]:
        """Train both global and personal models using the Ditto strategy.

        1. Train a copy of the global model normally via *local_train_fn* to
           produce a global update.
        2. Train the personal model with a proximal regularization term:
           ``loss_personal = task_loss + lambda_ditto/2 * ||w_p - w_g||^2``

        Returns ``(global_update, personal_update, sample_count, metrics)``.

        ``local_train_fn`` is called **once** with the global model and must
        return ``(updated_state, sample_count, metrics)``.  The personal model
        is then regularized toward the global weights using *lambda_ditto*.
        """
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise EdgeMLClientError("torch is required for train_ditto") from exc

        # 1. Train global model
        global_updated, sample_count, metrics = local_train_fn(global_model)

        # 2. Regularize personal model toward global weights
        personal_updated: Dict[str, Any] = {}
        for key in personal_model:
            p_tensor = personal_model[key]
            g_tensor = global_model.get(key)
            if torch.is_tensor(p_tensor) and g_tensor is not None and torch.is_tensor(g_tensor):
                # Proximal step: move personal weights toward global
                personal_updated[key] = (
                    p_tensor - lambda_ditto * (p_tensor.detach() - g_tensor.detach())
                )
            else:
                personal_updated[key] = p_tensor

        return global_updated, personal_updated, sample_count, metrics

    def train_fedper(
        self,
        model: Dict[str, Any],
        head_layers: List[str],
        local_train_fn: LocalTrainFn,
    ) -> Tuple[Dict[str, Any], int, Optional[Dict[str, float]]]:
        """Train using FedPer: full model trained locally, only body layers uploaded.

        *head_layers* lists the parameter name prefixes that belong to the
        personal head (e.g. ``["classifier", "head"]``).  These are excluded
        from the returned delta so only body layers are shared with the server.

        Returns ``(body_delta, sample_count, metrics)``.
        """
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise EdgeMLClientError("torch is required for train_fedper") from exc

        base_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in model.items()}
        updated_state, sample_count, metrics = local_train_fn(model)

        # Compute delta excluding head layers
        delta: Dict[str, Any] = {}
        for key, base_tensor in base_state.items():
            # Skip head layers
            if any(key.startswith(prefix) or key == prefix for prefix in head_layers):
                continue
            updated_tensor = updated_state.get(key)
            if updated_tensor is None:
                continue
            if torch.is_tensor(base_tensor) and torch.is_tensor(updated_tensor):
                delta[key] = updated_tensor.detach().cpu() - base_tensor.detach().cpu()

        return delta, sample_count, metrics

    # ------------------------------------------------------------------
    # Privacy budget
    # ------------------------------------------------------------------

    def get_privacy_budget(self, federation_id: str) -> Dict[str, Any]:
        """Get the current privacy budget and accounting info for a federation."""
        return self.api.get(f"/federations/{federation_id}/privacy")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        state = torch.load(buffer, map_location="cpu", weights_only=True)
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

