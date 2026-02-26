from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from octomil.device_info import get_battery_level, is_charging

from .api_client import OctomilClientError, _ApiClient
from .control_plane import ExperimentsAPI, RolloutsAPI
from .data_loader import DataLoadError, DataSource, load_data, validate_target
from .filters import FilterRegistry, FilterResult  # noqa: F401 -- re-export
from .filters import apply_filters as _apply_filters_impl
from .resilience import (
    check_network_quality,  # noqa: F401 -- re-export for tests
    check_training_eligibility,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

def _get_sdk_version() -> str:
    """Return the SDK version, avoiding circular imports."""
    from octomil import __version__

    return __version__


_MODEL_VERSION_ERROR = "Failed to resolve model version"
_TRAINING_WEIGHTS_ENDPOINT = "/training/weights"

WeightsData = Union[bytes, bytearray, Dict[str, Any], Any]
TrainResult = Tuple[Dict[str, Any], int, Optional[Dict[str, float]]]
LocalTrainFn = Callable[[Dict[str, Any]], TrainResult]


def _apply_fedprox_correction(delta: Dict[str, Any], mu: float) -> Dict[str, Any]:
    """Apply FedProx proximal correction to a weight delta.

    The FedProx objective adds ``(mu/2) * ||w - w_global||^2`` to the local
    loss.  When the user's training loop does not include this term, we
    approximate its effect by scaling: ``delta / (1 + mu)``.  This keeps local
    updates closer to the global model, reducing client drift.

    Args:
        delta: State-dict delta (updated - base).
        mu: Proximal regularization strength (> 0).

    Returns:
        A new delta dict with the correction applied.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise OctomilClientError("torch is required for FedProx correction") from exc

    scale = 1.0 / (1.0 + mu)
    corrected: Dict[str, Any] = {}
    for key, value in delta.items():
        if torch.is_tensor(value):
            corrected[key] = value * scale
        else:
            corrected[key] = value
    return corrected


def apply_filters(delta: Dict[str, Any], filters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply a composable filter pipeline to a state-dict delta.

    This is a backward-compatible wrapper around :func:`octomil.filters.apply_filters`.
    It accepts the same dict-based filter configs and returns a plain dict.

    For the full API with audit trail and :class:`~octomil.filters.DeltaFilter`
    support, use :func:`octomil.filters.apply_filters` directly.
    """
    result = _apply_filters_impl(delta, list(filters))
    return result.delta


class FederatedClient:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        org_id: str = "default",
        api_base: str = "https://api.octomil.com/api/v1",
        device_identifier: Optional[str] = None,
        platform: str = "python",
        secure_aggregation: bool = False,
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
        self.secure_aggregation = secure_aggregation
        self.rollouts = RolloutsAPI(self.api)
        self.experiments = ExperimentsAPI(self.api, org_id=self.org_id)

    @property
    def _reporter(self):
        """Return the global TelemetryReporter, or None."""
        try:
            from octomil import get_reporter
            return get_reporter()
        except Exception:
            return None

    def _report_funnel(self, **kwargs) -> None:
        """Best-effort funnel event reporting."""
        reporter = self._reporter
        if reporter:
            try:
                reporter.report_funnel_event(**kwargs)
            except Exception:
                logger.debug("Telemetry reporting failed in FederatedClient")

    @property
    def inference(self):
        """
        Returns a :class:`~octomil.inference.StreamingInferenceClient` for
        streaming generative inference with timing metrics.

        The device must be registered before using this property.

        Example::

            from octomil.inference import Modality

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
            "sdk_version": _get_sdk_version(),
            "app_version": "0.1.0",
            "metadata": {"client": "python-sdk"},
            "capabilities": {"training": True},
        }
        if feature_schema:
            payload["feature_schema"] = feature_schema

        response = self.api.post("/devices/register", payload)
        self.device_id = response.get("id")
        if not self.device_id:
            raise OctomilClientError("Device registration failed: missing device ID")
        return self.device_id

    def _get_model_info(self, model: str) -> Dict[str, Any]:
        if model in self._model_cache:
            return self._model_cache[model]

        model_id = self._resolve_model_id(model)
        try:
            model_info = self.api.get(f"/models/{model_id}")
            self._model_cache[model] = model_info
            return model_info
        except OctomilClientError:
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
            raise OctomilClientError(f"Failed to load data: {e}") from e

        architecture = self._get_model_architecture(model)
        if target_col is None:
            target_col = architecture.get("target_col", "target")

        if target_col not in df.columns:
            available = sorted(df.columns)
            raise OctomilClientError(
                f"Target column '{target_col}' not found. " f"Available columns: {available}"
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
        logger.info(f"Loaded data: {sample_count} samples, {len(detected_features)} features")
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

        is_data_source = isinstance(data, (str, Path)) or (
            HAS_PANDAS and isinstance(data, pd.DataFrame)
        )

        if is_data_source:
            df, detected_features, target_col, data_count = self._prepare_training_data(
                model,
                data,
                target_col,
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
            raise OctomilClientError(_MODEL_VERSION_ERROR)

        t0 = time.monotonic()
        session_id = uuid.uuid4().hex[:12]
        self._report_funnel(
            stage="training_started",
            model_id=model,
            session_id=session_id,
        )

        try:
            for round_idx in range(rounds):
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

                upload_t0 = time.monotonic()
                try:
                    result = self.api.post(_TRAINING_WEIGHTS_ENDPOINT, payload)
                    results.append(result)
                    self._report_funnel(
                        stage="weight_upload",
                        success=True,
                        model_id=model,
                        session_id=session_id,
                        duration_ms=int((time.monotonic() - upload_t0) * 1000),
                    )
                except Exception as exc:
                    self._report_funnel(
                        stage="weight_upload",
                        success=False,
                        model_id=model,
                        session_id=session_id,
                        failure_reason=str(exc),
                        failure_category="upload_error",
                        duration_ms=int((time.monotonic() - upload_t0) * 1000),
                    )
                    raise
        except Exception as exc:
            self._report_funnel(
                stage="training_failed",
                model_id=model,
                session_id=session_id,
                failure_reason=str(exc),
                failure_category="training_error",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            raise

        self._report_funnel(
            stage="training_completed",
            success=True,
            model_id=model,
            session_id=session_id,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
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
            raise OctomilClientError(_MODEL_VERSION_ERROR)
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
            raise OctomilClientError(_MODEL_VERSION_ERROR)

        t0 = time.monotonic()
        session_id = uuid.uuid4().hex[:12]
        self._report_funnel(
            stage="training_started",
            model_id=model,
            session_id=session_id,
        )

        results = []
        try:
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
                upload_t0 = time.monotonic()
                try:
                    result = self.api.post(_TRAINING_WEIGHTS_ENDPOINT, payload)
                    results.append(result)
                    self._report_funnel(
                        stage="weight_upload",
                        success=True,
                        model_id=model,
                        session_id=session_id,
                        duration_ms=int((time.monotonic() - upload_t0) * 1000),
                    )
                except Exception as exc:
                    self._report_funnel(
                        stage="weight_upload",
                        success=False,
                        model_id=model,
                        session_id=session_id,
                        failure_reason=str(exc),
                        failure_category="upload_error",
                        duration_ms=int((time.monotonic() - upload_t0) * 1000),
                    )
                    raise
        except Exception as exc:
            self._report_funnel(
                stage="training_failed",
                model_id=model,
                session_id=session_id,
                failure_reason=str(exc),
                failure_category="training_error",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            raise

        self._report_funnel(
            stage="training_completed",
            success=True,
            model_id=model,
            session_id=session_id,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
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
        except OctomilClientError:
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

        When ``secure_aggregation`` is enabled (via the constructor flag **or**
        the round config), the update is masked using Shamir secret sharing
        before being uploaded.
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
            raise OctomilClientError(_MODEL_VERSION_ERROR)

        # 2. Pull global model
        base_bytes = self.pull_model(model_id, version=version, format=format)
        base_state = self._deserialize_weights(base_bytes)

        # 3. Train locally
        updated_state, sample_count, metrics = local_train_fn(base_state)

        # 4. Compute delta
        delta = compute_state_dict_delta(base_state, updated_state)

        # 4b. Apply FedProx proximal correction if configured.
        # The proximal term (mu/2)*||w - w_global||^2 dampens the update so
        # local models stay closer to the global model.  When the inner
        # training loop doesn't include the proximal loss directly, we apply
        # the closed-form correction: delta_corrected = delta / (1 + mu).
        proximal_mu = config.get("proximal_mu")
        if proximal_mu is not None and float(proximal_mu) > 0:
            delta = _apply_fedprox_correction(delta, float(proximal_mu))

        # 5. Apply client-side filters from round config
        filter_list = config.get("filters", [])

        # If the round specifies gradient clipping via top-level config, inject
        # a gradient_clip filter so that the delta is always clipped.
        clip_norm = config.get("clip_norm")
        if clip_norm is not None:
            filter_list = [{"type": "gradient_clip", "max_norm": clip_norm}] + filter_list

        if filter_list:
            delta = apply_filters(delta, filter_list)

        # 6. Serialize weights
        weights_data = self._serialize_weights(delta)

        # 7. SecAgg: mask the update if enabled
        use_secagg = self.secure_aggregation or config.get("secure_aggregation", False)
        if use_secagg:
            weights_data = self._secagg_mask_and_share(round_id, weights_data)

        # 8. Upload
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
        upload_t0 = time.monotonic()
        try:
            result = self.api.post(_TRAINING_WEIGHTS_ENDPOINT, payload)
            self._report_funnel(
                stage="weight_upload",
                success=True,
                model_id=model_id,
                session_id=round_id,
                duration_ms=int((time.monotonic() - upload_t0) * 1000),
            )
            return result
        except Exception as exc:
            self._report_funnel(
                stage="weight_upload",
                success=False,
                model_id=model_id,
                session_id=round_id,
                failure_reason=str(exc),
                failure_category="upload_error",
                duration_ms=int((time.monotonic() - upload_t0) * 1000),
            )
            raise

    def train_if_eligible(
        self,
        round_id: str,
        local_train_fn: LocalTrainFn,
        min_battery: int = 15,
        gradient_cache: Optional[Any] = None,
        format: str = "pytorch",
    ) -> Dict[str, Any]:
        """Check eligibility, train if OK, cache gradient on upload failure.

        Returns a dict with ``skipped`` (bool), ``reason`` (str or None),
        and ``result`` (server response when training succeeded).
        """
        battery_level = get_battery_level()
        charging = is_charging()

        eligibility = check_training_eligibility(
            battery_level=battery_level,
            min_battery=min_battery,
            charging=charging,
        )
        if not eligibility.eligible:
            self._report_funnel(
                stage="training_started",
                success=False,
                session_id=round_id,
                failure_category="device_ineligible",
                failure_reason=eligibility.reason,
            )
            return {"skipped": True, "reason": eligibility.reason}

        try:
            result = self.participate_in_round(
                round_id=round_id,
                local_train_fn=local_train_fn,
                format=format,
            )
            return {"skipped": False, "reason": None, "result": result}
        except Exception as exc:
            logger.warning("Training upload failed for round %s: %s", round_id, exc)
            if gradient_cache is not None:
                weights_bytes = self._serialize_weights({"error_round": round_id})
                gradient_cache.store(
                    round_id=round_id,
                    device_id=self.device_id or self.device_identifier,
                    weights_data=weights_bytes,
                    sample_count=0,
                )
                self._report_funnel(
                    stage="weight_upload",
                    success=False,
                    session_id=round_id,
                    failure_category="upload_failed_cached",
                    failure_reason=str(exc),
                )
            return {"skipped": True, "reason": "upload_failed"}

    def _secagg_mask_and_share(self, round_id: str, weights_data: bytes) -> bytes:
        """Run the client side of the SecAgg protocol and return masked weights."""
        from .secagg import SecAggClient
        from .secagg import SecAggConfig as _SAConfig

        # Fetch session parameters from the server.
        assert self.device_id is not None, "Device must be registered before SecAgg"
        session_info = self.api.secagg_get_session(round_id, self.device_id)

        sa_config = _SAConfig(
            session_id=session_info.get("session_id", ""),
            round_id=round_id,
            threshold=session_info.get("threshold", 2),
            total_clients=session_info.get("total_clients", 3),
            field_size=session_info.get("field_size", (1 << 127) - 1),
            key_length=session_info.get("key_length", 256),
            noise_scale=session_info.get("noise_scale"),
        )

        sac = SecAggClient(sa_config)

        # Phase 1 -- share keys with all participants.
        shares = sac.generate_key_shares()
        shares_bytes = SecAggClient.serialize_shares(shares)
        self.api.secagg_submit_shares(round_id, self.device_id, shares_bytes)

        # Phase 2 -- mask the weights.
        masked_data = sac.mask_model_update(weights_data)

        logger.info("SecAgg: masked update for round %s", round_id)
        return masked_data

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
            raise OctomilClientError("torch is required for train_ditto") from exc

        # 1. Train global model
        global_updated, sample_count, metrics = local_train_fn(global_model)

        # 2. Regularize personal model toward global weights
        personal_updated: Dict[str, Any] = {}
        for key in personal_model:
            p_tensor = personal_model[key]
            g_tensor = global_model.get(key)
            if torch.is_tensor(p_tensor) and g_tensor is not None and torch.is_tensor(g_tensor):
                # Proximal step: move personal weights toward global
                personal_updated[key] = p_tensor - lambda_ditto * (
                    p_tensor.detach() - g_tensor.detach()
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
            raise OctomilClientError("torch is required for train_fedper") from exc

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

        raise OctomilClientError(
            "data must be bytes, numpy array, torch.nn.Module, state_dict dict, "
            "or a callable returning (weights, sample_count, metrics)"
        )

    def _deserialize_weights(self, payload: bytes) -> Dict[str, Any]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise OctomilClientError("torch is required to load remote weights") from exc
        buffer = io.BytesIO(payload)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise OctomilClientError("Remote payload was not a state_dict")
        return state


def compute_state_dict_delta(
    base_state: Dict[str, Any],
    updated_state: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise OctomilClientError("torch is required to compute state_dict deltas") from exc

    delta: Dict[str, Any] = {}
    for key, base_tensor in base_state.items():
        updated_tensor = updated_state.get(key)
        if updated_tensor is None:
            continue
        if torch.is_tensor(base_tensor) and torch.is_tensor(updated_tensor):
            delta[key] = updated_tensor.detach().cpu() - base_tensor.detach().cpu()
    return delta
