"""
Convenience client for the EdgeML platform.

Wraps the existing SDK classes (ModelRegistry, RolloutsAPI) behind a
simpler interface designed for CLI and script usage::

    import edgeml

    client = edgeml.Client()  # reads EDGEML_API_KEY from env
    client.push("model.pt", name="sentiment-v1", version="1.0.0")
    client.deploy("sentiment-v1", version="1.0.0", rollout=10)
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .python.edgeml.api_client import _ApiClient
from .python.edgeml.control_plane import RolloutsAPI
from .python.edgeml.registry import ModelRegistry


_DEFAULT_API_BASE = "https://api.edgeml.io/api/v1"


class Client:
    """High-level EdgeML client for push/pull/deploy workflows.

    Args:
        api_key: API key. Falls back to ``EDGEML_API_KEY`` env var.
        org_id: Organisation identifier. Falls back to ``EDGEML_ORG_ID`` env var.
        api_base: API base URL. Falls back to ``EDGEML_API_BASE`` env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        _key = api_key if api_key is not None else os.environ.get("EDGEML_API_KEY", "")
        _oid = (
            org_id if org_id is not None else os.environ.get("EDGEML_ORG_ID", "default")
        )
        _base = (
            api_base
            if api_base is not None
            else os.environ.get("EDGEML_API_BASE", _DEFAULT_API_BASE)
        )
        self._api_key: str = _key
        self._org_id: str = _oid
        self._api_base: str = _base

        def _token_provider() -> str:
            return self._api_key

        self._api = _ApiClient(
            auth_token_provider=_token_provider,
            api_base=self._api_base,
        )
        self._registry = ModelRegistry(
            auth_token_provider=_token_provider,
            org_id=self._org_id,
            api_base=self._api_base,
        )
        self._rollouts = RolloutsAPI(self._api)

    # ------------------------------------------------------------------
    # Push — upload a model file + trigger server-side conversion
    # ------------------------------------------------------------------

    def push(
        self,
        file_path: str,
        *,
        name: str,
        version: str,
        description: Optional[str] = None,
        formats: Optional[str] = None,
        framework: str = "pytorch",
        use_case: str = "general",
    ) -> dict[str, Any]:
        """Upload a model file and register a new version.

        Creates the model entry if it doesn't exist, then uploads the file
        and triggers server-side format conversion.

        Returns:
            Upload response dict with model_id, version, formats, checksums.
        """
        model = self._registry.ensure_model(
            name=name,
            framework=framework,
            use_case=use_case,
            description=description,
        )
        model_id = model["id"]
        return self._registry.upload_version_from_path(
            model_id=model_id,
            file_path=file_path,
            version=version,
            description=description,
            formats=formats,
        )

    # ------------------------------------------------------------------
    # Pull — download a model in a specific format
    # ------------------------------------------------------------------

    def pull(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        format: Optional[str] = None,
        destination: str = ".",
    ) -> dict[str, Any]:
        """Download a model to the local filesystem.

        Args:
            name: Model name or ID.
            version: Specific version. Defaults to latest.
            format: Target format (onnx, coreml, tflite). Auto-detects if omitted.
            destination: Local directory to save files into.

        Returns:
            Dict with ``model_path`` and optional ``mnn_config``.
        """
        model_id = self._registry.resolve_model_id(name)
        if version is None:
            version = self._registry.get_latest_version(model_id)
        return self._registry.download(
            model_id=model_id,
            version=version,
            destination=destination,
            format=format,
        )

    # ------------------------------------------------------------------
    # Deploy — create a rollout
    # ------------------------------------------------------------------

    def deploy(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        rollout: int = 100,
        strategy: str = "canary",
        increment: int = 10,
    ) -> dict[str, Any]:
        """Deploy a model version to devices via rollout.

        Args:
            name: Model name or ID.
            version: Version to deploy. Defaults to latest.
            rollout: Initial rollout percentage (1-100).
            strategy: Rollout strategy (canary, immediate).
            increment: Percentage increment per advance step.

        Returns:
            Rollout creation response.
        """
        model_id = self._registry.resolve_model_id(name)
        if version is None:
            version = self._registry.get_latest_version(model_id)
        return self._registry.deploy_version(
            model_id=model_id,
            version=version,
            rollout_percentage=rollout,
            target_percentage=100,
            increment_step=increment,
            start_immediately=(strategy == "immediate"),
        )

    # ------------------------------------------------------------------
    # Status — query inference metrics
    # ------------------------------------------------------------------

    def status(self, name: str) -> dict[str, Any]:
        """Get model status including active rollouts and version info.

        Returns:
            Dict with model info and active rollouts.
        """
        model_id = self._registry.resolve_model_id(name)
        model_info = self._registry.get_model(model_id)
        rollouts = self._rollouts.list_active(model_id)
        return {
            "model": model_info,
            "active_rollouts": rollouts,
        }

    # ------------------------------------------------------------------
    # Train — federated training
    # ------------------------------------------------------------------

    def train(
        self,
        name: str,
        *,
        strategy: str = "fedavg",
        rounds: int = 10,
        group: Optional[str] = None,
        privacy: Optional[str] = None,
        epsilon: Optional[float] = None,
        min_devices: int = 2,
    ) -> dict[str, Any]:
        """Start federated training across deployed devices.

        Args:
            name: Model name or ID.
            strategy: Aggregation strategy (fedavg, fedprox, etc.).
            rounds: Number of training rounds.
            group: Device group to train on.
            privacy: Privacy mechanism (dp-sgd, none).
            epsilon: Privacy budget (lower = more private).
            min_devices: Minimum devices required per round.

        Returns:
            Training start response with training_id.
        """
        model_id = self._registry.resolve_model_id(name)
        return self._api.post(
            "/training/start",
            {
                "model_id": model_id,
                "strategy": strategy,
                "num_rounds": rounds,
                "device_group": group,
                "privacy_mechanism": privacy,
                "epsilon": epsilon,
                "min_devices": min_devices,
            },
        )

    def train_status(self, name: str) -> dict[str, Any]:
        """Get training status for a model.

        Args:
            name: Model name or ID.

        Returns:
            Dict with current_round, total_rounds, active_devices, status,
            and optional loss/accuracy metrics.
        """
        model_id = self._registry.resolve_model_id(name)
        return self._api.get(f"/training/{model_id}/status")

    def train_stop(self, name: str) -> dict[str, Any]:
        """Stop active training for a model.

        Args:
            name: Model name or ID.

        Returns:
            Dict with last_round and stop confirmation.
        """
        model_id = self._registry.resolve_model_id(name)
        return self._api.post(f"/training/{model_id}/stop")

    # ------------------------------------------------------------------
    # List — list all models
    # ------------------------------------------------------------------

    def list_models(self, **kwargs: Any) -> dict[str, Any]:
        """List models in the registry."""
        return self._registry.list_models(**kwargs)
