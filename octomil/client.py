"""
Convenience client for the Octomil platform.

Wraps the existing SDK classes (ModelRegistry, RolloutsAPI) behind a
simpler interface designed for CLI and script usage::

    import octomil

    client = octomil.Client()  # reads OCTOMIL_API_KEY from env
    client.push("model.pt", name="sentiment-v1", version="1.0.0")
    client.deploy("sentiment-v1", version="1.0.0", rollout=10)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .python.octomil.api_client import OctomilClientError

from .models import (
    DeploymentPlan,
    DeploymentResult,
    DeviceDeployment,
    DeviceDeploymentStatus,
    RollbackResult,
    TrainingSession,
)
from .python.octomil.api_client import _ApiClient
from .python.octomil.control_plane import RolloutsAPI
from .python.octomil.registry import ModelRegistry


_DEFAULT_API_BASE = "https://api.octomil.com/api/v1"

_MODEL_EXTENSIONS = {".safetensors", ".gguf", ".pt", ".pth", ".bin", ".onnx"}


def _find_model_file(directory: str) -> str | None:
    """Find the primary model file in a directory (e.g. HuggingFace snapshot)."""
    dir_path = Path(directory)
    # Direct children first
    for child in sorted(dir_path.iterdir()):
        if child.is_file() and child.suffix in _MODEL_EXTENSIONS:
            return str(child)
    # Recurse into subdirectories (HF snapshots nest files)
    for child in sorted(dir_path.rglob("*")):
        if child.is_file() and child.suffix in _MODEL_EXTENSIONS:
            return str(child)
    return None


class Client:
    """High-level Octomil client for push/pull/deploy workflows.

    Args:
        api_key: API key. Falls back to ``OCTOMIL_API_KEY`` env var.
        org_id: Organisation identifier. Falls back to ``OCTOMIL_ORG_ID`` env var.
        api_base: API base URL. Falls back to ``OCTOMIL_API_BASE`` env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        _key = api_key if api_key is not None else os.environ.get("OCTOMIL_API_KEY", "")
        _oid = (
            org_id
            if org_id is not None
            else os.environ.get("OCTOMIL_ORG_ID", "default")
        )
        _base = (
            api_base
            if api_base is not None
            else os.environ.get("OCTOMIL_API_BASE", _DEFAULT_API_BASE)
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
        use_case: str = "other",
    ) -> dict[str, Any]:
        """Upload a model file and register a new version.

        If *file_path* is a directory (e.g. a HuggingFace snapshot), the
        primary model file is located automatically.

        Creates the model entry if it doesn't exist, then uploads the file
        and triggers server-side format conversion.

        Returns:
            Upload response dict with model_id, version, formats, checksums.
        """
        if os.path.isdir(file_path):
            resolved = _find_model_file(file_path)
            if not resolved:
                raise OctomilClientError(
                    f"No model file found in directory: {file_path}\n"
                    f"Expected extensions: {', '.join(sorted(_MODEL_EXTENSIONS))}"
                )
            file_path = resolved

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
    # Push from HuggingFace — server-side import
    # ------------------------------------------------------------------

    def import_from_hf(
        self,
        repo_id: str,
        *,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        use_case: Optional[str] = None,
        reference_only: bool = True,
    ) -> dict[str, Any]:
        """Import a model from HuggingFace via the server.

        The server downloads, converts, and stores the model — no local
        download needed.

        Args:
            repo_id: HuggingFace repo (e.g. ``microsoft/Phi-4-mini-instruct``).
            name: Model name in the registry. Defaults to repo name.
            version: Semantic version string.
            description: Optional version description.
            use_case: Model use case (e.g. text_generation).
            reference_only: If True, store HF reference only (devices pull
                from HF directly). If False, server downloads and stores.

        Returns:
            Import response with model_id, version, format, files.
        """
        payload: dict[str, Any] = {
            "repo_id": repo_id,
            "version": version,
            "org_id": self._org_id,
            "reference_only": reference_only,
        }
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if use_case:
            payload["use_case"] = use_case
        return self._api.post("/integrations/huggingface/import", payload)

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
    # Deploy — create a rollout with optional device targeting
    # ------------------------------------------------------------------

    def deploy(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        rollout: int = 100,
        strategy: str = "canary",
        increment: int = 10,
        devices: list[str] | None = None,
        group: str | None = None,
    ) -> dict[str, Any] | DeploymentResult:
        """Deploy a model version to devices via rollout.

        When ``devices`` or ``group`` is specified, uses the deploy
        orchestration endpoint for targeted deployment.  Otherwise falls
        back to the existing rollout creation flow.

        Args:
            name: Model name or ID.
            version: Version to deploy. Defaults to latest.
            rollout: Initial rollout percentage (1-100).
            strategy: Rollout strategy (canary, immediate, blue_green).
            increment: Percentage increment per advance step.
            devices: Specific device IDs to target.
            group: Device group name to target.

        Returns:
            ``DeploymentResult`` when using targeted deployment, otherwise
            the raw rollout creation response dict.
        """
        model_id = self._registry.resolve_model_id(name)
        if version is None:
            version = self._registry.get_latest_version(model_id)

        # Targeted deployment via orchestration endpoint
        if devices is not None or group is not None:
            payload: dict[str, Any] = {
                "model_id": model_id,
                "model_name": name,
                "version": version,
                "rollout_percentage": rollout,
                "strategy": strategy,
            }
            if devices is not None:
                payload["devices"] = devices
            if group is not None:
                payload["group"] = group

            resp = self._api.post("/deploy/execute", payload)
            return _parse_deployment_result(resp, name, version)

        # Fallback: existing rollout-based deploy
        return self._registry.deploy_version(
            model_id=model_id,
            version=version,
            rollout_percentage=rollout,
            target_percentage=100,
            increment_step=increment,
            start_immediately=(strategy == "immediate"),
        )

    # ------------------------------------------------------------------
    # Deploy prepare — dry-run / preview
    # ------------------------------------------------------------------

    def deploy_prepare(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        devices: list[str] | None = None,
        group: str | None = None,
    ) -> DeploymentPlan:
        """Preview a deployment without executing it.

        Calls the deploy preparation endpoint and returns a plan
        showing what format, executor, and conversion each device
        would need.

        Args:
            name: Model name or ID.
            version: Version to deploy. Defaults to latest.
            devices: Specific device IDs to target.
            group: Device group name to target.

        Returns:
            ``DeploymentPlan`` with per-device deployment details.
        """
        model_id = self._registry.resolve_model_id(name)
        if version is None:
            version = self._registry.get_latest_version(model_id)

        payload: dict[str, Any] = {
            "model_id": model_id,
            "model_name": name,
            "version": version,
        }
        if devices is not None:
            payload["devices"] = devices
        if group is not None:
            payload["group"] = group

        resp = self._api.post("/deploy/prepare", payload)
        return _parse_deployment_plan(resp, name, version)

    # ------------------------------------------------------------------
    # Train — create a federated training session
    # ------------------------------------------------------------------

    def train(
        self,
        name: str,
        *,
        group: str = "default",
        strategy: str = "fedavg",
        rounds: int = 10,
        min_updates: int = 1,
        **kwargs: Any,
    ) -> TrainingSession:
        """Start a federated training session.

        Creates a training session via the round management endpoint.

        Args:
            name: Model name or ID.
            group: Device group to train on.
            strategy: Aggregation strategy (fedavg, fedprox, etc.).
            rounds: Number of training rounds.
            min_updates: Minimum device updates required per round.
            **kwargs: Additional training parameters forwarded to the server.

        Returns:
            ``TrainingSession`` with session info and status.
        """
        model_id = self._registry.resolve_model_id(name)

        payload: dict[str, Any] = {
            "model_id": model_id,
            "group": group,
            "strategy": strategy,
            "rounds": rounds,
            "min_updates": min_updates,
            **kwargs,
        }

        resp = self._api.post("/training/sessions", payload)

        return TrainingSession(
            session_id=resp.get("session_id", resp.get("id", "")),
            model_name=name,
            group=group,
            strategy=strategy,
            rounds=rounds,
            status=resp.get("status", "created"),
        )

    # ------------------------------------------------------------------
    # Rollback — revert to a previous model version
    # ------------------------------------------------------------------

    def rollback(
        self,
        name: str,
        *,
        to_version: str | None = None,
    ) -> RollbackResult:
        """Rollback a model to a previous version.

        If ``to_version`` is not specified, the second-most-recent
        version is used (i.e. the version before the current one).

        Args:
            name: Model name or ID.
            to_version: Explicit version to rollback to.  ``None`` means
                the previous version.

        Returns:
            ``RollbackResult`` with rollback details and status.
        """
        model_id = self._registry.resolve_model_id(name)

        # Determine the current version
        current_version = self._registry.get_latest_version(model_id)

        # Determine the target version
        if to_version is None:
            versions_resp = self._registry.list_versions(model_id)
            versions = versions_resp.get("versions", [])
            if len(versions) < 2:
                from .python.octomil.api_client import OctomilClientError

                raise OctomilClientError(
                    f"Cannot rollback {name}: only one version exists"
                )
            # Versions come sorted newest-first from the API; pick the second.
            to_version = versions[1].get("version", "")

        # Deploy the target version at 100% immediately
        rollout_resp = self._registry.deploy_version(
            model_id=model_id,
            version=to_version,
            rollout_percentage=100,
            target_percentage=100,
            increment_step=100,
            start_immediately=True,
        )

        return RollbackResult(
            model_name=name,
            from_version=current_version,
            to_version=to_version,
            rollout_id=str(rollout_resp.get("id", "")),
            status=rollout_resp.get("status", "rolling_back"),
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

    def train_status(self, name: str) -> dict[str, Any]:
        """Get training status for a model.

        Queries ``GET /training/{model_id}/status`` for the current
        round progress, active devices, and metrics.

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

        Calls ``POST /training/{model_id}/stop`` to cancel the
        currently running training round.

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


# ------------------------------------------------------------------
# Parsing helpers (module-level)
# ------------------------------------------------------------------


def _parse_deployment_result(
    resp: dict[str, Any], name: str, version: str
) -> DeploymentResult:
    """Convert a raw API response into a ``DeploymentResult``."""
    device_statuses = [
        DeviceDeploymentStatus(
            device_id=ds.get("device_id", ""),
            status=ds.get("status", "unknown"),
            download_url=ds.get("download_url"),
            error=ds.get("error"),
        )
        for ds in resp.get("device_statuses", [])
    ]
    return DeploymentResult(
        deployment_id=resp.get("deployment_id", resp.get("id", "")),
        model_name=name,
        model_version=version,
        status=resp.get("status", "created"),
        device_statuses=device_statuses,
    )


def _parse_deployment_plan(
    resp: dict[str, Any], name: str, version: str
) -> DeploymentPlan:
    """Convert a raw API response into a ``DeploymentPlan``."""
    deployments = [
        DeviceDeployment(
            device_id=d.get("device_id", ""),
            format=d.get("format", ""),
            executor=d.get("executor", ""),
            quantization=d.get("quantization", "none"),
            download_url=d.get("download_url"),
            conversion_needed=d.get("conversion_needed", False),
            runtime_config=d.get("runtime_config", {}),
        )
        for d in resp.get("deployments", [])
    ]
    return DeploymentPlan(
        model_name=name,
        model_version=version,
        deployments=deployments,
    )
