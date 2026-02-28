"""
Convenience client for the Octomil platform.

Wraps the existing SDK classes (ModelRegistry, RolloutsAPI) behind a
simpler interface designed for CLI and script usage::

    import octomil

    client = octomil.OctomilClient()  # reads OCTOMIL_API_KEY from env
    client.push("model.pt", name="sentiment-v1", version="1.0.0")
    client.deploy("sentiment-v1", version="1.0.0", rollout=10)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional

if TYPE_CHECKING:
    from .embeddings import EmbeddingResult
    from .model import Model, Prediction
    from .serve import GenerationChunk
    from .streaming import StreamToken
    from .telemetry import TelemetryReporter

logger = logging.getLogger(__name__)

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

_logger = logging.getLogger(__name__)


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


class OctomilClient:
    """High-level OctomilClient for push/pull/deploy workflows.

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
        self._models: dict[str, "Model"] = {}

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

        # Telemetry — best-effort, never blocks or raises
        self._reporter: TelemetryReporter | None = None
        if self._api_key:
            try:
                from .telemetry import TelemetryReporter as _TR

                self._reporter = _TR(
                    api_key=self._api_key,
                    api_base=self._api_base,
                    org_id=self._org_id,
                )
            except Exception:
                logger.debug("Failed to initialise telemetry reporter", exc_info=True)

        self._models: dict[str, Model] = {}

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

        t0 = time.monotonic()
        try:
            model = self._registry.ensure_model(
                name=name,
                framework=framework,
                use_case=use_case,
                description=description,
            )
            model_id = model["id"]
            result = self._registry.upload_version_from_path(
                model_id=model_id,
                file_path=file_path,
                version=version,
                description=description,
                formats=formats,
            )
        except Exception as exc:
            if self._reporter:
                try:
                    self._reporter.report_funnel_event(
                        stage="model_push",
                        success=False,
                        model_id=name,
                        failure_reason=str(exc),
                        failure_category="upload_error",
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                except Exception:
                    _logger.debug("Telemetry reporting failed for push()")
            raise

        if self._reporter:
            try:
                self._reporter.report_funnel_event(
                    stage="model_push",
                    success=True,
                    model_id=name,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception:
                _logger.debug("Telemetry reporting failed for push()")
        return result

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

        t0 = time.monotonic()
        telemetry_model_id = name or repo_id
        try:
            result = self._api.post("/integrations/huggingface/import", payload)
        except Exception as exc:
            if self._reporter:
                try:
                    self._reporter.report_funnel_event(
                        stage="hf_import",
                        success=False,
                        model_id=telemetry_model_id,
                        failure_reason=str(exc),
                        failure_category="import_error",
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                except Exception:
                    _logger.debug("Telemetry reporting failed for import_from_hf()")
            raise

        if self._reporter:
            try:
                self._reporter.report_funnel_event(
                    stage="hf_import",
                    success=True,
                    model_id=telemetry_model_id,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception:
                _logger.debug("Telemetry reporting failed for import_from_hf()")
        return result

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

        t0 = time.monotonic()
        try:
            result = self._registry.download(
                model_id=model_id,
                version=version,
                destination=destination,
                format=format,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if self._reporter:
                try:
                    self._reporter.report_funnel_event(
                        stage="model_pull",
                        success=False,
                        model_id=name,
                        duration_ms=elapsed_ms,
                        failure_reason=str(exc),
                        failure_category="download_error",
                    )
                except Exception:
                    pass
            raise

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if self._reporter:
            try:
                self._reporter.report_funnel_event(
                    stage="model_pull",
                    success=True,
                    model_id=name,
                    duration_ms=elapsed_ms,
                )
            except Exception:
                pass

        return result

    def _get_model(self, name: str, **kwargs: Any) -> "Model":
        """Return a cached Model, or load one."""
        if name not in self._models:
            self._models[name] = self.load_model(name, **kwargs)
        return self._models[name]

    # ------------------------------------------------------------------
    # Predict — one-call download + load + infer
    # ------------------------------------------------------------------

    def predict(
        self,
        name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        version: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> "Prediction":
        """Download, load, and run inference in one call.

        The model is cached after the first call — subsequent calls
        with the same *name* reuse the loaded backend.

        Args:
            name: Model name (e.g. ``"phi-4-mini"``).
            messages: Chat-style messages (role + content dicts).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            version: Specific version. Defaults to latest.
            engine: Override the engine (e.g. ``"mlx-lm"``).

        Returns:
            A ``Prediction`` — a ``str`` with a ``.metrics`` attribute.
        """
        from .serve import GenerationRequest

        model = self._get_model(name, version=version, engine=engine)
        req = GenerationRequest(
            model=name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return model.predict(req)

    async def predict_stream(
        self,
        name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        version: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> "AsyncIterator[GenerationChunk]":
        """Download, load, and stream inference in one call.

        The model is cached after the first call.

        Args:
            name: Model name (e.g. ``"phi-4-mini"``).
            messages: Chat-style messages (role + content dicts).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            version: Specific version. Defaults to latest.
            engine: Override the engine (e.g. ``"mlx-lm"``).

        Yields:
            ``GenerationChunk`` objects with incremental text.
        """
        from .serve import GenerationRequest

        model = self._get_model(name, version=version, engine=engine)
        req = GenerationRequest(
            model=name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        async for chunk in model.predict_stream(req):
            yield chunk

    # ------------------------------------------------------------------
    # Cloud streaming inference (SSE)
    # ------------------------------------------------------------------

    def stream_predict(
        self,
        model_id: str,
        input_data: str | list[dict[str, str]],
        *,
        parameters: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> "Iterator[StreamToken]":
        """Stream tokens from the cloud inference endpoint (sync).

        Requires ``api_key`` and ``api_base`` to be configured.  This is
        a cloud-only method — it does **not** download or run models
        locally.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            input_data: A plain string prompt or chat-style messages.
            parameters: Generation parameters (temperature, max_tokens, etc.).
            timeout: HTTP timeout in seconds for the streaming connection.

        Yields:
            :class:`~octomil.streaming.StreamToken` for each SSE event.
        """
        from .streaming import stream_inference

        return stream_inference(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input_data=input_data,
            parameters=parameters,
            timeout=timeout,
        )

    async def stream_predict_async(
        self,
        model_id: str,
        input_data: str | list[dict[str, str]],
        *,
        parameters: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> "AsyncIterator[StreamToken]":
        """Stream tokens from the cloud inference endpoint (async).

        Async variant of :meth:`stream_predict`.
        """
        from .streaming import stream_inference_async

        async for token in stream_inference_async(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input_data=input_data,
            parameters=parameters,
            timeout=timeout,
        ):
            yield token

    # ------------------------------------------------------------------
    # Chat — OpenAI-compatible chat completion interface
    # ------------------------------------------------------------------

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion (non-streaming).

        Collects the full response from :meth:`stream_predict` and returns
        a dict with the assistant message and latency.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Returns:
            Dict with ``message`` (role + content), ``latency_ms``, and
            optional ``usage`` fields.
        """
        import time as _time

        params: dict[str, Any] = {**parameters}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p

        start = _time.monotonic()
        content = ""
        for token in self.stream_predict(
            model_id,
            messages,
            parameters=params or None,
            timeout=timeout,
        ):
            content += token.token

        latency_ms = (_time.monotonic() - start) * 1000
        return {
            "message": {"role": "assistant", "content": content},
            "latency_ms": latency_ms,
        }

    async def chat_stream(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> "AsyncIterator[dict[str, Any]]":
        """Streaming chat — yields chunks as they arrive (async generator).

        Each yielded dict contains ``index``, ``content``, ``done``, and
        ``role`` keys, matching the browser SDK's ``ChatChunk`` shape.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Yields:
            Dict with ``index``, ``content``, ``done``, and ``role``.
        """
        params: dict[str, Any] = {**parameters}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p

        idx = 0
        async for token in self.stream_predict_async(
            model_id,
            messages,
            parameters=params or None,
            timeout=timeout,
        ):
            yield {
                "index": idx,
                "content": token.token,
                "done": token.done,
                "role": "assistant",
            }
            idx += 1

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        model_id: str,
        input: str | list[str],
        *,
        timeout: float = 30.0,
    ) -> "EmbeddingResult":
        """Generate embeddings via the Octomil cloud endpoint.

        Requires ``api_key`` and ``api_base`` to be configured.

        Args:
            model_id: Embedding model identifier (e.g. ``"nomic-embed-text"``).
            input: A single string or list of strings to embed.
            timeout: HTTP timeout in seconds.

        Returns:
            :class:`~octomil.embeddings.EmbeddingResult` with dense vectors.
        """
        from .embeddings import embed

        return embed(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input=input,
            timeout=timeout,
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

        if self._reporter:
            try:
                self._reporter.report_deploy_started(
                    model_id=name,
                    version=version,
                    target_platform=strategy,
                )
            except Exception:
                _logger.debug("Telemetry reporting failed for deploy()")

        t0 = time.monotonic()
        try:
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
                result = _parse_deployment_result(resp, name, version)
            else:
                # Fallback: existing rollout-based deploy
                result = self._registry.deploy_version(
                    model_id=model_id,
                    version=version,
                    rollout_percentage=rollout,
                    target_percentage=100,
                    increment_step=increment,
                    start_immediately=(strategy == "immediate"),
                )
        except Exception as exc:
            if self._reporter:
                try:
                    self._reporter.report_funnel_event(
                        stage="deploy",
                        success=False,
                        model_id=name,
                        failure_reason=str(exc),
                        failure_category="deploy_error",
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                except Exception:
                    _logger.debug("Telemetry reporting failed for deploy()")
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000
        if self._reporter:
            try:
                self._reporter.report_deploy_completed(
                    model_id=name,
                    version=version,
                    duration_ms=elapsed_ms,
                )
            except Exception:
                _logger.debug("Telemetry reporting failed for deploy()")

        return result

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
        t0 = time.monotonic()
        try:
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

            result = RollbackResult(
                model_name=name,
                from_version=current_version,
                to_version=to_version,
                rollout_id=str(rollout_resp.get("id", "")),
                status=rollout_resp.get("status", "rolling_back"),
            )
        except Exception as exc:
            if self._reporter:
                try:
                    self._reporter.report_funnel_event(
                        stage="rollback",
                        success=False,
                        model_id=name,
                        failure_reason=str(exc),
                        failure_category="rollback_error",
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                except Exception:
                    _logger.debug("Telemetry reporting failed for rollback()")
            raise

        if self._reporter:
            try:
                self._reporter.report_funnel_event(
                    stage="rollback",
                    success=True,
                    model_id=name,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception:
                _logger.debug("Telemetry reporting failed for rollback()")
        return result

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
    # Load — pull + auto-select engine + create Model wrapper
    # ------------------------------------------------------------------

    def load_model(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        format: Optional[str] = None,
        destination: str = ".",
        engine: Optional[str] = None,
    ) -> Model:
        """Pull a model and return a ready-to-use ``Model`` instance.

        Combines :meth:`pull` with engine auto-selection and creates
        a ``Model`` object suitable for ``model.predict()`` calls.

        Args:
            name: Model name or ID.
            version: Specific version. Defaults to latest.
            format: Target format (onnx, coreml, tflite). Auto-detects if omitted.
            destination: Local directory to save files into.
            engine: Engine override (e.g. ``"mlx-lm"``). Auto-selects if omitted.

        Returns:
            A ``Model`` instance backed by the best available engine.
        """
        from .engines import get_registry
        from .model import Model as _Model
        from .model import ModelMetadata

        eng_registry = get_registry()
        selected_engine, _ = eng_registry.auto_select(name, engine_override=engine)

        # Skip registry pull for engines that manage their own downloads
        # (e.g. mlx-lm loads from HuggingFace, ollama pulls via its API).
        engine_kwargs: dict[str, Any] = {}
        if selected_engine.manages_own_download:
            pull_result = {}
        else:
            pull_result = self.pull(
                name, version=version, format=format, destination=destination
            )
            engine_kwargs["model_path"] = pull_result.get("model_path")

        model_id = self._registry.resolve_model_id(name)
        resolved_version = version
        if resolved_version is None:
            resolved_version = self._registry.get_latest_version(model_id)

        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=resolved_version,
        )

        t0 = time.monotonic()
        model = _Model(
            metadata=metadata,
            engine=selected_engine,
            engine_kwargs=engine_kwargs,
            _reporter=self._reporter,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if self._reporter:
            try:
                self._reporter.report_funnel_event(
                    stage="model_load",
                    success=True,
                    model_id=name,
                    duration_ms=elapsed_ms,
                )
            except Exception:
                pass

        self._models[name] = model
        return model

    # ------------------------------------------------------------------
    # Close — clean up models and telemetry
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all cached models and shut down the telemetry reporter."""
        self._models.clear()
        if self._reporter:
            try:
                self._reporter.close()
            except Exception:
                logger.debug("Failed to close telemetry reporter", exc_info=True)
            self._reporter = None

    def dispose(self) -> None:
        """Deprecated: use :meth:`close` instead."""
        import warnings

        warnings.warn(
            "dispose() is deprecated, use close() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.close()


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
