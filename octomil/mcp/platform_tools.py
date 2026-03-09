"""Platform MCP tools — model resolution, inference, deployment, metrics.

These tools expose the SDK's core operations as machine-callable MCP tools.
All imports are lazy to avoid ImportError when optional deps are missing.
All tools return JSON for structured data, plain text for errors.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def register_platform_tools(mcp: Any, backend: Any) -> None:
    """Register platform-level tools onto a FastMCP instance.

    Parameters
    ----------
    mcp:
        A ``FastMCP`` instance (from ``mcp.server.fastmcp``).
    backend:
        An ``OctomilMCPBackend`` instance for inference.
    """

    @mcp.tool()
    def resolve_model(name: str, engine: str = "") -> str:
        """Resolve a model name to engine-specific artifacts.

        Returns the HuggingFace repo, filename, engine, quantization,
        and architecture for a given model specifier.

        Args:
            name: Model specifier (e.g. "gemma-3b", "phi-mini:4bit", "qwen-coder-7b")
            engine: Force a specific engine (e.g. "mlx", "llama.cpp", "onnx"). Empty = auto-select.
        """
        try:
            from octomil.models.resolver import ModelResolutionError, resolve

            kwargs: dict[str, Any] = {}
            if engine:
                kwargs["engine"] = engine
            resolved = resolve(name, **kwargs)
            result: dict[str, Any] = {
                "family": resolved.family,
                "quant": resolved.quant,
                "engine": resolved.engine,
                "hf_repo": resolved.hf_repo,
                "filename": resolved.filename,
                "architecture": resolved.architecture,
                "raw": resolved.raw,
            }
            if resolved.mlx_repo:
                result["mlx_repo"] = resolved.mlx_repo
            if resolved.source_repo:
                result["source_repo"] = resolved.source_repo
            if resolved.is_moe and resolved.moe:
                result["moe"] = {
                    "num_experts": resolved.moe.num_experts,
                    "active_experts": resolved.moe.active_experts,
                    "expert_size": resolved.moe.expert_size,
                    "total_params": resolved.moe.total_params,
                    "active_params": resolved.moe.active_params,
                }
            return json.dumps(result, indent=2)
        except ModelResolutionError as exc:
            return json.dumps({"error": "model_resolution_error", "message": str(exc)})
        except Exception as exc:
            logger.exception("resolve_model failed")
            return json.dumps({"error": "internal_error", "message": str(exc)})

    @mcp.tool()
    def list_models() -> str:
        """List all available models in the Octomil catalog.

        Returns model names with publisher, parameter count, supported engines,
        default quantization, available variants, and architecture type.
        """
        try:
            from octomil.models.catalog import CATALOG

            models: list[dict[str, Any]] = []
            for name, entry in sorted(CATALOG.items()):
                model_info: dict[str, Any] = {
                    "name": name,
                    "publisher": entry.publisher,
                    "params": entry.params,
                    "default_quant": entry.default_quant,
                    "engines": sorted(entry.engines),
                    "variants": sorted(entry.variants.keys()),
                    "architecture": entry.architecture,
                }
                if entry.moe:
                    model_info["moe"] = {
                        "total_params": entry.moe.total_params,
                        "active_params": entry.moe.active_params,
                    }
                models.append(model_info)
            return json.dumps({"count": len(models), "models": models}, indent=2)
        except Exception as exc:
            logger.exception("list_models failed")
            return json.dumps({"error": "internal_error", "message": str(exc)})

    @mcp.tool()
    def detect_engines(model_name: str = "") -> str:
        """Detect which inference engines are available on this machine.

        Optionally filters by model compatibility. Returns each engine's
        name, availability, priority, and detection info.

        Args:
            model_name: If provided, also checks which engines support this model.
        """
        try:
            from octomil.engines.registry import get_registry

            registry = get_registry()
            results = registry.detect_all(model_name or None)
            engines: list[dict[str, Any]] = []
            for r in results:
                engines.append(
                    {
                        "engine": r.engine.name,
                        "display_name": r.engine.display_name,
                        "available": r.available,
                        "priority": r.engine.priority,
                        "info": r.info,
                    }
                )
            # Sort: available first, then by priority
            engines.sort(key=lambda e: (not e["available"], e["priority"]))
            return json.dumps(
                {
                    "model_filter": model_name or None,
                    "engines": engines,
                    "available_count": sum(1 for e in engines if e["available"]),
                },
                indent=2,
            )
        except Exception as exc:
            logger.exception("detect_engines failed")
            return json.dumps({"error": "internal_error", "message": str(exc)})

    @mcp.tool()
    def run_inference(prompt: str, model: str = "", max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Run raw inference through the local on-device model.

        Unlike the code-focused tools (generate_code, review_code, etc.),
        this sends your prompt directly without any system prompt wrapping.
        Use this for general-purpose inference.

        Args:
            prompt: The prompt to send to the model
            model: Model override (default: server's configured model)
            max_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (default: 0.7, lower = more deterministic)
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            text, metrics = backend.generate(messages, max_tokens=max_tokens, temperature=temperature)
            return json.dumps(
                {
                    "text": text,
                    "metrics": metrics,
                },
                indent=2,
            )
        except Exception as exc:
            logger.exception("run_inference failed")
            return json.dumps({"error": "inference_error", "message": str(exc)})

    @mcp.tool()
    def get_metrics() -> str:
        """Get current model and engine status.

        Returns the loaded model name, engine, and whether the backend
        is ready for inference.
        """
        try:
            status: dict[str, Any] = {
                "model": backend.model_name,
                "engine": backend._engine_name,
                "loaded": backend.is_loaded,
            }
            return json.dumps(status, indent=2)
        except Exception as exc:
            logger.exception("get_metrics failed")
            return json.dumps({"error": "internal_error", "message": str(exc)})

    @mcp.tool()
    def deploy_model(
        name: str,
        version: str = "",
        devices: str = "",
        group: str = "",
        strategy: str = "canary",
        rollout: int = 100,
    ) -> str:
        """Deploy a model to edge devices via the Octomil platform.

        Requires OCTOMIL_API_KEY environment variable.

        Args:
            name: Model name to deploy
            version: Model version (optional)
            devices: Comma-separated device IDs (optional)
            group: Device group name (optional)
            strategy: Deployment strategy: "canary" or "rolling" (default: canary)
            rollout: Rollout percentage 1-100 (default: 100)
        """
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return json.dumps(
                {
                    "error": "auth_required",
                    "message": "OCTOMIL_API_KEY environment variable is required for deployment.",
                    "hint": "Set OCTOMIL_API_KEY in your environment or pass it via the MCP server config.",
                }
            )

        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            kwargs: dict[str, Any] = {"strategy": strategy, "rollout": rollout}
            if version:
                kwargs["version"] = version
            if devices:
                kwargs["devices"] = [d.strip() for d in devices.split(",") if d.strip()]
            if group:
                kwargs["group"] = group

            result = client.deploy(name, **kwargs)

            # DeploymentResult or dict — normalize to dict
            if hasattr(result, "__dict__") and not isinstance(result, dict):
                result = {k: v for k, v in result.__dict__.items() if not k.startswith("_") and not callable(v)}

            return json.dumps({"status": "deployed", "result": result}, indent=2, default=str)
        except Exception as exc:
            logger.exception("deploy_model failed")
            return json.dumps({"error": "deploy_error", "message": str(exc)})
