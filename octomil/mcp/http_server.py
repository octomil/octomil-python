"""HTTP server exposing Octomil tools via REST + A2A agent card + OpenAPI.

This is a separate process from the stdio MCP server. It reuses the same
``OctomilMCPBackend`` and tool logic but serves them over HTTP instead
of stdin/stdout JSON-RPC.

Start via: ``octomil mcp serve --port 8402``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .a2a import AgentCardConfig, build_agent_card
from .auth import require_auth
from .backend import OctomilMCPBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response models for OpenAPI schema generation
# ---------------------------------------------------------------------------


class ResolveModelRequest(BaseModel):
    name: str = Field(..., description="Model specifier (e.g. 'gemma-3b', 'phi-mini:4bit')")
    engine: str = Field("", description="Force a specific engine (empty = auto-select)")


class ListModelsRequest(BaseModel):
    pass


class DetectEnginesRequest(BaseModel):
    model_name: str = Field("", description="Optional: filter engines by model compatibility")


class RunInferenceRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the model")
    model: str = Field("", description="Model override (default: server's configured model)")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")


class DeployModelRequest(BaseModel):
    name: str = Field(..., description="Model name to deploy")
    version: str = Field("", description="Model version")
    devices: str = Field("", description="Comma-separated device IDs")
    group: str = Field("", description="Device group name")
    strategy: str = Field("canary", description="Deployment strategy: canary or rolling")
    rollout: int = Field(100, description="Rollout percentage 1-100")


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


@dataclass
class HTTPServerConfig:
    """Configuration for the HTTP agent server."""

    host: str = "0.0.0.0"
    port: int = 8402
    model: str | None = None
    enable_x402: bool = False
    x402_address: str = ""
    x402_price: str = "0.001"
    x402_currency: str = "USDC"
    x402_network: str = "base"
    base_url: str = ""  # auto-detected if empty


# ---------------------------------------------------------------------------
# Tool definitions for agent card
# ---------------------------------------------------------------------------


def _get_tool_definitions() -> list[dict[str, Any]]:
    """Return tool definitions for the agent card.

    These are read from the platform_tools module's registered functions.
    """
    from .prompts import PLATFORM_TOOL_DESCRIPTIONS

    tools: list[dict[str, Any]] = []

    # Platform tools
    for name, desc in PLATFORM_TOOL_DESCRIPTIONS.items():
        tools.append({"name": name, "description": desc})

    # Code tools (existing 7)
    code_tools = {
        "generate_code": "Generate code from natural language description using on-device inference",
        "review_code": "Review code for bugs, security issues, and improvements",
        "explain_code": "Explain code in plain English",
        "write_tests": "Generate unit tests for code",
        "general_task": "Free-form prompt through the local model",
        "review_file": "Read a file from disk and review it locally",
        "analyze_files": "Read multiple files and answer a question about them",
    }
    for name, desc in code_tools.items():
        tools.append({"name": name, "description": desc})

    return tools


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_http_app(config: HTTPServerConfig | None = None) -> FastAPI:
    """Create the FastAPI application for the Octomil agent HTTP server.

    Parameters
    ----------
    config:
        Server configuration. Uses defaults if not provided.

    Returns
    -------
    FastAPI
        Configured app with all routes, middleware, and agent card.
    """
    if config is None:
        config = HTTPServerConfig()

    app = FastAPI(
        title="Octomil Agent",
        description="On-device ML inference, model resolution, and deployment — agent-callable via REST, A2A, and MCP",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for agent-to-agent communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # x402 middleware (opt-in)
    if config.enable_x402:
        from .x402 import X402Config, X402Middleware

        x402_config = X402Config(
            price_per_call=config.x402_price,
            currency=config.x402_currency,
            network=config.x402_network,
            payment_address=config.x402_address or os.environ.get("OCTOMIL_X402_ADDRESS", ""),
        )
        app.add_middleware(X402Middleware, config=x402_config)
        logger.info("x402 payment gating enabled (address=%s)", x402_config.payment_address)

    # Shared backend instance
    backend = OctomilMCPBackend(model=config.model)

    # Agent card — built once at startup
    base_url = config.base_url or f"http://{config.host}:{config.port}"
    card_config = AgentCardConfig(url=base_url)
    tool_defs = _get_tool_definitions()
    agent_card = build_agent_card(tool_defs, card_config)

    # Store on app state for testing access
    app.state.backend = backend
    app.state.agent_card = agent_card

    # ------------------------------------------------------------------
    # Discovery & health endpoints (no auth)
    # ------------------------------------------------------------------

    @app.get("/.well-known/agent-card.json", tags=["discovery"])
    async def get_agent_card() -> JSONResponse:
        """A2A agent card for discovery by other agents."""
        return JSONResponse(content=agent_card)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        """Health check."""
        return {
            "status": "ok",
            "model": backend.model_name,
            "loaded": backend.is_loaded,
        }

    # ------------------------------------------------------------------
    # REST API endpoints (auth required)
    # ------------------------------------------------------------------

    @app.post("/api/v1/resolve_model", tags=["models"], dependencies=[Depends(require_auth)])
    async def api_resolve_model(req: ResolveModelRequest) -> JSONResponse:
        """Resolve a model name to engine-specific artifacts."""
        try:
            from octomil.models.resolver import ModelResolutionError, resolve

            kwargs: dict[str, Any] = {}
            if req.engine:
                kwargs["engine"] = req.engine
            resolved = resolve(req.name, **kwargs)
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
            return JSONResponse(content=result)
        except ModelResolutionError as exc:
            return JSONResponse(status_code=404, content={"error": "model_resolution_error", "message": str(exc)})
        except Exception as exc:
            logger.exception("resolve_model failed")
            return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})

    @app.post("/api/v1/list_models", tags=["models"], dependencies=[Depends(require_auth)])
    async def api_list_models(req: ListModelsRequest | None = None) -> JSONResponse:
        """List all available models."""
        try:
            from octomil.models.catalog import CATALOG

            models: list[dict[str, Any]] = []
            for name, entry in sorted(CATALOG.items()):
                models.append(
                    {
                        "name": name,
                        "publisher": entry.publisher,
                        "params": entry.params,
                        "default_quant": entry.default_quant,
                        "engines": sorted(entry.engines),
                        "variants": sorted(entry.variants.keys()),
                        "architecture": entry.architecture,
                    }
                )
            return JSONResponse(content={"count": len(models), "models": models})
        except Exception as exc:
            logger.exception("list_models failed")
            return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})

    @app.post("/api/v1/detect_engines", tags=["engines"], dependencies=[Depends(require_auth)])
    async def api_detect_engines(req: DetectEnginesRequest) -> JSONResponse:
        """Detect available inference engines."""
        try:
            from octomil.engines.registry import get_registry

            registry = get_registry()
            results = registry.detect_all(req.model_name or None)
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
            engines.sort(key=lambda e: (not e["available"], e["priority"]))
            return JSONResponse(
                content={
                    "model_filter": req.model_name or None,
                    "engines": engines,
                    "available_count": sum(1 for e in engines if e["available"]),
                }
            )
        except Exception as exc:
            logger.exception("detect_engines failed")
            return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})

    @app.post("/api/v1/run_inference", tags=["inference"], dependencies=[Depends(require_auth)])
    async def api_run_inference(req: RunInferenceRequest) -> JSONResponse:
        """Run inference through the local on-device model."""
        try:
            messages = [{"role": "user", "content": req.prompt}]
            text, metrics = backend.generate(messages, max_tokens=req.max_tokens, temperature=req.temperature)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("run_inference failed")
            return JSONResponse(status_code=500, content={"error": "inference_error", "message": str(exc)})

    @app.get("/api/v1/metrics", tags=["monitoring"], dependencies=[Depends(require_auth)])
    async def api_metrics() -> JSONResponse:
        """Get model and engine status."""
        return JSONResponse(
            content={
                "model": backend.model_name,
                "engine": backend._engine_name,
                "loaded": backend.is_loaded,
            }
        )

    @app.post("/api/v1/deploy_model", tags=["deployment"], dependencies=[Depends(require_auth)])
    async def api_deploy_model(req: DeployModelRequest) -> JSONResponse:
        """Deploy a model to edge devices."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "auth_required",
                    "message": "OCTOMIL_API_KEY environment variable is required for deployment.",
                },
            )

        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            kwargs: dict[str, Any] = {"strategy": req.strategy, "rollout": req.rollout}
            if req.version:
                kwargs["version"] = req.version
            if req.devices:
                kwargs["devices"] = [d.strip() for d in req.devices.split(",") if d.strip()]
            if req.group:
                kwargs["group"] = req.group

            result = client.deploy(req.name, **kwargs)
            if hasattr(result, "__dict__") and not isinstance(result, dict):
                result = {k: v for k, v in result.__dict__.items() if not k.startswith("_") and not callable(v)}

            return JSONResponse(content={"status": "deployed", "result": result})
        except Exception as exc:
            logger.exception("deploy_model failed")
            return JSONResponse(status_code=500, content={"error": "deploy_error", "message": str(exc)})

    return app
