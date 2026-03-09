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
from .prompts import build_messages

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


# Phase 2 request models


class ConvertModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to PyTorch model file (.pt or .pth)")
    target: str = Field("onnx", description="Comma-separated formats: onnx, coreml, tflite")
    input_shape: str = Field("1,3,224,224", description="Comma-separated input tensor shape")


class OptimizeModelRequest(BaseModel):
    name: str = Field(..., description="Model name (must be registered on platform)")
    target_devices: str = Field("", description="Comma-separated target device types")
    accuracy_threshold: float = Field(0.95, description="Min acceptable accuracy ratio 0-1")
    size_budget_mb: float = Field(0, description="Max model size in MB (0 = no limit)")


class BenchmarkModelRequest(BaseModel):
    model_name: str = Field(..., description="Model to benchmark")
    n_tokens: int = Field(32, description="Tokens to generate per benchmark run")
    engine: str = Field("", description="Specific engine to benchmark (empty = all)")


class RecommendModelRequest(BaseModel):
    priority: str = Field("balanced", description="Optimization: speed, quality, or balanced")


class ScanCodebaseRequest(BaseModel):
    path: str = Field(..., description="Directory path to scan")
    platform: str = Field("", description="Filter: ios, android, python (empty = all)")


class CompressPromptRequest(BaseModel):
    messages: str = Field(..., description='JSON array of messages [{"role":"user","content":"..."}]')
    strategy: str = Field("token_pruning", description="Strategy: token_pruning or sliding_window")
    target_ratio: float = Field(0.5, description="Target compression ratio 0-1")


class PlanDeploymentRequest(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field("", description="Model version (default: latest)")
    devices: str = Field("", description="Comma-separated device IDs")
    group: str = Field("", description="Device group name")


class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed (string or JSON array of strings)")
    model: str = Field("", description="Model ID for embeddings")


# Code tool request models


class GenerateCodeRequest(BaseModel):
    description: str = Field(..., description="What the code should do")
    language: str = Field("", description="Target programming language")
    context: str = Field("", description="Additional context")


class ReviewCodeRequest(BaseModel):
    code: str = Field(..., description="Code to review")
    language: str = Field("", description="Programming language")
    focus: str = Field("", description="Focus area (security, performance, style)")


class ExplainCodeRequest(BaseModel):
    code: str = Field(..., description="Code to explain")
    language: str = Field("", description="Programming language")
    detail_level: str = Field("medium", description="Detail level: brief, medium, thorough")


class WriteTestsRequest(BaseModel):
    code: str = Field(..., description="Code to test")
    language: str = Field("", description="Programming language")
    framework: str = Field("", description="Test framework (pytest, jest, etc.)")
    focus: str = Field("", description="Focus areas to test")


class GeneralTaskRequest(BaseModel):
    prompt: str = Field(..., description="The prompt or question")
    context: str = Field("", description="Additional context")


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
            price_per_call=config.x402_price or os.environ.get("OCTOMIL_X402_PRICE", "0.001"),
            currency=config.x402_currency or os.environ.get("OCTOMIL_X402_CURRENCY", "USDC"),
            network=config.x402_network or os.environ.get("OCTOMIL_X402_NETWORK", "base"),
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

    # ------------------------------------------------------------------
    # Phase 2 REST API endpoints
    # ------------------------------------------------------------------

    @app.post("/api/v1/convert_model", tags=["conversion"], dependencies=[Depends(require_auth)])
    async def api_convert_model(req: ConvertModelRequest) -> JSONResponse:
        """Convert a PyTorch model to edge formats."""
        try:
            import tempfile

            formats = [f.strip().lower() for f in req.target.split(",")]
            shape = [int(d) for d in req.input_shape.split(",")]
            output_dir = tempfile.mkdtemp(prefix="octomil_convert_")
            model_name = os.path.splitext(os.path.basename(req.model_path))[0]
            results: dict[str, Any] = {}

            if "onnx" in formats or "coreml" in formats or "tflite" in formats:
                try:
                    import torch

                    model = torch.jit.load(req.model_path, map_location="cpu")
                    model.eval()
                    dummy = torch.randn(*shape)
                    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
                    torch.onnx.export(model, dummy, onnx_path, opset_version=13)
                    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
                    results["onnx"] = {"path": onnx_path, "size_mb": round(size_mb, 2)}
                except ImportError:
                    results["onnx"] = {"error": "torch not installed"}
                except Exception as exc:
                    results["onnx"] = {"error": str(exc)}

            return JSONResponse(content={"model": model_name, "output_dir": output_dir, "conversions": results})
        except Exception as exc:
            logger.exception("convert_model failed")
            return JSONResponse(status_code=500, content={"error": "convert_error", "message": str(exc)})

    @app.post("/api/v1/optimize_model", tags=["optimization"], dependencies=[Depends(require_auth)])
    async def api_optimize_model(req: OptimizeModelRequest) -> JSONResponse:
        """Optimize a model for on-device deployment."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "auth_required", "message": "OCTOMIL_API_KEY required."},
            )
        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            devices = [d.strip() for d in req.target_devices.split(",") if d.strip()] if req.target_devices else None
            model_id = client._registry.resolve_model_id(req.name)
            kwargs: dict[str, Any] = {"accuracy_threshold": req.accuracy_threshold}
            if devices:
                kwargs["target_devices"] = devices
            if req.size_budget_mb > 0:
                kwargs["size_budget_mb"] = req.size_budget_mb
            result = client._registry.optimize(model_id, **kwargs)
            return JSONResponse(content={"status": "optimized", "result": result})
        except Exception as exc:
            logger.exception("optimize_model failed")
            return JSONResponse(status_code=500, content={"error": "optimize_error", "message": str(exc)})

    @app.get("/api/v1/hardware_profile", tags=["hardware"], dependencies=[Depends(require_auth)])
    async def api_hardware_profile() -> JSONResponse:
        """Detect hardware capabilities."""
        try:
            from octomil.hardware._unified import detect_hardware

            hw = detect_hardware(force=True)
            result: dict[str, Any] = {
                "platform": hw.platform,
                "best_backend": hw.best_backend,
                "total_ram_gb": round(hw.total_ram_gb, 2),
                "available_ram_gb": round(hw.available_ram_gb, 2),
                "cpu": {
                    "brand": hw.cpu.brand,
                    "cores": hw.cpu.cores,
                    "threads": hw.cpu.threads,
                    "architecture": hw.cpu.architecture,
                },
            }
            if hw.gpu:
                result["gpu"] = {
                    "backend": hw.gpu.backend,
                    "total_vram_gb": round(hw.gpu.total_vram_gb, 2),
                    "speed_coefficient": hw.gpu.speed_coefficient,
                }
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("hardware_profile failed")
            return JSONResponse(status_code=500, content={"error": "hardware_error", "message": str(exc)})

    @app.post("/api/v1/benchmark_model", tags=["benchmark"], dependencies=[Depends(require_auth)])
    async def api_benchmark_model(req: BenchmarkModelRequest) -> JSONResponse:
        """Benchmark inference engines for a model."""
        try:
            from octomil.engines.registry import get_registry

            registry = get_registry()
            if req.engine:
                eng = registry.get_engine(req.engine)
                if eng is None:
                    return JSONResponse(
                        status_code=404,
                        content={"error": "unknown_engine", "available": [e.name for e in registry.engines]},
                    )
                ranked = registry.benchmark_all(req.model_name, n_tokens=req.n_tokens, engines=[eng])
            else:
                ranked = registry.benchmark_all(req.model_name, n_tokens=req.n_tokens)

            results = [
                {
                    "engine": r.engine.name,
                    "tokens_per_second": round(r.result.tokens_per_second, 2),
                    "ok": r.result.ok,
                }
                for r in ranked
            ]
            best = results[0]["engine"] if results and results[0]["ok"] else None
            return JSONResponse(content={"model": req.model_name, "best_engine": best, "results": results})
        except Exception as exc:
            logger.exception("benchmark_model failed")
            return JSONResponse(status_code=500, content={"error": "benchmark_error", "message": str(exc)})

    @app.post("/api/v1/recommend_model", tags=["intelligence"], dependencies=[Depends(require_auth)])
    async def api_recommend_model(req: RecommendModelRequest) -> JSONResponse:
        """Recommend optimal model configuration for this hardware."""
        try:
            from octomil.hardware._unified import detect_hardware
            from octomil.model_optimizer import ModelOptimizer

            hw = detect_hardware()
            optimizer = ModelOptimizer(hw)
            recs = optimizer.recommend(priority=req.priority)
            results = [
                {
                    "model_size": r.model_size,
                    "quantization": r.quantization,
                    "reason": r.reason,
                    "estimated_tps": round(r.speed.tokens_per_second, 1),
                    "confidence": r.speed.confidence,
                    "serve_command": r.serve_command,
                }
                for r in recs
            ]
            return JSONResponse(content={"priority": req.priority, "recommendations": results})
        except Exception as exc:
            logger.exception("recommend_model failed")
            return JSONResponse(status_code=500, content={"error": "recommend_error", "message": str(exc)})

    @app.post("/api/v1/scan_codebase", tags=["scanning"], dependencies=[Depends(require_auth)])
    async def api_scan_codebase(req: ScanCodebaseRequest) -> JSONResponse:
        """Scan codebase for ML inference points."""
        try:
            from octomil.scanner import scan_directory

            points = scan_directory(req.path, platform=req.platform or None)
            results = [
                {"file": p.file, "line": p.line, "type": p.type, "platform": p.platform, "suggestion": p.suggestion}
                for p in points
            ]
            return JSONResponse(content={"total_points": len(results), "inference_points": results})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": "not_found", "message": str(exc)})
        except Exception as exc:
            logger.exception("scan_codebase failed")
            return JSONResponse(status_code=500, content={"error": "scan_error", "message": str(exc)})

    @app.post("/api/v1/compress_prompt", tags=["optimization"], dependencies=[Depends(require_auth)])
    async def api_compress_prompt(req: CompressPromptRequest) -> JSONResponse:
        """Compress a prompt to reduce tokens."""
        try:
            import json as _json

            from octomil.compression import CompressionConfig, PromptCompressor

            parsed = _json.loads(req.messages)
            config = CompressionConfig(strategy=req.strategy, target_ratio=req.target_ratio)
            compressor = PromptCompressor(config=config)
            compressed, stats = compressor.compress(parsed)
            return JSONResponse(
                content={
                    "compressed_messages": compressed,
                    "stats": {
                        "original_tokens": stats.original_tokens,
                        "compressed_tokens": stats.compressed_tokens,
                        "tokens_saved": stats.tokens_saved,
                        "savings_pct": round(stats.savings_pct, 1),
                    },
                }
            )
        except Exception as exc:
            logger.exception("compress_prompt failed")
            return JSONResponse(status_code=500, content={"error": "compress_error", "message": str(exc)})

    @app.post("/api/v1/plan_deployment", tags=["deployment"], dependencies=[Depends(require_auth)])
    async def api_plan_deployment(req: PlanDeploymentRequest) -> JSONResponse:
        """Dry-run a deployment plan."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "auth_required", "message": "OCTOMIL_API_KEY required."},
            )
        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            kwargs: dict[str, Any] = {}
            if req.version:
                kwargs["version"] = req.version
            if req.devices:
                kwargs["devices"] = [d.strip() for d in req.devices.split(",") if d.strip()]
            if req.group:
                kwargs["group"] = req.group
            plan = client.deploy_prepare(req.name, **kwargs)
            plan_dict: Any = plan
            if hasattr(plan, "__dict__") and not isinstance(plan, dict):
                plan_dict = {k: v for k, v in plan.__dict__.items() if not k.startswith("_") and not callable(v)}
            return JSONResponse(content={"status": "planned", "plan": plan_dict})
        except Exception as exc:
            logger.exception("plan_deployment failed")
            return JSONResponse(status_code=500, content={"error": "plan_error", "message": str(exc)})

    @app.post("/api/v1/embed", tags=["inference"], dependencies=[Depends(require_auth)])
    async def api_embed(req: EmbedRequest) -> JSONResponse:
        """Generate embeddings."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "auth_required", "message": "OCTOMIL_API_KEY required."},
            )
        if not req.model:
            return JSONResponse(
                status_code=400,
                content={"error": "model_required", "message": "model parameter is required."},
            )
        try:
            import json as _json

            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            input_text: str | list[str] = req.text
            try:
                parsed = _json.loads(req.text)
                if isinstance(parsed, list):
                    input_text = parsed
            except (ValueError, TypeError):
                pass
            embed_result = client.embed(req.model, input_text)
            embed_dict: Any = embed_result
            if hasattr(embed_result, "__dict__") and not isinstance(embed_result, dict):
                embed_dict = {
                    k: v for k, v in embed_result.__dict__.items() if not k.startswith("_") and not callable(v)
                }
            return JSONResponse(content={"status": "ok", "result": embed_dict})
        except Exception as exc:
            logger.exception("embed failed")
            return JSONResponse(status_code=500, content={"error": "embed_error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Code tool endpoints
    # ------------------------------------------------------------------

    @app.post("/api/v1/generate_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_generate_code(req: GenerateCodeRequest) -> JSONResponse:
        """Generate code from a natural language description."""
        try:
            parts = [f"Generate {req.language + ' ' if req.language else ''}code: {req.description}"]
            if req.context:
                parts.append(f"\nContext:\n{req.context}")
            messages = build_messages("generate_code", "\n".join(parts))
            text, metrics = backend.generate(messages)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("generate_code failed")
            return JSONResponse(status_code=500, content={"error": "generate_code_error", "message": str(exc)})

    @app.post("/api/v1/review_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_review_code(req: ReviewCodeRequest) -> JSONResponse:
        """Review code for bugs, security issues, and improvements."""
        try:
            parts = [f"Review this {req.language + ' ' if req.language else ''}code:"]
            if req.focus:
                parts.append(f"Focus on: {req.focus}")
            parts.append(f"\n```{req.language}\n{req.code}\n```")
            messages = build_messages("review_code", "\n".join(parts))
            text, metrics = backend.generate(messages)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("review_code failed")
            return JSONResponse(status_code=500, content={"error": "review_code_error", "message": str(exc)})

    @app.post("/api/v1/explain_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_explain_code(req: ExplainCodeRequest) -> JSONResponse:
        """Explain code in plain English."""
        try:
            parts = [f"Explain this {req.language + ' ' if req.language else ''}code ({req.detail_level} detail):"]
            parts.append(f"\n```{req.language}\n{req.code}\n```")
            messages = build_messages("explain_code", "\n".join(parts))
            text, metrics = backend.generate(messages)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("explain_code failed")
            return JSONResponse(status_code=500, content={"error": "explain_code_error", "message": str(exc)})

    @app.post("/api/v1/write_tests", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_write_tests(req: WriteTestsRequest) -> JSONResponse:
        """Generate unit tests for code."""
        try:
            parts = [
                f"Write {req.framework + ' ' if req.framework else ''}tests for this {req.language + ' ' if req.language else ''}code:"
            ]
            if req.focus:
                parts.append(f"Focus on: {req.focus}")
            parts.append(f"\n```{req.language}\n{req.code}\n```")
            messages = build_messages("write_tests", "\n".join(parts))
            text, metrics = backend.generate(messages)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("write_tests failed")
            return JSONResponse(status_code=500, content={"error": "write_tests_error", "message": str(exc)})

    @app.post("/api/v1/general_task", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_general_task(req: GeneralTaskRequest) -> JSONResponse:
        """Run a free-form prompt through the local model."""
        try:
            content = req.prompt
            if req.context:
                content = f"{req.prompt}\n\nContext:\n{req.context}"
            messages = build_messages("general_task", content)
            text, metrics = backend.generate(messages)
            return JSONResponse(content={"text": text, "metrics": metrics})
        except Exception as exc:
            logger.exception("general_task failed")
            return JSONResponse(status_code=500, content={"error": "general_task_error", "message": str(exc)})

    return app
