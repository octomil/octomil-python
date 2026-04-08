"""Platform REST API endpoints (auth required).

Model resolution, listing, engine detection, deployment, conversion,
optimization, hardware profiling, benchmarking, recommendations,
codebase scanning, prompt compression, and embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from ..auth import require_auth
from ..backend import OctomilMCPBackend
from .models import (
    BenchmarkModelRequest,
    CompressPromptRequest,
    ConvertModelRequest,
    DeployModelRequest,
    DetectEnginesRequest,
    EmbedRequest,
    ListModelsRequest,
    OptimizeModelRequest,
    PlanDeploymentRequest,
    RecommendModelRequest,
    ResolveModelRequest,
    ScanCodebaseRequest,
)

logger = logging.getLogger(__name__)


def register_platform_routes(
    app: FastAPI,
    backend: OctomilMCPBackend,
) -> None:
    """Register platform API routes on *app*."""

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
    async def api_list_models(req: Optional[ListModelsRequest] = None) -> JSONResponse:
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
            from octomil.runtime.engines import get_registry

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

    @app.get("/api/v1/metrics", tags=["monitoring"], dependencies=[Depends(require_auth)])
    async def api_metrics() -> JSONResponse:
        """Get model, engine, and device status."""
        result: dict[str, Any] = {
            "model": backend.model_name,
            "engine": backend._engine_name,
            "loaded": backend.is_loaded,
        }

        # Hardware
        try:
            from octomil.hardware._unified import detect_hardware

            hw = detect_hardware()
            result["hardware"] = {
                "platform": hw.platform,
                "best_backend": hw.best_backend,
                "total_ram_gb": round(hw.total_ram_gb, 2),
                "available_ram_gb": round(hw.available_ram_gb, 2),
                "cpu": hw.cpu.brand,
                "architecture": hw.cpu.architecture,
            }
            if hw.gpu:
                result["hardware"]["gpu"] = hw.gpu.backend
                result["hardware"]["vram_gb"] = round(hw.gpu.total_vram_gb, 2)
        except Exception:
            pass

        # Available engines
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            detections = registry.detect_all()
            result["engines"] = [d.engine.name for d in detections if d.available and d.engine.name != "echo"]
        except Exception:
            pass

        return JSONResponse(content=result)

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
            from octomil.auth import OrgApiKeyAuth
            from octomil.client import OctomilClient

            client = OctomilClient(auth=OrgApiKeyAuth(api_key=api_key, org_id=os.getenv("OCTOMIL_ORG_ID", "default")))
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
            from octomil.auth import OrgApiKeyAuth
            from octomil.client import OctomilClient

            client = OctomilClient(auth=OrgApiKeyAuth(api_key=api_key, org_id=os.getenv("OCTOMIL_ORG_ID", "default")))
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
            from octomil.runtime.engines import get_registry

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
            from octomil.auth import OrgApiKeyAuth
            from octomil.client import OctomilClient

            client = OctomilClient(auth=OrgApiKeyAuth(api_key=api_key, org_id=os.getenv("OCTOMIL_ORG_ID", "default")))
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

            from octomil.auth import OrgApiKeyAuth
            from octomil.client import OctomilClient

            client = OctomilClient(auth=OrgApiKeyAuth(api_key=api_key, org_id=os.getenv("OCTOMIL_ORG_ID", "default")))
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
