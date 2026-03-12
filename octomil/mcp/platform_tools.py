"""Platform MCP tools — model resolution, inference, deployment, metrics, and more.

These tools expose the SDK's core operations as machine-callable MCP tools.
All imports are lazy to avoid ImportError when optional deps are missing.
All tools return JSON for structured data, plain text for errors.

Tools registered here:
  Phase 1: resolve_model, list_models, detect_engines, run_inference,
           get_metrics, deploy_model
  Phase 2: convert_model, optimize_model, detect_hardware, benchmark_model,
           recommend_model, scan_codebase, compress_prompt, plan_deployment, embed
"""

import json
import logging
import os
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

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

    @mcp.tool(
        annotations=ToolAnnotations(title="Resolve Model", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def resolve_model(
        name: Annotated[str, Field(description="Model specifier (e.g. 'gemma-3b', 'phi-mini:4bit', 'qwen-coder-7b')")],
        engine: Annotated[
            str, Field(description="Force a specific engine (e.g. 'mlx', 'llama.cpp', 'onnx'). Empty = auto-select.")
        ] = "",
    ) -> str:
        """Resolve a model name to engine-specific artifacts including HuggingFace repo, filename, engine, and architecture."""
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

    @mcp.tool(
        annotations=ToolAnnotations(title="List Models", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def list_models(
        filter_engine: Annotated[
            str, Field(description="Filter models by engine compatibility (e.g. 'mlx', 'llama.cpp'). Empty = all.")
        ] = "",
    ) -> str:
        """List all available models in the Octomil catalog with publisher, parameter count, engines, and variants."""
        try:
            from octomil.models.catalog import CATALOG

            models: list[dict[str, Any]] = []
            for name, entry in sorted(CATALOG.items()):
                if filter_engine and filter_engine.lower() not in {e.lower() for e in entry.engines}:
                    continue
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

    @mcp.tool(
        annotations=ToolAnnotations(title="Detect Engines", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def detect_engines(
        model_name: Annotated[str, Field(description="Filter engines by model compatibility. Empty = show all.")] = "",
    ) -> str:
        """Detect which inference engines are available on this machine."""
        try:
            from octomil.runtime.engines import get_registry

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

    @mcp.tool(
        annotations=ToolAnnotations(title="Run Inference", readOnlyHint=True, idempotentHint=False, openWorldHint=False)
    )
    def run_inference(
        prompt: Annotated[str, Field(description="The prompt to send to the model")],
        model: Annotated[str, Field(description="Model override (empty = server's configured model)")] = "",
        max_tokens: Annotated[int, Field(description="Maximum tokens to generate")] = 2048,
        temperature: Annotated[float, Field(description="Sampling temperature (lower = more deterministic)")] = 0.7,
    ) -> str:
        """Run raw inference through the local on-device model without system prompt wrapping."""
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

    @mcp.tool(
        annotations=ToolAnnotations(title="Get Metrics", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def get_metrics(
        include_hardware: Annotated[bool, Field(description="Include hardware detection info (CPU, RAM, GPU)")] = True,
    ) -> str:
        """Get current model, engine, hardware, and device status."""
        try:
            status: dict[str, Any] = {
                "model": backend.model_name,
                "engine": backend._engine_name,
                "loaded": backend.is_loaded,
            }

            if include_hardware:
                try:
                    from octomil.hardware._unified import detect_hardware

                    hw = detect_hardware()
                    status["hardware"] = {
                        "platform": hw.platform,
                        "best_backend": hw.best_backend,
                        "total_ram_gb": round(hw.total_ram_gb, 2),
                        "available_ram_gb": round(hw.available_ram_gb, 2),
                        "cpu": hw.cpu.brand,
                        "architecture": hw.cpu.architecture,
                    }
                    if hw.gpu:
                        status["hardware"]["gpu"] = hw.gpu.backend
                        status["hardware"]["vram_gb"] = round(hw.gpu.total_vram_gb, 2)
                except Exception:
                    pass

                try:
                    from octomil.runtime.engines import get_registry

                    registry = get_registry()
                    detections = registry.detect_all()
                    status["engines"] = [d.engine.name for d in detections if d.available and d.engine.name != "echo"]
                except Exception:
                    pass

            return json.dumps(status, indent=2)
        except Exception as exc:
            logger.exception("get_metrics failed")
            return json.dumps({"error": "internal_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Deploy Model", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
        )
    )
    def deploy_model(
        name: Annotated[str, Field(description="Model name to deploy")],
        version: Annotated[str, Field(description="Model version (empty = latest)")] = "",
        devices: Annotated[str, Field(description="Comma-separated device IDs")] = "",
        group: Annotated[str, Field(description="Device group name")] = "",
        strategy: Annotated[str, Field(description="Deployment strategy: 'canary' or 'rolling'")] = "canary",
        rollout: Annotated[int, Field(description="Rollout percentage 1-100")] = 100,
    ) -> str:
        """Deploy a model to edge devices via the Octomil platform. Requires OCTOMIL_API_KEY."""
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

    # ------------------------------------------------------------------
    # Phase 2 tools: convert, optimize, hardware, benchmark, recommend,
    #                scan, compress, plan_deployment, embed
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Convert Model", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
        )
    )
    def convert_model(
        model_path: Annotated[str, Field(description="Path to the PyTorch model file (.pt or .pth)")],
        target: Annotated[str, Field(description="Comma-separated target formats: onnx, coreml, tflite")] = "onnx",
        input_shape: Annotated[str, Field(description="Comma-separated input tensor shape")] = "1,3,224,224",
    ) -> str:
        """Convert a local PyTorch model to edge-optimized formats (ONNX, CoreML, TFLite)."""
        try:
            import tempfile

            formats = [f.strip().lower() for f in target.split(",")]
            shape = [int(d) for d in input_shape.split(",")]
            output_dir = tempfile.mkdtemp(prefix="octomil_convert_")
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            results: dict[str, Any] = {}

            if "onnx" in formats or "coreml" in formats or "tflite" in formats:
                try:
                    import torch

                    model = torch.jit.load(model_path, map_location="cpu")
                    model.eval()
                    dummy = torch.randn(*shape)
                    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
                    torch.onnx.export(model, dummy, onnx_path, opset_version=13)
                    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
                    results["onnx"] = {"path": onnx_path, "size_mb": round(size_mb, 2)}
                except ImportError:
                    results["onnx"] = {"error": "torch not installed — install with: pip install torch"}
                except Exception as exc:
                    results["onnx"] = {"error": str(exc)}

            if "coreml" in formats:
                onnx_path = results.get("onnx", {}).get("path")
                if onnx_path:
                    try:
                        import coremltools as ct

                        coreml_model = ct.converters.onnx.convert(model=onnx_path)
                        coreml_path = os.path.join(output_dir, f"{model_name}.mlmodel")
                        coreml_model.save(coreml_path)
                        size_mb = os.path.getsize(coreml_path) / 1024 / 1024
                        results["coreml"] = {"path": coreml_path, "size_mb": round(size_mb, 2)}
                    except ImportError:
                        results["coreml"] = {"error": "coremltools not installed"}
                    except Exception as exc:
                        results["coreml"] = {"error": str(exc)}
                else:
                    results["coreml"] = {"error": "ONNX conversion required first"}

            if "tflite" in formats:
                onnx_path = results.get("onnx", {}).get("path")
                if onnx_path:
                    try:
                        import tensorflow as tf
                        from onnx_tf.backend import prepare

                        tf_rep = prepare(onnx_path)
                        tf_path = os.path.join(output_dir, f"{model_name}_tf")
                        tf_rep.export_graph(tf_path)
                        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
                        tflite_model = converter.convert()
                        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
                        with open(tflite_path, "wb") as f:
                            f.write(tflite_model)
                        size_mb = len(tflite_model) / 1024 / 1024
                        results["tflite"] = {"path": tflite_path, "size_mb": round(size_mb, 2)}
                    except ImportError:
                        results["tflite"] = {
                            "error": "requires onnx-tf and tensorflow — pip install onnx-tf tensorflow"
                        }
                    except Exception as exc:
                        results["tflite"] = {"error": str(exc)}
                else:
                    results["tflite"] = {"error": "ONNX conversion required first"}

            return json.dumps(
                {"model": model_name, "output_dir": output_dir, "conversions": results},
                indent=2,
            )
        except Exception as exc:
            logger.exception("convert_model failed")
            return json.dumps({"error": "convert_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Optimize Model", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
        )
    )
    def optimize_model(
        name: Annotated[str, Field(description="Model name (must be registered on the platform)")],
        target_devices: Annotated[
            str, Field(description="Comma-separated target device types (e.g. 'ios,android')")
        ] = "",
        accuracy_threshold: Annotated[float, Field(description="Minimum acceptable accuracy ratio 0-1")] = 0.95,
        size_budget_mb: Annotated[float, Field(description="Maximum model size in MB (0 = no limit)")] = 0,
    ) -> str:
        """Optimize a model for on-device deployment via server-side pruning and quantization. Requires OCTOMIL_API_KEY."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return json.dumps(
                {
                    "error": "auth_required",
                    "message": "OCTOMIL_API_KEY is required for server-side optimization.",
                }
            )

        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            # Access the internal registry for the optimize call
            devices = [d.strip() for d in target_devices.split(",") if d.strip()] if target_devices else None
            model_id = client._registry.resolve_model_id(name)
            kwargs: dict[str, Any] = {"accuracy_threshold": accuracy_threshold}
            if devices:
                kwargs["target_devices"] = devices
            if size_budget_mb > 0:
                kwargs["size_budget_mb"] = size_budget_mb

            result = client._registry.optimize(model_id, **kwargs)
            return json.dumps({"status": "optimized", "result": result}, indent=2, default=str)
        except Exception as exc:
            logger.exception("optimize_model failed")
            return json.dumps({"error": "optimize_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Detect Hardware", readOnlyHint=True, idempotentHint=True, openWorldHint=False
        )
    )
    def detect_hardware(
        force_refresh: Annotated[
            bool, Field(description="Force re-detection instead of using cached hardware info")
        ] = False,
    ) -> str:
        """Detect full hardware capabilities including CPU, GPU, RAM, and recommended inference backend."""
        try:
            from octomil.hardware._unified import detect_hardware

            hw = detect_hardware(force=force_refresh)
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
                    "base_speed_ghz": hw.cpu.base_speed_ghz,
                    "has_avx2": hw.cpu.has_avx2,
                    "has_avx512": hw.cpu.has_avx512,
                    "has_neon": hw.cpu.has_neon,
                    "estimated_gflops": round(hw.cpu.estimated_gflops, 1),
                },
            }
            if hw.gpu:
                gpu_info: dict[str, Any] = {
                    "backend": hw.gpu.backend,
                    "total_vram_gb": round(hw.gpu.total_vram_gb, 2),
                    "is_multi_gpu": hw.gpu.is_multi_gpu,
                    "speed_coefficient": hw.gpu.speed_coefficient,
                }
                if hw.gpu.gpus:
                    gpu_info["gpus"] = [{"name": g.name, "memory_gb": round(g.memory.total_gb, 2)} for g in hw.gpu.gpus]
                if hw.gpu.driver_version:
                    gpu_info["driver_version"] = hw.gpu.driver_version
                if hw.gpu.cuda_version:
                    gpu_info["cuda_version"] = hw.gpu.cuda_version
                result["gpu"] = gpu_info
            else:
                result["gpu"] = None

            if hw.diagnostics:
                result["diagnostics"] = hw.diagnostics

            return json.dumps(result, indent=2)
        except Exception as exc:
            logger.exception("detect_hardware failed")
            return json.dumps({"error": "hardware_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Benchmark Model", readOnlyHint=True, idempotentHint=False, openWorldHint=False
        )
    )
    def benchmark_model(
        model_name: Annotated[str, Field(description="Model to benchmark (e.g. 'gemma-3b', 'phi-mini')")],
        n_tokens: Annotated[int, Field(description="Number of tokens to generate per benchmark run")] = 32,
        engine: Annotated[str, Field(description="Specific engine to benchmark (empty = all available)")] = "",
    ) -> str:
        """Benchmark inference engines for a specific model, measuring tokens/second and latency."""
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()

            if engine:
                eng = registry.get_engine(engine)
                if eng is None:
                    return json.dumps(
                        {
                            "error": "unknown_engine",
                            "message": f"Engine '{engine}' not found.",
                            "available": [e.name for e in registry.engines],
                        }
                    )
                ranked = registry.benchmark_all(model_name, n_tokens=n_tokens, engines=[eng])
            else:
                ranked = registry.benchmark_all(model_name, n_tokens=n_tokens)

            results: list[dict[str, Any]] = []
            for r in ranked:
                entry: dict[str, Any] = {
                    "engine": r.engine.name,
                    "tokens_per_second": round(r.result.tokens_per_second, 2),
                    "ok": r.result.ok,
                    "metadata": r.result.metadata,
                }
                if r.result.error:
                    entry["error"] = r.result.error
                results.append(entry)

            best = results[0]["engine"] if results and results[0]["ok"] else None
            return json.dumps(
                {"model": model_name, "n_tokens": n_tokens, "best_engine": best, "results": results},
                indent=2,
            )
        except Exception as exc:
            logger.exception("benchmark_model failed")
            return json.dumps({"error": "benchmark_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Recommend Model", readOnlyHint=True, idempotentHint=True, openWorldHint=False
        )
    )
    def recommend_model(
        priority: Annotated[
            str, Field(description="Optimization priority: 'speed', 'quality', or 'balanced'")
        ] = "balanced",
    ) -> str:
        """Recommend the best model configuration for this hardware based on available RAM, GPU, and CPU."""
        try:
            from octomil.hardware._unified import detect_hardware
            from octomil.model_optimizer import ModelOptimizer

            hw = detect_hardware()
            optimizer = ModelOptimizer(hw)
            recs = optimizer.recommend(priority=priority)

            results: list[dict[str, Any]] = []
            for rec in recs:
                results.append(
                    {
                        "model_size": rec.model_size,
                        "quantization": rec.quantization,
                        "reason": rec.reason,
                        "estimated_tps": round(rec.speed.tokens_per_second, 1),
                        "backend": rec.speed.backend,
                        "strategy": rec.config.strategy.value
                        if hasattr(rec.config.strategy, "value")
                        else str(rec.config.strategy),
                        "gpu_layers": rec.config.gpu_layers,
                        "vram_gb": round(rec.config.vram_gb, 2),
                        "ram_gb": round(rec.config.ram_gb, 2),
                        "confidence": rec.speed.confidence,
                        "serve_command": rec.serve_command,
                    }
                )

            return json.dumps(
                {
                    "priority": priority,
                    "hardware_summary": {
                        "platform": hw.platform,
                        "best_backend": hw.best_backend,
                        "total_ram_gb": round(hw.total_ram_gb, 2),
                    },
                    "recommendations": results,
                },
                indent=2,
            )
        except Exception as exc:
            logger.exception("recommend_model failed")
            return json.dumps({"error": "recommend_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(title="Scan Codebase", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def scan_codebase(
        path: Annotated[str, Field(description="Directory path to scan")],
        platform: Annotated[
            str, Field(description="Filter by platform: 'ios', 'android', 'python' (empty = all)")
        ] = "",
    ) -> str:
        """Scan a codebase to find all ML inference points across iOS, Android, and Python code."""
        try:
            from octomil.scanner import scan_directory

            points = scan_directory(path, platform=platform or None)
            results: list[dict[str, Any]] = []
            for p in points:
                results.append(
                    {
                        "file": p.file,
                        "line": p.line,
                        "type": p.type,
                        "platform": p.platform,
                        "pattern": p.pattern,
                        "suggestion": p.suggestion,
                        "context": p.context,
                    }
                )

            # Summary by type
            type_counts: dict[str, int] = {}
            for r in results:
                type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

            return json.dumps(
                {
                    "path": path,
                    "platform_filter": platform or None,
                    "total_points": len(results),
                    "by_type": type_counts,
                    "inference_points": results,
                },
                indent=2,
            )
        except FileNotFoundError as exc:
            return json.dumps({"error": "not_found", "message": str(exc)})
        except Exception as exc:
            logger.exception("scan_codebase failed")
            return json.dumps({"error": "scan_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Compress Prompt", readOnlyHint=True, idempotentHint=True, openWorldHint=False
        )
    )
    def compress_prompt(
        messages: Annotated[
            str, Field(description='JSON array of message objects [{"role": "user", "content": "..."}]')
        ],
        strategy: Annotated[
            str, Field(description="Compression strategy: 'token_pruning' or 'sliding_window'")
        ] = "token_pruning",
        target_ratio: Annotated[float, Field(description="Target compression ratio 0-1 (0.5 = reduce by half)")] = 0.5,
    ) -> str:
        """Compress a prompt to reduce token count before inference, saving cost without losing meaning."""
        try:
            from octomil.compression import CompressionConfig, PromptCompressor

            parsed_messages = json.loads(messages)
            if not isinstance(parsed_messages, list):
                return json.dumps({"error": "invalid_input", "message": "messages must be a JSON array"})

            config = CompressionConfig(
                strategy=strategy,
                target_ratio=target_ratio,
            )
            compressor = PromptCompressor(config=config)
            compressed, stats = compressor.compress(parsed_messages)

            return json.dumps(
                {
                    "compressed_messages": compressed,
                    "stats": {
                        "original_tokens": stats.original_tokens,
                        "compressed_tokens": stats.compressed_tokens,
                        "compression_ratio": round(stats.compression_ratio, 3),
                        "tokens_saved": stats.tokens_saved,
                        "savings_pct": round(stats.savings_pct, 1),
                        "strategy": stats.strategy,
                        "duration_ms": round(stats.duration_ms, 2),
                    },
                },
                indent=2,
            )
        except json.JSONDecodeError:
            return json.dumps({"error": "invalid_json", "message": "messages must be valid JSON"})
        except Exception as exc:
            logger.exception("compress_prompt failed")
            return json.dumps({"error": "compress_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(title="Plan Deployment", readOnlyHint=True, idempotentHint=True, openWorldHint=True)
    )
    def plan_deployment(
        name: Annotated[str, Field(description="Model name to plan deployment for")],
        version: Annotated[str, Field(description="Model version (empty = latest)")] = "",
        devices: Annotated[str, Field(description="Comma-separated device IDs")] = "",
        group: Annotated[str, Field(description="Device group name")] = "",
    ) -> str:
        """Dry-run a deployment to see the plan without executing it. Requires OCTOMIL_API_KEY."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return json.dumps(
                {
                    "error": "auth_required",
                    "message": "OCTOMIL_API_KEY is required for deployment planning.",
                }
            )

        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)
            kwargs: dict[str, Any] = {}
            if version:
                kwargs["version"] = version
            if devices:
                kwargs["devices"] = [d.strip() for d in devices.split(",") if d.strip()]
            if group:
                kwargs["group"] = group

            plan = client.deploy_prepare(name, **kwargs)

            # DeploymentPlan → dict
            if hasattr(plan, "__dict__") and not isinstance(plan, dict):
                plan_dict: Any = {k: v for k, v in plan.__dict__.items() if not k.startswith("_") and not callable(v)}
            else:
                plan_dict = plan

            return json.dumps({"status": "planned", "plan": plan_dict}, indent=2, default=str)
        except Exception as exc:
            logger.exception("plan_deployment failed")
            return json.dumps({"error": "plan_error", "message": str(exc)})

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Generate Embeddings", readOnlyHint=True, idempotentHint=True, openWorldHint=True
        )
    )
    def embed_text(
        text: Annotated[str, Field(description="Text to embed (single string or JSON array of strings)")],
        model: Annotated[str, Field(description="Model ID for embeddings (required)")] = "",
    ) -> str:
        """Generate text embeddings using the Octomil platform. Requires OCTOMIL_API_KEY."""
        api_key = os.environ.get("OCTOMIL_API_KEY")
        if not api_key:
            return json.dumps(
                {
                    "error": "auth_required",
                    "message": "OCTOMIL_API_KEY is required for embeddings.",
                }
            )

        try:
            from octomil.client import OctomilClient

            client = OctomilClient(api_key=api_key)

            # Support both single string and JSON array of strings
            input_text: str | list[str] = text
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    input_text = parsed
            except (json.JSONDecodeError, TypeError):
                pass

            if not model:
                return json.dumps(
                    {
                        "error": "model_required",
                        "message": "model parameter is required for embeddings.",
                    }
                )

            embed_result = client.embed(model, input_text)

            # EmbeddingResult → dict
            if hasattr(embed_result, "__dict__") and not isinstance(embed_result, dict):
                embed_dict: Any = {
                    k: v for k, v in embed_result.__dict__.items() if not k.startswith("_") and not callable(v)
                }
            else:
                embed_dict = embed_result

            return json.dumps({"status": "ok", "result": embed_dict}, indent=2, default=str)
        except Exception as exc:
            logger.exception("embed failed")
            return json.dumps({"error": "embed_error", "message": str(exc)})
