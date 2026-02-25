"""Serve command — local OpenAI-compatible inference server."""

from __future__ import annotations

import os
import sys
from typing import Optional

import click

from octomil.cli_helpers import (
    _auto_optimize,
    _complete_model_name,
    _get_api_key,
    _get_telemetry_reporter,
    _has_explicit_quant,
)


def register(cli: click.Group) -> None:
    cli.add_command(serve)


@click.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("--port", "-p", default=8080, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--benchmark", is_flag=True, help="Run latency benchmark on startup.")
@click.option(
    "--share", is_flag=True, help="Share anonymous benchmark data with Octomil."
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Default all responses to JSON output (response_format=json_object).",
)
@click.option(
    "--cache-size",
    default=2048,
    type=int,
    help="KV cache size in MB (default: 2048). Caches prompt prefixes to speed up multi-turn conversations.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable KV cache entirely.",
)
@click.option(
    "--engine",
    "-e",
    default=None,
    help="Force a specific engine (mlx-lm, llama.cpp, mnn, onnxruntime). "
    "Default: auto-benchmark all available engines and pick fastest.",
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated list of models to load (ordered small to large). "
    "Enables multi-model serving. Example: smollm-360m,phi-mini,llama-3b",
)
@click.option(
    "--auto-route",
    is_flag=True,
    help="Enable automatic query routing across loaded models. "
    "Requires --models with 2+ models.",
)
@click.option(
    "--route-strategy",
    default="complexity",
    type=click.Choice(["complexity"]),
    help="Routing strategy for --auto-route (default: complexity).",
)
@click.option(
    "--max-queue",
    default=32,
    type=int,
    help="Max pending requests in the queue (default: 32). Set to 0 to disable.",
)
@click.option(
    "--compress-context",
    is_flag=True,
    help="Enable prompt compression. Compresses long prompts before inference "
    "to reduce context window usage and speed up prefill.",
)
@click.option(
    "--compression-strategy",
    default="token_pruning",
    type=click.Choice(["token_pruning", "sliding_window"]),
    help="Compression strategy (default: token_pruning). "
    "token_pruning removes low-information tokens. "
    "sliding_window keeps recent turns verbatim and summarises older ones.",
)
@click.option(
    "--compression-ratio",
    default=0.5,
    type=float,
    help="Target compression ratio for token pruning (0.0-1.0, default: 0.5). "
    "Higher values prune more aggressively.",
)
@click.option(
    "--compression-max-turns",
    default=4,
    type=int,
    help="Number of recent conversation turns to keep verbatim "
    "when using sliding_window strategy (default: 4).",
)
@click.option(
    "--compression-threshold",
    default=256,
    type=int,
    help="Minimum estimated token count before compression kicks in (default: 256).",
)
@click.option(
    "--tool-use",
    is_flag=True,
    default=False,
    help="Pre-load coding agent tool schemas for structured output. "
    "Exposes tool definitions (read_file, write_file, edit_file, run_command, "
    "search_files) at /v1/tool-schemas for coding agents like Aider, Goose, "
    "and OpenCode.",
)
@click.option(
    "--early-exit-threshold",
    default=None,
    type=float,
    help="Enable early exit with this entropy threshold (0.0-1.0). "
    "Tokens exit early when intermediate logit entropy drops below this value. "
    "Lower = fewer exits (conservative), higher = more exits (aggressive). "
    "Example: --early-exit-threshold 0.3",
)
@click.option(
    "--speed-quality",
    default=None,
    type=click.Choice(["quality", "balanced", "fast"]),
    help="Speed-quality preset for early exit. "
    "quality: conservative (threshold=0.1), "
    "balanced: moderate (threshold=0.3), "
    "fast: aggressive (threshold=0.5). "
    "Overridden by --early-exit-threshold if both are set.",
)
def serve(
    model: str,
    port: int,
    host: str,
    benchmark: bool,
    share: bool,
    json_mode: bool,
    cache_size: int,
    no_cache: bool,
    engine: str | None,
    models: str | None,
    auto_route: bool,
    route_strategy: str,
    max_queue: int,
    compress_context: bool,
    compression_strategy: str,
    compression_ratio: float,
    compression_max_turns: int,
    compression_threshold: int,
    tool_use: bool,
    early_exit_threshold: float | None,
    speed_quality: str | None,
) -> None:
    """Start a local OpenAI-compatible inference server.

    MODEL accepts Ollama-style model:variant syntax:

    \b
        octomil serve gemma-1b              # auto-picks best quant for your hw
        octomil serve gemma-1b:8bit         # explicit 8-bit quantization
        octomil serve llama-8b:fp16         # full precision (no auto-optimize)
        octomil serve llama-3b:q4_k_m       # engine-specific quant (explicit)

    Without an explicit quantization variant, Octomil detects your GPU/RAM
    and picks the best quantization automatically.

    Auto-detects all available inference engines, benchmarks each,
    and picks the fastest for your hardware. Override with --engine.

    \b
    Multi-model with auto-routing:
        octomil serve smollm-360m --models smollm-360m,phi-mini,llama-3b --auto-route

    Simple queries (greetings, short factual) route to the smallest model.
    Complex queries (code, reasoning, multi-step) route to the largest.
    If a model fails, the next larger model is tried automatically.

    \b
    Force a specific engine:
        octomil serve gemma-1b --engine llama.cpp

    \b
    Use --json-mode to default all responses to valid JSON output:
        octomil serve gemma-1b --json-mode
    """
    api_key = _get_api_key() if share else None
    api_base: str = (
        os.environ.get("OCTOMIL_API_URL")
        or os.environ.get("OCTOMIL_API_BASE")
        or "https://api.octomil.com/api/v1"
    )
    cache_enabled = not no_cache

    # Determine if multi-model mode
    model_list: list[str] | None = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    if auto_route and not model_list:
        click.echo(
            "Error: --auto-route requires --models with 2+ models.",
            err=True,
        )
        sys.exit(1)

    if auto_route and model_list and len(model_list) < 2:
        click.echo(
            "Error: --auto-route requires at least 2 models in --models.",
            err=True,
        )
        sys.exit(1)

    # Multi-model routing mode
    if auto_route and model_list:
        _serve_multi_model(
            model_list=model_list,
            port=port,
            host=host,
            api_key=api_key,
            api_base=api_base,
            json_mode=json_mode,
            cache_size=cache_size,
            cache_enabled=cache_enabled,
            engine=engine,
            route_strategy=route_strategy,
        )
        return

    # Check if this is a whisper (speech-to-text) model
    from octomil.engines.whisper_engine import is_whisper_model

    is_whisper = is_whisper_model(model)

    # Auto-optimize: pick best quantization for hardware if not explicit
    if not is_whisper and not _has_explicit_quant(model):
        best_quant = _auto_optimize(model, context_length=cache_size)
        if best_quant:
            model = f"{model}:{best_quant.lower()}"
            click.echo(f"    Serving as: {model}")

    # Single-model mode (original behaviour)
    _print_engine_detection(model, engine)

    click.echo(f"\nStarting Octomil serve on {host}:{port}")
    if is_whisper:
        click.echo(f"Model: {model} (speech-to-text)")
        click.echo(f"POST http://localhost:{port}/v1/audio/transcriptions")
    else:
        click.echo(f"Model: {model}")
        if engine:
            click.echo(f"Engine: {engine} (manual override)")
        if json_mode:
            click.echo("JSON mode: enabled (all responses default to valid JSON)")
        click.echo(
            f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions"
        )
    click.echo(f"Engine info: http://localhost:{port}/v1/engines")
    click.echo(f"Health check: http://localhost:{port}/health")
    if not is_whisper:
        if cache_enabled:
            click.echo(f"KV cache: enabled ({cache_size} MB)")
            click.echo(f"Cache stats: http://localhost:{port}/v1/cache/stats")
        else:
            click.echo("KV cache: disabled")
        if max_queue > 0:
            click.echo(f"Request queue: enabled (max_depth={max_queue})")
            click.echo(f"Queue stats: http://localhost:{port}/v1/queue/stats")
        else:
            click.echo("Request queue: disabled")
        if compress_context:
            click.echo(
                f"Context compression: enabled "
                f"(strategy={compression_strategy}, "
                f"ratio={compression_ratio}, "
                f"threshold={compression_threshold} tokens)"
            )
        if tool_use:
            click.echo("Tool-use mode: enabled (coding agent tool schemas loaded)")
            click.echo(f"Tool schemas: http://localhost:{port}/v1/tool-schemas")

    # Build early exit config
    from octomil.early_exit import config_from_cli as _ee_config_from_cli

    ee_config = _ee_config_from_cli(
        early_exit_threshold=early_exit_threshold,
        speed_quality=speed_quality,
    )

    if not is_whisper and ee_config.enabled:
        preset_label = (
            f" (preset: {ee_config.preset.value})" if ee_config.preset else ""
        )
        click.echo(
            f"Early exit: enabled (threshold={ee_config.effective_threshold:.2f}"
            f", min_layers_frac={ee_config.effective_min_layers_fraction:.2f}"
            f"{preset_label})"
        )
        click.echo(f"Early exit stats: http://localhost:{port}/v1/early-exit/stats")

    if benchmark:
        click.echo("Benchmark mode: will run latency test after model loads.")

    if share and not api_key:
        click.echo(
            "Warning: --share requires an API key to upload benchmark data. "
            "Run `octomil login` or set OCTOMIL_API_KEY.",
            err=True,
        )

    from octomil.serve import run_server

    run_server(
        model,
        port=port,
        host=host,
        api_key=api_key,
        api_base=api_base,
        json_mode=json_mode,
        cache_size_mb=cache_size,
        cache_enabled=cache_enabled,
        engine=engine,
        max_queue_depth=max_queue,
        compress_context=compress_context,
        compression_strategy=compression_strategy,
        compression_ratio=compression_ratio,
        compression_max_turns=compression_max_turns,
        compression_threshold=compression_threshold,
        tool_use=tool_use,
        early_exit_config=ee_config if ee_config.enabled else None,
    )


def _serve_multi_model(
    model_list: list[str],
    port: int,
    host: str,
    api_key: str | None,
    api_base: str,
    json_mode: bool,
    cache_size: int,
    cache_enabled: bool,
    engine: str | None,
    route_strategy: str,
) -> None:
    """Start multi-model serving with query routing."""
    from octomil.routing import assign_tiers

    tiers = assign_tiers(model_list)
    tier_labels = {
        "fast": "<0.3 complexity",
        "balanced": "0.3-0.7 complexity",
        "quality": ">0.7 complexity",
    }

    click.echo(f"\nLoading {len(model_list)} models for auto-routing...")
    for name in model_list:
        info = tiers[name]
        label = tier_labels.get(info.tier, info.tier)
        click.echo(f"  {name}: tier={info.tier} ({label})")

    click.echo(f"\nRouting enabled (strategy: {route_strategy})")
    click.echo(f"Serving on {host}:{port}")
    click.echo(f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions")
    click.echo(f"Routing stats: http://localhost:{port}/v1/routing/stats")
    click.echo(f"Health check: http://localhost:{port}/health")
    if json_mode:
        click.echo("JSON mode: enabled (all responses default to valid JSON)")

    from octomil.serve import run_multi_model_server

    run_multi_model_server(
        model_list,
        port=port,
        host=host,
        api_key=api_key,
        api_base=api_base,
        json_mode=json_mode,
        cache_size_mb=cache_size,
        cache_enabled=cache_enabled,
        engine=engine,
        route_strategy=route_strategy,
    )


def _get_recommended_engines() -> list[tuple[str, str, str]]:
    """Return platform-aware engine recommendations as (package, pip_extra, description)."""
    import platform as _platform

    system = _platform.system()
    machine = _platform.machine()

    if system == "Darwin" and machine == "arm64":
        return [
            ("mlx-lm", "mlx", "MLX — best performance on Apple Silicon"),
            ("llama-cpp-python", "llama", "llama.cpp — good alternative"),
        ]
    elif system == "Darwin":
        return [
            ("llama-cpp-python", "llama", "llama.cpp"),
            ("onnxruntime", "onnx", "ONNX Runtime"),
        ]
    elif system == "Linux":
        return [
            ("llama-cpp-python", "llama", "llama.cpp"),
            ("onnxruntime", "onnx", "ONNX Runtime"),
        ]
    else:
        return [("llama-cpp-python", "llama", "llama.cpp")]


def _prompt_engine_install() -> bool:
    """Prompt user to install the recommended engine. Returns True if installed."""
    import subprocess

    recommendations = _get_recommended_engines()
    if not recommendations:
        return False

    top_pkg, top_extra, top_desc = recommendations[0]

    click.echo(
        click.style(
            "\nNo inference engines found.",
            fg="yellow",
        )
    )
    click.echo(f"  Recommended: {top_desc}\n")

    if not sys.stdin.isatty():
        click.echo(f"    pip install {top_pkg}\n")
        return False

    if click.confirm(f"  Install {top_pkg} now?", default=True):
        click.echo()
        if getattr(sys, "frozen", False):
            pip_cmd = ["pip", "install", top_pkg]
        else:
            pip_cmd = [sys.executable, "-m", "pip", "install", top_pkg]
        try:
            subprocess.check_call(pip_cmd)
            click.echo(
                click.style(f"\n  {top_pkg} installed successfully.\n", fg="green")
            )
            try:
                reporter = _get_telemetry_reporter()
                if reporter:
                    reporter.report_funnel_event("cli_install", success=True)
            except Exception:
                pass
            return True
        except subprocess.CalledProcessError:
            click.echo(
                click.style(f"\n  Failed to install {top_pkg}.", fg="red"),
                err=True,
            )
            click.echo(f"  Try manually: pip install {top_pkg}\n")
            try:
                reporter = _get_telemetry_reporter()
                if reporter:
                    reporter.report_funnel_event(
                        "cli_install",
                        success=False,
                        failure_reason=f"Failed to install {top_pkg}",
                        failure_category="engine_install",
                    )
            except Exception:
                pass
            return False
    else:
        click.echo("\n  Other options:\n")
        for pkg, _extra, desc in recommendations:
            click.echo(f"    pip install {pkg:<22s} # {desc}")
        click.echo()
        return False


def _print_engine_detection(model: str, engine_override: str | None) -> None:
    """Print engine detection results to terminal."""
    from octomil.engines import get_registry

    registry = get_registry()

    click.echo("\nDetecting engines...")
    detections = registry.detect_all(model)
    for d in detections:
        if d.engine.name == "echo":
            continue
        if d.available:
            info = f" ({d.info})" if d.info else ""
            click.echo(click.style(f"  + {d.engine.display_name}{info}", fg="green"))
        else:
            click.echo(click.style(f"  - {d.engine.display_name}", dim=True))

    if engine_override:
        click.echo(f"\nUsing {engine_override} (manual override)")
        return

    available = [d for d in detections if d.available and d.engine.name != "echo"]
    if len(available) > 1:
        click.echo(f"\nBenchmarking {model} across {len(available)} engines...")
        click.echo("(this runs a quick 32-token generation on each)")
    elif len(available) == 1:
        click.echo(
            f"\nUsing {available[0].engine.display_name} (only available engine)"
        )
    else:
        installed = _prompt_engine_install()
        if installed:
            detections = registry.detect_all(model)
            available = [
                d for d in detections if d.available and d.engine.name != "echo"
            ]
            if available:
                click.echo(f"  Using {available[0].engine.display_name}")
                return
        click.echo(
            click.style(
                "  Using echo backend for now (mirrors input, no real inference).\n",
                dim=True,
            )
        )
