"""
EdgeML command-line interface.

Usage::

    edgeml serve gemma-1b --port 8080
    edgeml deploy gemma-1b --phone
    edgeml dashboard
    edgeml push model.pt --name sentiment-v1 --version 1.0.0
    edgeml pull sentiment-v1 --version 1.0.0 --format coreml
    edgeml check model.pt
    edgeml convert model.pt --target ios,android
    edgeml status sentiment-v1
    edgeml benchmark gemma-1b
    edgeml benchmark gemma-1b --local
    edgeml login
    edgeml init "Acme Corp" --compliance hipaa --region us
    edgeml team add alice@acme.com --role admin
    edgeml team list
    edgeml team set-policy --require-mfa --session-hours 8
    edgeml keys create deploy-key --scope devices:write --scope models:read
    edgeml keys list
    edgeml keys revoke <key-id>
    edgeml scan ./MyApp
    edgeml scan ./MyApp --format json --platform ios
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Any, Optional

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Read API key from env, keychain, or raise."""
    key = os.environ.get("EDGEML_API_KEY", "")
    if not key:
        config_path = os.path.expanduser("~/.edgeml/credentials")
        if os.path.exists(config_path):
            with open(config_path) as f:
                for line in f:
                    if line.startswith("api_key="):
                        key = line.split("=", 1)[1].strip()
                        break
    return key


def _require_api_key() -> str:
    key = _get_api_key()
    if not key:
        click.echo("No API key found. Run `edgeml login` first.", err=True)
        sys.exit(1)
    return key


def _get_client():  # type: ignore[no-untyped-def]
    from .client import Client

    return Client(api_key=_require_api_key())


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="1.0.0", prog_name="edgeml")
def main() -> None:
    """EdgeML — serve, deploy, and observe ML models on edge devices."""


# ---------------------------------------------------------------------------
# edgeml serve
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model")
@click.option("--port", "-p", default=8080, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--benchmark", is_flag=True, help="Run latency benchmark on startup.")
@click.option(
    "--share", is_flag=True, help="Share anonymous benchmark data with EdgeML."
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
) -> None:
    """Start a local OpenAI-compatible inference server.

    MODEL accepts Ollama-style model:variant syntax:

    \b
        edgeml serve gemma-1b              # default (4bit)
        edgeml serve gemma-1b:8bit         # 8-bit quantization
        edgeml serve llama-8b:fp16         # full precision
        edgeml serve llama-3b:q4_k_m       # engine-specific quant

    Auto-detects all available inference engines, benchmarks each,
    and picks the fastest for your hardware. Override with --engine.

    \b
    Multi-model with auto-routing:
        edgeml serve smollm-360m --models smollm-360m,phi-mini,llama-3b --auto-route

    Simple queries (greetings, short factual) route to the smallest model.
    Complex queries (code, reasoning, multi-step) route to the largest.
    If a model fails, the next larger model is tried automatically.

    \b
    Force a specific engine:
        edgeml serve gemma-1b --engine llama.cpp

    \b
    Use --json-mode to default all responses to valid JSON output:
        edgeml serve gemma-1b --json-mode
    """
    api_key = _get_api_key() if share else None
    api_base: str = (
        os.environ.get("EDGEML_API_URL")
        or os.environ.get("EDGEML_API_BASE")
        or "https://api.edgeml.io/api/v1"
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

    # Single-model mode (original behaviour)
    _print_engine_detection(model, engine)

    click.echo(f"\nStarting EdgeML serve on {host}:{port}")
    click.echo(f"Model: {model}")
    if engine:
        click.echo(f"Engine: {engine} (manual override)")
    if json_mode:
        click.echo("JSON mode: enabled (all responses default to valid JSON)")
    click.echo(f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions")
    click.echo(f"Engine info: http://localhost:{port}/v1/engines")
    click.echo(f"Health check: http://localhost:{port}/health")
    if cache_enabled:
        click.echo(f"KV cache: enabled ({cache_size} MB)")
        click.echo(f"Cache stats: http://localhost:{port}/v1/cache/stats")
    else:
        click.echo("KV cache: disabled")

    if benchmark:
        click.echo("Benchmark mode: will run latency test after model loads.")

    if share and not api_key:
        click.echo(
            "Warning: --share requires an API key to upload benchmark data. "
            "Run `edgeml login` or set EDGEML_API_KEY.",
            err=True,
        )

    from .serve import run_server

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
    from .routing import assign_tiers

    # Show tier assignment
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

    from .serve import run_multi_model_server

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


def _print_engine_detection(model: str, engine_override: str | None) -> None:
    """Print engine detection results to terminal."""
    from .engines import get_registry

    registry = get_registry()

    click.echo("\nDetecting engines...")
    detections = registry.detect_all(model)
    for d in detections:
        if d.available:
            info = f" ({d.info})" if d.info else ""
            click.echo(click.style(f"  + {d.engine.display_name}{info}", fg="green"))
        else:
            click.echo(click.style(f"  - {d.engine.display_name}", fg="red"))

    if engine_override:
        click.echo(f"\nUsing {engine_override} (manual override)")
        return

    # Show which engines will be benchmarked
    available = [d for d in detections if d.available and d.engine.name != "echo"]
    if len(available) > 1:
        click.echo(f"\nBenchmarking {model} across {len(available)} engines...")
        click.echo("(this runs a quick 32-token generation on each)")
    elif len(available) == 1:
        click.echo(
            f"\nUsing {available[0].engine.display_name} (only available engine)"
        )
    else:
        click.echo(
            click.style(
                "\nNo inference engines found. Using echo backend (testing only).",
                fg="yellow",
            )
        )


# ---------------------------------------------------------------------------
# edgeml check
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--devices",
    "-d",
    default=None,
    help="Comma-separated device profiles (e.g. iphone_15_pro,pixel_8).",
)
def check(model_path: str, devices: Optional[str]) -> None:
    """Check device compatibility for a local model file.

    Analyzes the model and reports file size, format, and basic
    compatibility with edge devices.
    """
    file_size = os.path.getsize(model_path)
    ext = os.path.splitext(model_path)[1].lower()

    click.echo(f"Checking: {model_path}")
    click.echo(f"Size: {file_size / 1024 / 1024:.1f} MB")

    format_map = {
        ".pt": "PyTorch",
        ".pth": "PyTorch",
        ".onnx": "ONNX",
        ".mlmodel": "CoreML",
        ".mlpackage": "CoreML",
        ".tflite": "TFLite",
        ".gguf": "GGUF",
    }
    fmt = format_map.get(ext, f"Unknown ({ext})")
    click.echo(f"Format: {fmt}")

    # Size-based compatibility assessment
    size_mb = file_size / 1024 / 1024
    click.echo("\nDevice compatibility:")
    if size_mb < 50:
        click.echo("  iPhone 15 Pro:  compatible (NPU)")
        click.echo("  Pixel 8:        compatible (NNAPI)")
        click.echo("  Raspberry Pi 4: compatible (CPU)")
    elif size_mb < 500:
        click.echo("  iPhone 15 Pro:  compatible (NPU)")
        click.echo("  Pixel 8:        compatible (NNAPI)")
        click.echo("  Raspberry Pi 4: may require quantization")
    else:
        click.echo("  iPhone 15 Pro:  may require quantization")
        click.echo("  Pixel 8:        may require quantization")
        click.echo("  Raspberry Pi 4: too large — quantize or prune")

    click.echo("\nRecommendations:")
    if ext in (".pt", ".pth"):
        click.echo("  - Convert to ONNX: edgeml convert model.pt --target onnx")
        click.echo(
            "  - Convert to CoreML (iOS): edgeml convert model.pt --target coreml"
        )
        click.echo(
            "  - Convert to TFLite (Android): edgeml convert model.pt --target tflite"
        )
    elif ext == ".onnx":
        click.echo("  - ONNX is cross-platform — ready for deployment")
        click.echo(
            "  - Convert to CoreML (iOS): edgeml convert model.onnx --target coreml"
        )
    elif ext == ".gguf":
        click.echo("  - GGUF models work with llama.cpp backend")
        click.echo("  - Serve locally: edgeml serve model.gguf")
    else:
        click.echo("  - No specific recommendations for this format")


# ---------------------------------------------------------------------------
# edgeml scan
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format (text or json).",
)
@click.option(
    "--platform",
    "-p",
    default=None,
    type=click.Choice(["ios", "android", "python"]),
    help="Filter by platform. Default: scan all platforms.",
)
def scan(path: str, output_format: str, platform: str | None) -> None:
    """Scan a codebase for ML inference points.

    Walks PATH looking for model loading, inference calls, and model files
    across iOS (CoreML), Android (TFLite), and Python (PyTorch, ONNX,
    OpenAI, HuggingFace, MLX).

    Reports where you can add EdgeML.wrap() for telemetry.

    \b
    Examples:
        edgeml scan ./MyApp
        edgeml scan ./MyApp --format json
        edgeml scan ./MyApp --platform ios
        edgeml scan ./MyApp --platform android
    """
    from .scanner import format_json, format_text, scan_directory

    click.echo("Scanning for inference points...\n")

    try:
        points = scan_directory(path, platform=platform)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(format_json(points))
    else:
        click.echo(format_text(points))


# ---------------------------------------------------------------------------
# edgeml convert
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    default="onnx",
    help="Comma-separated target formats: onnx, coreml, tflite.",
)
@click.option("--output", "-o", default="./converted", help="Output directory.")
@click.option(
    "--input-shape",
    default="1,3,224,224",
    help="Comma-separated input tensor shape (e.g. 1,3,224,224).",
)
def convert(model_path: str, target: str, output: str, input_shape: str) -> None:
    """Convert a model to edge formats locally.

    Converts MODEL_PATH (PyTorch .pt/.pth) to target formats. Runs entirely
    on your machine — no account needed.

    Examples:

        edgeml convert model.pt --target onnx
        edgeml convert model.pt --target onnx,coreml --input-shape 1,3,224,224
    """
    formats = [f.strip().lower() for f in target.split(",")]
    shape = [int(d) for d in input_shape.split(",")]
    os.makedirs(output, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    click.echo(f"Converting {model_path} → {', '.join(formats)}")
    click.echo(f"Input shape: {shape}")
    click.echo(f"Output: {output}")

    results: dict[str, str] = {}

    if "onnx" in formats:
        try:
            import torch

            model = torch.jit.load(model_path)
            model.eval()
            sample = torch.randn(*shape)
            onnx_path = os.path.join(output, f"{model_name}.onnx")
            torch.onnx.export(model, sample, onnx_path, opset_version=13)
            results["onnx"] = onnx_path
        except ImportError:
            click.echo("  onnx: requires torch — pip install torch", err=True)
        except Exception as exc:
            click.echo(f"  onnx: failed — {exc}", err=True)

    if "coreml" in formats:
        try:
            import coremltools as ct  # type: ignore[import-untyped]

            onnx_src = results.get("onnx")
            if not onnx_src:
                click.echo(
                    "  coreml: requires onnx conversion first — "
                    "include onnx in --target",
                    err=True,
                )
            else:
                ml_model = ct.converters.onnx.convert(model=onnx_src)
                coreml_path = os.path.join(output, f"{model_name}.mlmodel")
                ml_model.save(coreml_path)
                results["coreml"] = coreml_path
        except ImportError:
            click.echo(
                "  coreml: requires coremltools — pip install coremltools", err=True
            )
        except Exception as exc:
            click.echo(f"  coreml: failed — {exc}", err=True)

    if "tflite" in formats:
        try:
            import onnx  # type: ignore[import-untyped]
            from onnx_tf.backend import prepare  # type: ignore[import-untyped]

            onnx_src = results.get("onnx")
            if not onnx_src:
                click.echo(
                    "  tflite: requires onnx conversion first — "
                    "include onnx in --target",
                    err=True,
                )
            else:
                import tensorflow as tf  # type: ignore[import-untyped]

                onnx_model = onnx.load(onnx_src)
                tf_rep = prepare(onnx_model)
                tf_dir = os.path.join(output, f"{model_name}_tf")
                tf_rep.export_graph(tf_dir)
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
                tflite_model = converter.convert()
                tflite_path = os.path.join(output, f"{model_name}.tflite")
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)
                results["tflite"] = tflite_path
        except ImportError:
            click.echo(
                "  tflite: requires onnx, onnx-tf, tensorflow — "
                "pip install onnx onnx-tf tensorflow",
                err=True,
            )
        except Exception as exc:
            click.echo(f"  tflite: failed — {exc}", err=True)

    if results:
        click.echo("\nConverted:")
        for fmt, path in results.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            click.echo(f"  {fmt}: {path} ({size_mb:.1f} MB)")
    else:
        click.echo("\nNo conversions succeeded.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# edgeml login
# ---------------------------------------------------------------------------


def _save_credentials(api_key: str) -> None:
    """Save an API key to ~/.edgeml/credentials with restrictive permissions."""
    config_dir = os.path.expanduser("~/.edgeml")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "credentials")
    with open(config_path, "w") as f:
        f.write(f"api_key={api_key}\n")
    os.chmod(config_path, 0o600)


def _browser_login() -> None:
    """Run the browser-based OAuth-style login flow.

    1. Spins up a temporary HTTP server on localhost.
    2. Opens the EdgeML dashboard auth page in the browser.
    3. Waits for the dashboard to redirect back with an API key.
    4. Falls back to manual paste on timeout.
    """
    import http.server
    import secrets
    import socket
    import threading
    import urllib.parse

    state = secrets.token_urlsafe(32)

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    dashboard_url = os.environ.get("EDGEML_DASHBOARD_URL", "https://app.edgeml.io")
    callback_url = f"http://127.0.0.1:{port}"
    auth_url = (
        f"{dashboard_url}/cli/auth"
        f"?callback={urllib.parse.quote(callback_url, safe='')}"
        f"&state={state}"
    )

    received_key: str | None = None
    received_org: str | None = None
    got_callback = threading.Event()

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        """Handle the single GET callback from the dashboard."""

        def do_GET(self) -> None:  # noqa: N802
            nonlocal received_key, received_org
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            cb_state = params.get("state", [None])[0]
            if cb_state != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state parameter.")
                return

            received_key = params.get("key", [None])[0]
            received_org = params.get("org", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Success! You can close this tab.</h2></body></html>"
            )
            got_callback.set()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # Suppress server log noise

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webbrowser.open(auth_url)
    click.echo("Opening browser for authentication...")
    click.echo("Waiting for authorization (Ctrl+C to cancel)...")

    got_callback.wait(timeout=300)
    server.shutdown()

    if received_key:
        _save_credentials(received_key)
        org_display = received_org or "unknown"
        click.echo(
            click.style(f"\u2713 Authenticated as org: {org_display}", fg="green")
        )
        click.echo("API key saved to ~/.edgeml/credentials")
    else:
        click.echo("Timed out. You can paste your API key manually:")
        manual_key: str = click.prompt("API key", hide_input=True)
        _save_credentials(manual_key)
        click.echo("API key saved to ~/.edgeml/credentials")


@main.command()
@click.option(
    "--api-key",
    default=None,
    hide_input=True,
    help="Paste API key directly (skip browser flow).",
)
def login(api_key: Optional[str]) -> None:
    """Authenticate with EdgeML Cloud.

    Opens your browser to authenticate, then saves the API key locally.
    Use --api-key to paste a key directly (for CI/headless environments).
    """
    if api_key:
        _save_credentials(api_key)
        click.echo("API key saved to ~/.edgeml/credentials")
        return

    _browser_login()


# ---------------------------------------------------------------------------
# edgeml push
# ---------------------------------------------------------------------------


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Model name.")
@click.option("--version", "-v", required=True, help="Semantic version (e.g. 1.0.0).")
@click.option("--description", "-d", default=None, help="Version description.")
@click.option(
    "--formats",
    "-f",
    default=None,
    help="Comma-separated target formats for server-side conversion.",
)
def push(
    file_path: str,
    name: str,
    version: str,
    description: Optional[str],
    formats: Optional[str],
) -> None:
    """Upload a model and trigger server-side conversion.

    Uploads FILE_PATH, registers it as NAME at VERSION, and optionally
    triggers conversion to mobile formats on the server.

    Example:

        edgeml push model.pt --name sentiment-v1 --version 1.0.0 --formats coreml,tflite
    """
    client = _get_client()
    click.echo(f"Pushing {file_path} as {name} v{version}...")

    result = client.push(
        file_path,
        name=name,
        version=version,
        description=description,
        formats=formats,
    )

    click.echo(f"Uploaded: {name} v{version}")
    for fmt, info in result.get("formats", {}).items():
        click.echo(f"  {fmt}: {info}")


# ---------------------------------------------------------------------------
# edgeml pull
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option(
    "--version", "-v", default=None, help="Version to download. Defaults to latest."
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default=None,
    help="Model format (onnx, coreml, tflite).",
)
@click.option("--output", "-o", default=".", help="Output directory.")
def pull(name: str, version: Optional[str], fmt: Optional[str], output: str) -> None:
    """Download a model from the registry.

    NAME accepts Ollama-style model:variant syntax:

    \b
        edgeml pull gemma-1b                  # default variant
        edgeml pull gemma-1b:8bit             # 8-bit quantization
        edgeml pull sentiment-v1 --version 1.0.0 --format coreml
    """
    client = _get_client()
    ver_str = version or "latest"
    click.echo(f"Pulling {name} v{ver_str}...")

    result = client.pull(name, version=version, format=fmt, destination=output)
    click.echo(f"Downloaded: {result['model_path']}")


# ---------------------------------------------------------------------------
# edgeml deploy
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option(
    "--version", "-v", default=None, help="Version to deploy. Defaults to latest."
)
@click.option("--phone", is_flag=True, help="Deploy to your connected phone.")
@click.option("--rollout", "-r", default=100, help="Rollout percentage (1-100).")
@click.option(
    "--strategy",
    "-s",
    default="canary",
    type=click.Choice(["canary", "immediate", "blue_green"]),
    help="Rollout strategy.",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Comma-separated target formats: ios, android.",
)
@click.option(
    "--devices",
    default=None,
    help="Comma-separated device IDs to deploy to.",
)
@click.option(
    "--group",
    "-g",
    default=None,
    help="Device group name to deploy to.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would happen without deploying.",
)
def deploy(
    name: str,
    version: Optional[str],
    phone: bool,
    rollout: int,
    strategy: str,
    target: Optional[str],
    devices: Optional[str],
    group: Optional[str],
    dry_run: bool,
) -> None:
    """Deploy a model to edge devices.

    Deploys NAME at VERSION to devices. Use --phone for quick
    phone deployment, --devices/--group for targeted deployment,
    or --rollout for fleet percentage rollouts.

    Examples:

        edgeml deploy gemma-1b --phone
        edgeml deploy sentiment-v1 --rollout 10 --strategy canary
        edgeml deploy gemma-1b --devices device_1,device_2
        edgeml deploy gemma-1b --group production
        edgeml deploy gemma-1b --group production --dry-run
    """
    if phone:
        import httpx

        from .qr import build_deep_link, render_qr_terminal

        # Detect ollama models before deploying
        from .ollama import get_ollama_model

        ollama_model = get_ollama_model(name)
        if ollama_model:
            click.echo(
                f"Detected ollama model: {ollama_model.name} "
                f"({ollama_model.size_display}, {ollama_model.quantization})"
            )

        # Try mDNS network scan before falling back to QR code
        from .discovery import scan_for_devices

        click.echo("Scanning for EdgeML devices on local network...")
        discovered = scan_for_devices(timeout=5.0)

        if discovered:
            if len(discovered) == 1:
                dev = discovered[0]
                click.echo(
                    click.style(
                        f"  \u2713 Found: {dev.name} ({dev.platform}, {dev.ip})",
                        fg="green",
                    )
                )
                if click.confirm(f"\nDeploy {name} to this device?", default=True):
                    # Direct deployment — create pairing session targeting this device
                    pass
            else:
                click.echo(f"  Found {len(discovered)} devices:")
                for i, dev in enumerate(discovered, 1):
                    click.echo(f"    {i}. {dev.name} ({dev.platform}, {dev.ip})")
                choice = click.prompt("Select device", type=int, default=1)
                dev = discovered[choice - 1]
                # Deploy to selected device (pairing session will target it)
        else:
            click.echo("  No devices found. Falling back to QR code pairing.\n")

        api_key = _require_api_key()
        api_base: str = (
            os.environ.get("EDGEML_API_URL")
            or os.environ.get("EDGEML_API_BASE")
            or "https://api.edgeml.io/api/v1"
        )
        dashboard_url = os.environ.get("EDGEML_DASHBOARD_URL", "https://app.edgeml.io")

        click.echo(f"Creating pairing session for {name}...")
        resp = httpx.post(
            f"{api_base}/deploy/pair",
            json={"model_name": name, "model_version": version},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code >= 400:
            click.echo(
                f"Failed to create pairing session: {resp.status_code}", err=True
            )
            click.echo(resp.text, err=True)
            sys.exit(1)

        session = resp.json()
        code = session["code"]

        # Build the deep link URL for the EdgeML mobile app.
        # The edgeml:// scheme is handled by DeepLinkHandler in the
        # iOS and Android SDKs, opening the app directly to pairing.
        pair_url = build_deep_link(token=code, host=api_base)

        # Render QR code in a styled box
        qr_art = render_qr_terminal(pair_url)
        qr_lines = qr_art.split("\n")
        # Determine box width: widest QR line or minimum for text
        max_qr_width = max((len(line) for line in qr_lines), default=0)
        box_inner = max(max_qr_width + 4, 45)

        click.echo()
        click.echo("\u256d" + "\u2500" * box_inner + "\u256e")
        click.echo(
            "\u2502"
            + "  Scan this QR code with your phone camera:".ljust(box_inner)
            + "\u2502"
        )
        click.echo("\u2502" + " " * box_inner + "\u2502")
        for line in qr_lines:
            padded = ("  " + line).ljust(box_inner)
            click.echo("\u2502" + padded + "\u2502")
        click.echo("\u2502" + " " * box_inner + "\u2502")
        click.echo(
            "\u2502" + f"  Or open manually: {pair_url}".ljust(box_inner) + "\u2502"
        )
        click.echo("\u2502" + "  Expires in 5 minutes".ljust(box_inner) + "\u2502")
        click.echo("\u2570" + "\u2500" * box_inner + "\u256f")
        click.echo()

        webbrowser.open(pair_url)

        click.echo("Waiting for device to connect (Ctrl+C to cancel)...")
        last_status = ""
        try:
            while True:
                import time

                time.sleep(2)
                poll = httpx.get(f"{api_base}/deploy/pair/{code}", timeout=5.0)
                if poll.status_code != 200:
                    continue
                data = poll.json()
                status_val = data.get("status", "pending")
                if status_val == last_status:
                    continue
                last_status = status_val
                if status_val == "connected":
                    device = data.get("device_name") or data.get("device_id", "unknown")
                    platform = data.get("device_platform", "unknown")
                    click.echo(
                        click.style(
                            f"  \u2713 Device connected: {device} ({platform})",
                            fg="green",
                        )
                    )
                elif status_val == "converting":
                    click.echo(
                        click.style(
                            "  \u2713 Converting model for device...", fg="yellow"
                        )
                    )
                elif status_val == "deploying":
                    click.echo(
                        click.style("  \u2713 Deploying to device...", fg="yellow")
                    )
                elif status_val == "done":
                    device = data.get("device_name") or data.get("device_id", "device")
                    click.echo(
                        click.style(
                            f"  \u2713 Deployment complete! Model running on {device}",
                            fg="green",
                            bold=True,
                        )
                    )
                    click.echo(f"  Open dashboard: {dashboard_url}")
                    break
                elif status_val in ("expired", "cancelled"):
                    click.echo(f"Session {status_val}.", err=True)
                    sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\nCancelled.")
            httpx.post(
                f"{api_base}/deploy/pair/{code}/cancel",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
        return

    client = _get_client()
    device_list = [d.strip() for d in devices.split(",")] if devices else None

    # Dry-run: preview deployment plan
    if dry_run:
        click.echo(f"Preparing deployment plan for {name}...")
        plan = client.deploy_prepare(
            name, version=version, devices=device_list, group=group
        )
        click.echo(f"Model: {plan.model_name} v{plan.model_version}")
        click.echo(f"Devices: {len(plan.deployments)}")
        for d in plan.deployments:
            conv = " (conversion needed)" if d.conversion_needed else ""
            click.echo(
                f"  {d.device_id}: {d.format} via {d.executor} [{d.quantization}]{conv}"
            )
        return

    # Targeted deployment
    if device_list or group:
        target_desc = f"devices={devices}" if devices else f"group={group}"
        click.echo(f"Deploying {name} to {target_desc} ({strategy})...")
        result = client.deploy(
            name,
            version=version,
            rollout=rollout,
            strategy=strategy,
            devices=device_list,
            group=group,
        )
        # result is a DeploymentResult
        from .models import DeploymentResult

        if isinstance(result, DeploymentResult):
            click.echo(f"Deployment: {result.deployment_id}")
            click.echo(f"Status: {result.status}")
            for ds in result.device_statuses:
                err = f" — {ds.error}" if ds.error else ""
                click.echo(f"  {ds.device_id}: {ds.status}{err}")
        return

    # Default: rollout-based deploy
    click.echo(f"Deploying {name} at {rollout}% rollout ({strategy})...")
    result = client.deploy(
        name,
        version=version,
        rollout=rollout,
        strategy=strategy,
    )
    click.echo(f"Rollout created: {result.get('id', 'ok')}")
    click.echo(f"Status: {result.get('status', 'started')}")


# ---------------------------------------------------------------------------
# edgeml rollback
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option(
    "--to-version",
    default=None,
    help="Version to rollback to. Defaults to previous version.",
)
def rollback(name: str, to_version: Optional[str]) -> None:
    """Rollback a model to a previous version.

    Reverts NAME to the specified version, or the previous version
    if --to-version is not provided.

    Examples:

        edgeml rollback gemma-1b
        edgeml rollback gemma-1b --to-version 1.0.0
    """
    client = _get_client()
    target = to_version or "previous"
    click.echo(f"Rolling back {name} to {target}...")

    try:
        result = client.rollback(name, to_version=to_version)
    except Exception as exc:
        click.echo(f"Rollback failed: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Rolled back: {result.from_version} -> {result.to_version}")
    click.echo(f"Rollout ID: {result.rollout_id}")
    click.echo(f"Status: {result.status}")


# ---------------------------------------------------------------------------
# edgeml pair (device-side: connect to a pairing session)
# ---------------------------------------------------------------------------


@main.command()
@click.argument("code")
@click.option(
    "--device-id", default=None, help="Device identifier. Auto-generated if omitted."
)
@click.option(
    "--platform",
    "-p",
    default=None,
    help="Device platform (ios, android, python). Auto-detected if omitted.",
)
@click.option("--device-name", default=None, help="Friendly device name.")
def pair(
    code: str,
    device_id: Optional[str],
    platform: Optional[str],
    device_name: Optional[str],
) -> None:
    """Connect to a pairing session as a device.

    Enter the CODE displayed by `edgeml deploy --phone` to pair
    this device and receive the model deployment.

    Example:

        edgeml pair ABC123
    """
    import platform as _platform
    import uuid

    import httpx

    api_base: str = (
        os.environ.get("EDGEML_API_URL")
        or os.environ.get("EDGEML_API_BASE")
        or "https://api.edgeml.io/api/v1"
    )
    device_id = device_id or f"device-{uuid.uuid4().hex[:8]}"
    platform = platform or f"python-{_platform.system().lower()}"
    device_name = device_name or f"{_platform.node()}"

    click.echo(f"Connecting to pairing session {code.upper()}...")
    click.echo(f"Device: {device_name} ({platform})")

    resp = httpx.post(
        f"{api_base}/deploy/pair/{code}/connect",
        json={
            "device_id": device_id,
            "platform": platform,
            "device_name": device_name,
        },
        timeout=10.0,
    )

    if resp.status_code == 404:
        click.echo("Pairing session not found. Check the code and try again.", err=True)
        sys.exit(1)
    elif resp.status_code == 410:
        click.echo("Pairing session has expired.", err=True)
        sys.exit(1)
    elif resp.status_code == 409:
        click.echo(
            f"Session conflict: {resp.json().get('detail', 'already connected')}",
            err=True,
        )
        sys.exit(1)
    elif resp.status_code >= 400:
        click.echo(f"Failed to connect: {resp.status_code}", err=True)
        sys.exit(1)

    session = resp.json()
    click.echo(f"Connected to session for model: {session['model_name']}")
    click.echo(f"Status: {session['status']}")
    click.echo("Waiting for deployment...")

    import time

    while True:
        time.sleep(2)
        poll = httpx.get(f"{api_base}/deploy/pair/{code}", timeout=5.0)
        if poll.status_code != 200:
            continue
        data = poll.json()
        st = data.get("status", "connected")
        if st == "deploying":
            click.echo("Deployment in progress...")
        elif st == "done":
            click.echo("Deployment complete. Model received.")
            break
        elif st in ("expired", "cancelled"):
            click.echo(f"Session {st}.", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# edgeml status
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
def status(name: str) -> None:
    """Show model status, active rollouts, and inference metrics.

    Example:

        edgeml status sentiment-v1
    """
    client = _get_client()
    info = client.status(name)

    model = info.get("model", {})
    click.echo(f"Model: {model.get('name', name)}")
    click.echo(f"ID: {model.get('id', 'unknown')}")
    click.echo(f"Framework: {model.get('framework', 'unknown')}")

    rollouts = info.get("active_rollouts", [])
    if rollouts:
        click.echo(f"\nActive rollouts: {len(rollouts)}")
        for r in rollouts:
            click.echo(
                f"  v{r.get('version', '?')} — "
                f"{r.get('rollout_percentage', 0)}% — "
                f"{r.get('status', 'unknown')}"
            )
    else:
        click.echo("\nNo active rollouts.")


# ---------------------------------------------------------------------------
# edgeml dashboard
# ---------------------------------------------------------------------------


@main.command()
def dashboard() -> None:
    """Open the EdgeML dashboard in your browser.

    Shows inference metrics across all devices — latency,
    throughput, errors, model versions side-by-side.
    """
    dashboard_url = os.environ.get("EDGEML_DASHBOARD_URL", "https://app.edgeml.io")
    click.echo(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)


# ---------------------------------------------------------------------------
# edgeml benchmark
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a sorted list."""
    s = sorted(data)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@main.command()
@click.argument("model")
@click.option(
    "--local",
    is_flag=True,
    help="Keep results local — do not upload to EdgeML Cloud.",
)
@click.option("--iterations", "-n", default=10, help="Number of inference iterations.")
@click.option("--max-tokens", default=50, help="Max tokens to generate per iteration.")
@click.option(
    "--engine",
    "-e",
    default=None,
    help="Force a specific engine for benchmarking. Default: benchmark all available.",
)
@click.option(
    "--all-engines",
    is_flag=True,
    help="Benchmark ALL available engines and compare (ignores --engine).",
)
def benchmark(
    model: str,
    local: bool,
    iterations: int,
    max_tokens: int,
    engine: str | None,
    all_engines: bool,
) -> None:
    """Run inference benchmarks on a model.

    MODEL accepts Ollama-style model:variant syntax:

    \b
        edgeml benchmark gemma-1b            # default (4bit)
        edgeml benchmark gemma-1b:8bit       # 8-bit quantization
        edgeml benchmark llama-8b:fp16       # full precision

    Measures TTFT, TPOT, latency distribution (min/avg/median/p90/p95/p99/max),
    throughput, and memory usage across multiple iterations.

    By default, anonymous benchmark data (model name, hardware, throughput — no
    PII) is shared with EdgeML Cloud.  Use --local to opt out.

    Use --all-engines to compare performance across all available engines:

        edgeml benchmark gemma-1b --all-engines

    Example:

        edgeml benchmark gemma-1b --iterations 20
        edgeml benchmark gemma-1b --local
    """
    import platform as _platform
    import time

    import psutil

    click.echo(
        f"Benchmarking {model} ({iterations} iterations, {max_tokens} max tokens)..."
    )
    click.echo(f"Platform: {_platform.system()} {_platform.machine()}")

    # Quick engine comparison if --all-engines
    if all_engines:
        _benchmark_all_engines(model, iterations, max_tokens)
        return

    from .serve import _detect_backend

    backend = _detect_backend(model, engine_override=engine)
    click.echo(f"Backend: {backend.name}")

    from .serve import GenerationRequest

    req = GenerationRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=max_tokens,
    )

    latencies: list[float] = []
    tps_list: list[float] = []
    ttft_list: list[float] = []
    prompt_tokens_list: list[int] = []
    completion_tokens_list: list[int] = []

    process = psutil.Process()
    mem_before = process.memory_info().rss

    for i in range(iterations):
        start = time.monotonic()
        _text, metrics = backend.generate(req)
        elapsed = (time.monotonic() - start) * 1000
        latencies.append(elapsed)
        tps_list.append(metrics.tokens_per_second)
        if metrics.ttfc_ms > 0:
            ttft_list.append(metrics.ttfc_ms)
        if metrics.prompt_tokens > 0:
            prompt_tokens_list.append(metrics.prompt_tokens)
        if metrics.total_tokens > 0:
            completion_tokens_list.append(metrics.total_tokens)
        click.echo(
            f"  [{i + 1}/{iterations}] {elapsed:.1f}ms, "
            f"{metrics.tokens_per_second:.1f} tok/s, "
            f"{metrics.total_tokens} tokens"
        )

    peak_mem = process.memory_info().rss
    peak_mem_delta = peak_mem - mem_before

    # Latency stats
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50 = _percentile(latencies, 50)
    p90 = _percentile(latencies, 90)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    # Throughput stats
    avg_tps = sum(tps_list) / len(tps_list)
    peak_tps = max(tps_list) if tps_list else 0

    # Token-level timing
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_prompt = (
        sum(prompt_tokens_list) // len(prompt_tokens_list) if prompt_tokens_list else 0
    )
    avg_completion = (
        sum(completion_tokens_list) // len(completion_tokens_list)
        if completion_tokens_list
        else 0
    )
    # TPOT = (total_latency - TTFT) / completion_tokens
    tpot = (
        (avg_latency - avg_ttft) / avg_completion
        if avg_completion > 0 and avg_ttft > 0
        else 0
    )

    click.echo("\nResults:")
    click.echo(f"  Backend:          {backend.name}")
    click.echo(f"  Iterations:       {iterations}")
    click.echo(f"  Avg prompt:       {avg_prompt} tokens")
    click.echo(f"  Avg completion:   {avg_completion} tokens")
    click.echo("")
    click.echo(f"  TTFT (avg):       {avg_ttft:.1f}ms")
    click.echo(f"  TPOT (avg):       {tpot:.2f}ms/token")
    click.echo("")
    click.echo(f"  Latency min:      {min_latency:.1f}ms")
    click.echo(f"  Latency avg:      {avg_latency:.1f}ms")
    click.echo(f"  Latency p50:      {p50:.1f}ms")
    click.echo(f"  Latency p90:      {p90:.1f}ms")
    click.echo(f"  Latency p95:      {p95:.1f}ms")
    click.echo(f"  Latency p99:      {p99:.1f}ms")
    click.echo(f"  Latency max:      {max_latency:.1f}ms")
    click.echo("")
    click.echo(f"  Throughput avg:   {avg_tps:.1f} tok/s")
    click.echo(f"  Throughput peak:  {peak_tps:.1f} tok/s")
    click.echo(
        f"  Peak memory:      {peak_mem / 1024 / 1024:.0f} MB (+{peak_mem_delta / 1024 / 1024:.0f} MB)"
    )

    if not local:
        api_key = _get_api_key()
        if not api_key:
            click.echo(
                "\nSkipping share: no API key. Run `edgeml login` first, "
                "or use --local to keep results local.",
                err=True,
            )
            return

        click.echo("\nSharing anonymous benchmark data...")
        try:
            import httpx

            payload = {
                "model": model,
                "backend": backend.name,
                "platform": _platform.system(),
                "arch": _platform.machine(),
                "os_version": _platform.platform(),
                "accelerator": "Metal" if _platform.system() == "Darwin" else "CPU",
                "ram_total_bytes": psutil.virtual_memory().total,
                "iterations": iterations,
                "prompt_tokens": avg_prompt,
                "completion_tokens": avg_completion,
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "p50_latency_ms": round(p50, 2),
                "p90_latency_ms": round(p90, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "ttft_ms": round(avg_ttft, 2) if avg_ttft > 0 else None,
                "tpot_ms": round(tpot, 2) if tpot > 0 else None,
                "avg_tokens_per_second": round(avg_tps, 1),
                "peak_tokens_per_second": round(peak_tps, 1),
                "peak_memory_bytes": peak_mem,
            }
            api_base: str = (
                os.environ.get("EDGEML_API_URL")
                or os.environ.get("EDGEML_API_BASE")
                or "https://api.edgeml.io/api/v1"
            )
            resp = httpx.post(
                f"{api_base}/benchmarks",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if resp.status_code < 400:
                click.echo("Benchmark data shared successfully.")
            else:
                click.echo(f"Failed to share: {resp.status_code}", err=True)
        except Exception as exc:
            click.echo(f"Failed to share: {exc}", err=True)
    else:
        click.echo("\nResults kept local (--local).")


def _benchmark_all_engines(model: str, iterations: int, max_tokens: int) -> None:
    """Benchmark all available engines and print a comparison table."""
    import time

    from .engines import get_registry
    from .serve import GenerationRequest

    registry = get_registry()
    detections = registry.detect_all(model)
    available = [d for d in detections if d.available and d.engine.name != "echo"]

    if not available:
        click.echo("No inference engines available for this model.", err=True)
        return

    click.echo(f"\nDetected {len(available)} engine(s):")
    for d in available:
        info = f" ({d.info})" if d.info else ""
        click.echo(f"  + {d.engine.display_name}{info}")

    results: list[dict[str, Any]] = []
    for d in available:
        click.echo(f"\nBenchmarking {d.engine.display_name}...")
        try:
            backend = d.engine.create_backend(model, cache_enabled=False)
            req = GenerationRequest(
                model=model,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=max_tokens,
            )

            latencies: list[float] = []
            tps_list: list[float] = []
            for i in range(iterations):
                start = time.monotonic()
                _text, metrics = backend.generate(req)
                elapsed = (time.monotonic() - start) * 1000
                latencies.append(elapsed)
                tps_list.append(metrics.tokens_per_second)
                click.echo(
                    f"  [{i + 1}/{iterations}] {elapsed:.1f}ms, "
                    f"{metrics.tokens_per_second:.1f} tok/s"
                )

            avg_tps = sum(tps_list) / len(tps_list)
            avg_lat = sum(latencies) / len(latencies)
            results.append(
                {
                    "engine": d.engine.display_name,
                    "avg_tps": avg_tps,
                    "avg_latency_ms": avg_lat,
                    "min_latency_ms": min(latencies),
                    "error": None,
                }
            )
        except Exception as exc:
            click.echo(f"  Failed: {exc}")
            results.append(
                {
                    "engine": d.engine.display_name,
                    "avg_tps": 0,
                    "avg_latency_ms": 0,
                    "min_latency_ms": 0,
                    "error": str(exc),
                }
            )

    # Print comparison table
    click.echo("\n" + "=" * 65)
    click.echo(
        f"{'Engine':<30s} {'Avg tok/s':>10s} {'Avg latency':>12s} {'Status':>10s}"
    )
    click.echo("-" * 65)

    # Sort by tok/s descending
    results.sort(key=lambda r: r["avg_tps"], reverse=True)
    best = results[0] if results and not results[0]["error"] else None

    for i, r in enumerate(results):
        if r["error"]:
            status = "error"
            click.echo(f"  {r['engine']:<28s} {'---':>10s} {'---':>12s} {status:>10s}")
        else:
            marker = " <-- fastest" if i == 0 and best else ""
            click.echo(
                f"  {r['engine']:<28s} {r['avg_tps']:>10.1f} "
                f"{r['avg_latency_ms']:>9.1f}ms {marker}"
            )

    click.echo("=" * 65)
    if best:
        click.echo(f"\nFastest engine: {best['engine']} ({best['avg_tps']:.1f} tok/s)")


# ---------------------------------------------------------------------------
# edgeml train
# ---------------------------------------------------------------------------


@main.group()
def train() -> None:
    """Federated training across deployed devices."""


@train.command("start")
@click.argument("name")
@click.option(
    "--strategy",
    "-s",
    default="fedavg",
    type=click.Choice(
        [
            "fedavg",
            "fedprox",
            "fedopt",
            "fedadam",
            "krum",
            "scaffold",
            "ditto",
            "fedmedian",
            "fedtrimmedavg",
        ]
    ),
    help="Aggregation strategy.",
)
@click.option("--rounds", "-r", default=10, help="Number of training rounds.")
@click.option("--group", "-g", default=None, help="Device group to train on.")
@click.option(
    "--privacy",
    default=None,
    type=click.Choice(["dp-sgd", "none"]),
    help="Privacy mechanism.",
)
@click.option(
    "--epsilon", default=None, type=float, help="Privacy budget (lower = more private)."
)
@click.option("--min-devices", default=2, help="Minimum devices required per round.")
def train_start(
    name: str,
    strategy: str,
    rounds: int,
    group: Optional[str],
    privacy: Optional[str],
    epsilon: Optional[float],
    min_devices: int,
) -> None:
    """Start federated training.

    Example:

        edgeml train start sentiment-v1 --strategy fedavg --rounds 50
    """
    client = _get_client()
    click.echo(f"Starting federated training for {name}")
    click.echo(f"Strategy: {strategy} | Rounds: {rounds} | Min devices: {min_devices}")
    if privacy:
        click.echo(f"Privacy: {privacy} (e={epsilon})")

    result = client.train(
        name,
        strategy=strategy,
        rounds=rounds,
        group=group,
        privacy=privacy,
        epsilon=epsilon,
        min_devices=min_devices,
    )
    click.echo(f"Training started: {result.session_id}")
    click.echo(f"Status: {result.status}")
    click.echo(f"Monitor: edgeml train status {name}")


@train.command("status")
@click.argument("name")
def train_status_cmd(name: str) -> None:
    """Show training progress.

    Example:

        edgeml train status sentiment-v1
    """
    client = _get_client()
    info = client.train_status(name)

    current = info.get("current_round", 0)
    total = info.get("total_rounds", 0)
    devices = info.get("active_devices", 0)
    status_val = info.get("status", "unknown")
    loss = info.get("loss")
    accuracy = info.get("accuracy")

    click.echo(f"Model: {name}")
    click.echo(f"Status: {status_val}")
    click.echo(f"Round: {current}/{total}")
    click.echo(f"Active devices: {devices}")
    if loss is not None:
        click.echo(f"Loss: {loss:.4f}")
    if accuracy is not None:
        click.echo(f"Accuracy: {accuracy:.1%}")


@train.command("stop")
@click.argument("name")
def train_stop_cmd(name: str) -> None:
    """Stop active training.

    Example:

        edgeml train stop sentiment-v1
    """
    client = _get_client()
    click.echo(f"Stopping training for {name}...")
    result = client.train_stop(name)
    click.echo(f"Training stopped. Last round: {result.get('last_round', '?')}")


# ---------------------------------------------------------------------------
# edgeml list
# ---------------------------------------------------------------------------


@main.command("list")
@click.argument("model_family", required=False, default=None)
def list_models_cmd(model_family: Optional[str]) -> None:
    """List available model families and their variants.

    Without arguments, shows all available model families with
    default variant, parameter count, and supported engines.

    With a model family name, shows all variants for that family
    with engine-specific artifacts (MLX repos, GGUF files).

    \b
    Examples:
        edgeml list
        edgeml list gemma-4b
        edgeml list llama-8b

    \b
    Use model:variant syntax with serve/benchmark/pull:
        edgeml serve gemma-4b:8bit
        edgeml benchmark llama-8b:fp16
    """
    from .models.catalog import CATALOG, get_model

    if model_family is None:
        # Show all families
        click.echo(
            f"{'Model':<18s} {'Publisher':<14s} {'Params':<8s} "
            f"{'Default':<10s} {'Variants'}"
        )
        click.echo("-" * 76)
        for name, entry in sorted(CATALOG.items()):
            variant_tags = ", ".join(sorted(entry.variants.keys()))
            click.echo(
                f"  {name:<16s} {entry.publisher:<14s} {entry.params:<8s} "
                f"{entry.default_quant:<10s} {variant_tags}"
            )
        click.echo(f"\n{len(CATALOG)} model families available.")
        click.echo("Use `edgeml list <model>` to see variants and engine artifacts.")
        click.echo("Use `edgeml serve <model>:<variant>` to serve a specific variant.")
    else:
        entry = get_model(model_family)
        if entry is None:
            # Try to suggest close matches
            import difflib

            suggestions = difflib.get_close_matches(
                model_family, CATALOG.keys(), n=3, cutoff=0.4
            )
            click.echo(f"Unknown model family: {model_family}", err=True)
            if suggestions:
                click.echo(f"Did you mean: {', '.join(suggestions)}?", err=True)
            click.echo(
                f"Available: {', '.join(sorted(CATALOG))}",
                err=True,
            )
            sys.exit(1)

        click.echo(f"{model_family} ({entry.publisher}, {entry.params})")
        click.echo(f"Default variant: {entry.default_quant}")
        engines_str = ", ".join(sorted(entry.engines))
        click.echo(f"Engines: {engines_str}")
        click.echo("")

        for quant, variant in sorted(entry.variants.items()):
            default_marker = " (default)" if quant == entry.default_quant else ""
            click.echo(f"  {model_family}:{quant}{default_marker}")
            if variant.mlx:
                click.echo(f"    mlx-lm:    {variant.mlx}")
            if variant.gguf:
                click.echo(f"    llama.cpp: {variant.gguf.repo} ({variant.gguf.filename})")
            if variant.source_repo:
                click.echo(f"    source:    {variant.source_repo}")
            click.echo("")


# ---------------------------------------------------------------------------
# edgeml models
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--source",
    type=click.Choice(["all", "ollama", "registry"]),
    default="all",
    help="Filter model source.",
)
def models(source: str) -> None:
    """List available models from ollama and the EdgeML registry."""
    if source in ("all", "ollama"):
        from .ollama import is_ollama_running, list_ollama_models

        if is_ollama_running():
            ollama_models = list_ollama_models()
            if ollama_models:
                click.echo("Local (ollama):")
                for m in ollama_models:
                    click.echo(
                        f"  {m.name:<20s}{m.size_display:>8s}   "
                        f"{m.quantization:<9s}{m.family}"
                    )
            else:
                click.echo("Local (ollama): no models found")
        else:
            click.echo("Local (ollama): not running")

    if source in ("all", "registry"):
        api_key = _get_api_key()
        if api_key:
            try:
                client = _get_client()
                registry_data = client.list_models()
                registry_models: list[dict[str, Any]] = registry_data.get("models", [])
                if registry_models:
                    click.echo("Registry (edgeml):")
                    for rm in registry_models:
                        name_val = rm.get("name", "unknown")
                        size = rm.get("size", 0)
                        fmt = rm.get("format", "unknown")
                        framework = rm.get("framework", "unknown")
                        size_mb = size / (1024 * 1024) if size else 0
                        click.echo(
                            f"  {name_val:<20s}{size_mb:>5.0f} MB   {fmt:<9s}{framework}"
                        )
                else:
                    click.echo("Registry (edgeml): no models found")
            except Exception:
                click.echo("Registry (edgeml): unable to fetch", err=True)
        elif source == "registry":
            click.echo("Registry (edgeml): no API key — run `edgeml login` first")


# ---------------------------------------------------------------------------
# edgeml demo
# ---------------------------------------------------------------------------


@main.group()
def demo() -> None:
    """Run interactive demos showcasing EdgeML capabilities."""


@demo.command("code-assistant")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to serve (default: gemma-2b, or EDGEML_MODEL env var).",
)
@click.option(
    "--url",
    default=None,
    help="Connect to an existing edgeml serve instance instead of auto-starting.",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8099,
    help="Port for auto-started server (default: 8099).",
)
@click.option(
    "--no-auto-start",
    is_flag=True,
    help="Don't auto-start edgeml serve; fail if no server is running.",
)
def demo_code_assistant(
    model: Optional[str],
    url: Optional[str],
    port: int,
    no_auto_start: bool,
) -> None:
    """Local code assistant — 100% on-device, zero cloud, zero cost.

    Starts an interactive chat powered by a local LLM through edgeml serve.
    Shows real-time performance metrics: latency, throughput, cost savings.

    \b
    Examples:
        edgeml demo code-assistant
        edgeml demo code-assistant --model phi-3-mini
        edgeml demo code-assistant --url http://localhost:8080
    """
    from .demos.code_assistant import run_demo

    effective_model: str = (
        model if model else os.environ.get("EDGEML_MODEL", "gemma-2b")
    )
    effective_url: str = url if url else f"http://localhost:{port}"
    run_demo(url=effective_url, model=effective_model, auto_start=not no_auto_start)


# ---------------------------------------------------------------------------
# Enterprise helpers
# ---------------------------------------------------------------------------


def _get_enterprise_client(api_base: Optional[str] = None):  # type: ignore[no-untyped-def]
    """Build an EnterpriseClient with the current API key."""
    from .enterprise import EnterpriseClient

    key = _require_api_key()
    return EnterpriseClient(api_key=key, api_base=api_base)


def _require_org_id() -> str:
    """Read org_id from config/env or exit."""
    from .enterprise import get_org_id

    org_id = get_org_id()
    if not org_id:
        click.echo(
            "No org_id configured. Run `edgeml init <name>` first, or "
            "set EDGEML_ORG_ID.",
            err=True,
        )
        sys.exit(1)
    return org_id


# ---------------------------------------------------------------------------
# edgeml init
# ---------------------------------------------------------------------------


@main.command()
@click.argument("org_name")
@click.option(
    "--compliance",
    type=click.Choice(["hipaa", "gdpr", "pci", "soc2"]),
    default=None,
    help="Enable a compliance preset (hipaa, gdpr, pci, soc2).",
)
@click.option("--region", default="us", help="Data region (us, eu, ap).")
@click.option("--api-base", default=None, help="Override API base URL.")
def init(
    org_name: str,
    compliance: Optional[str],
    region: str,
    api_base: Optional[str],
) -> None:
    """Initialize an EdgeML organization for enterprise use.

    Creates a new organization, optionally applies a compliance preset,
    and saves the org_id to ~/.edgeml/config.json for subsequent commands.

    Examples:

        edgeml init "Acme Corp" --compliance hipaa --region us
        edgeml init "Startup Inc" --region eu
    """
    from .enterprise import EnterpriseClient, save_config, load_config

    key = _get_api_key()
    if not key:
        click.echo("No API key found. Run `edgeml login` first.", err=True)
        sys.exit(1)

    client = EnterpriseClient(api_key=key, api_base=api_base)

    # 1. Create org
    click.echo(f"Creating organization: {org_name} (region={region})")
    try:
        result = client.create_org(org_name, region=region, workspace_type="enterprise")
    except Exception as exc:
        click.echo(f"Failed to create organization: {exc}", err=True)
        sys.exit(1)

    org_id = result.get("org_id", "")
    click.echo(click.style(f"  Organization created: {org_id}", fg="green"))

    # 2. Apply compliance preset if specified
    if compliance:
        click.echo(f"Applying {compliance.upper()} compliance preset...")
        try:
            client.set_compliance(org_id, compliance)
            click.echo(
                click.style(f"  {compliance.upper()} compliance applied", fg="green")
            )
        except Exception as exc:
            click.echo(f"  Warning: compliance preset failed: {exc}", err=True)

    # 3. Save org_id to config
    config = load_config()
    config["org_id"] = org_id
    config["org_name"] = org_name
    config["region"] = region
    if compliance:
        config["compliance"] = compliance
    save_config(config)
    click.echo("  Config saved to ~/.edgeml/config.json")

    # 4. Print next steps
    click.echo("")
    click.echo("Next steps:")
    click.echo(
        f"  1. Invite team members:  edgeml team add alice@{org_name.lower().replace(' ', '')}.com --role admin"
    )
    click.echo("  2. Create an API key:    edgeml keys create deploy-key")
    click.echo("  3. Set security policy:  edgeml team set-policy --require-mfa")
    click.echo(
        "  4. Push a model:         edgeml push model.pt --name my-model --version 1.0.0"
    )


# ---------------------------------------------------------------------------
# edgeml org (informational)
# ---------------------------------------------------------------------------


@main.command("org")
def org_info() -> None:
    """Show current organization info and settings."""
    from .enterprise import get_org_id, load_config

    org_id = get_org_id()
    if not org_id:
        click.echo("No organization configured. Run `edgeml init <name>` first.")
        return

    config = load_config()
    click.echo(f"Organization: {config.get('org_name', org_id)}")
    click.echo(f"Org ID: {org_id}")
    click.echo(f"Region: {config.get('region', 'unknown')}")
    if config.get("compliance"):
        click.echo(f"Compliance: {config['compliance'].upper()}")

    # Try to fetch live settings
    key = _get_api_key()
    if key:
        try:
            client = _get_enterprise_client()
            settings = client.get_settings(org_id)
            click.echo("")
            click.echo("Settings:")
            click.echo(
                f"  Audit retention:     {settings.get('audit_retention_days', '?')} days"
            )
            click.echo(
                f"  Require MFA:         {settings.get('require_mfa_for_admin', '?')}"
            )
            click.echo(
                f"  Admin approval:      {settings.get('require_admin_approval', '?')}"
            )
            click.echo(
                f"  Model approval:      {settings.get('require_model_approval', '?')}"
            )
            click.echo(
                f"  Auto rollback:       {settings.get('auto_rollback_enabled', '?')}"
            )
            click.echo(
                f"  Session duration:    {settings.get('session_duration_hours', '?')}h"
            )
            click.echo(
                f"  Reauth interval:     {settings.get('reauth_interval_minutes', '?')}min"
            )
        except Exception:
            click.echo("\n  (unable to fetch live settings)")


# ---------------------------------------------------------------------------
# edgeml team
# ---------------------------------------------------------------------------


@main.group()
def team() -> None:
    """Manage organization team members."""


@team.command("add")
@click.argument("email")
@click.option(
    "--role",
    type=click.Choice(["member", "admin", "owner"]),
    default="member",
    help="Role to assign.",
)
@click.option("--name", default=None, help="Display name for the member.")
def team_add(email: str, role: str, name: Optional[str]) -> None:
    """Invite a team member to the organization.

    Example:

        edgeml team add alice@acme.com --role admin
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    click.echo(f"Inviting {email} as {role}...")
    try:
        result = client.invite_member(org_id, email, role=role, name=name)
        click.echo(
            click.style(
                f"  Invited: {result.get('email', email)} ({result.get('role', role)})",
                fg="green",
            )
        )
    except Exception as exc:
        click.echo(f"Failed to invite member: {exc}", err=True)
        sys.exit(1)


@team.command("list")
def team_list() -> None:
    """List team members.

    Example:

        edgeml team list
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    try:
        members = client.list_members(org_id)
    except Exception as exc:
        click.echo(f"Failed to list members: {exc}", err=True)
        sys.exit(1)

    if not members:
        click.echo("No team members found.")
        return

    # Table header
    click.echo(f"{'Email':<35s} {'Role':<10s} {'Name':<20s}")
    click.echo("-" * 65)
    for m in members:
        email = m.get("email", "?")
        role = m.get("role", "?")
        name = m.get("name", "") or ""
        click.echo(f"{email:<35s} {role:<10s} {name:<20s}")
    click.echo(f"\nTotal: {len(members)} member(s)")


@team.command("set-policy")
@click.option(
    "--min-privacy-budget", type=float, default=None, help="Minimum DP epsilon budget."
)
@click.option(
    "--require-mfa", is_flag=True, default=None, help="Require MFA for admins."
)
@click.option(
    "--no-require-mfa",
    is_flag=True,
    default=None,
    help="Disable MFA requirement for admins.",
)
@click.option(
    "--auto-rollback/--no-auto-rollback",
    default=None,
    help="Auto-rollback on model drift.",
)
@click.option(
    "--session-hours", type=int, default=None, help="Session duration in hours."
)
@click.option(
    "--reauth-minutes",
    type=int,
    default=None,
    help="Re-authentication interval in minutes.",
)
@click.option(
    "--audit-retention-days",
    type=int,
    default=None,
    help="Audit log retention in days.",
)
@click.option(
    "--require-model-approval",
    is_flag=True,
    default=None,
    help="Require approval for model deployments.",
)
@click.option(
    "--no-require-model-approval",
    is_flag=True,
    default=None,
    help="Disable model deployment approval.",
)
def team_set_policy(
    min_privacy_budget: Optional[float],
    require_mfa: Optional[bool],
    no_require_mfa: Optional[bool],
    auto_rollback: Optional[bool],
    session_hours: Optional[int],
    reauth_minutes: Optional[int],
    audit_retention_days: Optional[int],
    require_model_approval: Optional[bool],
    no_require_model_approval: Optional[bool],
) -> None:
    """Set organization security policies.

    Examples:

        edgeml team set-policy --require-mfa --session-hours 8
        edgeml team set-policy --auto-rollback --audit-retention-days 365
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    updates: dict[str, Any] = {}

    # Resolve MFA flag (--require-mfa / --no-require-mfa)
    if require_mfa:
        updates["require_mfa_for_admin"] = True
    elif no_require_mfa:
        updates["require_mfa_for_admin"] = False

    if auto_rollback is not None:
        updates["auto_rollback_enabled"] = auto_rollback

    if session_hours is not None:
        updates["session_duration_hours"] = session_hours

    if reauth_minutes is not None:
        updates["reauth_interval_minutes"] = reauth_minutes

    if audit_retention_days is not None:
        updates["audit_retention_days"] = audit_retention_days

    # Resolve model approval flag
    if require_model_approval:
        updates["require_model_approval"] = True
    elif no_require_model_approval:
        updates["require_model_approval"] = False

    if not updates:
        click.echo("No policy changes specified. Use --help to see options.")
        return

    click.echo("Updating security policies...")
    try:
        client.update_settings(org_id, **updates)
        for key, value in updates.items():
            label = key.replace("_", " ").title()
            click.echo(click.style(f"  {label}: {value}", fg="green"))
    except Exception as exc:
        click.echo(f"Failed to update policies: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# edgeml keys
# ---------------------------------------------------------------------------


@main.group()
def keys() -> None:
    """Manage API keys."""


@keys.command("create")
@click.argument("name")
@click.option(
    "--scope",
    multiple=True,
    help="Key scopes (e.g., devices:write, models:read). Repeat for multiple.",
)
def keys_create(name: str, scope: tuple[str, ...]) -> None:
    """Create a new API key.

    Examples:

        edgeml keys create deploy-key --scope devices:write --scope models:read
        edgeml keys create admin-key
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    # Convert scope tuple to dict format expected by the API
    scopes: dict[str, Any] | None = None
    if scope:
        scopes = {}
        for s in scope:
            if ":" in s:
                resource, permission = s.split(":", 1)
                scopes[resource] = permission
            else:
                scopes[s] = "read"

    click.echo(f"Creating API key: {name}")
    try:
        result = client.create_api_key(org_id, name, scopes=scopes)
        raw_key = result.get("api_key", "")
        prefix = result.get("prefix", "")
        click.echo(click.style(f"  Key created: {prefix}...", fg="green"))
        click.echo("")
        click.echo(click.style(f"  API Key: {raw_key}", fg="yellow", bold=True))
        click.echo("")
        click.echo("  Save this key securely — it will not be shown again.")
        click.echo(f"  Set it as: export EDGEML_API_KEY={raw_key}")
    except Exception as exc:
        click.echo(f"Failed to create API key: {exc}", err=True)
        sys.exit(1)


@keys.command("list")
def keys_list() -> None:
    """List API keys.

    Example:

        edgeml keys list
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    try:
        api_keys = client.list_api_keys(org_id)
    except Exception as exc:
        click.echo(f"Failed to list API keys: {exc}", err=True)
        sys.exit(1)

    if not api_keys:
        click.echo("No API keys found.")
        return

    click.echo(f"{'Name':<25s} {'Prefix':<15s} {'Created':<20s} {'Status':<10s}")
    click.echo("-" * 70)
    for k in api_keys:
        name = k.get("name", "?")
        prefix = k.get("prefix", "?")
        created = k.get("created_at", "?")[:10]
        revoked = k.get("revoked_at")
        status_str = (
            click.style("revoked", fg="red")
            if revoked
            else click.style("active", fg="green")
        )
        click.echo(f"{name:<25s} {prefix:<15s} {created:<20s} {status_str}")
    click.echo(f"\nTotal: {len(api_keys)} key(s)")


@keys.command("revoke")
@click.argument("key_id")
@click.confirmation_option(prompt="Are you sure you want to revoke this API key?")
def keys_revoke(key_id: str) -> None:
    """Revoke an API key.

    Example:

        edgeml keys revoke abc-123-def
    """
    client = _get_enterprise_client()

    click.echo(f"Revoking API key: {key_id}")
    try:
        result = client.revoke_api_key(key_id)
        click.echo(
            click.style(
                f"  Revoked: {result.get('name', key_id)} (prefix: {result.get('prefix', '?')})",
                fg="yellow",
            )
        )
    except Exception as exc:
        click.echo(f"Failed to revoke API key: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# edgeml federation
# ---------------------------------------------------------------------------


def _get_org_id() -> str:
    """Read org_id from EDGEML_ORG_ID env var, defaulting to 'default'."""
    return os.environ.get("EDGEML_ORG_ID", "default")


def _resolve_federation_id(client, name_or_id: str) -> str:
    """Resolve a federation name or ID to an ID.

    First tries listing federations filtered by name.  If no match,
    assumes ``name_or_id`` is already an ID and returns it as-is.
    """
    org_id = client._org_id
    results = client._api.get(
        "/federations",
        params={"org_id": org_id, "name": name_or_id},
    )
    if results:
        return results[0]["id"]
    return name_or_id


@main.group()
def federation() -> None:
    """Manage cross-org federations."""


@federation.command("create")
@click.argument("name")
@click.option("--description", "-d", default=None, help="Federation description.")
def federation_create(name: str, description: Optional[str]) -> None:
    """Create a new federation.

    Example:

        edgeml federation create healthcare-consortium --description "Cross-hospital FL"
    """
    client = _get_client()
    payload: dict[str, Any] = {
        "name": name,
        "org_id": client._org_id,
    }
    if description:
        payload["description"] = description

    result = client._api.post("/federations", payload)
    fed_id = result.get("id", "unknown")
    click.echo(f"Federation created: {name}")
    click.echo(f"ID: {fed_id}")


@federation.command("invite")
@click.argument("federation_name")
@click.option(
    "--org",
    "org_ids",
    multiple=True,
    required=True,
    help="Org ID to invite (repeatable).",
)
def federation_invite(federation_name: str, org_ids: tuple[str, ...]) -> None:
    """Invite organisations to a federation.

    Example:

        edgeml federation invite healthcare-consortium --org org_abc --org org_def
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    result = client._api.post(
        f"/federations/{fed_id}/invite",
        {"org_ids": list(org_ids)},
    )
    invited = result if isinstance(result, list) else result.get("invited", [])
    click.echo(f"Invited {len(invited)} org(s) to federation {federation_name}")


@federation.command("join")
@click.argument("federation_name")
def federation_join(federation_name: str) -> None:
    """Accept an invitation and join a federation.

    Example:

        edgeml federation join healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    client._api.post(
        f"/federations/{fed_id}/join",
        {"org_id": client._org_id},
    )
    click.echo(f"Joined federation: {federation_name}")


@federation.command("list")
def federation_list() -> None:
    """List federations visible to your organisation.

    Example:

        edgeml federation list
    """
    client = _get_client()
    results = client._api.get(
        "/federations",
        params={"org_id": client._org_id},
    )

    if not results:
        click.echo("No federations found.")
        return

    click.echo(f"{'Name':<30s} {'ID':<40s} {'Description':<30s}")
    click.echo("-" * 100)
    for fed in results:
        name = fed.get("name", "")
        fed_id = fed.get("id", "")
        desc = fed.get("description", "") or ""
        click.echo(f"{name:<30s} {fed_id:<40s} {desc:<30s}")


@federation.command("show")
@click.argument("federation_name")
def federation_show(federation_name: str) -> None:
    """Show details of a federation.

    Example:

        edgeml federation show healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    data = client._api.get(f"/federations/{fed_id}")

    click.echo(f"Name:        {data.get('name', '')}")
    click.echo(f"ID:          {data.get('id', '')}")
    click.echo(f"Description: {data.get('description', '') or '-'}")
    click.echo(f"Created:     {data.get('created_at', '')}")
    click.echo(f"Org ID:      {data.get('org_id', '')}")


@federation.command("members")
@click.argument("federation_name")
def federation_members(federation_name: str) -> None:
    """List members of a federation.

    Example:

        edgeml federation members healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    members = client._api.get(f"/federations/{fed_id}/members")

    if not members:
        click.echo("No members found.")
        return

    click.echo(f"{'Org ID':<40s} {'Status':<15s} {'Joined':<25s}")
    click.echo("-" * 80)
    for m in members:
        org = m.get("org_id", "")
        status = m.get("status", "")
        joined = m.get("joined_at", "") or "-"
        click.echo(f"{org:<40s} {status:<15s} {joined:<25s}")


@federation.command("share")
@click.argument("model_name")
@click.option(
    "--federation",
    "-f",
    "federation_name",
    required=True,
    help="Federation name or ID to share the model with.",
)
def federation_share(model_name: str, federation_name: str) -> None:
    """Share a model with a federation for cross-org training.

    The model must belong to the federation owner's organization.
    Once shared, all active federation members can contribute training
    updates and deploy the model.

    Example:

        edgeml federation share radiology-v1 --federation healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)

    # Resolve model name to model ID via the registry
    models = client._api.get("/models", params={"org_id": client._org_id})
    model_id = None
    if isinstance(models, list):
        for m in models:
            if m.get("name") == model_name:
                model_id = m.get("id")
                break
    if not model_id:
        click.echo(
            f"Model '{model_name}' not found in your organization. "
            "Push it first with `edgeml push`.",
            err=True,
        )
        sys.exit(1)

    client._api.post(
        f"/federations/{fed_id}/models",
        {"model_id": model_id},
    )
    click.echo(f"Model '{model_name}' shared with federation '{federation_name}'")
    click.echo(
        "Active federation members can now contribute training updates "
        "and deploy this model."
    )


if __name__ == "__main__":
    main()
