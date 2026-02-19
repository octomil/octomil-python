"""
Octomil command-line interface.

Usage::

    octomil serve gemma-1b --port 8080
    octomil deploy gemma-1b --phone
    octomil dashboard
    octomil push model.pt --name sentiment-v1 --version 1.0.0
    octomil pull sentiment-v1 --version 1.0.0 --format coreml
    octomil check model.pt
    octomil convert model.pt --target ios,android
    octomil status sentiment-v1
    octomil benchmark gemma-1b --share
    octomil login
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Read API key from env, keychain, or raise."""
    key = os.environ.get("OCTOMIL_API_KEY", "")
    if not key:
        config_path = os.path.expanduser("~/.octomil/credentials")
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
        click.echo("No API key found. Run `octomil login` first.", err=True)
        sys.exit(1)
    return key


def _get_client():  # type: ignore[no-untyped-def]
    from .client import Client

    return Client(api_key=_require_api_key())


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="1.0.0", prog_name="octomil")
def main() -> None:
    """Octomil — serve, deploy, and observe ML models on edge devices."""


# ---------------------------------------------------------------------------
# octomil serve
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model")
@click.option("--port", "-p", default=8080, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--benchmark", is_flag=True, help="Run latency benchmark on startup.")
@click.option(
    "--share", is_flag=True, help="Share anonymous benchmark data with Octomil."
)
def serve(model: str, port: int, host: str, benchmark: bool, share: bool) -> None:
    """Start a local OpenAI-compatible inference server.

    Serves MODEL via the best available backend (mlx-lm on Apple Silicon,
    llama.cpp on other platforms). No account required.

    Example:

        octomil serve gemma-1b --port 8080

        curl localhost:8080/v1/chat/completions \\
            -d '{"model":"gemma-1b","messages":[{"role":"user","content":"Hi"}]}'
    """
    api_key = _get_api_key() if share else None
    api_base = os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.io/api/v1")

    click.echo(f"Starting Octomil serve on {host}:{port}")
    click.echo(f"Model: {model}")
    click.echo(f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions")
    click.echo(f"Health check: http://localhost:{port}/health")

    if benchmark:
        click.echo("Benchmark mode: will run latency test after model loads.")

    if share and not api_key:
        click.echo(
            "Warning: --share requires an API key to upload benchmark data. "
            "Run `octomil login` or set OCTOMIL_API_KEY.",
            err=True,
        )

    from .serve import run_server

    run_server(model, port=port, host=host, api_key=api_key, api_base=api_base)


# ---------------------------------------------------------------------------
# octomil check
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
        click.echo("  - Convert to ONNX: octomil convert model.pt --target onnx")
        click.echo(
            "  - Convert to CoreML (iOS): octomil convert model.pt --target coreml"
        )
        click.echo(
            "  - Convert to TFLite (Android): octomil convert model.pt --target tflite"
        )
    elif ext == ".onnx":
        click.echo("  - ONNX is cross-platform — ready for deployment")
        click.echo(
            "  - Convert to CoreML (iOS): octomil convert model.onnx --target coreml"
        )
    elif ext == ".gguf":
        click.echo("  - GGUF models work with llama.cpp backend")
        click.echo("  - Serve locally: octomil serve model.gguf")
    else:
        click.echo("  - No specific recommendations for this format")


# ---------------------------------------------------------------------------
# octomil convert
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

        octomil convert model.pt --target onnx
        octomil convert model.pt --target onnx,coreml --input-shape 1,3,224,224
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
# octomil login
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--api-key", prompt="API key", hide_input=True, help="Your Octomil API key."
)
def login(api_key: str) -> None:
    """Authenticate with Octomil and store your API key.

    The key is stored in ~/.octomil/credentials. You can also set the
    OCTOMIL_API_KEY environment variable instead.
    """
    config_dir = os.path.expanduser("~/.octomil")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "credentials")

    with open(config_path, "w") as f:
        f.write(f"api_key={api_key}\n")

    os.chmod(config_path, 0o600)
    click.echo(f"API key saved to {config_path}")


# ---------------------------------------------------------------------------
# octomil push
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

        octomil push model.pt --name sentiment-v1 --version 1.0.0 --formats coreml,tflite
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
# octomil pull
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

    Downloads NAME at VERSION in the specified FORMAT to OUTPUT directory.

    Example:

        octomil pull sentiment-v1 --version 1.0.0 --format coreml
    """
    client = _get_client()
    ver_str = version or "latest"
    click.echo(f"Pulling {name} v{ver_str}...")

    result = client.pull(name, version=version, format=fmt, destination=output)
    click.echo(f"Downloaded: {result['model_path']}")


# ---------------------------------------------------------------------------
# octomil deploy
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
    type=click.Choice(["canary", "immediate"]),
    help="Rollout strategy.",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Comma-separated target formats: ios, android.",
)
def deploy(
    name: str,
    version: Optional[str],
    phone: bool,
    rollout: int,
    strategy: str,
    target: Optional[str],
) -> None:
    """Deploy a model to edge devices.

    Deploys NAME at VERSION to devices. Use --phone for quick
    phone deployment, or --rollout for fleet percentage rollouts.

    Examples:

        octomil deploy gemma-1b --phone
        octomil deploy sentiment-v1 --rollout 10 --strategy canary
    """
    if phone:
        click.echo(f"Deploying {name} to phone...")
        click.echo("Scan the QR code in your Octomil dashboard to connect your device.")
        click.echo("Opening dashboard...")
        dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.io")
        webbrowser.open(f"{dashboard_url}/deploy/phone?model={name}")
        return

    client = _get_client()
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
# octomil status
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
def status(name: str) -> None:
    """Show model status, active rollouts, and inference metrics.

    Example:

        octomil status sentiment-v1
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
# octomil dashboard
# ---------------------------------------------------------------------------


@main.command()
def dashboard() -> None:
    """Open the Octomil dashboard in your browser.

    Shows inference metrics across all devices — latency,
    throughput, errors, model versions side-by-side.
    """
    dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.io")
    click.echo(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)


# ---------------------------------------------------------------------------
# octomil benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model")
@click.option(
    "--share", is_flag=True, help="Upload anonymous benchmark results to Octomil."
)
@click.option("--iterations", "-n", default=10, help="Number of inference iterations.")
def benchmark(model: str, share: bool, iterations: int) -> None:
    """Run inference benchmarks on a model.

    Measures time-to-first-chunk, throughput, and memory usage
    across multiple iterations.

    Example:

        octomil benchmark gemma-1b --share --iterations 20
    """
    import platform as _platform
    import time

    click.echo(f"Benchmarking {model} ({iterations} iterations)...")
    click.echo(f"Platform: {_platform.system()} {_platform.machine()}")

    from .serve import _detect_backend

    backend = _detect_backend(model)
    click.echo(f"Backend: {backend.name}")

    from .serve import GenerationRequest

    req = GenerationRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=50,
    )

    latencies: list[float] = []
    tps_list: list[float] = []

    for i in range(iterations):
        start = time.monotonic()
        text, metrics = backend.generate(req)
        elapsed = (time.monotonic() - start) * 1000
        latencies.append(elapsed)
        tps_list.append(metrics.tokens_per_second)
        click.echo(
            f"  [{i+1}/{iterations}] {elapsed:.1f}ms, {metrics.tokens_per_second:.1f} tok/s"
        )

    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    avg_tps = sum(tps_list) / len(tps_list)

    click.echo("\nResults:")
    click.echo(f"  Avg latency: {avg_latency:.1f}ms")
    click.echo(f"  P50 latency: {p50:.1f}ms")
    click.echo(f"  P95 latency: {p95:.1f}ms")
    click.echo(f"  Avg throughput: {avg_tps:.1f} tok/s")
    click.echo(f"  Backend: {backend.name}")

    if share:
        api_key = _get_api_key()
        if not api_key:
            click.echo(
                "\nSkipping share: no API key. Run `octomil login` first.",
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
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "avg_tokens_per_second": avg_tps,
                "iterations": iterations,
            }
            api_base = os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.io/api/v1")
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


if __name__ == "__main__":
    main()
