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
    edgeml benchmark gemma-1b --share
    edgeml login
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
def serve(model: str, port: int, host: str, benchmark: bool, share: bool) -> None:
    """Start a local OpenAI-compatible inference server.

    Serves MODEL via the best available backend (mlx-lm on Apple Silicon,
    llama.cpp on other platforms). No account required.

    Example:

        edgeml serve gemma-1b --port 8080

        curl localhost:8080/v1/chat/completions \\
            -d '{"model":"gemma-1b","messages":[{"role":"user","content":"Hi"}]}'
    """
    api_key = _get_api_key() if share else None
    api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")

    click.echo(f"Starting EdgeML serve on {host}:{port}")
    click.echo(f"Model: {model}")
    click.echo(f"OpenAI-compatible API: http://localhost:{port}/v1/chat/completions")
    click.echo(f"Health check: http://localhost:{port}/health")

    if benchmark:
        click.echo("Benchmark mode: will run latency test after model loads.")

    if share and not api_key:
        click.echo(
            "Warning: --share requires an API key to upload benchmark data. "
            "Run `edgeml login` or set EDGEML_API_KEY.",
            err=True,
        )

    from .serve import run_server

    run_server(model, port=port, host=host, api_key=api_key, api_base=api_base)


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


@main.command()
@click.option(
    "--api-key", prompt="API key", hide_input=True, help="Your EdgeML API key."
)
def login(api_key: str) -> None:
    """Authenticate with EdgeML and store your API key.

    The key is stored in ~/.edgeml/credentials. You can also set the
    EDGEML_API_KEY environment variable instead.
    """
    config_dir = os.path.expanduser("~/.edgeml")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "credentials")

    with open(config_path, "w") as f:
        f.write(f"api_key={api_key}\n")

    os.chmod(config_path, 0o600)
    click.echo(f"API key saved to {config_path}")


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

    Downloads NAME at VERSION in the specified FORMAT to OUTPUT directory.

    Example:

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

        edgeml deploy gemma-1b --phone
        edgeml deploy sentiment-v1 --rollout 10 --strategy canary
    """
    if phone:
        import httpx

        # Detect ollama models before deploying
        from .ollama import get_ollama_model

        ollama_model = get_ollama_model(name)
        if ollama_model:
            click.echo(
                f"Detected ollama model: {ollama_model.name} "
                f"({ollama_model.size_display}, {ollama_model.quantization})"
            )


        api_key = _require_api_key()
        api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")
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

        click.echo(f"\nPairing code: {code}")
        click.echo("Enter this code in the EdgeML app on your phone.")
        click.echo(f"Expires: {session['expires_at']}")
        click.echo("\nOpening dashboard...")
        webbrowser.open(f"{dashboard_url}/deploy/phone?code={code}&model={name}")

        click.echo("\nWaiting for device to connect (Ctrl+C to cancel)...")
        try:
            while True:
                import time

                time.sleep(2)
                poll = httpx.get(f"{api_base}/deploy/pair/{code}", timeout=5.0)
                if poll.status_code != 200:
                    continue
                data = poll.json()
                status_val = data.get("status", "pending")
                if status_val == "connected":
                    device = data.get("device_name") or data.get("device_id", "unknown")
                    platform = data.get("device_platform", "unknown")
                    click.echo(f"Device connected: {device} ({platform})")
                    click.echo("Deployment in progress...")
                elif status_val == "deploying":
                    click.echo("Deploying...")
                elif status_val == "done":
                    click.echo("Deployment complete.")
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

    api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")
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
    "--share", is_flag=True, help="Upload anonymous benchmark results to EdgeML."
)
@click.option("--iterations", "-n", default=10, help="Number of inference iterations.")
@click.option("--max-tokens", default=50, help="Max tokens to generate per iteration.")
def benchmark(model: str, share: bool, iterations: int, max_tokens: int) -> None:
    """Run inference benchmarks on a model.

    Measures TTFT, TPOT, latency distribution (min/avg/median/p90/p95/p99/max),
    throughput, and memory usage across multiple iterations.

    Example:

        edgeml benchmark gemma-1b --share --iterations 20
    """
    import platform as _platform
    import time

    import psutil

    click.echo(
        f"Benchmarking {model} ({iterations} iterations, {max_tokens} max tokens)..."
    )
    click.echo(f"Platform: {_platform.system()} {_platform.machine()}")

    from .serve import _detect_backend

    backend = _detect_backend(model)
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

    if share:
        api_key = _get_api_key()
        if not api_key:
            click.echo(
                "\nSkipping share: no API key. Run `edgeml login` first.",
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
            api_base = os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")
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
                registry_models = client.list_models()
                if registry_models:
                    click.echo("Registry (edgeml):")
                    for m in registry_models:
                        name = m.get("name", "unknown")
                        size = m.get("size", 0)
                        fmt = m.get("format", "unknown")
                        framework = m.get("framework", "unknown")
                        size_mb = size / (1024 * 1024) if size else 0
                        click.echo(
                            f"  {name:<20s}{size_mb:>5.0f} MB   {fmt:<9s}{framework}"
                        )
                else:
                    click.echo("Registry (edgeml): no models found")
            except Exception:
                click.echo("Registry (edgeml): unable to fetch", err=True)
        elif source == "registry":
            click.echo("Registry (edgeml): no API key — run `edgeml login` first")


if __name__ == "__main__":
    main()
