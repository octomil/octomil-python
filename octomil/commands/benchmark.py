"""octomil benchmark -- inference benchmarking command and hardware helpers."""

from __future__ import annotations

import os
from typing import Any

import click

from octomil.cli_helpers import _complete_model_name, _get_api_key, http_request


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _get_gpu_core_count() -> int | None:
    """Return GPU core count on macOS via system_profiler, else None."""
    import platform as _platform

    if _platform.system() != "Darwin":
        return None
    try:
        import json
        import subprocess

        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        data = json.loads(out.stdout)
        for gpu in data.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            if cores is not None:
                return int(str(cores).replace(" ", ""))
    except Exception:
        pass
    return None


def _get_thermal_state() -> int | None:
    """Return thermal pressure as 1-4 on macOS, else None."""
    import platform as _platform

    if _platform.system() != "Darwin":
        return None
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "kern.thermal_pressure"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        val = int(out.stdout.strip())
        return val + 1  # 0-3 -> 1-4
    except Exception:
        pass
    return None


def _get_battery_level() -> int | None:
    """Return battery percentage (0-100), None for desktops."""
    try:
        import psutil

        bat = psutil.sensors_battery()
        if bat is not None:
            return int(bat.percent)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a sorted list."""
    s = sorted(data)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# _benchmark_all_engines (internal)
# ---------------------------------------------------------------------------


def _benchmark_all_engines(model: str, iterations: int, max_tokens: int) -> None:
    """Benchmark all available engines and print a comparison table."""
    import time

    from octomil.engines import get_registry
    from octomil.serve import GenerationRequest

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
# benchmark command
# ---------------------------------------------------------------------------


@click.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option(
    "--local",
    is_flag=True,
    help="Keep results local — do not upload to Octomil Cloud.",
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
        octomil benchmark gemma-1b            # default (4bit)
        octomil benchmark gemma-1b:8bit       # 8-bit quantization
        octomil benchmark llama-8b:fp16       # full precision

    Measures TTFT, TPOT, latency distribution (min/avg/median/p90/p95/p99/max),
    throughput, and memory usage across multiple iterations.

    By default, anonymous benchmark data (model name, hardware, throughput — no
    PII) is shared with Octomil Cloud.  Use --local to opt out.

    Use --all-engines to compare performance across all available engines:

        octomil benchmark gemma-1b --all-engines

    Example:

        octomil benchmark gemma-1b --iterations 20
        octomil benchmark gemma-1b --local
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

    from octomil.serve import _detect_backend

    backend = _detect_backend(model, engine_override=engine)
    click.echo(f"Backend: {backend.name}")

    from octomil.serve import GenerationRequest

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
        click.echo("\nSharing anonymous benchmark data...")
        try:
            gpu_cores = _get_gpu_core_count()
            thermal = _get_thermal_state()
            battery = _get_battery_level()

            payload = {
                "model": model,
                "backend": backend.name,
                "platform": _platform.system(),
                "arch": _platform.machine(),
                "os_version": _platform.platform(),
                "accelerator": "Metal" if _platform.system() == "Darwin" else "CPU",
                "ram_total_bytes": psutil.virtual_memory().total,
                "gpu_core_count": gpu_cores,
                "thermal_state": thermal,
                "battery_level": battery,
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
                os.environ.get("OCTOMIL_API_URL")
                or os.environ.get("OCTOMIL_API_BASE")
                or "https://api.octomil.com/api/v1"
            )
            headers = {}
            api_key = _get_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = http_request(
                "POST",
                f"{api_base}/benchmarks",
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code < 400:
                data = resp.json()
                share_url = data.get("share_url", "")
                rank = data.get("percentile_rank")
                chip = payload.get("accelerator", "CPU")
                cores = payload.get("gpu_core_count")

                click.echo(f"\nYour device: {chip}" + (f" ({cores} GPU cores)" if cores else ""))
                click.echo(f"  {model}: {avg_tps:.1f} tok/s", nl=False)
                if rank is not None:
                    click.echo(f" — top {100 - rank:.0f}% worldwide")
                else:
                    click.echo()
                click.echo("\n  Leaderboard: https://octomil.com/benchmarks")
                if share_url:
                    click.echo(f"  Share: {share_url}")
            else:
                click.echo(f"Failed to share: {resp.status_code}", err=True)
        except Exception as exc:
            click.echo(f"Failed to share: {exc}", err=True)
    else:
        click.echo("\nResults kept local (--local).")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register the benchmark command with the CLI group."""
    cli.add_command(benchmark)
