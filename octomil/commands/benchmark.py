"""octomil benchmark -- inference benchmarking command and hardware helpers."""

from __future__ import annotations

import os
from typing import Any

import click

from octomil.cli_helpers import (
    _complete_model_name,
    _get_api_key,
    cli_header,
    cli_kv,
    cli_metric,
    cli_section,
    cli_success,
    cli_table_header,
    http_request,
)

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
    import sys
    import time

    from octomil.runtime.engines import get_registry
    from octomil.serve import GenerationRequest

    registry = get_registry()
    detections = registry.detect_all(model)
    available = [d for d in detections if d.available and d.engine.name != "echo"]

    # Frozen binary with no native engines: try managed venv
    if getattr(sys, "frozen", False):
        native = [d for d in available if d.engine.name != "ollama"]
        if not native:
            from octomil.venv_reexec import try_venv_reexec

            try_venv_reexec()  # replaces process via os.execv if venv ready

    if not available:
        click.echo(click.style("  No inference engines available for this model.", fg="red"), err=True)
        return

    cli_section(f"Detected {len(available)} engine(s)")
    for d in available:
        info = f" ({d.info})" if d.info else ""
        click.echo(f"    {click.style('+', fg='green')} {d.engine.display_name}{info}")

    results: list[dict[str, Any]] = []
    for d in available:
        click.echo()
        cli_section(f"Benchmarking {d.engine.display_name}")
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
                progress = click.style(f"  [{i + 1}/{iterations}]", dim=True)
                tps_str = click.style(f"{metrics.tokens_per_second:.1f} tok/s", fg="white", bold=True)
                click.echo(f"{progress} {elapsed:.0f}ms  {tps_str}")

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
            click.echo(click.style(f"    Failed: {exc}", fg="red"))
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
    click.echo()
    cli_section("Comparison")
    cli_table_header(("ENGINE", 30), ("AVG TOK/S", 12), ("AVG LATENCY", 14), ("", 12))

    # Sort by tok/s descending
    results.sort(key=lambda r: r["avg_tps"], reverse=True)
    best = results[0] if results and not results[0]["error"] else None

    for i, r in enumerate(results):
        if r["error"]:
            click.echo(
                f"    {r['engine']:<30s}"
                f"{click.style('---', dim=True):>12s}"
                f"{click.style('---', dim=True):>14s}"
                f"{click.style('error', fg='red'):>12s}"
            )
        else:
            marker = click.style(" fastest", fg="cyan", bold=True) if i == 0 and best else ""
            tps_val = click.style(f"{r['avg_tps']:.1f}", fg="white", bold=True) if i == 0 else f"{r['avg_tps']:.1f}"
            click.echo(f"    {r['engine']:<30s}{tps_val:>12s}{r['avg_latency_ms']:>11.1f}ms {marker}")

    click.echo()
    if best:
        cli_success(f"Fastest: {best['engine']} ({best['avg_tps']:.1f} tok/s)")


# ---------------------------------------------------------------------------
# Hardware profiling helper
# ---------------------------------------------------------------------------


def _run_profile(model: str, engine_override: str | None) -> None:
    """Run hardware profiling and print accelerator stats."""
    from octomil.runtime.engines import get_registry

    registry = get_registry()
    detections = registry.detect_all(model)
    available = [d for d in detections if d.available and d.engine.name != "echo"]

    if engine_override:
        available = [d for d in available if d.engine.name == engine_override]

    for d in available:
        result = d.engine.profile(model)
        if result is None:
            continue

        device = result.metadata.get("device", "")
        device_str = f" ({device})" if device and device != "unknown" else ""
        active_mem = result.metadata.get("active_memory_mb")

        click.echo()
        cli_section("Hardware Profile")
        cli_metric("Accelerator", f"{result.accelerator_used}{device_str}")
        cli_metric("Peak memory", f"{result.memory_peak_mb:,.1f} MB")
        if active_mem is not None:
            cli_metric("Active memory", f"{active_mem:,.1f} MB")
        cli_metric(
            "Ops on GPU",
            f"{result.ops_on_accelerator}/{result.ops_total} ({result.utilization_pct:.1f}%)",
        )
        return  # Only show profile for the first engine that supports it


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
@click.option(
    "--profile/--no-profile",
    default=False,
    help="Run hardware utilization profiling after benchmark.",
)
def benchmark(
    model: str,
    local: bool,
    iterations: int,
    max_tokens: int,
    engine: str | None,
    all_engines: bool,
    profile: bool,
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
    import sys
    import time

    import psutil

    # Frozen binary: re-exec into managed venv if no native engines available
    if getattr(sys, "frozen", False):
        from octomil.runtime.engines import get_registry

        registry = get_registry()
        detections = registry.detect_all(model)
        available = [d for d in detections if d.available and d.engine.name != "echo"]
        native = [d for d in available if d.engine.name != "ollama"]
        if not native:
            from octomil.venv_reexec import try_venv_reexec

            try_venv_reexec()  # replaces process via os.execv if venv ready

    cli_header(f"Benchmark — {model}")
    cli_kv("Platform", f"{_platform.system()} {_platform.machine()}")
    cli_kv("Iterations", str(iterations))
    cli_kv("Max tokens", str(max_tokens))

    # Quick engine comparison if --all-engines
    if all_engines:
        _benchmark_all_engines(model, iterations, max_tokens)
        return

    from octomil.serve import _detect_backend

    backend = _detect_backend(model, engine_override=engine)
    cli_kv("Engine", backend.name)
    click.echo()

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
        progress = click.style(f"  [{i + 1}/{iterations}]", dim=True)
        tps_str = click.style(f"{metrics.tokens_per_second:.1f} tok/s", fg="white", bold=True)
        click.echo(f"{progress} {elapsed:.0f}ms  {tps_str}  {click.style(f'{metrics.total_tokens} tokens', dim=True)}")

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
    avg_prompt = sum(prompt_tokens_list) // len(prompt_tokens_list) if prompt_tokens_list else 0
    avg_completion = sum(completion_tokens_list) // len(completion_tokens_list) if completion_tokens_list else 0
    # TPOT = (total_latency - TTFT) / completion_tokens
    tpot = (avg_latency - avg_ttft) / avg_completion if avg_completion > 0 and avg_ttft > 0 else 0

    click.echo()
    cli_section("Results")
    cli_kv("Engine", backend.name)
    cli_kv("Iterations", str(iterations))
    cli_kv("Avg prompt", f"{avg_prompt} tokens")
    cli_kv("Avg completion", f"{avg_completion} tokens")
    click.echo()
    cli_section("Timing")
    cli_metric("TTFT (avg)", f"{avg_ttft:.1f}ms", highlight=True)
    cli_metric("TPOT (avg)", f"{tpot:.2f}ms/token", highlight=True)
    click.echo()
    cli_section("Latency")
    cli_metric("min", f"{min_latency:.1f}ms")
    cli_metric("avg", f"{avg_latency:.1f}ms")
    cli_metric("p50", f"{p50:.1f}ms")
    cli_metric("p90", f"{p90:.1f}ms")
    cli_metric("p95", f"{p95:.1f}ms")
    cli_metric("p99", f"{p99:.1f}ms")
    cli_metric("max", f"{max_latency:.1f}ms")
    click.echo()
    cli_section("Throughput")
    cli_metric("avg", f"{avg_tps:.1f} tok/s", highlight=True)
    cli_metric("peak", f"{peak_tps:.1f} tok/s", highlight=True)
    cli_metric("Memory", f"{peak_mem / 1024 / 1024:.0f} MB (+{peak_mem_delta / 1024 / 1024:.0f} MB)")

    if profile:
        _run_profile(model, engine)

    if not local:
        click.echo()
        click.echo(click.style("  Sharing anonymous benchmark data...", dim=True))
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

                click.echo()
                cli_section("Leaderboard")
                device_str = str(chip) + (f" ({cores} cores)" if cores else "")
                cli_kv("Device", device_str)
                tps_str = click.style(f"{avg_tps:.1f} tok/s", fg="white", bold=True)
                rank_str = ""
                if rank is not None:
                    rank_str = click.style(f"  top {100 - rank:.0f}%", fg="cyan", bold=True)
                cli_kv("Score", f"{tps_str}{rank_str}")
                click.echo()
                cli_kv("Leaderboard", "https://octomil.com/benchmarks")
                if share_url:
                    cli_kv("Share", share_url)
            else:
                click.echo(click.style(f"  Failed to share: {resp.status_code}", fg="red"), err=True)
        except Exception as exc:
            click.echo(click.style(f"  Failed to share: {exc}", fg="red"), err=True)
    else:
        click.echo()
        click.echo(click.style("  Results kept local (--local).", dim=True))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register the benchmark command with the CLI group."""
    cli.add_command(benchmark)
