#!/usr/bin/env python3
"""EdgeML Code Assistant — local AI that never phones home.

A fully on-device code assistant powered by EdgeML.  Runs a local LLM
through ``edgeml serve``, exposes an OpenAI-compatible API, and renders
a rich terminal chat with live performance metrics.

Usage::

    # Option A: launch everything automatically
    edgeml demo code-assistant

    # Option B: run against an existing edgeml serve instance
    python examples/code-assistant/demo.py --url http://localhost:8080

    # Option C: specify a model
    edgeml demo code-assistant --model gemma-2b
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Optional rich dependency — graceful fallback to plain text
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemma-2b"
DEFAULT_PORT = 8099
SYSTEM_PROMPT = (
    "You are a helpful code assistant running entirely on the user's device. "
    "Your inference is local — no data leaves this machine. "
    "Be concise. Use code blocks with syntax highlighting when showing code."
)

# Approximate OpenAI GPT-4o pricing per 1K tokens (for cost comparison)
CLOUD_COST_PER_1K_INPUT = 0.0025
CLOUD_COST_PER_1K_OUTPUT = 0.01


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Accumulates session-level inference metrics."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0.0
        self.ttfc_samples: list[float] = []
        self.cloud_api_calls = 0
        self.session_start = time.time()

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        ttfc_ms: float | None = None,
    ) -> None:
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        if ttfc_ms is not None:
            self.ttfc_samples.append(ttfc_ms)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_requests, 1)

    @property
    def avg_ttfc_ms(self) -> float:
        return sum(self.ttfc_samples) / max(len(self.ttfc_samples), 1)

    @property
    def avg_tok_per_sec(self) -> float:
        total_sec = self.total_latency_ms / 1000.0
        return self.total_output_tokens / max(total_sec, 0.001)

    @property
    def cloud_cost_saved(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * CLOUD_COST_PER_1K_INPUT
        output_cost = (self.total_output_tokens / 1000) * CLOUD_COST_PER_1K_OUTPUT
        return input_cost + output_cost

    @property
    def session_duration_min(self) -> float:
        return (time.time() - self.session_start) / 60.0


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _wait_for_server(url: str, timeout: float = 120.0) -> bool:
    """Poll the health endpoint until the server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            pass
        time.sleep(1.0)
    return False


def _start_server(model: str, port: int) -> subprocess.Popen[bytes] | None:
    """Start ``edgeml serve`` in a subprocess."""
    cmd = [sys.executable, "-m", "edgeml.cli", "serve", model, "--port", str(port)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Chat client (OpenAI-compatible)
# ---------------------------------------------------------------------------


def stream_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[dict[str, Any]]:
    """Stream chat completions from the local edgeml serve instance."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Server returned {response.status_code}")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Rich terminal UI
# ---------------------------------------------------------------------------


def _build_metrics_panel(metrics: MetricsTracker, model: str) -> Panel:
    """Build the metrics sidebar panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Model", model)
    table.add_row("Requests", str(metrics.total_requests))
    table.add_row("Avg latency", f"{metrics.avg_latency_ms:.0f} ms")
    table.add_row("Avg TTFC", f"{metrics.avg_ttfc_ms:.0f} ms")
    table.add_row("Throughput", f"{metrics.avg_tok_per_sec:.1f} tok/s")
    table.add_row(
        "Tokens (in/out)", f"{metrics.total_input_tokens}/{metrics.total_output_tokens}"
    )
    table.add_row("", "")
    table.add_row("Cloud API calls", "[green]0[/green]")
    table.add_row("Data sent to cloud", "[green]0 bytes[/green]")
    table.add_row(
        "Cost", f"[green]$0.00[/green] (saved ${metrics.cloud_cost_saved:.4f})"
    )
    table.add_row("Session", f"{metrics.session_duration_min:.1f} min")

    return Panel(
        table, title="[bold cyan]EdgeML Metrics[/bold cyan]", border_style="cyan"
    )


def _print_metrics_plain(metrics: MetricsTracker, model: str) -> None:
    """Print metrics without rich."""
    print("\n--- EdgeML Metrics ---")
    print(f"Model:          {model}")
    print(f"Requests:       {metrics.total_requests}")
    print(f"Avg latency:    {metrics.avg_latency_ms:.0f} ms")
    print(f"Throughput:     {metrics.avg_tok_per_sec:.1f} tok/s")
    print("Cloud API calls: 0")
    print(f"Cost:           $0.00 (saved ${metrics.cloud_cost_saved:.4f})")
    print("---------------------\n")


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------


def run_demo(
    url: str = f"http://localhost:{DEFAULT_PORT}",
    model: str = DEFAULT_MODEL,
    auto_start: bool = True,
) -> None:
    """Run the interactive code assistant demo."""

    console = Console() if HAS_RICH else None
    metrics = MetricsTracker()
    server_proc: subprocess.Popen[bytes] | None = None

    def _cleanup(sig: int | None = None, frame: Any = None) -> None:
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Banner
    banner = (
        "\n"
        "  ╔══════════════════════════════════════════════════════╗\n"
        "  ║         EdgeML Code Assistant                       ║\n"
        "  ║         100% on-device · zero cloud · zero cost     ║\n"
        "  ╚══════════════════════════════════════════════════════╝\n"
    )

    if console:
        console.print(banner, style="bold cyan")
    else:
        print(banner)

    # Check if server is already running
    try:
        resp = httpx.get(f"{url}/health", timeout=3.0)
        if resp.status_code == 200:
            health = resp.json()
            model = health.get("model", model)
            engine = health.get("engine", "unknown")
            msg = f"Connected to edgeml serve ({model} on {engine})"
            if console:
                console.print(f"  [green]✓[/green] {msg}")
            else:
                print(f"  ✓ {msg}")
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        if not auto_start:
            print(f"  ✗ No server at {url}. Start one with: edgeml serve {model}")
            sys.exit(1)

        msg = f"Starting edgeml serve {model} on port {url.split(':')[-1]}..."
        if console:
            console.print(f"  [yellow]⟳[/yellow] {msg}")
        else:
            print(f"  ⟳ {msg}")

        port = int(url.split(":")[-1])
        server_proc = _start_server(model, port)
        if server_proc is None:
            print(
                "  ✗ Failed to start edgeml serve. Install with: pip install edgeml[serve]"
            )
            sys.exit(1)

        if console:
            console.print("  Waiting for model to load (this may take a minute)...")
        else:
            print("  Waiting for model to load (this may take a minute)...")

        if not _wait_for_server(url, timeout=180.0):
            print("  ✗ Server failed to start. Check: edgeml serve --help")
            _cleanup()
            sys.exit(1)

        # Re-check health for model info
        try:
            resp = httpx.get(f"{url}/health", timeout=3.0)
            health = resp.json()
            model = health.get("model", model)
            engine = health.get("engine", "unknown")
            if console:
                console.print(f"  [green]✓[/green] Ready ({model} on {engine})")
            else:
                print(f"  ✓ Ready ({model} on {engine})")
        except Exception:
            pass

    if console:
        console.print()
        console.print(
            "  Type your question and press Enter. Type [bold]/quit[/bold] to exit."
        )
        console.print("  Type [bold]/metrics[/bold] to see session stats.\n")
    else:
        print("\n  Type your question and press Enter. Type /quit to exit.")
        print("  Type /metrics to see session stats.\n")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            _cleanup()
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            if console:
                console.print(_build_metrics_panel(metrics, model))
            else:
                _print_metrics_plain(metrics, model)
            _cleanup()
            break

        if user_input.lower() in ("/metrics", "/stats"):
            if console:
                console.print(_build_metrics_panel(metrics, model))
            else:
                _print_metrics_plain(metrics, model)
            continue

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  Context cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        # Stream the response
        start = time.perf_counter()
        ttfc_ms: float | None = None
        full_response = ""
        output_tokens = 0

        try:
            if console:
                # Rich streaming output
                console.print()
                with Live(Text(""), console=console, refresh_per_second=15) as live:
                    for chunk in stream_chat(url, model, messages):
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            if ttfc_ms is None:
                                ttfc_ms = (time.perf_counter() - start) * 1000
                            full_response += delta
                            output_tokens += 1
                            try:
                                live.update(Markdown(full_response))
                            except Exception:
                                live.update(Text(full_response))
                console.print()
            else:
                # Plain text streaming
                print()
                for chunk in stream_chat(url, model, messages):
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        if ttfc_ms is None:
                            ttfc_ms = (time.perf_counter() - start) * 1000
                        full_response += delta
                        output_tokens += 1
                        print(delta, end="", flush=True)
                print("\n")

        except (RuntimeError, httpx.HTTPError) as exc:
            print(f"  Error: {exc}")
            messages.pop()  # Remove failed user message
            continue

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Rough input token estimate (4 chars per token)
        input_tokens = sum(len(m["content"]) for m in messages) // 4

        metrics.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            ttfc_ms=ttfc_ms,
        )

        # Show inline metrics
        tok_s = output_tokens / max(elapsed_ms / 1000, 0.001)
        cost_saved = (output_tokens / 1000) * CLOUD_COST_PER_1K_OUTPUT
        ttfc_str = f"  TTFC: {ttfc_ms:.0f}ms" if ttfc_ms else ""
        stats_line = (
            f"  [{output_tokens} tokens · {tok_s:.1f} tok/s · "
            f"{elapsed_ms:.0f}ms{ttfc_str} · "
            f"cloud calls: 0 · saved: ${cost_saved:.4f}]"
        )
        if console:
            console.print(stats_line, style="dim")
            console.print()
        else:
            print(stats_line)
            print()

        messages.append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeML Code Assistant — 100% on-device, zero cloud",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EDGEML_MODEL", DEFAULT_MODEL),
        help=f"Model to serve (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("EDGEML_URL"),
        help="Connect to an existing edgeml serve instance",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for auto-started server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start edgeml serve; fail if not running",
    )
    args = parser.parse_args()

    url = args.url or f"http://localhost:{args.port}"
    run_demo(url=url, model=args.model, auto_start=not args.no_auto_start)


if __name__ == "__main__":
    main()
