"""``octomil local`` commands — status / stop / endpoint for the invisible runner."""

from __future__ import annotations

import json
from typing import Optional

import click


@click.group("local")
def local_group() -> None:
    """Manage the invisible local inference runner."""


@local_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output JSON.")
def local_status(output_json: bool) -> None:
    """Show the current local runner status."""
    from octomil.local_runner.manager import LocalRunnerManager

    mgr = LocalRunnerManager()
    status = mgr.status()

    if output_json:
        d = {
            "running": status.running,
            "pid": status.pid,
            "port": status.port,
            "model": status.model,
            "engine": status.engine,
            "base_url": status.base_url,
            "uptime_seconds": round(status.uptime_seconds, 1) if status.uptime_seconds else None,
            "idle_timeout_seconds": status.idle_timeout_seconds,
        }
        click.echo(json.dumps(d, indent=2))
    else:
        if not status.running:
            click.echo("Local runner is not running.")
            return

        click.echo(f"Local runner is running (pid {status.pid})")
        click.echo(f"  Model:        {status.model}")
        click.echo(f"  Engine:       {status.engine}")
        click.echo(f"  Endpoint:     {status.base_url}")
        click.echo(f"  Port:         {status.port}")
        if status.uptime_seconds is not None:
            mins = int(status.uptime_seconds // 60)
            click.echo(f"  Uptime:       {mins}m")
        if status.idle_timeout_seconds is not None:
            click.echo(f"  Idle timeout: {status.idle_timeout_seconds}s")


@local_group.command("stop")
def local_stop() -> None:
    """Stop the local runner if running."""
    from octomil.local_runner.manager import LocalRunnerManager

    mgr = LocalRunnerManager()
    status = mgr.status()
    if not status.running:
        click.echo("Local runner is not running.")
        return

    mgr.stop()
    click.echo(f"Local runner stopped (was pid {status.pid}, model {status.model}).")


@local_group.command("endpoint")
@click.option("--model", "-m", default=None, help="Model to ensure is running.")
@click.option("--engine", default=None, help="Engine to use.")
@click.option("--show-token", is_flag=True, help="Print the bearer token.")
@click.option("--json", "output_json", is_flag=True, help="Output JSON.")
def local_endpoint(
    model: Optional[str],
    engine: Optional[str],
    show_token: bool,
    output_json: bool,
) -> None:
    """Ensure a runner is running and print its endpoint.

    If no --model is given, uses the default chat model from config.
    """
    from octomil.local_runner.manager import LocalRunnerManager

    # Resolve default model from config if not specified
    if model is None:
        try:
            from octomil.execution.kernel import ExecutionKernel

            kernel = ExecutionKernel()
            defaults = kernel.resolve_chat_defaults()
            model = defaults.model if defaults and defaults.model else None
        except Exception:
            pass

    if not model:
        raise click.UsageError(
            "No model specified and no default chat model configured.\n"
            "Pass --model or set a default in .octomil.toml."
        )

    mgr = LocalRunnerManager()
    try:
        handle = mgr.ensure(model=model, engine=engine)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        d = {
            "base_url": handle.base_url,
            "port": handle.port,
            "model": handle.model,
            "engine": handle.engine,
            "pid": handle.pid,
            "token_file": str(mgr._token_path),
        }
        if show_token:
            d["token"] = handle.token
        click.echo(json.dumps(d, indent=2))
    else:
        click.echo(handle.base_url)
        click.echo(f"Token file: {mgr._token_path}")
        if show_token:
            click.echo(f"Token: {handle.token}")


# ---------------------------------------------------------------------------
# Hidden subprocess entry point: octomil _local-runner-serve
# ---------------------------------------------------------------------------


@click.command("_local-runner-serve", hidden=True)
@click.option("--model", required=True, help="Model to serve.")
@click.option("--engine", default=None, help="Engine to use.")
@click.option("--port", type=int, required=True, help="Port to bind to.")
@click.option("--token-file", required=True, type=click.Path(), help="Path to bearer token file.")
@click.option("--idle-timeout", type=int, default=1800, help="Idle timeout in seconds.")
def local_runner_serve_cmd(
    model: str,
    engine: str | None,
    port: int,
    token_file: str,
    idle_timeout: int,
) -> None:
    """Internal: start the local runner server process."""
    from pathlib import Path

    from octomil.local_runner.server import run_local_runner

    run_local_runner(
        model=model,
        engine=engine,
        port=port,
        token_file=Path(token_file),
        idle_timeout=idle_timeout,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register local runner commands on the main CLI."""
    cli.add_command(local_group)
    cli.add_command(local_runner_serve_cmd)
