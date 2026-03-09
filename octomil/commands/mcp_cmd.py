"""CLI command: ``octomil mcp`` — register, unregister, and check MCP status."""

from __future__ import annotations

from typing import Optional

import click


@click.group()
def mcp() -> None:
    """Manage the Octomil MCP server for Claude Code."""


@mcp.command()
@click.option("--model", "-m", default=None, help="Model to use (default: qwen-coder-7b).")
def register(model: Optional[str]) -> None:
    """Register the Octomil MCP server with Claude Code."""
    try:
        import mcp as _mcp  # noqa: F401
    except ImportError:
        click.echo("Error: the 'mcp' package is required.", err=True)
        click.echo("Install with: pip install 'mcp[cli]>=1.2.0'", err=True)
        raise SystemExit(1)

    from octomil.mcp.registration import RegistrationError, register_mcp_server

    try:
        path = register_mcp_server(model=model)
        click.echo(f"Registered Octomil MCP server in {path}")
        if model:
            click.echo(f"Model: {model}")
        else:
            click.echo("Model: qwen-coder-7b (default, set with --model)")
    except RegistrationError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


@mcp.command()
def unregister() -> None:
    """Remove the Octomil MCP server from Claude Code."""
    from octomil.mcp.registration import RegistrationError, unregister_mcp_server

    try:
        removed = unregister_mcp_server()
        if removed:
            click.echo("Unregistered Octomil MCP server.")
        else:
            click.echo("Octomil MCP server was not registered.")
    except RegistrationError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


@mcp.command()
def status() -> None:
    """Show the current MCP registration status."""
    from octomil.mcp.registration import get_registration_info, get_settings_path, is_registered

    click.echo(f"Settings file: {get_settings_path()}")

    if is_registered():
        info = get_registration_info()
        click.echo("Status: registered")
        if info:
            click.echo(f"  command: {info.get('command', '?')}")
            click.echo(f"  args: {info.get('args', [])}")
            env = info.get("env", {})
            if env:
                click.echo(f"  model: {env.get('OCTOMIL_MCP_MODEL', 'default')}")
    else:
        click.echo("Status: not registered")
        click.echo("Run 'octomil mcp register' to set up.")


@mcp.command()
@click.option("--port", "-p", default=8402, help="Port to listen on (default: 8402).")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0).")
@click.option("--model", "-m", default=None, help="Model to use (default: qwen-coder-7b).")
@click.option("--x402", is_flag=True, default=False, help="Enable x402 payment gating.")
def serve(port: int, host: str, model: Optional[str], x402: bool) -> None:
    """Start the Octomil HTTP agent server.

    Exposes all tools via REST, serves an A2A agent card for discovery,
    and auto-generates OpenAPI docs at /docs.

    Examples:

        octomil mcp serve

        octomil mcp serve --port 9000 --model gemma-3b

        OCTOMIL_X402_ADDRESS=0x... octomil mcp serve --x402
    """
    try:
        import fastapi as _fa  # noqa: F401
        import uvicorn as _uv  # noqa: F401
    except ImportError:
        click.echo("Error: fastapi and uvicorn are required for the HTTP server.", err=True)
        click.echo("Install with: pip install 'octomil-sdk[serve]'", err=True)
        raise SystemExit(1)

    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    config = HTTPServerConfig(
        host=host,
        port=port,
        model=model,
        enable_x402=x402,
    )
    app = create_http_app(config)

    click.echo(f"Starting Octomil agent server on {host}:{port}")
    click.echo(f"  Agent card: http://{host}:{port}/.well-known/agent-card.json")
    click.echo(f"  OpenAPI docs: http://{host}:{port}/docs")
    click.echo(f"  Health: http://{host}:{port}/health")
    if x402:
        click.echo("  x402 payment gating: enabled")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def register_cmd(cli: click.Group) -> None:
    """Register the mcp command group with the CLI."""
    cli.add_command(mcp)
