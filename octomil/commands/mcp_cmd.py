"""CLI command: ``octomil mcp`` — register MCP server across AI coding tools."""

from __future__ import annotations

from typing import Optional

import click

from octomil.cli_helpers import cli_header, cli_kv, cli_success, cli_warn


@click.group()
def mcp() -> None:
    """Set up the Octomil MCP server for AI coding tools."""


@mcp.command()
@click.option("--model", "-m", default=None, help="Model to use (default: qwen-coder-7b).")
@click.option(
    "--target",
    "-t",
    default=None,
    type=click.Choice(["claude", "cursor", "vscode", "codex"]),
    help="Register with a specific tool only.",
)
def register(model: Optional[str], target: Optional[str]) -> None:
    """Register the Octomil MCP server with your AI tools.

    \b
    Configures Claude Code, Cursor, VS Code, and Codex CLI
    so they can use Octomil's local inference tools via MCP.
    """
    from octomil.mcp.registration import register_mcp_server

    results = register_mcp_server(model=model, target=target)

    cli_header("MCP Setup")
    any_success = False
    for r in results:
        if r.success:
            cli_success(f"{r.display:<16s}{r.path}")
            any_success = True
        elif r.error:
            cli_warn(f"{r.display:<16s}{r.error}")
        else:
            cli_warn(f"{r.display:<16s}skipped")

    if any_success:
        click.echo()
        if model:
            cli_kv("Model", model)
        else:
            cli_kv("Model", "qwen-coder-7b (default, set with --model)")
        click.echo()


@mcp.command()
@click.option(
    "--target",
    "-t",
    default=None,
    type=click.Choice(["claude", "cursor", "vscode", "codex"]),
    help="Unregister from a specific tool only.",
)
def unregister(target: Optional[str]) -> None:
    """Remove the Octomil MCP server from your AI tools."""
    from octomil.mcp.registration import unregister_mcp_server

    results = unregister_mcp_server(target=target)

    cli_header("MCP Unregister")
    for r in results:
        if r.success:
            cli_success(f"Removed from {r.display}")
        else:
            click.echo(click.style(f"    {r.display:<16s}", dim=True) + "not registered")


@mcp.command()
def status() -> None:
    """Show MCP registration status across all tools."""
    from octomil.mcp.registration import TARGETS, get_all_status

    cli_header("MCP Status")
    statuses = get_all_status()
    for t in TARGETS:
        registered = statuses.get(t.name, False)
        if registered:
            cli_success(f"{t.display:<16s}{t.path}")
        else:
            click.echo(click.style(f"    -  {t.display:<16s}", dim=True) + click.style(str(t.path), dim=True))


@mcp.command()
@click.option("--port", "-p", default=8402, help="Port to listen on (default: 8402).")
@click.option("--host", "-H", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0).")
@click.option("--model", "-m", default=None, help="Model to use (default: qwen-coder-7b).")
@click.option("--x402", is_flag=True, default=False, help="Enable x402 payment gating.")
@click.option(
    "--x402-price",
    default="1000",
    envvar="OCTOMIL_X402_PRICE",
    help="Price per call in base units (default: 1000 = $0.001 USDC).",
)
@click.option(
    "--x402-currency", default="USDC", envvar="OCTOMIL_X402_CURRENCY", help="Payment currency (default: USDC)."
)
@click.option(
    "--x402-address",
    default="0x7BeEa3e83033e5399FfAAdfb8bf731eBb36126F3",
    envvar="OCTOMIL_X402_ADDRESS",
    help="Payment receiving address.",
)
@click.option(
    "--x402-threshold",
    default="1.0",
    envvar="OCTOMIL_X402_THRESHOLD",
    help="Settlement threshold in USD (default: 1.0).",
)
@click.option(
    "--settler-url",
    default="https://api.settle402.dev",
    envvar="OCTOMIL_SETTLER_URL",
    help="settle402 batch settlement URL.",
)
@click.option(
    "--settler-token",
    default="",
    envvar="OCTOMIL_SETTLER_TOKEN",
    help="settle402 API key (X-Settler-Token).",
)
def serve(
    port: int,
    host: str,
    model: Optional[str],
    x402: bool,
    x402_price: str,
    x402_currency: str,
    x402_address: str,
    x402_threshold: str,
    settler_url: str,
    settler_token: str,
) -> None:
    """Start the Octomil HTTP agent server.

    Exposes all tools via REST, serves an A2A agent card for discovery,
    and auto-generates OpenAPI docs at /docs. When --x402 is enabled,
    accepted payments are batched and settled on-chain via settle402.

    Examples:

        octomil mcp serve

        octomil mcp serve --port 9000 --model gemma-3b

        OCTOMIL_X402_ADDRESS=0x... OCTOMIL_SETTLER_TOKEN=s402_... octomil mcp serve --x402
    """
    try:
        import fastapi as _fa  # noqa: F401
        import uvicorn as _uv  # noqa: F401
    except ImportError:
        click.echo("Error: fastapi and uvicorn are required for the HTTP server.", err=True)
        click.echo("Install with: pip install 'octomil-sdk[serve]'", err=True)
        raise SystemExit(1)

    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    # Convert USD threshold to USDC base units (6 decimals)
    threshold_base_units = int(float(x402_threshold) * 1_000_000)

    config = HTTPServerConfig(
        host=host,
        port=port,
        model=model,
        enable_x402=x402,
        x402_address=x402_address,
        x402_price=x402_price,
        x402_currency=x402_currency,
        x402_threshold=threshold_base_units,
        settler_url=settler_url,
        settler_token=settler_token,
    )
    app = create_http_app(config)

    click.echo(f"Starting Octomil agent server on {host}:{port}")
    click.echo(f"  Agent card: http://{host}:{port}/.well-known/agent-card.json")
    click.echo(f"  OpenAPI docs: http://{host}:{port}/docs")
    click.echo(f"  Health: http://{host}:{port}/health")
    if x402:
        click.echo("  x402 payment gating: enabled")
        click.echo(f"  settle402: {settler_url or 'not configured'}")
        if settler_token:
            click.echo(f"  settler token: {settler_token[:10]}...")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def register_cmd(cli: click.Group) -> None:
    """Register the mcp command group with the CLI."""
    cli.add_command(mcp)
