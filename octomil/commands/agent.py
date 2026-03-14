"""Run an AI agent with on-device inference and server-side tools."""

from __future__ import annotations

import asyncio
import os
import sys

import click

from octomil.cli_helpers import cli_header, cli_kv, cli_section, cli_success


def register(cli: click.Group) -> None:
    cli.add_command(agent)


@click.command()
@click.argument("agent_type")
@click.argument("query")
@click.option(
    "--model",
    "-m",
    default="qwen2.5:7b",
    help="Model to run on-device.",
)
@click.option(
    "--api-url",
    envvar="OCTOMIL_API_URL",
    default=None,
    help="Octomil API URL. Defaults to OCTOMIL_API_URL env var.",
)
@click.option(
    "--api-key",
    envvar="OCTOMIL_API_KEY",
    default=None,
    help="Octomil API key. Defaults to OCTOMIL_API_KEY env var.",
)
def agent(
    agent_type: str,
    query: str,
    model: str,
    api_url: str | None,
    api_key: str | None,
) -> None:
    """Run an AI agent with on-device inference.

    The model runs locally on your device. The server provides
    tool execution (database queries, deployment actions, etc.).

    \b
    Examples
    ─────────────────────────────────────────────
    octomil agent deployment-advisor "Deploy phi-mini to iOS staging"
    octomil agent deployment-advisor "Is model X ready?" --model phi-4
    """
    base_url = api_url or os.environ.get("OCTOMIL_API_BASE") or "https://api.octomil.com"
    token = api_key
    if not token:
        click.echo(
            click.style("Error: OCTOMIL_API_KEY required. Set it or pass --api-key.", fg="red"),
            err=True,
        )
        sys.exit(1)

    # Normalize agent_type: allow hyphens in CLI, convert to underscores
    agent_type = agent_type.replace("-", "_")

    cli_header(f"Agent — {agent_type}")
    cli_kv("Model", f"{model} (on-device)")
    cli_kv("Query", query)
    click.echo()

    click.echo(click.style("  Running agent loop...", dim=True))

    result = asyncio.run(_run(base_url, token, agent_type, query, model))

    click.echo()
    cli_section("Result")
    click.echo()
    click.echo(f"  {result.summary}")

    if result.confidence is not None:
        click.echo()
        cli_kv("Confidence", f"{result.confidence:.0%}")

    if result.evidence:
        click.echo()
        cli_section("Evidence")
        for item in result.evidence:
            click.echo(f"    - {item}")

    if result.next_steps:
        click.echo()
        cli_section("Next Steps")
        for i, step in enumerate(result.next_steps, 1):
            click.echo(f"    {i}. {step}")

    click.echo()
    cli_success(f"Session: {result.session_id}")


async def _run(base_url: str, token: str, agent_type: str, query: str, model: str):
    from octomil.agents.session import AgentSession

    session = AgentSession(base_url=base_url, auth_token=token)
    return await session.run(agent_type, query, model=model)
