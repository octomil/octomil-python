"""
Octomil command-line interface.

Usage::

    octomil serve gemma-1b --port 8080
    octomil deploy gemma-1b --phone
    octomil dashboard
    octomil push phi-4-mini --version 1.0.0
    octomil pull sentiment-v1 --version 1.0.0 --format coreml
    octomil check model.pt
    octomil convert model.pt --target ios,android
    octomil status sentiment-v1
    octomil benchmark gemma-1b
    octomil benchmark gemma-1b --local
    octomil login
    octomil init "Acme Corp" --compliance hipaa --region us
    octomil team add alice@acme.com --role admin
    octomil team list
    octomil team set-policy --require-mfa --session-hours 8
    octomil keys create deploy-key --scope devices:write --scope models:read
    octomil keys list
    octomil keys revoke <key-id>
    octomil scan ./MyApp
    octomil scan ./MyApp --format json --platform ios
"""

from __future__ import annotations

import click

from octomil.cli_helpers import WELCOME_MESSAGE


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.version_option(version="2.1.8", prog_name="octomil")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Octomil — serve, deploy, and observe ML models on edge devices."""
    if ctx.invoked_subcommand is None:
        click.echo(WELCOME_MESSAGE)


# ---------------------------------------------------------------------------
# Register command modules
# ---------------------------------------------------------------------------

from octomil.commands import (  # noqa: E402
    serve,
    model_ops,
    deploy,
    benchmark,
    enterprise,
    federation,
    interactive,
)

for _mod in [serve, model_ops, deploy, benchmark, enterprise, federation, interactive]:
    _mod.register(main)
