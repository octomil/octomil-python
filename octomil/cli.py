"""
Octomil command-line interface.

Usage::

    octomil serve gemma3-1b --port 8080
    octomil deploy gemma3-1b --phone
    octomil dashboard
    octomil push phi-4-mini --version 1.0.0
    octomil pull sentiment-v1 --version 1.0.0 --format coreml
    octomil check model.pt
    octomil convert model.pt --target ios,android
    octomil status sentiment-v1
    octomil benchmark gemma3-1b
    octomil benchmark gemma3-1b --local
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

from octomil import __version__
from octomil.cli_helpers import print_welcome
from octomil.sectioned_group import SectionedGroup

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(cls=SectionedGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="octomil")
@click.pass_context
def main(ctx: click.Context) -> None:
    """\b
    \U0001f419 Octomil — on-device AI inference
    """
    if ctx.invoked_subcommand is None:
        print_welcome()


# ---------------------------------------------------------------------------
# Register command modules
# ---------------------------------------------------------------------------

from octomil.commands import (  # noqa: E402
    agent,
    benchmark,
    completions,
    deploy,
    doctor,
    enterprise,
    federation,
    inference,
    interactive,
    local,
    model_ops,
    prepare,
    serve,
    setup,
    warmup,
)

# Register inference commands first so they appear before serve in help output.
for _mod in [
    inference,
    prepare,
    warmup,
    serve,
    local,
    model_ops,
    deploy,
    benchmark,
    enterprise,
    federation,
    interactive,
    completions,
    setup,
    agent,
    doctor,
]:
    _mod.register(main)

from octomil.commands import mcp_cmd  # noqa: E402

mcp_cmd.register_cmd(main)
