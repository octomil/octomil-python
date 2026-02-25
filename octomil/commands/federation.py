"""Federation management commands.

Extracted from cli.py – manages cross-org federations (create, invite,
join, list, show, members, share).
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import click

from octomil.cli_helpers import _get_client


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _resolve_federation_id(client, name_or_id: str) -> str:
    """Resolve a federation name or ID to an ID.

    First tries listing federations filtered by name.  If no match,
    assumes ``name_or_id`` is already an ID and returns it as-is.
    """
    org_id = client._org_id
    results = client._api.get(
        "/federations",
        params={"org_id": org_id, "name": name_or_id},
    )
    if results:
        return results[0]["id"]
    return name_or_id


# ---------------------------------------------------------------------------
# Click group & subcommands
# ---------------------------------------------------------------------------


@click.group()
def federation() -> None:
    """Manage cross-org federations."""


@federation.command("create")
@click.argument("name")
@click.option("--description", "-d", default=None, help="Federation description.")
def federation_create(name: str, description: Optional[str]) -> None:
    """Create a new federation.

    Example:

        octomil federation create healthcare-consortium --description "Cross-hospital FL"
    """
    client = _get_client()
    payload: dict[str, Any] = {
        "name": name,
        "org_id": client._org_id,
    }
    if description:
        payload["description"] = description

    result = client._api.post("/federations", payload)
    fed_id = result.get("id", "unknown")
    click.echo(f"Federation created: {name}")
    click.echo(f"ID: {fed_id}")


@federation.command("invite")
@click.argument("federation_name")
@click.option(
    "--org",
    "org_ids",
    multiple=True,
    required=True,
    help="Org ID to invite (repeatable).",
)
def federation_invite(federation_name: str, org_ids: tuple[str, ...]) -> None:
    """Invite organisations to a federation.

    Example:

        octomil federation invite healthcare-consortium --org org_abc --org org_def
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    result = client._api.post(
        f"/federations/{fed_id}/invite",
        {"org_ids": list(org_ids)},
    )
    invited = result if isinstance(result, list) else result.get("invited", [])
    click.echo(f"Invited {len(invited)} org(s) to federation {federation_name}")


@federation.command("join")
@click.argument("federation_name")
def federation_join(federation_name: str) -> None:
    """Accept an invitation and join a federation.

    Example:

        octomil federation join healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    client._api.post(
        f"/federations/{fed_id}/join",
        {"org_id": client._org_id},
    )
    click.echo(f"Joined federation: {federation_name}")


@federation.command("list")
def federation_list() -> None:
    """List federations visible to your organisation.

    Example:

        octomil federation list
    """
    client = _get_client()
    results = client._api.get(
        "/federations",
        params={"org_id": client._org_id},
    )

    if not results:
        click.echo("No federations found.")
        return

    click.echo(f"{'Name':<30s} {'ID':<40s} {'Description':<30s}")
    click.echo("-" * 100)
    for fed in results:
        name = fed.get("name", "")
        fed_id = fed.get("id", "")
        desc = fed.get("description", "") or ""
        click.echo(f"{name:<30s} {fed_id:<40s} {desc:<30s}")


@federation.command("show")
@click.argument("federation_name")
def federation_show(federation_name: str) -> None:
    """Show details of a federation.

    Example:

        octomil federation show healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    data = client._api.get(f"/federations/{fed_id}")

    click.echo(f"Name:        {data.get('name', '')}")
    click.echo(f"ID:          {data.get('id', '')}")
    click.echo(f"Description: {data.get('description', '') or '-'}")
    click.echo(f"Created:     {data.get('created_at', '')}")
    click.echo(f"Org ID:      {data.get('org_id', '')}")


@federation.command("members")
@click.argument("federation_name")
def federation_members(federation_name: str) -> None:
    """List members of a federation.

    Example:

        octomil federation members healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)
    members = client._api.get(f"/federations/{fed_id}/members")

    if not members:
        click.echo("No members found.")
        return

    click.echo(f"{'Org ID':<40s} {'Status':<15s} {'Joined':<25s}")
    click.echo("-" * 80)
    for m in members:
        org = m.get("org_id", "")
        status = m.get("status", "")
        joined = m.get("joined_at", "") or "-"
        click.echo(f"{org:<40s} {status:<15s} {joined:<25s}")


@federation.command("share")
@click.argument("model_name")
@click.option(
    "--federation",
    "-f",
    "federation_name",
    required=True,
    help="Federation name or ID to share the model with.",
)
def federation_share(model_name: str, federation_name: str) -> None:
    """Share a model with a federation for cross-org training.

    The model must belong to the federation owner's organization.
    Once shared, all active federation members can contribute training
    updates and deploy the model.

    Example:

        octomil federation share radiology-v1 --federation healthcare-consortium
    """
    client = _get_client()
    fed_id = _resolve_federation_id(client, federation_name)

    # Resolve model name to model ID via the registry
    models = client._api.get("/models", params={"org_id": client._org_id})
    model_id = None
    if isinstance(models, list):
        for m in models:
            if m.get("name") == model_name:
                model_id = m.get("id")
                break
    if not model_id:
        click.echo(
            f"Model '{model_name}' not found in your organization. "
            "Push it first with `octomil push`.",
            err=True,
        )
        sys.exit(1)

    client._api.post(
        f"/federations/{fed_id}/models",
        {"model_id": model_id},
    )
    click.echo(f"Model '{model_name}' shared with federation '{federation_name}'")
    click.echo(
        "Active federation members can now contribute training updates "
        "and deploy this model."
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    cli.add_command(federation)
