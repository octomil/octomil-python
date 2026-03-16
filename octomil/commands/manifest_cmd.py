"""CLI commands for ``octomil manifest``."""

from __future__ import annotations

from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Template for ``octomil manifest init``
# ---------------------------------------------------------------------------

_INIT_TEMPLATE = """\
# Octomil App Manifest
# Declare the models your app needs.
version: 1
models:
  - id: "smolvlm2-500m"
    capability: chat
    delivery: managed

  - id: "whisper-tiny"
    capability: transcription
    delivery: managed

  # - id: "smollm2-135m"
  #   capability: keyboard_prediction
  #   delivery: bundled
  #   bundled_path: "smollm2-135m.gguf"
"""

# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group()
def manifest() -> None:
    """Manage Octomil app manifests."""


@manifest.command("init")
@click.option(
    "-o",
    "--output",
    default="octomil.yaml",
    show_default=True,
    help="Output file path for the scaffold manifest.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the file if it already exists.",
)
def manifest_init(output: str, force: bool) -> None:
    """Scaffold a template octomil.yaml manifest."""
    dest = Path(output)
    if dest.exists() and not force:
        click.echo(click.style(f"File already exists: {dest}", fg="yellow"))
        click.echo("Use --force to overwrite.")
        raise SystemExit(1)

    dest.write_text(_INIT_TEMPLATE, encoding="utf-8")
    click.echo(click.style(f"Created {dest}", fg="green"))


@manifest.command("validate")
@click.argument("path", type=click.Path(exists=True))
def manifest_validate(path: str) -> None:
    """Validate an octomil.yaml manifest file."""
    from octomil.manifest_validator import validate_manifest_file

    errors = validate_manifest_file(Path(path))

    if not errors:
        click.echo(click.style("Manifest is valid.", fg="green"))
        return

    click.echo(click.style(f"Found {len(errors)} error(s):", fg="red"))
    for err in errors:
        click.echo(f"  - {err}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register the manifest command group with the top-level CLI."""
    cli.add_command(manifest)
