"""``octomil warmup`` — prepare AND load an on-device model into memory.

Strict superset of ``octomil prepare``. After ``warmup`` returns, the
artifact bytes are on disk *and* the local engine has constructed +
``load_model``'d a backend instance, so the next inference dispatch in
the same process can reuse it without paying cold-start latency.

Today this command supports the same capabilities as ``prepare`` —
``tts`` and ``transcription`` — because those are the cells whose
dispatch path actually threads ``model_dir`` into the backend AND
checks the kernel's warmup cache before constructing a fresh one.

Note that warmup runs a *one-shot* CLI process: the cached backend
lives on the kernel instance for that process only. To get
across-process warmup savings, run warmup from inside the same
long-lived host that will service inference (e.g. ``octomil serve``,
or your own embedded :class:`Octomil` client).

Usage::

    octomil warmup @app/eternum/tts
    octomil warmup @app/notes/transcription --capability transcription
    octomil warmup kokoro-en-v0_19 --capability tts --policy local_first
"""

from __future__ import annotations

import sys

import click


@click.command("warmup")
@click.argument("model", required=True)
@click.option(
    "--capability",
    default="tts",
    type=click.Choice(["tts", "transcription"]),
    show_default=True,
    help=(
        "Which capability's planner candidate to warm. Mirrors "
        "``octomil prepare`` plus the load + cache step. Today: 'tts' and "
        "'transcription'. Chat / responses and embedding are added once their "
        "backends consume the prepared dir."
    ),
)
@click.option(
    "--policy",
    default=None,
    help="Override the routing policy preset (e.g. local_only, local_first).",
)
@click.option(
    "--app",
    default=None,
    help="Optional explicit app slug for @app/<slug>/<capability> resolution.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    default=None,
    help="Override the artifact cache directory (defaults to OCTOMIL_CACHE_DIR or ~/.cache/octomil).",
)
def warmup_cmd(model: str, capability: str, policy: str | None, app: str | None, cache_dir: str | None) -> None:
    """Prepare + load ``MODEL`` so first inference is instant in this process."""
    from pathlib import Path

    from octomil.errors import OctomilError
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    pm = None
    if cache_dir is not None:
        pm = PrepareManager(cache_dir=Path(cache_dir))
    kernel = ExecutionKernel(prepare_manager=pm)

    try:
        outcome = kernel.warmup(model=model, capability=capability, policy=policy, app=app)
    except OctomilError as exc:
        click.echo(f"warmup failed: {exc}", err=True)
        sys.exit(1)

    prepare_state = "cached" if outcome.prepare_outcome.cached else "downloaded"
    load_state = "loaded" if outcome.backend_loaded else "load_skipped"
    click.echo(
        f"{prepare_state}+{load_state}: {outcome.prepare_outcome.artifact_id} "
        f"-> {outcome.prepare_outcome.artifact_dir} ({outcome.latency_ms:.0f} ms)"
    )
    if not outcome.backend_loaded:
        # Prepare succeeded but the backend constructor refused — the
        # bytes are on disk, the next inference call still pays load
        # latency. Tell the user explicitly so this isn't a silent
        # half-success.
        click.echo(
            "  note: backend was not loaded (engine missing or runtime imports failed). "
            "Inference will fall through the cold path.",
            err=True,
        )


def register(cli: click.Group) -> None:
    """Register the warmup command on the main CLI."""
    cli.add_command(warmup_cmd)
