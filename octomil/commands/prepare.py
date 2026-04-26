"""``octomil prepare`` — pre-warm an on-device model artifact.

Wraps :meth:`octomil.execution.kernel.ExecutionKernel.prepare` so users
can download an artifact once, explicitly, instead of paying first-call
latency or hitting the ``prepare_policy='explicit_only'`` actionable
error.

Today this command supports the capabilities whose dispatch paths
actually consume the prepared ``model_dir``:

- ``tts``           — ``SherpaTtsEngine`` loads from the prepared dir.
- ``transcription`` — ``_WhisperBackend`` loads the prepared
  ``<dir>/artifact`` sentinel (or the matching ``.bin`` / ``.gguf`` /
  ``.ggml``) instead of triggering pywhispercpp's HuggingFace download.
- ``chat`` / ``responses`` — ``MLXBackend`` and ``LlamaCppBackend``
  accept ``model_dir`` and load from the prepared directory
  (mlx_lm.load reads the path like an HF repo id; llama_cpp.Llama
  opens the ``<dir>/artifact`` sentinel by GGUF magic bytes).

Embedding will be added once its local backend learns to consume the
prepared directory; until then, calling prepare for it would download
bytes the next inference ignores. The CLI mirrors the kernel's
``_PREPAREABLE_CAPABILITIES`` set via the ``--capability`` choice list.

Usage::

    octomil prepare @app/eternum/tts
    octomil prepare @app/notes/transcription --capability transcription
    octomil prepare gemma3-1b --capability chat --policy local_first
    octomil prepare kokoro-en-v0_19 --capability tts --policy local_first
"""

from __future__ import annotations

import sys

import click


@click.command("prepare")
@click.argument("model", required=True)
@click.option(
    "--capability",
    default="tts",
    type=click.Choice(["tts", "transcription", "chat", "responses"]),
    show_default=True,
    help=(
        "Which capability's planner candidate to prepare. Today: 'tts' "
        "(SherpaTtsEngine), 'transcription' (whisper.cpp loads the prepared file "
        "instead of triggering its own download), and 'chat' / 'responses' "
        "(mlx-lm / llama.cpp load the prepared directory). Embedding is added "
        "once its backend consumes the prepared dir."
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
def prepare_cmd(model: str, capability: str, policy: str | None, app: str | None, cache_dir: str | None) -> None:
    """Pre-warm the on-disk artifact for ``MODEL`` so first inference is instant."""
    # Lazy imports keep `octomil --help` fast for users who never prepare.
    from pathlib import Path

    from octomil.errors import OctomilError
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    pm = None
    if cache_dir is not None:
        pm = PrepareManager(cache_dir=Path(cache_dir))
    kernel = ExecutionKernel(prepare_manager=pm)

    try:
        outcome = kernel.prepare(model=model, capability=capability, policy=policy, app=app)
    except OctomilError as exc:
        # Render the manager's actionable message verbatim. No traceback.
        click.echo(f"prepare failed: {exc}", err=True)
        sys.exit(1)

    state = "cached" if outcome.cached else "downloaded"
    click.echo(f"{state}: {outcome.artifact_id} -> {outcome.artifact_dir}")
    if outcome.files:
        for rel, path in outcome.files.items():
            display = rel or "(single file)"
            click.echo(f"  {display}: {path}")


def register(cli: click.Group) -> None:
    """Register the prepare command on the main CLI."""
    cli.add_command(prepare_cmd)
