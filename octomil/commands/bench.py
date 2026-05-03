"""``octomil bench`` — runtime selection bench inspection + control (v0.5 PR D).

Exposes the on-disk cache from ``octomil.runtime.bench.cache`` as a
set of operator-facing CLI verbs. Read-only verbs (``list``, ``show``,
``status``) work without the ``OCTOMIL_RUNTIME_BENCH=experimental``
env-var gate; mutating verbs (``run``, ``reset``) work regardless of
the gate because the operator is asking for them explicitly.

Verbs (per ``strategy/runtime-selection-bench.md`` §6 "PR D scope"):

  * ``octomil bench list`` — list every cached entry in this device's
    cache root, grouped by ``(model, capability)``.
  * ``octomil bench show <model>`` — pretty-print every leaf JSON for
    a model.
  * ``octomil bench reset [<model>]`` — clear cache for one model or
    ``--all``.
  * ``octomil bench run <model> --capability <cap>`` — foreground bench
    for one ``(model, capability)``. v0.5 stub: refuses with a clear
    message until the SDK-level wiring PR plugs the engine factories
    into ``BenchScheduler.run_foreground``.
  * ``octomil bench status`` — show cache root, entry counts, env-var
    gate state.

Hard cutover: this CLI ships the v0.5 cache layout; v1's runtime core
takes over the writer role and this module is removed in the same
release. No deprecation aliases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from octomil import __version__ as _OCTOMIL_VERSION
from octomil.runtime.bench.cache import (
    CACHE_DIR_NAME,
    CacheStore,
    HardwareFingerprint,
    default_cache_root,
)

#: Runtime build tag MUST match the value the SDK writes with so the
#: CLI lands on the same on-disk hardware directory.
#: ``HardwareFingerprint.full_digest()`` hashes ``runtime_build_tag``
#: into the path component, so a CLI-specific tag would silently
#: split the cache namespace and the CLI would not see SDK-written
#: entries.
SDK_RUNTIME_BUILD_TAG: str = f"octomil-python:{_OCTOMIL_VERSION}"

# ---------------------------------------------------------------------------
# Group root
# ---------------------------------------------------------------------------


@click.group("bench")
def bench_cmd() -> None:
    """Runtime selection bench — cache inspection and one-shot runs.

    The bench picks the empirical winning ``(engine, provider, threads,
    quantization)`` config per ``(capability, model, device, dispatch
    shape)``. v0.5 ships the cache + harness + scheduler in Python; the
    v1 runtime core takes over the writer role on the same on-disk
    schema.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_store(
    cache_root: Optional[Path] = None,
    runtime_build_tag: str = SDK_RUNTIME_BUILD_TAG,
) -> CacheStore:
    """Construct a :class:`CacheStore` against the resolved cache root.

    Default ``runtime_build_tag`` is ``octomil-python:<sdk-version>``
    — matches what the SDK writes with, so the CLI lands on the same
    on-disk hardware directory. Tests may pass a custom tag for
    isolation."""
    root = cache_root if cache_root is not None else default_cache_root()
    hardware = HardwareFingerprint.detect(runtime_build_tag=runtime_build_tag)
    return CacheStore(cache_root=root, hardware=hardware)


def _print_json(payload: Any) -> None:
    """Stable, human-readable JSON dump."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def _model_dir_for(store: CacheStore, model_id: str) -> Path:
    """Return the on-disk directory for ``model_id`` under the
    current hardware fingerprint, regardless of whether it exists.
    Uses the same path-resolution helpers ``CacheStore`` uses
    internally so URL-encoding round-trips correctly."""
    # `clear_model` constructs a synthetic key for path resolution
    # only; we replicate that pattern here (private API call is
    # intentional — this CLI is the cache's owner).
    from octomil.runtime.bench.cache import _path_only_cache_key

    return store._model_dir(_path_only_cache_key(model_id))  # noqa: SLF001 — see docstring


def _list_leaf_files(model_dir: Path) -> list[Path]:
    """Every leaf JSON in a model directory, EXCLUDING ``index.json``
    and any ``.lock`` files. Sorted for stable output."""
    if not model_dir.is_dir():
        return []
    leaves = []
    for child in sorted(model_dir.iterdir()):
        if child.suffix != ".json" or child.name == "index.json":
            continue
        leaves.append(child)
    return leaves


def _read_json_or_none(path: Path) -> Optional[dict[str, Any]]:
    """Read + parse one JSON file. Returns None on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _entry_counts(store: CacheStore) -> dict[str, int]:
    """Walk every model dir, count leaves split by ``incomplete``."""
    total = 0
    incomplete = 0
    for model_id in store.list_models():
        for entry in store.list_cache_keys(model_id=model_id):
            total += 1
            if entry.get("incomplete", False):
                incomplete += 1
    return {
        "total": total,
        "committed": total - incomplete,
        "incomplete": incomplete,
    }


# ---------------------------------------------------------------------------
# `octomil bench list`
# ---------------------------------------------------------------------------


@bench_cmd.command("list")
@click.option(
    "--cache-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override cache root (default: per-device platform cache dir).",
)
@click.option(
    "--capability",
    type=str,
    default=None,
    help="Filter to one capability (e.g. tts, embeddings).",
)
@click.option(
    "--include-incomplete",
    is_flag=True,
    default=False,
    help="Include entries with incomplete=true (skipped by default).",
)
def list_cmd(cache_root: Optional[Path], capability: Optional[str], include_incomplete: bool) -> None:
    """List cached entries by ``(model, capability)``.

    Reads each model's ``index.json`` sidecar via
    :meth:`CacheStore.list_cache_keys` — fast even for caches with
    thousands of dispatch-shape variants."""
    store = _open_store(cache_root)
    rows = 0
    for model_id in store.list_models():
        for entry in store.list_cache_keys(model_id=model_id):
            entry_cap = entry.get("capability") or entry.get("cache_key", {}).get("capability", "?")
            if capability and entry_cap != capability:
                continue
            entry_incomplete = bool(entry.get("incomplete", False))
            if entry_incomplete and not include_incomplete:
                continue
            click.echo(
                f"{model_id}  cap={entry_cap}  "
                f"winner={entry.get('winner_summary', '<none>') or '<none>'}  "
                f"confidence={entry.get('confidence', '?')}  "
                f"incomplete={entry_incomplete}"
            )
            rows += 1
    if rows == 0:
        click.echo("No cached entries found.", err=True)
        # Exit cleanly even on empty — the CLI is read-only and a
        # zero-row response is a valid answer, not a failure.


# ---------------------------------------------------------------------------
# `octomil bench show`
# ---------------------------------------------------------------------------


@bench_cmd.command("show")
@click.argument("model", required=True)
@click.option(
    "--cache-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--capability",
    type=str,
    default=None,
    help="Filter to one capability (default: all).",
)
@click.option(
    "--max",
    "max_blocks",
    type=int,
    default=50,
    show_default=True,
    help="Maximum leaf JSONs to print (0 = unlimited).",
)
def show_cmd(model: str, cache_root: Optional[Path], capability: Optional[str], max_blocks: int) -> None:
    """Pretty-print every cache leaf for a model.

    Emits one JSON object per leaf, separated by ``---`` on its own
    line — keeps the output greppable while still parseable per
    block."""
    store = _open_store(cache_root)
    model_dir = _model_dir_for(store, model)
    leaves = _list_leaf_files(model_dir)
    printed = 0
    truncated = False
    for path in leaves:
        if max_blocks > 0 and printed >= max_blocks:
            truncated = True
            break
        payload = _read_json_or_none(path)
        if payload is None:
            click.echo(f"<unreadable: {path.name}>", err=True)
            continue
        if capability:
            ck = payload.get("cache_key") or {}
            if ck.get("capability") != capability:
                continue
        if printed > 0:
            click.echo("---")
        _print_json(payload)
        printed += 1
    if printed == 0:
        click.echo(f"No cache entries for model={model!r}.", err=True)
        sys.exit(1)
    if truncated:
        click.echo(
            f"... output truncated at --max={max_blocks}; pass --max=0 for all.",
            err=True,
        )


# ---------------------------------------------------------------------------
# `octomil bench reset`
# ---------------------------------------------------------------------------


@bench_cmd.command("reset")
@click.argument("model", required=False)
@click.option("--all", "all_models", is_flag=True, default=False, help="Clear cache for every model.")
@click.option(
    "--cache-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation prompt.",
)
def reset_cmd(model: Optional[str], all_models: bool, cache_root: Optional[Path], yes: bool) -> None:
    """Clear cache for one model or ``--all``.

    Confirmation prompt is interactive by default; pass ``--yes`` for
    scripted/CI use."""
    if not model and not all_models:
        raise click.UsageError("Pass a <model> arg or --all.")
    if model and all_models:
        raise click.UsageError("Pass <model> OR --all, not both.")

    if not yes:
        target = "ALL models" if all_models else f"model={model!r}"
        click.confirm(
            f"This deletes cached bench winners for {target}. Continue?",
            abort=True,
        )

    store = _open_store(cache_root)
    if all_models:
        # CacheStore.clear_all returns None; count via list_models
        # before+after for a useful CLI summary. Compute AFTER the
        # clear runs so a partial failure (disk full, perms) doesn't
        # silently report the original count.
        before = sum(len(store.list_cache_keys(model_id=m)) for m in store.list_models())
        try:
            store.clear_all()
        except Exception as exc:  # noqa: BLE001 — operator action; surface specifics
            after = sum(len(store.list_cache_keys(model_id=m)) for m in store.list_models())
            click.echo(
                f"Cleared {before - after} of {before} entries before failure: {exc!r}",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Cleared {before} cache entries across all models.")
    else:
        assert model is not None
        n = store.clear_model(model_id=model)
        click.echo(f"Cleared {n} cache entries for model={model!r}.")


# ---------------------------------------------------------------------------
# `octomil bench run`
# ---------------------------------------------------------------------------


@bench_cmd.command("run")
@click.argument("model", required=True)
@click.option(
    "--capability",
    type=str,
    default="tts",
    show_default=True,
    help="Capability to bench (v0.5 supports tts only).",
)
def run_cmd(model: str, capability: str) -> None:
    """Force a foreground bench cycle for one model.

    \b
    v0.5 stub. Exit codes:
      * 2  — usage error (e.g. --capability != tts).
      * 65 — feature not yet wired (data unavailable, EX_NOINPUT).
      * 0  — once the candidate-enumeration step lands.

    The CLI surface is shipped now so the contract is reviewable
    independently; the wiring step (a separate follow-on PR) plugs
    the engine factories into :meth:`BenchScheduler.run_foreground`.
    Flags that depend on the wiring (``--budget-s``,
    ``--allow-placeholder``) deliberately do NOT appear in this
    surface — they'll land alongside the implementation that
    actually honors them."""
    if capability != "tts":
        raise click.UsageError(f"v0.5 supports capability=tts only; got {capability!r}.")

    click.echo(
        f"octomil bench run {model} --capability {capability}: the candidate-enumeration "
        "wiring (SDK integration step) is not yet shipped. The harness + scheduler "
        "are ready (PR D wires the CLI); the next PR plugs the engine factories "
        "into BenchScheduler.run_foreground.",
        err=True,
    )
    # 65 = sysexits.h EX_NOINPUT — distinguishes "missing wiring" from
    # "usage error" (Click default 2). Scripted callers can branch on
    # the exit code.
    sys.exit(65)


# ---------------------------------------------------------------------------
# `octomil bench status`
# ---------------------------------------------------------------------------


@bench_cmd.command("status")
@click.option(
    "--cache-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def status_cmd(cache_root: Optional[Path]) -> None:
    """Show cache root + entry counts + env-var gate state."""
    from octomil.runtime.bench.scheduler import (
        ENV_ALLOW_PLACEHOLDER,
        ENV_BENCH_GATE,
        is_bench_enabled,
        is_placeholder_bypassed,
    )

    store = _open_store(cache_root)
    counts = _entry_counts(store)
    payload = {
        # Bumped lockstep with the JSON shape. Scripted callers should
        # reject newer schema_versions they don't recognize.
        "schema_version": 1,
        "cache_root": str(store.cache_root),
        "cache_dir_name": CACHE_DIR_NAME,
        "hardware": store.hardware.descriptor_dict(),
        "runtime_build_tag": store.hardware.runtime_build_tag,
        "entry_count_total": counts["total"],
        "entry_count_committed": counts["committed"],
        "entry_count_incomplete": counts["incomplete"],
        "env": {
            ENV_BENCH_GATE: "experimental" if is_bench_enabled() else "<off>",
            ENV_ALLOW_PLACEHOLDER: "1" if is_placeholder_bypassed() else "<off>",
        },
    }
    _print_json(payload)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register the bench command group on the main CLI."""
    cli.add_command(bench_cmd)
