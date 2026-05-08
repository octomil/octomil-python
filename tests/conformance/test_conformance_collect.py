"""Conformance collector (Lane G PR3).

For each non-cross-cutting capability YAML in the fetched conformance
cache, run the bundled ``scripts/generate_conformance.py --target python``
and execute the generated module in-process. Cross-cutting YAMLs
(model_lifecycle / error_mapping / event_sequence) are exercised
separately by their own stubs in PR4 — this collector skips them.

Generated Python sources are written to ``tmp_path`` (pytest's
per-test temporary directory) and are NEVER committed.

Soft-skip policy: when ``conformance_cache`` is ``None`` (fetch
unreachable, no local checkout), the collector marks itself skipped
with a clear reason. Lane G PR3 spec: "Don't fail pytest when fetch
can't reach the artifact — soft skip."
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Optional

import pytest
import yaml  # type: ignore[import-untyped]

# Cross-cutting YAMLs encode declarative tables, not lifecycle drivers.
# Their executable form lives in PR4 (error-code introspection,
# event-taxonomy assertions). The generator emits stubs for them; we
# skip them in the per-capability collector so collection failures are
# scoped to real capabilities.
_CROSS_CUTTING_YAML_NAMES: frozenset[str] = frozenset(
    {
        "model_lifecycle.yaml",
        "error_mapping.yaml",
        "event_sequence.yaml",
    }
)


def _capability_yamls(cache_dir: Path) -> list[Path]:
    """Return per-capability YAMLs from the fetched cache, sorted by
    name. Cross-cutting and legacy YAMLs are excluded so the collector
    only drives real capabilities."""
    conf_dir = cache_dir / "conformance"
    if not conf_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(conf_dir.glob("*.yaml")):
        if p.name in _CROSS_CUTTING_YAML_NAMES:
            continue
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and "capability" in doc:
            out.append(p)
    return out


def _safe_module_name(yaml_name: str) -> str:
    """Generated module name. ``audio.vad.yaml`` → ``test_conf_audio_vad``.
    Matches the spec's ``test_conf_<capability>.py`` convention."""
    stem = yaml_name.removesuffix(".yaml")
    return "test_conf_" + stem.replace(".", "_")


def _generate_one(yaml_path: Path, cache_dir: Path, out_dir: Path) -> Path:
    """Drive the bundled generator. Returns the path to the generated
    Python source."""
    generator = cache_dir / "scripts" / "generate_conformance.py"
    if not generator.is_file():
        pytest.skip(f"bundled generator not found at {generator} " f"(conformance cache layout unexpected)")
    out_path = out_dir / (_safe_module_name(yaml_path.name) + ".py")
    proc = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--capability",
            str(yaml_path),
            "--target",
            "python",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"generator failed for {yaml_path.name}: " f"rc={proc.returncode} stderr={proc.stderr[-512:]!r}")
    if not out_path.is_file():
        pytest.fail(f"generator returned 0 but {out_path} was not written")
    return out_path


def _load_module(path: Path, name: str) -> ModuleType:
    """Import a generated module by file path. Each test gets its own
    module name so pytest doesn't conflate state across capabilities."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"could not build module spec for {path}")
    # Narrow for mypy: pytest.fail raises NoReturn so spec is non-None below.
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.fail(f"loading generated module {name} raised: {exc!r}")
    return module


def _iter_module_tests(module: ModuleType) -> Iterator[tuple[str, Callable[[], None]]]:
    for attr in sorted(dir(module)):
        if not attr.startswith("test_"):
            continue
        fn = getattr(module, attr)
        if callable(fn):
            yield attr, fn


def test_conformance_collector_runs(conformance_cache: Optional[Path], tmp_path: Path) -> None:
    """Top-level collector entry: generate + run each capability's
    Python conformance test in-process. Soft-skips when the cache is
    unreachable (Lane G PR3 spec)."""
    if conformance_cache is None:
        pytest.skip(
            "conformance cache not populated (no local octomil-contracts "
            "checkout + GitHub fetch unavailable). Soft-skip per Lane G "
            "PR3 spec — run scripts/fetch_contracts_dev.py with a "
            "valid GH_TOKEN, or build conformance/ in a sibling "
            "octomil-contracts checkout, to exercise this rail."
        )
    # Narrow for mypy: pytest.skip raises so conformance_cache is non-None below.
    assert conformance_cache is not None

    yaml_paths = _capability_yamls(conformance_cache)
    if not yaml_paths:
        pytest.skip(f"no capability YAMLs found under {conformance_cache}/conformance/")

    out_dir = tmp_path / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    skips: list[str] = []
    passes: list[str] = []

    for yaml_path in yaml_paths:
        gen_path = _generate_one(yaml_path, conformance_cache, out_dir)
        mod_name = _safe_module_name(yaml_path.name)
        module = _load_module(gen_path, mod_name)

        for test_name, fn in _iter_module_tests(module):
            qualified = f"{yaml_path.name}::{test_name}"
            try:
                fn()
            except pytest.skip.Exception as exc:
                skips.append(f"{qualified} [skip: {exc}]")
            except AssertionError as exc:
                failures.append(f"{qualified}: AssertionError {exc!r}")
            except Exception as exc:  # noqa: BLE001
                # Any other exception in a generated test is a hard
                # failure — generated tests should only raise pytest's
                # skip exception or AssertionError.
                failures.append(f"{qualified}: {type(exc).__name__} {exc!r}")
            else:
                passes.append(qualified)

    sys.stderr.write(
        f"\n[conformance] capabilities driven: {len(yaml_paths)}\n"
        f"[conformance] tests passed: {len(passes)}\n"
        f"[conformance] tests skipped: {len(skips)}\n"
        f"[conformance] tests failed: {len(failures)}\n"
    )
    for s in skips:
        sys.stderr.write(f"[conformance] SKIP {s}\n")

    if failures:
        joined = "\n  ".join(failures)
        pytest.fail(f"{len(failures)} generated conformance test(s) failed:\n  " f"{joined}")
