"""Conformance pytest hook (Lane G PR3).

Provides a session-scoped ``conformance_cache`` fixture that:
    1. resolves the conformance artifact version from
       ``octomil-python/CONFORMANCE_PIN``,
    2. populates ``~/.cache/octomil-conformance/<version>/`` by invoking
       ``scripts/fetch_contracts_dev.py`` if the cache is missing,
    3. returns the cache path.

When the fetch is soft-skipped (no local checkout + no GitHub access),
the fixture returns ``None`` and tests SHOULD use the cache path's
absence as a clean ``pytest.skip`` signal — see
``test_conformance_collect.py`` for the canonical pattern. This is
deliberate: dev environments without network access still run the rest
of the test suite without spurious failures (Lane G PR3 spec: "Don't
fail pytest when fetch can't reach the artifact — soft skip").
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIN_PATH = REPO_ROOT / "CONFORMANCE_PIN"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_contracts_dev.py"
CACHE_ROOT = Path.home() / ".cache" / "octomil-conformance"


def _read_pin() -> str | None:
    if not PIN_PATH.is_file():
        return None
    pin = PIN_PATH.read_text(encoding="utf-8").strip()
    return pin or None


def _cache_dir(version: str) -> Path:
    return CACHE_ROOT / version


def _cache_populated(version: str) -> bool:
    """A cache is "populated" only if the fetch script's sentinel exists.
    Mirrors fetch_contracts_dev.py's ``.extracted-ok`` check; partial
    extractions don't count as populated."""
    return (_cache_dir(version) / ".extracted-ok").is_file()


@pytest.fixture(scope="session")
def conformance_cache() -> Optional[Path]:
    """Session-scoped fixture: cache path of the fetched conformance
    artifact. Returns ``None`` when the cache is unreachable
    (soft-skip).

    The fetch is invoked exactly once per test session. ``--force``
    re-fetch is left to the operator running the script directly; the
    fixture is read-mostly."""
    version = _read_pin()
    if not version:
        return None

    if _cache_populated(version):
        return _cache_dir(version)

    if not FETCH_SCRIPT.is_file():
        return None

    # Run the fetch script. We deliberately do NOT raise on a non-zero
    # exit — the script soft-skips on unreachable sources (returning 0
    # with no sentinel written). A hard failure is integrity-related
    # (sha mismatch / suspicious tar entry) and surfaces as exit != 0;
    # we still don't want to crash the whole pytest session because of
    # it — log and degrade to None so the rest of the suite runs.
    try:
        proc = subprocess.run(
            [sys.executable, str(FETCH_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env={**os.environ},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"[conformance] fetch_contracts_dev.py failed to launch: {exc!r}\n")
        return None

    if proc.returncode != 0:
        sys.stderr.write(
            f"[conformance] fetch_contracts_dev.py exited {proc.returncode}; "
            f"degrading to None.\n"
            f"[conformance] stderr: {proc.stderr[-1024:]!r}\n"
        )
        return None

    if _cache_populated(version):
        return _cache_dir(version)

    # Script soft-skipped (no source reachable). Honest signal: None.
    return None
