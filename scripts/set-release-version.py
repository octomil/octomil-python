#!/usr/bin/env python3
"""Set the package version in release workspaces."""

from __future__ import annotations

import pathlib
import re
import sys

VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def _replace_once(path: pathlib.Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"expected exactly one version field in {path}")
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: set-release-version.py <version>")

    version = sys.argv[1].strip()
    if version.startswith("v"):
        version = version[1:]
    if VERSION_RE.fullmatch(version) is None:
        raise SystemExit(f"expected stable SemVer version, got {sys.argv[1]!r}")

    root = pathlib.Path(__file__).resolve().parents[1]
    _replace_once(
        root / "pyproject.toml",
        r'^version = "[^"]+"$',
        f'version = "{version}"',
    )
    _replace_once(
        root / "octomil" / "__init__.py",
        r'^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
    )
    print(f"Set release version to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
