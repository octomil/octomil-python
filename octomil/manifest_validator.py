"""Manifest validation utilities."""

from __future__ import annotations

from pathlib import Path

from octomil.manifest import AppManifest


def validate_manifest(manifest: AppManifest) -> list[str]:
    """Validate an :class:`AppManifest` and return a list of errors.

    An empty list means the manifest is valid.
    """
    return manifest.validate()


def validate_manifest_file(path: Path | str) -> list[str]:
    """Load a YAML manifest from *path* and validate it.

    Returns a list of validation error strings.  An empty list means
    the file is valid.  Structural parse errors (bad YAML, missing
    required keys) are returned as single-element error lists.
    """
    path = Path(path)
    if not path.exists():
        return [f"File not found: {path}"]

    try:
        manifest = AppManifest.from_yaml(path)
    except ImportError as exc:
        return [str(exc)]
    except Exception as exc:
        return [f"Failed to parse {path}: {exc}"]

    return validate_manifest(manifest)
