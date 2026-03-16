"""Persistent install ID for the Octomil SDK.

Generates a UUID on first SDK initialization and persists it to
``~/.octomil/install_id``. On subsequent inits, reads from the
persisted file. This provides a stable anonymous identifier for
telemetry without requiring user registration.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".octomil"
_DEFAULT_FILE = _DEFAULT_DIR / "install_id"

_cached_install_id: str | None = None


def get_install_id(*, install_dir: Path | None = None) -> str:
    """Return the persistent install ID, creating it if necessary.

    Args:
        install_dir: Override the directory for the install_id file.
            Defaults to ``~/.octomil``.  Useful for testing.

    Returns:
        A stable UUID string that persists across SDK sessions.
    """
    global _cached_install_id
    if _cached_install_id is not None:
        return _cached_install_id

    id_file = (install_dir or _DEFAULT_DIR) / "install_id"

    # Try to read existing
    try:
        if id_file.exists():
            stored = id_file.read_text().strip()
            if stored:
                _cached_install_id = stored
                return stored
    except Exception:
        logger.debug("Failed to read install_id from %s", id_file, exc_info=True)

    # Generate new
    new_id = uuid.uuid4().hex
    try:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(new_id + "\n")
    except Exception:
        logger.debug("Failed to persist install_id to %s", id_file, exc_info=True)

    _cached_install_id = new_id
    return new_id


def reset_cache() -> None:
    """Clear the in-memory cache. Primarily for testing."""
    global _cached_install_id
    _cached_install_id = None
