"""Cache status tracking for planner artifacts.

Does NOT download or manage model files — that is a future follow-up.
This module provides cache status tracking (hit/miss) for RouteMetadata
based on whether a file matching the artifact digest already exists on disk.
The actual download implementation depends on the artifact URI scheme and
is not yet implemented.
"""

# TODO(managed-lifecycle): Implement actual artifact download using planner
# artifact_candidates + preferred_engines. Current implementation only tracks
# cache status — it does not download, verify, or prepare model files.

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "octomil" / "artifacts"


class ArtifactCache:
    """Cache status tracking for planner artifacts.

    Checks whether a file matching a given digest exists on disk.
    Does NOT download or manage model files — that is a future follow-up.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(os.environ.get("OCTOMIL_ARTIFACT_CACHE", str(_DEFAULT_CACHE_DIR)))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    def artifact_path(self, digest: str) -> Path:
        """Return the cached path for an artifact by digest."""
        return self._cache_dir / digest

    def is_cached(self, digest: str) -> bool:
        """Check if an artifact is already cached."""
        return self.artifact_path(digest).exists()

    def acquire_lock(self, digest: str):
        """Acquire a file lock for downloading an artifact.

        Returns an open file descriptor with an exclusive advisory lock.
        The caller is responsible for closing the fd when done (which
        releases the lock).  Blocks until the lock is acquired.
        """
        lock_path = self._cache_dir / f"{digest}.lock"
        fd = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def cache_status(self, digest: str | None) -> str:
        """Return cache status for route metadata.

        Returns one of ``"hit"``, ``"miss"``, or ``"not_applicable"``.
        """
        if digest is None:
            return "not_applicable"
        if self.is_cached(digest):
            return "hit"
        return "miss"


def _warn_if_large_download_non_tty(artifact_size_bytes: int | None) -> None:
    """Emit a warning for large artifacts in non-interactive environments.

    For artifacts >100 MB, emits a ``UserWarning`` when stdout is not a TTY
    so that CI/script environments can pre-cache artifacts via
    ``OCTOMIL_ARTIFACT_CACHE``.  This is advisory only — it does not block
    execution.
    """
    if artifact_size_bytes and artifact_size_bytes > 100_000_000:  # 100 MB
        if not sys.stdout.isatty():
            import warnings

            warnings.warn(
                f"About to download {artifact_size_bytes / 1e9:.1f}GB artifact in non-interactive mode. "
                "Set OCTOMIL_ARTIFACT_CACHE to pre-cache artifacts.",
                stacklevel=2,
            )
