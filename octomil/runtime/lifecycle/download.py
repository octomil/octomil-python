"""Download manager — fetch model artifacts with progress, resume, and verification.

Supports:
- Progress bar (tqdm if available, suppressed in non-TTY)
- HTTP range requests for resume
- SHA-256 digest verification after download
- Async-compatible via httpx
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.lifecycle.artifact_cache import ArtifactCache, _compute_sha256
from octomil.runtime.lifecycle.file_lock import FileLock

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0  # 10 minutes
_DOWNLOAD_CHUNK_SIZE = 65536  # 64KB


class DownloadManager:
    """Manages artifact downloads with locking, resume, progress, and verification.

    Usage::

        mgr = DownloadManager(cache=ArtifactCache())
        path = mgr.download(
            artifact_id="gemma-2b-q4",
            url="https://models.octomil.com/gemma-2b-q4.gguf",
            expected_digest="sha256:abc123...",
        )
    """

    def __init__(
        self,
        cache: ArtifactCache | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        show_progress: bool | None = None,
    ) -> None:
        self._cache = cache or ArtifactCache()
        self._timeout = timeout
        # Auto-detect TTY unless explicitly set
        if show_progress is None:
            self._show_progress = _is_tty()
        else:
            self._show_progress = show_progress

    @property
    def cache(self) -> ArtifactCache:
        return self._cache

    def download(
        self,
        artifact_id: str,
        url: str,
        expected_digest: str,
        filename: str | None = None,
    ) -> Path:
        """Download an artifact, using cache and file locks.

        1. Check cache first — return immediately on hit.
        2. Acquire file lock to prevent concurrent downloads.
        3. Re-check cache (another process may have finished).
        4. Download with optional resume support.
        5. Verify digest.
        6. Store in cache.

        Raises OctomilError on failure with actionable messages.
        """
        # 1. Cache hit check
        cached = self._cache.get(artifact_id, expected_digest)
        if cached is not None:
            logger.info("Artifact %s found in cache: %s", artifact_id, cached)
            return cached

        # 2. Acquire lock
        lock = FileLock(artifact_id, lock_dir=self._cache.cache_dir / ".locks")
        with lock:
            # 3. Re-check cache (another process may have completed download)
            cached = self._cache.get(artifact_id, expected_digest)
            if cached is not None:
                logger.info("Artifact %s appeared in cache after lock: %s", artifact_id, cached)
                return cached

            # 4. Download
            dest_name = filename or _filename_from_url(url, artifact_id)
            tmp_dir = self._cache.cache_dir / ".tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                dir=str(tmp_dir),
                prefix=f"{artifact_id.replace('/', '_')}_",
                suffix=f"_{dest_name}",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                self._download_file(url, tmp_path)
            except Exception as exc:
                # Clean up partial download
                tmp_path.unlink(missing_ok=True)
                raise OctomilError(
                    code=OctomilErrorCode.DOWNLOAD_FAILED,
                    message=f"Failed to download artifact '{artifact_id}' from {url}: {exc}",
                    cause=exc,
                ) from exc

            # 5. Verify digest
            actual_digest = _compute_sha256(tmp_path)
            expected_hex = expected_digest[7:] if expected_digest.startswith("sha256:") else expected_digest
            if actual_digest != expected_hex.lower():
                tmp_path.unlink(missing_ok=True)
                raise OctomilError(
                    code=OctomilErrorCode.CHECKSUM_MISMATCH,
                    message=(
                        f"Digest mismatch for artifact '{artifact_id}'. "
                        f"Expected: {expected_hex}, got: {actual_digest}. "
                        f"The file may be corrupt. Try downloading again."
                    ),
                )

            # 6. Store in cache
            # Rename tmp file to final filename for cleaner cache paths
            final_tmp = tmp_path.parent / dest_name
            if final_tmp != tmp_path:
                tmp_path.rename(final_tmp)
                tmp_path = final_tmp

            result = self._cache.put(artifact_id, expected_digest, tmp_path)

            # Clean up tmp file (put() copies it)
            if tmp_path.exists() and tmp_path != result:
                tmp_path.unlink(missing_ok=True)

            logger.info("Artifact %s downloaded and cached: %s", artifact_id, result)
            return result

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file with optional resume and progress bar."""
        headers: dict[str, str] = {}
        existing_size = 0

        # Check for partial download (resume support)
        if dest.exists():
            existing_size = dest.stat().st_size
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                logger.debug("Resuming download from byte %d", existing_size)

        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as response:
                # Handle resume response
                if response.status_code == 416:
                    # Range not satisfiable — file is complete or server doesn't support resume
                    logger.debug("Range not satisfiable — re-downloading from scratch")
                    existing_size = 0
                    # Retry without range header
                    response.close()
                    with client.stream("GET", url) as retry_response:
                        retry_response.raise_for_status()
                        self._write_stream(retry_response, dest, 0)
                    return

                if response.status_code not in (200, 206):
                    response.raise_for_status()

                resuming = response.status_code == 206
                total_size = _get_content_length(response, existing_size if resuming else 0)

                self._write_stream(response, dest, existing_size if resuming else 0, total_size)

    def _write_stream(
        self,
        response: Any,
        dest: Path,
        offset: int,
        total_size: int | None = None,
    ) -> None:
        """Write response stream to file with optional progress bar."""
        mode = "ab" if offset > 0 else "wb"
        progress_bar = None

        try:
            if self._show_progress and total_size and total_size > 0:
                progress_bar = _make_progress_bar(total_size, offset, dest.name)

            with open(dest, mode) as f:
                for chunk in response.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    if progress_bar is not None:
                        progress_bar.update(len(chunk))
        finally:
            if progress_bar is not None:
                progress_bar.close()


def _filename_from_url(url: str, fallback: str) -> str:
    """Extract filename from URL, falling back to artifact_id."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if "/" in path:
        name = path.rsplit("/", 1)[1]
        if name:
            return name
    return fallback.replace("/", "_")


def _get_content_length(response: Any, offset: int) -> int | None:
    """Extract total file size from response headers."""
    cl = response.headers.get("content-length")
    if cl:
        try:
            return int(cl) + offset
        except ValueError:
            pass
    return None


def _is_tty() -> bool:
    """Check if stdout is a TTY (interactive terminal)."""
    try:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        return False


def _make_progress_bar(total: int, initial: int, desc: str) -> Any:
    """Create a tqdm progress bar if available, else return a no-op."""
    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        )
    except ImportError:
        # No tqdm — return a no-op object
        return _NoOpProgressBar()


class _NoOpProgressBar:
    """Stub progress bar when tqdm is not available."""

    def update(self, n: int) -> None:
        pass

    def close(self) -> None:
        pass
