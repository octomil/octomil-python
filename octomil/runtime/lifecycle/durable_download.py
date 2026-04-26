"""Durable, resumable, multi-URL artifact downloader.

Implements the prepare-lifecycle download contract:
- ``required_files``: list of files comprising the artifact (relative paths)
- ``download_urls``: ordered fallback list of ``ArtifactDownloadEndpoint``
- ``manifest_uri``: optional per-file digest manifest

Persists per-file progress (bytes-written, chosen-endpoint, last-error) in a
SQLite journal so a download interrupted by process exit, network drop, or
endpoint expiry resumes from the last committed offset on the next call.

Single-file artifacts are supported by passing ``required_files=[""]`` (the
empty relative path), in which case each endpoint URL is treated as the file
URL directly. Multi-file artifacts are fetched as ``{endpoint.url}/{file}``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import httpx

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.runtime.lifecycle.file_lock import FileLock

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 16  # 64 KiB
_DEFAULT_TIMEOUT = 600.0
_PROGRESS_FLUSH_BYTES = 4 * 1024 * 1024  # flush sqlite progress every 4 MiB


@dataclass
class DownloadEndpoint:
    """One fallback download endpoint.

    ``url`` is treated as a base for multi-file artifacts (full URL is
    ``{url}/{relative_path}``) or as a direct file URL when the artifact has
    a single file with empty relative path.

    ``expires_at`` is an ISO-8601 timestamp; endpoints whose ``expires_at``
    is in the past at fetch time are skipped before any HTTP request.
    """

    url: str
    expires_at: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        try:
            ts = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        return (now or datetime.now(timezone.utc)) >= ts


@dataclass
class RequiredFile:
    """A single file within an artifact.

    ``relative_path`` is the path within the artifact root; "" means the
    artifact is single-file and the endpoint URL points directly at it.
    ``digest`` is ``sha256:<hex>`` or bare hex; verification runs after the
    last byte is written.
    """

    relative_path: str
    digest: str
    size_bytes: int | None = None


@dataclass
class ArtifactDescriptor:
    """Everything needed to fetch one artifact: identity, files, endpoints."""

    artifact_id: str
    required_files: list[RequiredFile]
    endpoints: list[DownloadEndpoint]


@dataclass
class DownloadResult:
    """Resolved on-disk paths keyed by relative_path."""

    artifact_id: str
    files: dict[str, Path]


class _ProgressJournal:
    """SQLite-backed per-file progress journal.

    Schema is one row per (artifact_id, relative_path) carrying the byte
    offset that has been *durably* written to disk and the index of the
    endpoint that produced those bytes. The journal is *advisory*: at
    open time we cross-check the row against the on-disk ``.part`` file
    and clamp ``bytes_written`` to the smaller of the two values.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS download_progress (
            artifact_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            bytes_written INTEGER NOT NULL DEFAULT 0,
            endpoint_index INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            PRIMARY KEY (artifact_id, relative_path)
        )
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def get(self, artifact_id: str, relative_path: str) -> tuple[int, int]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT bytes_written, endpoint_index FROM download_progress "
                "WHERE artifact_id = ? AND relative_path = ?",
                (artifact_id, relative_path),
            ).fetchone()
        if row is None:
            return (0, 0)
        return (int(row[0]), int(row[1]))

    def record(
        self,
        artifact_id: str,
        relative_path: str,
        bytes_written: int,
        endpoint_index: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO download_progress "
                "(artifact_id, relative_path, bytes_written, endpoint_index, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(artifact_id, relative_path) DO UPDATE SET "
                "  bytes_written = excluded.bytes_written, "
                "  endpoint_index = excluded.endpoint_index, "
                "  updated_at = excluded.updated_at",
                (artifact_id, relative_path, bytes_written, endpoint_index, time.time()),
            )

    def clear(self, artifact_id: str, relative_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM download_progress WHERE artifact_id = ? AND relative_path = ?",
                (artifact_id, relative_path),
            )


class DurableDownloader:
    """Resumable multi-URL multi-file artifact downloader.

    Designed to be wrapped by ``PrepareManager`` in PR 3. Single responsibility:
    given an :class:`ArtifactDescriptor` and a destination directory, return
    verified on-disk paths or raise :class:`OctomilError`.
    """

    def __init__(
        self,
        cache_dir: Path,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._timeout = timeout
        self._client = client
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._journal = _ProgressJournal(cache_dir / ".progress.sqlite")

    def download(self, descriptor: ArtifactDescriptor, dest_dir: Path) -> DownloadResult:
        """Download every file in the descriptor; return resolved paths.

        Raises :class:`OctomilError` if every endpoint is exhausted, a digest
        verification fails after a complete download, or no endpoints are
        usable (all expired before being tried).
        """
        if not descriptor.endpoints:
            raise OctomilError(
                code=ErrorCode.DOWNLOAD_FAILED,
                message=f"Artifact '{descriptor.artifact_id}' has no download endpoints.",
            )
        if not descriptor.required_files:
            raise OctomilError(
                code=ErrorCode.DOWNLOAD_FAILED,
                message=f"Artifact '{descriptor.artifact_id}' has no required_files.",
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = dest_dir / ".parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        lock = FileLock(descriptor.artifact_id, lock_dir=self._cache_dir / ".locks")
        with lock:
            results: dict[str, Path] = {}
            for required in descriptor.required_files:
                results[required.relative_path] = self._download_one(
                    descriptor=descriptor,
                    required=required,
                    dest_dir=dest_dir,
                    parts_dir=parts_dir,
                )
            return DownloadResult(artifact_id=descriptor.artifact_id, files=results)

    def _download_one(
        self,
        descriptor: ArtifactDescriptor,
        required: RequiredFile,
        dest_dir: Path,
        parts_dir: Path,
    ) -> Path:
        final_path = dest_dir / required.relative_path if required.relative_path else dest_dir / "artifact"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        if final_path.exists() and _digest_matches(final_path, required.digest):
            logger.debug(
                "Artifact %s file %r already present and verified", descriptor.artifact_id, required.relative_path
            )
            return final_path

        part_name = (required.relative_path or "artifact").replace("/", "_") + ".part"
        part_path = parts_dir / part_name

        journal_offset, journal_endpoint = self._journal.get(descriptor.artifact_id, required.relative_path)
        on_disk = part_path.stat().st_size if part_path.exists() else 0
        # Trust the smaller of the two — journal may be ahead of disk if the
        # process died mid-flush, or disk may be ahead if the journal flush
        # was rate-limited and we recovered after a clean exit.
        offset = min(journal_offset, on_disk)
        if offset != on_disk and part_path.exists():
            with part_path.open("r+b") as fh:
                fh.truncate(offset)

        last_error: Exception | None = None
        endpoints = list(enumerate(descriptor.endpoints))
        # Start with the journal's last endpoint, then fall through the rest.
        ordered = sorted(endpoints, key=lambda e: 0 if e[0] == journal_endpoint else 1)

        for index, endpoint in ordered:
            if endpoint.is_expired(self._clock()):
                logger.info("Skipping expired endpoint %d for %s", index, required.relative_path)
                continue
            try:
                self._fetch(endpoint, required, part_path, offset, descriptor.artifact_id, index)
                # Verify before moving into place; on mismatch, drop the .part
                # and let the next endpoint retry from zero.
                if not _digest_matches(part_path, required.digest):
                    part_path.unlink(missing_ok=True)
                    self._journal.clear(descriptor.artifact_id, required.relative_path)
                    offset = 0
                    last_error = OctomilError(
                        code=ErrorCode.CHECKSUM_MISMATCH,
                        message=(
                            f"Digest mismatch for '{descriptor.artifact_id}' file "
                            f"'{required.relative_path}' from endpoint {index}."
                        ),
                    )
                    continue
                part_path.replace(final_path)
                self._journal.clear(descriptor.artifact_id, required.relative_path)
                return final_path
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                logger.warning(
                    "Endpoint %d failed for %s/%s: %s",
                    index,
                    descriptor.artifact_id,
                    required.relative_path,
                    exc,
                )
                # On 4xx that suggests the URL itself is dead (403/404/410),
                # drop progress so the next endpoint starts clean.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403, 404, 410):
                    part_path.unlink(missing_ok=True)
                    self._journal.clear(descriptor.artifact_id, required.relative_path)
                    offset = 0
                else:
                    # Re-read disk for next attempt (partial bytes may have landed).
                    offset = part_path.stat().st_size if part_path.exists() else 0

        raise OctomilError(
            code=ErrorCode.DOWNLOAD_FAILED,
            message=(
                f"Exhausted all endpoints for '{descriptor.artifact_id}' file "
                f"'{required.relative_path}'. Last error: {last_error}"
            ),
            cause=last_error,
        ) from last_error

    def _fetch(
        self,
        endpoint: DownloadEndpoint,
        required: RequiredFile,
        part_path: Path,
        offset: int,
        artifact_id: str,
        endpoint_index: int,
    ) -> None:
        url = self._resolve_url(endpoint.url, required.relative_path)
        headers = dict(endpoint.headers or {})
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"

        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        owns_client = self._client is None
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 416:
                    # Server says our offset is past the file; treat as fresh.
                    part_path.unlink(missing_ok=True)
                    offset = 0
                    response.raise_for_status()
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                mode = "ab" if response.status_code == 206 and offset > 0 else "wb"
                if mode == "wb":
                    offset = 0
                bytes_written = offset
                last_flush = bytes_written
                with part_path.open(mode) as fh:
                    for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written - last_flush >= _PROGRESS_FLUSH_BYTES:
                            fh.flush()
                            self._journal.record(artifact_id, required.relative_path, bytes_written, endpoint_index)
                            last_flush = bytes_written
                    fh.flush()
                self._journal.record(artifact_id, required.relative_path, bytes_written, endpoint_index)
        finally:
            if owns_client:
                client.close()

    @staticmethod
    def _resolve_url(base: str, relative_path: str) -> str:
        if not relative_path:
            return base
        return f"{base.rstrip('/')}/{relative_path.lstrip('/')}"


def _digest_matches(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    expected_hex = expected[7:] if expected.startswith("sha256:") else expected
    expected_hex = expected_hex.lower()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest() == expected_hex
