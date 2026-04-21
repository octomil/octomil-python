"""Artifact cache — local disk cache for downloaded model artifacts.

Manages the Octomil artifact cache directory with a JSON manifest
tracking cached files, their digests, sizes, and access timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    """Return the managed artifact cache directory.

    Large artifacts live under the platform cache root, not under
    ``~/.octomil`` where we keep credentials/config-style state.
    """

    cache_root = os.environ.get("OCTOMIL_CACHE_DIR")
    if cache_root:
        return Path(cache_root).expanduser() / "artifacts"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "octomil" / "artifacts"

    return Path.home() / ".cache" / "octomil" / "artifacts"


_MANIFEST_FILENAME = "manifest.json"
_CHUNK_SIZE = 8192  # 8KB read chunks for digest calculation


@dataclass
class ArtifactEntry:
    """A single cached artifact entry in the manifest."""

    artifact_id: str
    digest: str  # SHA-256 hex digest
    path: str  # Relative to cache dir
    size_bytes: int
    last_used: float  # Unix timestamp
    created_at: float  # Unix timestamp


@dataclass
class ArtifactManifest:
    """The on-disk manifest tracking all cached artifacts."""

    version: int = 1
    entries: dict[str, ArtifactEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        data = {"version": self.version, "entries": {k: asdict(v) for k, v in self.entries.items()}}
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ArtifactManifest:
        data = json.loads(text)
        entries: dict[str, ArtifactEntry] = {}
        for key, val in data.get("entries", {}).items():
            entries[key] = ArtifactEntry(**val)
        return cls(version=data.get("version", 1), entries=entries)


class ArtifactCache:
    """Local disk cache for model artifacts.

    Artifacts are stored as files under ``<cache-root>/artifacts/{artifact_id}/``.
    A manifest.json in the cache root tracks metadata.

    Usage::

        cache = ArtifactCache()
        path = cache.get("my-model-q4", "sha256:abc123...")
        if path:
            # Use cached artifact at path
            ...
        else:
            # Download, then store
            cache.put("my-model-q4", "sha256:abc123...", downloaded_path)
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._cache_dir / _MANIFEST_FILENAME
        self._manifest = self._load_manifest()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def manifest(self) -> ArtifactManifest:
        return self._manifest

    def get(self, artifact_id: str, digest: str) -> Path | None:
        """Return the cached artifact path if it exists and passes verification.

        Returns None if the artifact is not cached or if the digest does not match.
        On digest mismatch, the corrupt entry is removed.
        """
        key = self._make_key(artifact_id, digest)
        entry = self._manifest.entries.get(key)
        if entry is None:
            return None

        artifact_path = self._cache_dir / entry.path
        if not artifact_path.exists():
            # File was deleted externally
            logger.debug("Cached artifact file missing: %s", artifact_path)
            del self._manifest.entries[key]
            self._save_manifest()
            return None

        # Verify digest
        if not self.verify(artifact_path, digest):
            logger.warning(
                "Digest mismatch for cached artifact %s — removing corrupt entry",
                artifact_id,
            )
            self._remove_artifact_file(artifact_path)
            del self._manifest.entries[key]
            self._save_manifest()
            return None

        # Update last_used timestamp
        entry.last_used = time.time()
        self._save_manifest()
        return artifact_path

    def put(self, artifact_id: str, digest: str, data_path: Path) -> Path:
        """Store an artifact in the cache.

        Copies/moves the file from ``data_path`` into the cache directory
        and registers it in the manifest.

        Returns the final cache path.
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Source artifact not found: {data_path}")

        # Determine destination
        safe_id = artifact_id.replace("/", "_").replace("\\", "_")
        artifact_dir = self._cache_dir / safe_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / data_path.name

        # Copy the file (we don't move in case caller still needs it)
        if dest != data_path:
            import shutil

            shutil.copy2(str(data_path), str(dest))

        # Register in manifest
        now = time.time()
        key = self._make_key(artifact_id, digest)
        self._manifest.entries[key] = ArtifactEntry(
            artifact_id=artifact_id,
            digest=digest,
            path=str(dest.relative_to(self._cache_dir)),
            size_bytes=dest.stat().st_size,
            last_used=now,
            created_at=now,
        )
        self._save_manifest()
        return dest

    def remove(self, artifact_id: str, digest: str) -> bool:
        """Remove an artifact from the cache. Returns True if it was found."""
        key = self._make_key(artifact_id, digest)
        entry = self._manifest.entries.get(key)
        if entry is None:
            return False

        artifact_path = self._cache_dir / entry.path
        self._remove_artifact_file(artifact_path)
        del self._manifest.entries[key]
        self._save_manifest()
        return True

    def list_entries(self) -> list[ArtifactEntry]:
        """List all cached artifact entries."""
        return list(self._manifest.entries.values())

    def total_size_bytes(self) -> int:
        """Total size of all cached artifacts in bytes."""
        return sum(e.size_bytes for e in self._manifest.entries.values())

    @staticmethod
    def verify(path: Path, expected_digest: str) -> bool:
        """Verify SHA-256 digest of a file.

        Accepts digest in formats:
        - "sha256:<hex>"
        - "<hex>" (bare hex, assumed SHA-256)
        """
        if not path.exists():
            return False

        # Parse expected digest
        if expected_digest.startswith("sha256:"):
            expected_hex = expected_digest[7:]
        else:
            expected_hex = expected_digest

        actual_hex = _compute_sha256(path)
        return actual_hex == expected_hex.lower()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(artifact_id: str, digest: str) -> str:
        """Deterministic cache key from id + digest."""
        return f"{artifact_id}::{digest}"

    def _load_manifest(self) -> ArtifactManifest:
        """Load manifest from disk, or create empty if not present."""
        if self._manifest_path.exists():
            try:
                text = self._manifest_path.read_text(encoding="utf-8")
                return ArtifactManifest.from_json(text)
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Corrupt manifest at %s — resetting", self._manifest_path)
        return ArtifactManifest()

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        try:
            self._manifest_path.write_text(self._manifest.to_json(), encoding="utf-8")
        except OSError:
            logger.warning("Failed to write artifact manifest", exc_info=True)

    @staticmethod
    def _remove_artifact_file(path: Path) -> None:
        """Remove an artifact file, ignoring errors."""
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil

                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
