"""Storage garbage collector for device model versions.

Periodically scans for model versions that are no longer active, staged,
or needed for rollback, and deletes their on-disk artifacts to reclaim
storage space.

Rules:
  - Always keep the active version + previous version (rollback target).
  - Never collect a version that has active inference sessions (refcount > 0).
  - Never collect a version that is currently staged.
  - Everything else that the registry marks as gc_eligible can be collected.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .inference_session_manager import InferenceSessionManager
from .model_registry import DeviceModelRegistry

logger = logging.getLogger(__name__)


class StorageGC:
    """Garbage collector that removes unused model version directories."""

    def __init__(
        self,
        registry: DeviceModelRegistry,
        session_manager: InferenceSessionManager,
        models_dir: str | Path = "/models",
    ) -> None:
        self._registry = registry
        self._sessions = session_manager
        self._models_dir = Path(models_dir)

    # -- Scanning --

    def scan(self) -> list[tuple[str, str]]:
        """Return a list of ``(model_id, version)`` pairs eligible for collection.

        A version is eligible if:
          1. The registry reports it as gc_eligible (not active, not previous).
          2. It has zero active inference sessions.
          3. It is not currently staged.
        """
        eligible: list[tuple[str, str]] = []
        model_ids = self._discover_model_ids()

        for model_id in model_ids:
            gc_versions = self._registry.gc_eligible_versions(model_id)
            staged_versions = {v["version"] for v in self._registry.get_staged_versions(model_id)}
            for version in gc_versions:
                if version in staged_versions:
                    continue
                if self._sessions.get_refcount(model_id, version) > 0:
                    continue
                eligible.append((model_id, version))

        return eligible

    # -- Collection --

    def collect(self, model_id: str, version: str) -> int:
        """Delete the on-disk directory for a single model version.

        Returns bytes freed (0 if nothing deleted or version is protected).
        """
        # Guard: never collect active or previous version
        active = self._registry.get_active_model(model_id)
        if active is not None:
            if version == active["active_version"]:
                logger.warning("Refusing to collect active version %s/%s", model_id, version)
                return 0
            if version == active.get("previous_version"):
                logger.warning("Refusing to collect previous version %s/%s", model_id, version)
                return 0

        # Guard: never collect if sessions still hold a reference
        if self._sessions.get_refcount(model_id, version) > 0:
            logger.warning("Refusing to collect %s/%s — active sessions", model_id, version)
            return 0

        # Guard: never collect staged versions
        staged_versions = {v["version"] for v in self._registry.get_staged_versions(model_id)}
        if version in staged_versions:
            logger.warning("Refusing to collect staged version %s/%s", model_id, version)
            return 0

        model_path = self._registry.get_model_path(model_id, version)
        if not model_path.exists():
            return 0

        size = self._dir_size(model_path)
        shutil.rmtree(model_path)
        logger.info("Collected %s/%s — freed %d bytes", model_id, version, size)
        return size

    # -- Full GC pass --

    def run(self, dry_run: bool = False) -> int:
        """Execute a full GC pass. Returns total bytes freed.

        If *dry_run* is True, scans and logs but does not delete anything.
        """
        eligible = self.scan()
        total_freed = 0
        for model_id, version in eligible:
            model_path = self._registry.get_model_path(model_id, version)
            size = self._dir_size(model_path) if model_path.exists() else 0
            if dry_run:
                logger.info("[dry-run] Would collect %s/%s (%d bytes)", model_id, version, size)
            else:
                total_freed += self.collect(model_id, version)
        return total_freed

    # -- Storage usage reporting --

    def get_storage_usage(self) -> dict[str, Any]:
        """Return a summary of storage usage across all models.

        Returns a dict with ``total_bytes`` and ``by_model`` breakdown.
        """
        total = 0
        by_model: dict[str, int] = {}

        if not self._models_dir.exists():
            return {"total_bytes": 0, "by_model": {}}

        for model_dir in self._models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_total = 0
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                model_total += self._dir_size(version_dir)
            by_model[model_dir.name] = model_total
            total += model_total

        return {"total_bytes": total, "by_model": by_model}

    # -- Helpers --

    def _discover_model_ids(self) -> list[str]:
        """Discover model IDs from the on-disk directory structure."""
        if not self._models_dir.exists():
            return []
        return [d.name for d in sorted(self._models_dir.iterdir()) if d.is_dir()]

    @staticmethod
    def _dir_size(path: Path) -> int:
        """Recursively compute the total size of a directory in bytes."""
        if not path.exists():
            return 0
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
