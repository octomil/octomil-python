"""Managed local runtime lifecycle — detection, caching, downloading, preparation.

Public API::

    from octomil.runtime.lifecycle import (
        detect_installed_runtimes,
        ArtifactCache,
        DownloadManager,
        prepare_runtime,
    )
"""

from __future__ import annotations

from octomil.runtime.lifecycle.artifact_cache import ArtifactCache
from octomil.runtime.lifecycle.detection import InstalledRuntime, detect_installed_runtimes
from octomil.runtime.lifecycle.download import DownloadManager
from octomil.runtime.lifecycle.prepare import PrepareResult, prepare_runtime

__all__ = [
    "ArtifactCache",
    "DownloadManager",
    "InstalledRuntime",
    "PrepareResult",
    "detect_installed_runtimes",
    "prepare_runtime",
]
