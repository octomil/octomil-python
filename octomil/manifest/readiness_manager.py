"""ModelReadinessManager — download tracking for managed models."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from octomil._generated.delivery_mode import DeliveryMode
from octomil.manifest.types import AppModelEntry

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"


@dataclass
class DownloadUpdate:
    """Event emitted as managed models download."""

    model_id: str
    status: DownloadStatus
    progress: float = 0.0  # 0.0 to 1.0
    path: Optional[Path] = None
    error: Optional[str] = None


# Type alias for progress callbacks
ProgressCallback = Callable[[DownloadUpdate], None]


class ModelReadinessManager:
    """Orchestrates background downloads for managed models.

    Thread-safe. Tracks download state and provides readiness queries.
    """

    def __init__(
        self,
        download_dir: Optional[Path] = None,
        download_fn: Optional[Callable[[AppModelEntry, ProgressCallback], Path]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._download_dir = download_dir or Path.home() / ".octomil" / "models"
        self._download_fn = download_fn
        self._statuses: dict[str, DownloadStatus] = {}
        self._paths: dict[str, Path] = {}
        self._errors: dict[str, str] = {}
        self._callbacks: list[ProgressCallback] = []
        self._ready_events: dict[str, threading.Event] = {}

    def add_callback(self, callback: ProgressCallback) -> None:
        """Register a callback for download updates."""
        with self._lock:
            self._callbacks.append(callback)

    def enqueue(self, entry: AppModelEntry) -> None:
        """Queue a managed model entry for background download."""
        if entry.delivery != DeliveryMode.MANAGED:
            return

        with self._lock:
            if entry.id in self._statuses and self._statuses[entry.id] in (
                DownloadStatus.READY,
                DownloadStatus.IN_PROGRESS,
            ):
                return
            self._statuses[entry.id] = DownloadStatus.PENDING
            self._ready_events.setdefault(entry.id, threading.Event())

        thread = threading.Thread(
            target=self._download_worker,
            args=(entry,),
            daemon=True,
            name=f"octomil-download-{entry.id}",
        )
        thread.start()

    def is_ready(self, model_id: str) -> bool:
        """Check if a model is ready for inference."""
        with self._lock:
            return self._statuses.get(model_id) == DownloadStatus.READY

    def get_path(self, model_id: str) -> Optional[Path]:
        """Get the local path of a ready model, or None."""
        with self._lock:
            return self._paths.get(model_id)

    def await_ready(self, model_id: str, timeout: Optional[float] = None) -> Path:
        """Block until a model is ready and return its local path.

        Raises RuntimeError if the download fails or times out.
        """
        with self._lock:
            event = self._ready_events.get(model_id)
            if event is None:
                # Not enqueued — check if already ready
                if model_id in self._paths:
                    return self._paths[model_id]
                raise RuntimeError(f"Model '{model_id}' not enqueued for download")

        if not event.wait(timeout=timeout):
            raise RuntimeError(f"Timeout waiting for model '{model_id}' to be ready")

        with self._lock:
            if self._statuses.get(model_id) == DownloadStatus.FAILED:
                error = self._errors.get(model_id, "unknown error")
                raise RuntimeError(f"Download failed for model '{model_id}': {error}")
            path = self._paths.get(model_id)
            if path is None:
                raise RuntimeError(f"Model '{model_id}' not found after download")
            return path

    def mark_ready(self, model_id: str, path: Path) -> None:
        """Manually mark a model as ready (e.g. from cache)."""
        with self._lock:
            self._statuses[model_id] = DownloadStatus.READY
            self._paths[model_id] = path
            event = self._ready_events.get(model_id)
            if event:
                event.set()
        self._emit(DownloadUpdate(model_id=model_id, status=DownloadStatus.READY, progress=1.0, path=path))

    def _download_worker(self, entry: AppModelEntry) -> None:
        """Background worker that downloads a model."""
        model_id = entry.id
        with self._lock:
            self._statuses[model_id] = DownloadStatus.IN_PROGRESS
        self._emit(DownloadUpdate(model_id=model_id, status=DownloadStatus.IN_PROGRESS))

        try:
            if self._download_fn is not None:
                path = self._download_fn(entry, self._on_progress)
            else:
                path = self._default_download(entry)

            with self._lock:
                self._statuses[model_id] = DownloadStatus.READY
                self._paths[model_id] = path
                event = self._ready_events.get(model_id)
                if event:
                    event.set()
            self._emit(DownloadUpdate(model_id=model_id, status=DownloadStatus.READY, progress=1.0, path=path))
            logger.info("Model '%s' ready at %s", model_id, path)

        except Exception as exc:
            error_msg = str(exc)
            with self._lock:
                self._statuses[model_id] = DownloadStatus.FAILED
                self._errors[model_id] = error_msg
                event = self._ready_events.get(model_id)
                if event:
                    event.set()
            self._emit(DownloadUpdate(model_id=model_id, status=DownloadStatus.FAILED, error=error_msg))
            logger.error("Model '%s' download failed: %s", model_id, error_msg)

    def _default_download(self, entry: AppModelEntry) -> Path:
        """Placeholder download — in production, fetches from download_url."""
        if entry.download_url is None:
            raise RuntimeError(f"No download_url for managed model '{entry.id}'")
        # Placeholder: in real implementation, this downloads via httpx
        dest = self._download_dir / entry.id
        dest.parent.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(
            f"Default download not implemented. Configure a download_fn or provide a cached model for '{entry.id}'."
        )

    def _on_progress(self, update: DownloadUpdate) -> None:
        """Progress callback from download function."""
        self._emit(update)

    def _emit(self, update: DownloadUpdate) -> None:
        """Emit a download update to all registered callbacks."""
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(update)
            except Exception:
                logger.debug("Callback error for model '%s'", update.model_id, exc_info=True)
