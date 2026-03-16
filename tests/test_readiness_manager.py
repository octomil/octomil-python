"""Tests for octomil.manifest.readiness_manager — ModelReadinessManager."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pytest

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.model_capability import ModelCapability
from octomil.manifest.readiness_manager import (
    DownloadStatus,
    DownloadUpdate,
    ModelReadinessManager,
    ProgressCallback,
)
from octomil.manifest.types import AppModelEntry


def _make_entry(
    model_id: str = "test-model",
    delivery: DeliveryMode = DeliveryMode.MANAGED,
    download_url: Optional[str] = "https://example.com/model.bin",
) -> AppModelEntry:
    return AppModelEntry(
        id=model_id,
        capability=ModelCapability.CHAT,
        delivery=delivery,
        download_url=download_url,
    )


class TestReadinessManagerBasic:
    def test_initial_state(self) -> None:
        mgr = ModelReadinessManager()
        assert not mgr.is_ready("test-model")
        assert mgr.get_path("test-model") is None

    def test_mark_ready(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_text("fake")

        mgr = ModelReadinessManager()
        mgr.mark_ready("test-model", model_file)

        assert mgr.is_ready("test-model")
        assert mgr.get_path("test-model") == model_file

    def test_mark_ready_with_callback(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_text("fake")

        updates: list[DownloadUpdate] = []
        mgr = ModelReadinessManager()
        mgr.add_callback(updates.append)
        mgr.mark_ready("test-model", model_file)

        assert len(updates) == 1
        assert updates[0].model_id == "test-model"
        assert updates[0].status == DownloadStatus.READY
        assert updates[0].path == model_file

    def test_await_ready_already_ready(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_text("fake")

        mgr = ModelReadinessManager()
        mgr.mark_ready("test-model", model_file)

        result = mgr.await_ready("test-model", timeout=1.0)
        assert result == model_file

    def test_await_ready_not_enqueued(self) -> None:
        mgr = ModelReadinessManager()
        with pytest.raises(RuntimeError, match="not enqueued"):
            mgr.await_ready("unknown-model", timeout=0.1)

    def test_enqueue_ignores_non_managed(self) -> None:
        entry = _make_entry(delivery=DeliveryMode.BUNDLED)
        mgr = ModelReadinessManager()
        mgr.enqueue(entry)
        assert not mgr.is_ready(entry.id)


class TestReadinessManagerDownload:
    def test_enqueue_with_custom_download_fn(self, tmp_path: Path) -> None:
        model_file = tmp_path / "downloaded.bin"
        model_file.write_text("downloaded model")

        def download_fn(entry: AppModelEntry, progress_cb: ProgressCallback) -> Path:
            progress_cb(DownloadUpdate(model_id=entry.id, status=DownloadStatus.IN_PROGRESS, progress=0.5))
            return model_file

        mgr = ModelReadinessManager(download_fn=download_fn)
        entry = _make_entry()
        mgr.enqueue(entry)

        result = mgr.await_ready(entry.id, timeout=5.0)
        assert result == model_file
        assert mgr.is_ready(entry.id)

    def test_enqueue_download_failure(self) -> None:
        def download_fn(entry: AppModelEntry, progress_cb: ProgressCallback) -> Path:
            raise RuntimeError("Download failed intentionally")

        mgr = ModelReadinessManager(download_fn=download_fn)
        entry = _make_entry()
        mgr.enqueue(entry)

        with pytest.raises(RuntimeError, match="Download failed"):
            mgr.await_ready(entry.id, timeout=5.0)

    def test_await_ready_timeout(self) -> None:
        def download_fn(entry: AppModelEntry, progress_cb: ProgressCallback) -> Path:
            time.sleep(10)
            return Path("/never")

        mgr = ModelReadinessManager(download_fn=download_fn)
        entry = _make_entry()
        mgr.enqueue(entry)

        with pytest.raises(RuntimeError, match="Timeout"):
            mgr.await_ready(entry.id, timeout=0.1)

    def test_duplicate_enqueue_ignored(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_text("data")
        call_count = 0

        def download_fn(entry: AppModelEntry, progress_cb: ProgressCallback) -> Path:
            nonlocal call_count
            call_count += 1
            return model_file

        mgr = ModelReadinessManager(download_fn=download_fn)
        entry = _make_entry()
        mgr.enqueue(entry)
        mgr.await_ready(entry.id, timeout=5.0)

        mgr.enqueue(entry)
        assert call_count == 1
