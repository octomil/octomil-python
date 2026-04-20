"""Tests for the managed local runtime lifecycle.

Covers: detection, artifact cache, file locks, download manager, and prepare.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.lifecycle.artifact_cache import ArtifactCache, _compute_sha256
from octomil.runtime.lifecycle.detection import InstalledRuntime, detect_installed_runtimes
from octomil.runtime.lifecycle.download import DownloadManager, _is_tty
from octomil.runtime.lifecycle.file_lock import FileLock
from octomil.runtime.lifecycle.prepare import PrepareResult, RuntimeCandidate, prepare_runtime

# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetection:
    """Tests for runtime detection."""

    def test_detect_mlx_when_installed(self) -> None:
        """mlx-lm is detected when importlib.util.find_spec returns a valid spec."""
        mock_spec = MagicMock()
        with (
            patch("octomil.runtime.lifecycle.detection.importlib.util.find_spec", return_value=mock_spec),
            patch("octomil.runtime.lifecycle.detection.platform.system", return_value="Darwin"),
            patch("octomil.runtime.lifecycle.detection.platform.machine", return_value="arm64"),
            patch("octomil.runtime.lifecycle.detection._get_package_version", return_value="0.19.0"),
        ):
            results = detect_installed_runtimes()
            mlx_results = [r for r in results if r.engine_id == "mlx-lm"]
            assert len(mlx_results) >= 1
            assert mlx_results[0].version == "0.19.0"

    def test_detect_mlx_not_on_linux(self) -> None:
        """mlx-lm detection is skipped on non-Darwin systems."""
        with (
            patch("octomil.runtime.lifecycle.detection.platform.system", return_value="Linux"),
            patch("octomil.runtime.lifecycle.detection.platform.machine", return_value="x86_64"),
            patch("octomil.runtime.lifecycle.detection.importlib.util.find_spec", return_value=None),
            patch("octomil.runtime.lifecycle.detection.shutil.which", return_value=None),
        ):
            results = detect_installed_runtimes()
            mlx_results = [r for r in results if r.engine_id == "mlx-lm"]
            assert len(mlx_results) == 0

    def test_detect_llamacpp_when_in_path(self) -> None:
        """llama.cpp CLI is detected when shutil.which finds it on PATH."""
        with (
            patch("octomil.runtime.lifecycle.detection.platform.system", return_value="Linux"),
            patch("octomil.runtime.lifecycle.detection.platform.machine", return_value="x86_64"),
            patch("octomil.runtime.lifecycle.detection.importlib.util.find_spec", return_value=None),
            patch(
                "octomil.runtime.lifecycle.detection.shutil.which",
                side_effect=lambda name: "/usr/local/bin/llama-cli" if name == "llama-cli" else None,
            ),
        ):
            results = detect_installed_runtimes()
            llama_results = [r for r in results if r.engine_id == "llama.cpp"]
            assert len(llama_results) >= 1
            assert llama_results[0].path == "/usr/local/bin/llama-cli"
            assert llama_results[0].extras.get("binding") == "cli"

    def test_detect_llamacpp_python_binding(self) -> None:
        """llama-cpp-python package is detected via importlib."""
        mock_spec = MagicMock()
        with (
            patch("octomil.runtime.lifecycle.detection.platform.system", return_value="Linux"),
            patch("octomil.runtime.lifecycle.detection.platform.machine", return_value="x86_64"),
            patch(
                "octomil.runtime.lifecycle.detection.importlib.util.find_spec",
                side_effect=lambda name: mock_spec if name == "llama_cpp" else None,
            ),
            patch("octomil.runtime.lifecycle.detection._get_package_version", return_value="0.2.90"),
            patch("octomil.runtime.lifecycle.detection.shutil.which", return_value=None),
        ):
            results = detect_installed_runtimes()
            llama_results = [r for r in results if r.engine_id == "llama.cpp"]
            assert len(llama_results) >= 1
            assert llama_results[0].version == "0.2.90"
            assert llama_results[0].extras.get("binding") == "python"

    def test_installed_runtime_display(self) -> None:
        """InstalledRuntime.display formats a readable summary."""
        rt = InstalledRuntime(engine_id="mlx-lm", version="0.19.0")
        assert "mlx-lm" in rt.display
        assert "v0.19.0" in rt.display


# ---------------------------------------------------------------------------
# Artifact cache tests
# ---------------------------------------------------------------------------


class TestArtifactCache:
    """Tests for the artifact cache."""

    def _make_test_file(self, tmp_dir: Path, content: bytes = b"test content") -> tuple[Path, str]:
        """Create a test file and return (path, sha256_hex)."""
        f = tmp_dir / "test_artifact.gguf"
        f.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return f, digest

    def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        """Cache hit returns path without downloading."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"model weights data here"
        source_file, digest = self._make_test_file(tmp_path, content)

        # Put into cache
        cached_path = cache.put("model-q4", f"sha256:{digest}", source_file)
        assert cached_path.exists()

        # Get should return without any download
        result = cache.get("model-q4", f"sha256:{digest}")
        assert result is not None
        assert result.exists()
        assert result.read_bytes() == content

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """Cache miss returns None."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        result = cache.get("nonexistent", "sha256:deadbeef")
        assert result is None

    def test_digest_verification_passes(self, tmp_path: Path) -> None:
        """verify() returns True for matching digest."""
        content = b"verified content"
        f = tmp_path / "file.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert ArtifactCache.verify(f, f"sha256:{expected}") is True

    def test_digest_verification_fails_deletes_artifact(self, tmp_path: Path) -> None:
        """Cache get with mismatched digest removes the corrupt entry."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"original content"
        source_file, digest = self._make_test_file(tmp_path, content)

        # Store with correct digest
        cached_path = cache.put("model-q4", f"sha256:{digest}", source_file)
        assert cached_path.exists()

        # Corrupt the cached file
        cached_path.write_bytes(b"corrupted!!!")

        # Get with original digest should fail and remove entry
        result = cache.get("model-q4", f"sha256:{digest}")
        assert result is None
        # The corrupt file should be removed
        assert not cached_path.exists()
        # Manifest should no longer have the entry
        assert cache.list_entries() == []

    def test_digest_verification_bare_hex(self, tmp_path: Path) -> None:
        """verify() accepts bare hex without sha256: prefix."""
        content = b"bare hex test"
        f = tmp_path / "file.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert ArtifactCache.verify(f, expected) is True

    def test_put_and_list(self, tmp_path: Path) -> None:
        """put() stores artifacts and list_entries() returns them."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"test data"
        f, digest = self._make_test_file(tmp_path, content)

        cache.put("art-1", f"sha256:{digest}", f)
        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0].artifact_id == "art-1"
        assert entries[0].size_bytes == len(content)

    def test_remove(self, tmp_path: Path) -> None:
        """remove() deletes artifact from cache and manifest."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"removable"
        f, digest = self._make_test_file(tmp_path, content)

        cache.put("removable-model", f"sha256:{digest}", f)
        assert cache.remove("removable-model", f"sha256:{digest}") is True
        assert cache.get("removable-model", f"sha256:{digest}") is None

    def test_total_size(self, tmp_path: Path) -> None:
        """total_size_bytes() accumulates across entries."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        for i in range(3):
            content = f"data-{i}".encode()
            f = tmp_path / f"art_{i}.bin"
            f.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            cache.put(f"art-{i}", f"sha256:{digest}", f)

        total = cache.total_size_bytes()
        assert total > 0

    def test_compute_sha256(self, tmp_path: Path) -> None:
        """_compute_sha256 returns correct hex digest."""
        content = b"hello sha256"
        f = tmp_path / "hash_test.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _compute_sha256(f) == expected


# ---------------------------------------------------------------------------
# File lock tests
# ---------------------------------------------------------------------------


class TestFileLock:
    """Tests for cross-platform file locking."""

    def test_lock_acquire_release(self, tmp_path: Path) -> None:
        """Basic acquire/release cycle."""
        lock = FileLock("test-artifact", lock_dir=tmp_path / "locks")
        assert not lock.is_locked
        lock.acquire()
        assert lock.is_locked
        assert lock.lock_path.exists()
        lock.release()
        assert not lock.is_locked

    def test_lock_context_manager(self, tmp_path: Path) -> None:
        """Context manager acquires and releases."""
        lock = FileLock("ctx-test", lock_dir=tmp_path / "locks")
        with lock:
            assert lock.is_locked
        assert not lock.is_locked

    def test_file_lock_prevents_concurrent_download(self, tmp_path: Path) -> None:
        """Two threads contending for the same lock — only one holds it at a time."""
        lock_dir = tmp_path / "locks"
        results: list[tuple[str, float, float]] = []

        def worker(name: str) -> None:
            lock = FileLock("shared-artifact", lock_dir=lock_dir, timeout=10.0)
            with lock:
                start = time.monotonic()
                time.sleep(0.1)  # Simulate work
                end = time.monotonic()
                results.append((name, start, end))

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert len(results) == 2
        # One must finish before the other starts (non-overlapping)
        results.sort(key=lambda x: x[1])
        first_end = results[0][2]
        second_start = results[1][1]
        # Allow small tolerance for thread scheduling
        assert second_start >= first_end - 0.01

    def test_lock_timeout(self, tmp_path: Path) -> None:
        """Lock times out if another process holds it indefinitely."""
        lock_dir = tmp_path / "locks"
        lock1 = FileLock("timeout-test", lock_dir=lock_dir, timeout=0.3)
        lock1.acquire()

        # Second lock should timeout (same process can re-acquire on some OSes,
        # so we test with a separate fd)
        lock2 = FileLock("timeout-test", lock_dir=lock_dir, timeout=0.3, poll_interval=0.05)
        with pytest.raises(TimeoutError, match="Could not acquire lock"):
            lock2.acquire()

        lock1.release()

    def test_lock_sanitizes_name(self, tmp_path: Path) -> None:
        """Lock name with special chars is sanitized for filesystem."""
        lock = FileLock("org/model:v1.0", lock_dir=tmp_path / "locks")
        assert "/" not in lock.lock_path.name
        assert ":" not in lock.lock_path.name


# ---------------------------------------------------------------------------
# Download manager tests
# ---------------------------------------------------------------------------


class TestDownloadManager:
    """Tests for the download manager."""

    def test_artifact_cache_miss_triggers_download(self, tmp_path: Path) -> None:
        """When artifact is not in cache, download is triggered."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"downloaded model data"
        digest = hashlib.sha256(content).hexdigest()

        # Mock httpx.Client to return our content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_bytes = MagicMock(return_value=iter([content]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        mgr = DownloadManager(cache=cache, show_progress=False)

        with patch("octomil.runtime.lifecycle.download.httpx.Client", return_value=mock_client):
            result = mgr.download(
                artifact_id="test-model",
                url="https://models.example.com/test-model.gguf",
                expected_digest=f"sha256:{digest}",
            )

        assert result.exists()
        assert result.read_bytes() == content
        # Should now be in cache
        cached = cache.get("test-model", f"sha256:{digest}")
        assert cached is not None

    def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        """When artifact is already cached, no download occurs."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"already cached"
        digest = hashlib.sha256(content).hexdigest()

        # Pre-populate cache
        source = tmp_path / "source.gguf"
        source.write_bytes(content)
        cache.put("cached-model", f"sha256:{digest}", source)

        mgr = DownloadManager(cache=cache, show_progress=False)

        # No httpx mock needed — download should not be called
        result = mgr.download(
            artifact_id="cached-model",
            url="https://should-not-be-called.com/file",
            expected_digest=f"sha256:{digest}",
        )

        assert result.exists()
        assert result.read_bytes() == content

    def test_digest_mismatch_raises_error(self, tmp_path: Path) -> None:
        """Download with wrong digest raises CHECKSUM_MISMATCH error."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"actual data"
        wrong_digest = "0" * 64  # Wrong digest

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_bytes = MagicMock(return_value=iter([content]))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        mgr = DownloadManager(cache=cache, show_progress=False)

        with patch("octomil.runtime.lifecycle.download.httpx.Client", return_value=mock_client):
            with pytest.raises(OctomilError) as exc_info:
                mgr.download(
                    artifact_id="bad-model",
                    url="https://models.example.com/bad.gguf",
                    expected_digest=f"sha256:{wrong_digest}",
                )

        assert exc_info.value.code == OctomilErrorCode.CHECKSUM_MISMATCH

    def test_non_tty_skips_progress_bar(self) -> None:
        """_is_tty returns False when stdout is not a terminal."""
        mock_stdout = MagicMock()
        mock_stdout.isatty = MagicMock(return_value=False)
        with patch.object(sys, "stdout", mock_stdout):
            assert _is_tty() is False

    def test_tty_shows_progress(self) -> None:
        """_is_tty returns True when stdout is a terminal."""
        mock_stdout = MagicMock()
        mock_stdout.isatty = MagicMock(return_value=True)
        with patch.object(sys, "stdout", mock_stdout):
            assert _is_tty() is True

    def test_download_failure_raises_download_error(self, tmp_path: Path) -> None:
        """Network failure raises DOWNLOAD_FAILED error."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")

        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=Exception("Connection refused"))
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        mgr = DownloadManager(cache=cache, show_progress=False)

        with patch("octomil.runtime.lifecycle.download.httpx.Client", return_value=mock_client):
            with pytest.raises(OctomilError) as exc_info:
                mgr.download(
                    artifact_id="fail-model",
                    url="https://models.example.com/fail.gguf",
                    expected_digest="sha256:" + "a" * 64,
                )

        assert exc_info.value.code == OctomilErrorCode.DOWNLOAD_FAILED


# ---------------------------------------------------------------------------
# Prepare tests
# ---------------------------------------------------------------------------


class TestPrepare:
    """Tests for prepare_runtime."""

    def test_prepare_returns_actionable_error_when_engine_missing(self, tmp_path: Path) -> None:
        """When engine is not installed, error includes install instructions."""
        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[],
        ):
            candidate = RuntimeCandidate(engine_id="mlx-lm", artifact_id="model-q4")
            result = prepare_runtime(candidate, cache=ArtifactCache(cache_dir=tmp_path / "cache"))

        assert result.ok is False
        assert result.error_code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "mlx-lm" in (result.error or "")
        assert "pip install" in (result.error or "")

    def test_no_fake_output_on_failure(self, tmp_path: Path) -> None:
        """On failure, prepare_runtime returns error — NEVER fake inference output."""
        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[],
        ):
            candidate = RuntimeCandidate(engine_id="llama.cpp")
            result = prepare_runtime(candidate, cache=ArtifactCache(cache_dir=tmp_path / "cache"))

        assert result.ok is False
        assert result.artifact_path is None
        # Must not contain anything resembling model output
        assert result.error is not None
        assert "Install" in result.error or "install" in result.error

    def test_prepare_success_when_engine_manages_download(self, tmp_path: Path) -> None:
        """Engine that manages its own download returns success without artifact."""
        runtime = InstalledRuntime(engine_id="mlx-lm", version="0.19.0")
        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[runtime],
        ):
            candidate = RuntimeCandidate(engine_id="mlx-lm")
            result = prepare_runtime(candidate, cache=ArtifactCache(cache_dir=tmp_path / "cache"))

        assert result.ok is True
        assert result.installed_runtime == runtime
        assert result.artifact_path is None

    def test_prepare_with_cached_artifact(self, tmp_path: Path) -> None:
        """When artifact is already cached, prepare succeeds immediately."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        content = b"cached weights"
        digest = hashlib.sha256(content).hexdigest()
        source = tmp_path / "model.gguf"
        source.write_bytes(content)
        cache.put("model-q4", f"sha256:{digest}", source)

        runtime = InstalledRuntime(engine_id="llama.cpp", version="0.2.90")
        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[runtime],
        ):
            candidate = RuntimeCandidate(
                engine_id="llama.cpp",
                artifact_id="model-q4",
                expected_digest=f"sha256:{digest}",
            )
            result = prepare_runtime(candidate, cache=cache)

        assert result.ok is True
        assert result.artifact_path is not None
        assert result.artifact_path.exists()

    def test_prepare_skip_download_returns_error(self, tmp_path: Path) -> None:
        """With skip_download=True, uncached artifact returns actionable error."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        runtime = InstalledRuntime(engine_id="llama.cpp", version="0.2.90")

        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[runtime],
        ):
            candidate = RuntimeCandidate(
                engine_id="llama.cpp",
                artifact_id="model-q4",
                artifact_url="https://example.com/model.gguf",
                expected_digest="sha256:" + "b" * 64,
            )
            result = prepare_runtime(candidate, cache=cache, skip_download=True)

        assert result.ok is False
        assert "--yes" in (result.error or "")

    def test_prepare_no_url_returns_error(self, tmp_path: Path) -> None:
        """Without a download URL, uncached artifact returns helpful error."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        runtime = InstalledRuntime(engine_id="llama.cpp", version="0.2.90")

        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[runtime],
        ):
            candidate = RuntimeCandidate(
                engine_id="llama.cpp",
                artifact_id="model-q4",
                expected_digest="sha256:" + "c" * 64,
            )
            result = prepare_runtime(candidate, cache=cache)

        assert result.ok is False
        assert "no download URL" in (result.error or "") or "no download url" in (result.error or "").lower()

    def test_prepare_missing_digest_returns_error(self, tmp_path: Path) -> None:
        """Without expected digest, download is refused."""
        cache = ArtifactCache(cache_dir=tmp_path / "cache")
        runtime = InstalledRuntime(engine_id="llama.cpp", version="0.2.90")

        with patch(
            "octomil.runtime.lifecycle.prepare.detect_installed_runtimes",
            return_value=[runtime],
        ):
            candidate = RuntimeCandidate(
                engine_id="llama.cpp",
                artifact_id="model-q4",
                artifact_url="https://example.com/model.gguf",
                # No digest
            )
            result = prepare_runtime(candidate, cache=cache)

        assert result.ok is False
        assert "digest" in (result.error or "").lower()

    def test_prepare_result_dataclass(self) -> None:
        """PrepareResult fields are accessible."""
        r = PrepareResult(ok=True, engine_id="mlx-lm", artifact_path=Path("/tmp/test"))
        assert r.ok is True
        assert r.engine_id == "mlx-lm"
        assert r.artifact_path == Path("/tmp/test")
        assert r.error is None
