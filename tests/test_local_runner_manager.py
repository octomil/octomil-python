"""Tests for the invisible local runner manager and manifest."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import patch

from octomil.local_runner.manager import LocalRunnerManager
from octomil.local_runner.manifest import RunnerManifest

# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestRunnerManifest:
    def test_save_and_load(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        m = RunnerManifest(
            pid=12345,
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file="/tmp/tok",
            model="gemma-1b",
            engine="mlx-lm",
            started_at=1000.0,
        )
        m.save(manifest_path)

        loaded = RunnerManifest.load(manifest_path)
        assert loaded is not None
        assert loaded.pid == 12345
        assert loaded.port == 51200
        assert loaded.model == "gemma-1b"
        assert loaded.engine == "mlx-lm"
        assert loaded.started_at == 1000.0

    def test_load_missing(self, tmp_path: Path) -> None:
        assert RunnerManifest.load(tmp_path / "nonexistent.json") is None

    def test_load_corrupt(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json!!!")
        assert RunnerManifest.load(p) is None

    def test_load_ignores_extra_keys(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        data = {
            "pid": 1,
            "port": 2,
            "base_url": "http://127.0.0.1:2",
            "token_file": "/t",
            "model": "m",
            "engine": "e",
            "some_future_field": True,
        }
        p.write_text(json.dumps(data))
        loaded = RunnerManifest.load(p)
        assert loaded is not None
        assert loaded.model == "m"

    def test_remove(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        p.write_text("{}")
        RunnerManifest.remove(p)
        assert not p.exists()

    def test_remove_missing(self, tmp_path: Path) -> None:
        RunnerManifest.remove(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Manager tests
# ---------------------------------------------------------------------------


class TestLocalRunnerManager:
    def _make_manager(self, tmp_path: Path) -> LocalRunnerManager:
        return LocalRunnerManager(
            manifest_path=tmp_path / "manifest.json",
            token_path=tmp_path / "token",
            lock_path=tmp_path / "lock",
        )

    def test_status_no_runner(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        status = mgr.status()
        assert status.running is False
        assert status.pid is None

    def test_status_dead_pid(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        manifest = RunnerManifest(
            pid=999999999,  # almost certainly not alive
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(tmp_path / "token"),
            model="gemma-1b",
            engine="auto",
            started_at=time.time(),
        )
        manifest.save(tmp_path / "manifest.json")

        status = mgr.status()
        assert status.running is False

    def test_stop_no_runner(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        result = mgr.stop()
        assert result is False

    def test_stop_cleans_manifest_and_token(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        token_path = tmp_path / "token"

        # Write a manifest with a dead pid
        manifest = RunnerManifest(
            pid=999999999,
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="auto",
        )
        manifest.save(manifest_path)
        token_path.write_text("tok123")

        mgr.stop()
        assert not manifest_path.exists()
        assert not token_path.exists()

    @patch("octomil.local_runner.manager.LocalRunnerManager._is_alive", return_value=True)
    @patch("octomil.local_runner.manager.LocalRunnerManager._pid_exists", return_value=True)
    def test_ensure_reuses_compatible_runner(self, mock_pid, mock_alive, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        token_path = tmp_path / "token"
        token_path.write_text("tok123")

        manifest = RunnerManifest(
            pid=os.getpid(),  # current process as standin
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="auto",
            started_at=time.time(),
        )
        manifest.save(tmp_path / "manifest.json")

        handle = mgr.ensure(model="gemma-1b")
        assert handle.base_url == "http://127.0.0.1:51200"
        assert handle.token == "tok123"
        assert handle.model == "gemma-1b"

    def test_ensure_detects_incompatible_model(self, tmp_path: Path) -> None:
        """When the running runner has a different model, _is_compatible returns False."""
        mgr = self._make_manager(tmp_path)
        manifest = RunnerManifest(
            pid=os.getpid(),
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(tmp_path / "token"),
            model="gemma-1b",
            engine="auto",
        )
        assert not mgr._is_compatible(manifest, "phi-4-mini", None)
        assert mgr._is_compatible(manifest, "gemma-1b", None)
        assert mgr._is_compatible(manifest, "gemma-1b", "auto")
        assert not mgr._is_compatible(manifest, "gemma-1b", "llama.cpp")

    def test_ensure_restarts_on_model_mismatch(self, tmp_path: Path) -> None:
        """When model differs, ensure should stop old runner and start new one."""
        mgr = self._make_manager(tmp_path)
        token_path = tmp_path / "token"
        token_path.write_text("tok123")

        manifest = RunnerManifest(
            pid=os.getpid(),
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="auto",
            started_at=time.time(),
        )
        manifest.save(tmp_path / "manifest.json")

        # _is_alive returns True for old runner, _start_runner returns new manifest
        with (
            patch.object(mgr, "_is_alive", return_value=True),
            patch.object(mgr, "_kill_runner") as mock_kill,
            patch.object(mgr, "_start_runner") as mock_start,
            patch.object(mgr, "_find_free_port", return_value=51201),
            patch.object(mgr, "_read_token", return_value="newtok"),
        ):
            new_manifest = RunnerManifest(
                pid=99999,
                port=51201,
                base_url="http://127.0.0.1:51201",
                token_file=str(token_path),
                model="phi-4-mini",
                engine="auto",
            )
            mock_start.return_value = new_manifest

            handle = mgr.ensure(model="phi-4-mini")
            assert handle.model == "phi-4-mini"
            mock_kill.assert_called_once()
            mock_start.assert_called_once()

    def test_ensure_restarts_on_engine_mismatch(self, tmp_path: Path) -> None:
        """When engine differs, ensure should stop old runner and start new one."""
        mgr = self._make_manager(tmp_path)
        token_path = tmp_path / "token"
        token_path.write_text("tok123")

        manifest = RunnerManifest(
            pid=os.getpid(),
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="mlx-lm",
            started_at=time.time(),
        )
        manifest.save(tmp_path / "manifest.json")

        with (
            patch.object(mgr, "_is_alive", return_value=True),
            patch.object(mgr, "_kill_runner") as mock_kill,
            patch.object(mgr, "_start_runner") as mock_start,
            patch.object(mgr, "_find_free_port", return_value=51201),
            patch.object(mgr, "_read_token", return_value="newtok"),
        ):
            new_manifest = RunnerManifest(
                pid=99999,
                port=51201,
                base_url="http://127.0.0.1:51201",
                token_file=str(token_path),
                model="gemma-1b",
                engine="llama.cpp",
            )
            mock_start.return_value = new_manifest

            handle = mgr.ensure(model="gemma-1b", engine="llama.cpp")
            assert handle.engine == "llama.cpp"
            mock_kill.assert_called_once()

    def test_ensure_restarts_on_stale_pid(self, tmp_path: Path) -> None:
        """When the PID is dead, ensure should start a new runner."""
        mgr = self._make_manager(tmp_path)
        token_path = tmp_path / "token"
        token_path.write_text("tok123")

        manifest = RunnerManifest(
            pid=999999999,  # dead PID
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="auto",
            started_at=time.time(),
        )
        manifest.save(tmp_path / "manifest.json")

        with (
            patch.object(mgr, "_start_runner") as mock_start,
            patch.object(mgr, "_find_free_port", return_value=51201),
            patch.object(mgr, "_read_token", return_value="newtok"),
        ):
            new_manifest = RunnerManifest(
                pid=88888,
                port=51201,
                base_url="http://127.0.0.1:51201",
                token_file=str(token_path),
                model="gemma-1b",
                engine="auto",
            )
            mock_start.return_value = new_manifest

            handle = mgr.ensure(model="gemma-1b")
            assert handle.base_url == "http://127.0.0.1:51201"
            mock_start.assert_called_once()

    @patch("octomil.local_runner.manager.LocalRunnerManager._is_alive", return_value=True)
    @patch("octomil.local_runner.manager.LocalRunnerManager._pid_exists", return_value=True)
    def test_status_running(self, mock_pid, mock_alive, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        manifest = RunnerManifest(
            pid=os.getpid(),
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(tmp_path / "token"),
            model="gemma-1b",
            engine="mlx-lm",
            started_at=time.time() - 60,
            idle_timeout_seconds=1800,
        )
        manifest.save(tmp_path / "manifest.json")

        status = mgr.status()
        assert status.running is True
        assert status.pid == os.getpid()
        assert status.port == 51200
        assert status.model == "gemma-1b"
        assert status.engine == "mlx-lm"
        assert status.uptime_seconds >= 59
        assert status.idle_timeout_seconds == 1800
        assert status.warm is True

    def test_find_free_port(self) -> None:
        port = LocalRunnerManager._find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    @patch.object(LocalRunnerManager, "_start_runner")
    @patch.object(LocalRunnerManager, "_find_free_port", return_value=51200)
    def test_ensure_starts_runner_when_no_manifest(self, mock_port, mock_start, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        token_path = tmp_path / "token"
        token_path.write_text("newtok")

        new_manifest = RunnerManifest(
            pid=12345,
            port=51200,
            base_url="http://127.0.0.1:51200",
            token_file=str(token_path),
            model="gemma-1b",
            engine="auto",
        )
        mock_start.return_value = new_manifest

        handle = mgr.ensure(model="gemma-1b")
        assert handle.base_url == "http://127.0.0.1:51200"
        assert handle.model == "gemma-1b"
        mock_start.assert_called_once()


class TestTokenSecurity:
    def test_token_file_permissions(self, tmp_path: Path) -> None:
        from octomil.local_runner.server import generate_token

        token_path = tmp_path / "token"
        token = generate_token(token_path)

        assert token_path.exists()
        assert len(token) > 32  # token_urlsafe(48) produces >32 chars
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600, f"Token file should be 0600, got {oct(mode)}"

    def test_token_is_random(self, tmp_path: Path) -> None:
        from octomil.local_runner.server import generate_token

        t1 = generate_token(tmp_path / "tok1")
        t2 = generate_token(tmp_path / "tok2")
        assert t1 != t2


class TestIdleWatchdog:
    def test_touch_resets_timer(self) -> None:
        from octomil.local_runner.server import _IdleWatchdog

        wd = _IdleWatchdog(idle_timeout=100)
        wd.touch()
        # No assertion needed beyond no crash — the touch should reset monotonic timestamp

    def test_watchdog_starts_and_stops(self) -> None:
        from octomil.local_runner.server import _IdleWatchdog

        wd = _IdleWatchdog(idle_timeout=9999)
        wd.start()
        assert wd._thread is not None
        assert wd._thread.is_alive()
        wd.stop()
        assert not wd._thread.is_alive()
