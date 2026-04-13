"""Manages the invisible local runner lifecycle.

Handles starting, stopping, health-checking, and reusing the background
runner process. Uses a file lock to prevent concurrent starts.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .manifest import _DEFAULT_MANIFEST_PATH, _DEFAULT_TOKEN_PATH, RunnerManifest

logger = logging.getLogger(__name__)

_LOCK_PATH = Path.home() / ".cache" / "octomil" / "local-runner.lock"
_HEALTH_POLL_INTERVAL = 0.3
_HEALTH_POLL_TIMEOUT = 60.0

# ---------------------------------------------------------------------------
# File lock (uses filelock if available, falls back to fcntl)
# ---------------------------------------------------------------------------

try:
    from filelock import FileLock
except ImportError:
    import fcntl

    class FileLock:  # type: ignore[no-redef]
        """Minimal file lock using fcntl."""

        def __init__(self, lock_file: str | Path) -> None:
            self._path = str(lock_file)
            self._fd: int | None = None

        def __enter__(self) -> "FileLock":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self._path, os.O_WRONLY | os.O_CREAT)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, *args: object) -> None:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LocalRunnerHandle:
    """Opaque handle returned by ``ensure()`` for making requests."""

    base_url: str
    token: str
    model: str
    engine: str


@dataclass
class LocalRunnerStatus:
    """Status snapshot of the local runner."""

    running: bool
    pid: int | None = None
    port: int | None = None
    model: str | None = None
    engine: str | None = None
    uptime_seconds: float = 0.0
    idle_timeout_seconds: int = 0
    warm: bool = False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LocalRunnerManager:
    """Manages the invisible local runner lifecycle.

    Thread-safe across processes via a file lock.
    """

    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        token_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        self._manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
        self._token_path = token_path or _DEFAULT_TOKEN_PATH
        self._lock_path = lock_path or _LOCK_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure(
        self,
        *,
        model: str,
        engine: str | None = None,
        idle_timeout: int = 1800,
        restart: bool = False,
    ) -> LocalRunnerHandle:
        """Ensure a compatible runner is running, starting one if needed.

        Parameters
        ----------
        model:
            Model to serve.
        engine:
            Engine override, or None for auto-detect.
        idle_timeout:
            Auto-shutdown after this many seconds of inactivity.
        restart:
            Force a fresh runner even if one is already running.

        Returns
        -------
        LocalRunnerHandle with base_url and token for making requests.
        """
        with FileLock(str(self._lock_path)):
            manifest = RunnerManifest.load(self._manifest_path)

            # If restart requested, stop existing runner first
            if restart and manifest is not None:
                self._kill_runner(manifest)
                RunnerManifest.remove(self._manifest_path)
                manifest = None

            # Check if existing runner is compatible and alive
            if manifest is not None:
                if self._is_compatible(manifest, model, engine) and self._is_alive(manifest):
                    token = self._read_token()
                    return LocalRunnerHandle(
                        base_url=manifest.base_url,
                        token=token,
                        model=manifest.model,
                        engine=manifest.engine,
                    )
                # Incompatible or dead -- clean up
                if self._pid_exists(manifest.pid):
                    self._kill_runner(manifest)
                RunnerManifest.remove(self._manifest_path)

            # Start a new runner
            port = self._find_free_port()
            manifest = self._start_runner(model=model, engine=engine, port=port, idle_timeout=idle_timeout)
            token = self._read_token()
            return LocalRunnerHandle(
                base_url=manifest.base_url,
                token=token,
                model=manifest.model,
                engine=manifest.engine,
            )

    def status(self) -> LocalRunnerStatus:
        """Get current runner status."""
        manifest = RunnerManifest.load(self._manifest_path)
        if manifest is None:
            return LocalRunnerStatus(running=False)

        alive = self._is_alive(manifest)
        if not alive:
            return LocalRunnerStatus(running=False)

        uptime = time.time() - manifest.started_at if manifest.started_at else 0.0
        return LocalRunnerStatus(
            running=True,
            pid=manifest.pid,
            port=manifest.port,
            model=manifest.model,
            engine=manifest.engine,
            uptime_seconds=uptime,
            idle_timeout_seconds=manifest.idle_timeout_seconds,
            warm=alive,
        )

    def stop(self) -> bool:
        """Stop the running runner. Returns True if a runner was stopped."""
        manifest = RunnerManifest.load(self._manifest_path)
        if manifest is None:
            return False

        stopped = False
        if self._pid_exists(manifest.pid):
            self._kill_runner(manifest)
            stopped = True

        RunnerManifest.remove(self._manifest_path)
        # Also clean up token file
        self._token_path.unlink(missing_ok=True)
        return stopped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_alive(self, manifest: RunnerManifest) -> bool:
        """Check if the runner process is alive and responds to /health."""
        if not self._pid_exists(manifest.pid):
            return False
        try:
            import httpx

            resp = httpx.get(f"{manifest.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _is_compatible(self, manifest: RunnerManifest, model: str, engine: str | None) -> bool:
        """Check if the existing runner matches the requested model/engine."""
        if manifest.model != model:
            return False
        if engine is not None and manifest.engine != engine:
            return False
        return True

    def _start_runner(
        self,
        *,
        model: str,
        engine: str | None,
        port: int,
        idle_timeout: int,
    ) -> RunnerManifest:
        """Start a new runner subprocess and wait for it to become healthy."""
        from .server import generate_token

        # Generate token (writes to self._token_path; read back via _read_token)
        generate_token(self._token_path)

        # Build the command
        cmd = [
            sys.executable,
            "-m",
            "octomil",
            "_local-runner-serve",
            "--model",
            model,
            "--port",
            str(port),
            "--token-file",
            str(self._token_path),
            "--idle-timeout",
            str(idle_timeout),
        ]
        if engine:
            cmd.extend(["--engine", engine])

        logger.info("Starting local runner: %s", " ".join(cmd))

        # Start the subprocess (detached, stdout/stderr to devnull)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        base_url = f"http://127.0.0.1:{port}"

        # Write manifest immediately so other processes can see it
        from octomil import __version__

        manifest = RunnerManifest(
            pid=proc.pid,
            port=port,
            base_url=base_url,
            token_file=str(self._token_path),
            model=model,
            engine=engine or "auto",
            started_at=time.time(),
            idle_timeout_seconds=idle_timeout,
            octomil_version=__version__,
        )
        manifest.save(self._manifest_path)

        # Poll /health until ready
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            # Check if process died
            if proc.poll() is not None:
                RunnerManifest.remove(self._manifest_path)
                raise RuntimeError(f"Local runner process exited with code {proc.returncode} before becoming healthy.")
            try:
                import httpx

                resp = httpx.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    logger.info("Local runner healthy on port %d (PID %d).", port, proc.pid)
                    return manifest
            except Exception:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)

        # Timed out -- kill the process
        self._kill_runner(manifest)
        RunnerManifest.remove(self._manifest_path)
        raise RuntimeError(f"Local runner failed to become healthy within {_HEALTH_POLL_TIMEOUT:.0f}s.")

    def _kill_runner(self, manifest: RunnerManifest) -> None:
        """Send SIGTERM to the runner process, then SIGKILL if needed."""
        pid = manifest.pid
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait briefly for graceful shutdown
            for _ in range(20):
                time.sleep(0.1)
                if not self._pid_exists(pid):
                    return
            # Force kill
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning("Permission denied killing PID %d", pid)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        """Check if a process with the given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    @staticmethod
    def _find_free_port() -> int:
        """Find a free TCP port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _read_token(self) -> str:
        """Read the bearer token from the token file."""
        return self._token_path.read_text().strip()
