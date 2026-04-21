"""Cross-platform file locking for artifact downloads.

Prevents concurrent downloads of the same artifact across processes.
Uses fcntl.flock on Unix and msvcrt.locking on Windows.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

_DEFAULT_LOCK_DIR = Path.home() / ".octomil" / "artifacts" / ".locks"


class FileLock:
    """Cross-platform file lock using OS-level advisory locking.

    Usage::

        lock = FileLock("my-artifact-id")
        with lock:
            # download the artifact ...
            pass

    The lock file is created at ``~/.octomil/artifacts/.locks/{name}.lock``
    by default. The lock is released on context exit or explicit ``release()``.
    """

    def __init__(
        self,
        name: str,
        lock_dir: Path | None = None,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> None:
        self._lock_dir = lock_dir or _DEFAULT_LOCK_DIR
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        # Sanitise name for filesystem safety
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        self._lock_path = self._lock_dir / f"{safe_name}.lock"
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def is_locked(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        """Acquire the file lock, blocking up to ``timeout`` seconds."""
        deadline = time.monotonic() + self._timeout
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Open/create the lock file
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT)

        while True:
            try:
                self._platform_lock(fd)
                self._fd = fd
                logger.debug("Acquired lock: %s", self._lock_path)
                return
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(
                        f"Could not acquire lock {self._lock_path} within {self._timeout}s. "
                        f"Another process may be downloading this artifact."
                    )
                time.sleep(self._poll_interval)

    def release(self) -> None:
        """Release the file lock."""
        if self._fd is None:
            return
        try:
            self._platform_unlock(self._fd)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        logger.debug("Released lock: %s", self._lock_path)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Platform-specific locking
    # ------------------------------------------------------------------

    @staticmethod
    def _platform_lock(fd: int) -> None:
        """Non-blocking lock attempt. Raises OSError/BlockingIOError on failure."""
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _platform_unlock(fd: int) -> None:
        """Release the platform lock."""
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
