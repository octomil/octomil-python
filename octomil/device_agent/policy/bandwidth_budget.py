"""Token bucket bandwidth budgeting for device agent I/O."""

from __future__ import annotations

import threading
import time


class BandwidthBudget:
    """Simple token bucket rate limiter.

    Tokens represent bytes.  The bucket is refilled at *refill_rate*
    bytes per second up to *max_rate* bytes.  ``consume()`` returns
    ``True`` and deducts tokens if sufficient budget is available.
    """

    def __init__(self, max_rate: int, refill_rate: float, name: str = "") -> None:
        self._max_rate = max_rate
        self._refill_rate = refill_rate
        self._name = name
        self._tokens = float(max_rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_rate, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def consume(self, bytes_count: int) -> bool:
        """Attempt to consume *bytes_count* tokens.

        Returns ``True`` if sufficient budget was available.
        """
        with self._lock:
            self._refill()
            if self._tokens >= bytes_count:
                self._tokens -= bytes_count
                return True
            return False

    def available(self) -> int:
        """Return the number of bytes currently available."""
        with self._lock:
            self._refill()
            return int(self._tokens)

    def reset(self) -> None:
        """Refill the bucket to maximum capacity."""
        with self._lock:
            self._tokens = float(self._max_rate)
            self._last_refill = time.monotonic()

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_rate(self) -> int:
        return self._max_rate


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_1MB = 1_048_576
_64KB = 65_536
_512KB = 524_288
_UNLIMITED = 1_073_741_824  # 1 GB — effectively unlimited


def foreground_download_budget() -> BandwidthBudget:
    """Download budget while inference is active: max 1 MB/s."""
    return BandwidthBudget(max_rate=_1MB, refill_rate=_1MB, name="fg-download")


def foreground_upload_budget() -> BandwidthBudget:
    """Upload budget while inference is active: max 64 KB/s."""
    return BandwidthBudget(max_rate=_64KB, refill_rate=_64KB, name="fg-upload")


def idle_download_budget() -> BandwidthBudget:
    """Download budget when idle: effectively unlimited."""
    return BandwidthBudget(max_rate=_UNLIMITED, refill_rate=_UNLIMITED, name="idle-download")


def idle_upload_budget() -> BandwidthBudget:
    """Upload budget when idle: 512 KB/s."""
    return BandwidthBudget(max_rate=_512KB, refill_rate=_512KB, name="idle-upload")
