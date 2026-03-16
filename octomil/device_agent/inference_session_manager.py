"""Thread-safe inference session manager with refcounting.

Pins a model version per request so mid-flight model switches never
affect active inference sessions.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionHandle:
    """Opaque handle representing a pinned inference session."""

    model_id: str
    version: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    acquired_at: float = field(default_factory=time.monotonic)


class InferenceSessionManager:
    """Thread-safe refcounting for active inference sessions.

    Ensures old model versions are retained until all in-flight
    sessions referencing them have completed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (model_id, version) -> set of session_ids
        self._sessions: dict[tuple[str, str], set[str]] = {}
        # session_id -> SessionHandle
        self._handles: dict[str, SessionHandle] = {}

    def acquire(self, model_id: str, version: str) -> SessionHandle:
        """Pin the given model version and return a session handle."""
        handle = SessionHandle(model_id=model_id, version=version)
        key = (model_id, version)
        with self._lock:
            if key not in self._sessions:
                self._sessions[key] = set()
            self._sessions[key].add(handle.session_id)
            self._handles[handle.session_id] = handle
        return handle

    def release(self, handle: SessionHandle) -> None:
        """Release a session handle, decrementing the refcount."""
        key = (handle.model_id, handle.version)
        with self._lock:
            sessions = self._sessions.get(key)
            if sessions and handle.session_id in sessions:
                sessions.discard(handle.session_id)
                if not sessions:
                    del self._sessions[key]
            self._handles.pop(handle.session_id, None)

    def get_refcount(self, model_id: str, version: str) -> int:
        """Return the number of active sessions pinning a specific version."""
        key = (model_id, version)
        with self._lock:
            sessions = self._sessions.get(key)
            return len(sessions) if sessions else 0

    def get_active_sessions(self) -> list[SessionHandle]:
        """Return a snapshot of all active session handles."""
        with self._lock:
            return list(self._handles.values())
