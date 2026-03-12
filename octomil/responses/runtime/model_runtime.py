"""ModelRuntime protocol — typed interface for on-device inference."""

from __future__ import annotations

import abc
from typing import AsyncIterator, Callable, Optional

from .types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse


class ModelRuntime(abc.ABC):
    """Typed interface for on-device model inference (Layer 1).

    Each concrete runtime wraps a specific engine and exposes
    a uniform request/response API.
    """

    @property
    @abc.abstractmethod
    def capabilities(self) -> RuntimeCapabilities: ...

    @abc.abstractmethod
    async def run(self, request: RuntimeRequest) -> RuntimeResponse: ...

    @abc.abstractmethod
    def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]: ...

    def close(self) -> None:
        """Release resources held by this runtime."""


RuntimeFactory = Callable[[str], Optional[ModelRuntime]]
