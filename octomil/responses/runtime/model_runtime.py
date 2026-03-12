"""ModelRuntime protocol — typed interface for on-device inference."""

from __future__ import annotations

import abc
from typing import AsyncIterator, Callable, Optional

from .types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse


class ModelRuntime(abc.ABC):
    """Any execution backend that satisfies the run/stream interface -- local or remote.

    ModelRuntime is the foundational abstraction in the Octomil inference
    stack (Layer 1).  It represents any execution backend -- a local
    engine (MLX, llama.cpp, ONNX Runtime, Core ML, etc.), a remote cloud
    API, or a custom user-provided backend -- as long as it implements
    the ``run()`` and ``stream()`` methods with the prescribed
    ``RuntimeRequest -> RuntimeResponse / AsyncIterator[RuntimeChunk]``
    contract.

    Concrete implementations are registered with ``ModelRuntimeRegistry``
    and resolved at request time by model name.  The higher-level
    ``OctomilResponses`` layer (Layer 2) consumes runtimes through this
    interface, meaning new backends can be added without modifying any
    orchestration code.

    Subclasses MUST implement:
        - ``capabilities`` property  -- advertise supported features
        - ``run(request)``           -- single-shot inference
        - ``stream(request)``        -- streaming token-by-token inference

    Subclasses MAY override:
        - ``close()``                -- release GPU memory / file handles
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
