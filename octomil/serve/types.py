"""Core types for the serve package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..telemetry import TelemetryReporter


@dataclass(frozen=True)
class BackendCapabilities:
    """Cutover follow-up #71: declared capability flags for an
    ``InferenceBackend``. Replaces ``isinstance(backend,
    LlamaCppBackend)`` checks in the serve layer.

    Each backend subclass overrides ``InferenceBackend.capabilities``
    with its own instance. Callers query via
    ``backend.capabilities.grammar_supported`` rather than checking
    type identity — type is brittle (e.g., post-cutover the type
    became ``NativeChatBackend`` and every isinstance check broke
    silently for grammar routing) while the capability is the
    actual question being asked.

    Defaults are conservative: a backend that doesn't override is
    assumed not to support grammar / json_mode and to support
    streaming (because most backends do). Subclasses MUST override
    when their actual support diverges.
    """

    # Does the backend handle GBNF grammar internally? When False,
    # the serve layer falls back to system-prompt-based JSON
    # nudging (``_inject_json_system_prompt``) and does NOT pass
    # ``GenerationRequest.grammar`` through.
    grammar_supported: bool = False
    # Does the backend handle ``json_mode`` natively (via grammar
    # constraint or other mechanism)? Independent of
    # ``grammar_supported`` because some backends might constrain
    # JSON via a different mechanism (e.g., guided decoding).
    json_mode_supported: bool = False
    # Does the backend support streaming via ``generate_stream``?
    # When False, the serve layer rejects ``stream=True`` requests
    # rather than silently degrading to non-streaming.
    streaming_supported: bool = True
    # Does the backend support OpenAI-style tool / function calling?
    # v0.1.2 native is False; legacy LlamaCppBackend is False
    # (grammar-based tool routing is a different surface).
    tools_supported: bool = False
    # Free-form attention-backend identity. Kept for telemetry /
    # verbose-event payloads. Mirrors the existing
    # ``InferenceBackend.attention_backend`` class attribute; the
    # capabilities object carries it so a refactor that moves the
    # attribute keeps both views in sync.
    attention_backend: str = "standard"

    def supports(self, feature: str) -> bool:
        """Generic accessor for a feature flag by name. Returns
        False for unknown flags (forward-compat: a future feature
        flag the binding doesn't yet know about defaults to "no")."""
        return bool(getattr(self, feature, False))


# Sentinel default. Backends that don't override get conservative
# False/False/True/False/standard.
_DEFAULT_BACKEND_CAPABILITIES = BackendCapabilities()


def resolve_backend_capabilities(backend: object) -> BackendCapabilities:
    """Cutover follow-up #71 (R4 Codex): return a backend's declared
    ``capabilities``, falling back to conservative defaults for duck-
    typed backends that don't subclass ``InferenceBackend``.

    Some engine plugins (e.g. ORT, Ollama) construct duck-typed backends
    without inheriting from ``InferenceBackend``. ``backend.capabilities``
    on those raises ``AttributeError`` even though they implement the
    rest of the protocol. Callers in the serve layer use this helper to
    avoid breaking when a duck-typed backend reaches a capability query.
    Defensive insurance — every shipped backend SHOULD declare
    capabilities, but the helper makes refactor mistakes graceful.
    """
    return getattr(backend, "capabilities", _DEFAULT_BACKEND_CAPABILITIES)


@dataclass
class GenerationRequest:
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    grammar: Optional[str] = None
    json_mode: bool = False
    enable_thinking: Optional[bool] = None


@dataclass
class GenerationChunk:
    text: str
    token_count: int = 0
    tokens_per_second: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""

    ttfc_ms: float = 0.0
    prompt_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    total_duration_ms: float = 0.0
    cache_hit: bool = False
    attention_backend: str = "standard"
    early_exit_tokens: int = 0
    avg_layers_used: float = 0.0
    # Cutover follow-up #73: cache + latency telemetry surfaced from
    # the runtime's CACHE_HIT / CACHE_MISS / SESSION_COMPLETED events.
    # Default 0 so existing call sites that don't populate them stay
    # backwards-compatible.
    cache_hits: int = 0
    cache_misses: int = 0
    cache_saved_tokens: int = 0
    queued_ms: float = 0.0
    setup_ms: float = 0.0
    engine_first_chunk_ms: float = 0.0


class InferenceBackend:
    """Base class for inference backends.

    Provides a shared ``ThreadPoolExecutor`` for sync->async bridging and
    a ``warmup()`` hook that subclasses can override to pre-compile GPU
    kernels and prime caches at model load time.

    Capability declarations: subclasses override the
    ``capabilities`` class attribute with a ``BackendCapabilities``
    instance describing what the backend actually supports. Callers
    in the serve layer query
    ``backend.capabilities.grammar_supported`` etc. rather than
    checking ``isinstance(_, LlamaCppBackend)``, which broke after
    the v0.1.2 hard-cutover when the chat path's class identity
    changed to ``NativeChatBackend`` (cutover follow-up #71).
    """

    name: str = "base"
    attention_backend: str = "standard"
    # Class-level default; subclasses override.
    capabilities: ClassVar[BackendCapabilities] = _DEFAULT_BACKEND_CAPABILITIES

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self.name}-inference",
        )
        # Pre-spawn the executor thread so the first streaming request
        # doesn't pay OS thread creation cost (~50ms measured).
        self._executor.submit(lambda: None).result()

    def load_model(self, model_name: str) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        """Override to run a cheap prefill after model load.

        This compiles GPU shaders/kernels and primes caches so the first
        real request doesn't pay cold-start cost.  Called automatically
        at the end of ``load_model`` in subclasses that opt in.
        """

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        raise NotImplementedError

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
        yield  # pragma: no cover -- makes this an async generator

    def get_verbose_metadata(
        self,
        event_name: str,
        *,
        request: GenerationRequest | None = None,
        metrics: InferenceMetrics | None = None,
    ) -> dict[str, Any]:
        """Override to provide engine-specific metadata for verbose events."""
        return {}

    def list_models(self) -> list[str]:
        raise NotImplementedError


@runtime_checkable
class StreamableState(Protocol):
    """Protocol satisfied by both ServerState and _MultiModelStateAdapter.

    Used as the type for the ``state`` parameter in streaming functions
    to break the circular dependency between streaming.py and multi_model.py.
    """

    backend: Optional[InferenceBackend]
    reporter: Optional["TelemetryReporter"]
