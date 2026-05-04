"""NativeChatBackend — chat.completion via octomil-runtime v0.1.2+.

Hard-cutover backend for local GGUF chat. Replaces the v0.1.x
Python-local ``LlamaCppBackend`` for product `chat.completion`
flow when the planner selects ``native-llama-cpp`` as the engine.

Lifecycle:
- One ``NativeChatBackend`` instance per (planner-selected) engine.
- ``load_model(model_name)`` resolves the GGUF path (from the
  PrepareManager-materialized ``model_dir`` first, then from a
  bare ``*.gguf`` ``model_name``), opens a ``NativeRuntime`` and a
  warmed ``NativeModel``, and caches both on the instance. The
  cached model is reused across requests on this backend instance.
- ``generate(request)`` opens one session per request, calls
  ``send_chat(...)`` with the request's options, drains the event
  stream, closes the session deterministically. The model handle
  stays warm.
- ``close()`` closes the model + runtime in correct order.

Hard rules (per the cutover spec):
1. No silent fallback to the Python-local llama_cpp path. If the
   request asks for a feature native doesn't support, raise a
   bounded ``OctomilError`` — the planner should not have selected
   native for that request, and execution-time fallback would
   re-introduce the coupling the cutover removes.
2. Native supports the v0.1.2 subset only:
     - messages with roles ∈ {system, user, assistant}
     - max_tokens / max_completion_tokens (1..n_ctx=4096)
     - temperature (0.0 only — greedy)
     - top_p (1.0 only — greedy)
   Anything else is a bounded ``INVALID_INPUT`` (caller-misuse) or
   ``UNSUPPORTED_MODALITY`` (capability gap).
3. Streaming (chat.stream) is not implemented in v0.1.2 and is
   gated out by ``generate_stream`` raising ``UNSUPPORTED_MODALITY``.
   A streaming-aware variant ships when the runtime adds it.
4. Artifact path comes from the PrepareManager's ``model_dir`` —
   no ``OCTOMIL_LLAMA_CPP_GGUF`` requirement on the product flow.

Bounded-error mapping (runtime → SDK):
- ``OCT_STATUS_NOT_FOUND`` (model file missing) →
  ``MODEL_NOT_FOUND``.
- ``OCT_STATUS_INVALID_INPUT`` →
  ``INVALID_INPUT``.
- ``OCT_STATUS_UNSUPPORTED`` →
  ``UNSUPPORTED_MODALITY``.
- ``OCT_STATUS_VERSION_MISMATCH`` →
  ``RUNTIME_UNAVAILABLE`` (binding/dylib skew).
- ``OCT_STATUS_BUSY`` →
  ``SERVER_ERROR`` (shouldn't reach the chat path; defensive).
- Any other / non-OK terminal_status →
  ``INFERENCE_FAILED``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator

from ...errors import OctomilError, OctomilErrorCode
from ...serve.types import (
    GenerationChunk,
    GenerationRequest,
    InferenceBackend,
    InferenceMetrics,
)
from .loader import (
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_TRANSCRIPT_CHUNK,
    OCT_STATUS_BUSY,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_OK,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)

# Roles the v0.1.2 runtime accepts on chat.completion.
_SUPPORTED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})

# Engine identity surfaced through `InferenceBackend.name` and
# benchmarks. Distinct from the legacy "llama.cpp" name so logs /
# metrics / planner candidate lists make the cutover visible.
_BACKEND_NAME = "native-llama-cpp"


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Map a runtime ``oct_status_t`` to the SDK's bounded error
    taxonomy. Used both at construction (model_open / warm) and at
    request time (session_open / send_text terminal status)."""
    if status == OCT_STATUS_NOT_FOUND:
        code = OctomilErrorCode.MODEL_NOT_FOUND
    elif status == OCT_STATUS_INVALID_INPUT:
        code = OctomilErrorCode.INVALID_INPUT
    elif status == OCT_STATUS_UNSUPPORTED:
        code = OctomilErrorCode.UNSUPPORTED_MODALITY
    elif status == OCT_STATUS_VERSION_MISMATCH:
        code = OctomilErrorCode.RUNTIME_UNAVAILABLE
    elif status == OCT_STATUS_BUSY:
        code = OctomilErrorCode.SERVER_ERROR
    else:
        code = OctomilErrorCode.INFERENCE_FAILED
    full_message = message
    if last_error:
        full_message = f"{message}: {last_error}"
    return OctomilError(code=code, message=full_message)


def _validate_messages_for_native(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Pre-flight the messages list against the v0.1.2 native
    subset. Raises bounded ``OctomilError(INVALID_INPUT)`` on any
    shape outside the cutover scope.

    The runtime would also reject these shapes via the event stream,
    but doing it Python-side lets us:
      - Surface a precise diagnostic before paying session_open cost.
      - Avoid wasted round-trips for obvious caller misuse.
      - Keep the "no silent fallback" contract — the planner must not
        have selected native for these requests; if execution sees
        them, it's a planner bug we want to surface loudly.
    """
    if not messages:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="messages is empty; chat.completion requires at least one message",
        )
    cleaned: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"messages[{i}] is not a dict",
            )
        if set(msg.keys()) - {"role", "content"}:
            extras = sorted(set(msg.keys()) - {"role", "content"})
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"messages[{i}] has unsupported keys {extras}; "
                    f"native chat.completion accepts only 'role' and 'content'"
                ),
            )
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _SUPPORTED_ROLES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"messages[{i}].role={role!r} is not supported; "
                    f"native chat.completion supports {sorted(_SUPPORTED_ROLES)}"
                ),
            )
        if not isinstance(content, str):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"messages[{i}].content must be a string",
            )
        cleaned.append({"role": role, "content": content})
    return cleaned


def _gate_unsupported_request_features(request: GenerationRequest) -> None:
    """Reject request shapes the v0.1.2 native runtime cannot serve.
    The planner SHOULD have routed these elsewhere; if execution
    sees them, raising bounded errors is the documented
    "no silent fallback" contract.
    """
    if request.grammar:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                "GBNF grammar is not supported by the native chat backend in v0.1.2. "
                "Planner should not have selected the native route for grammar requests; "
                "configure routing policy to disable native for this model or use a "
                "non-grammar request."
            ),
        )
    if request.json_mode:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message="json_mode is not supported by the native chat backend in v0.1.2",
        )
    # request.enable_thinking is a model-side feature (chain-of-thought
    # tag handling) — not part of the v0.1.2 native subset. Reject
    # explicitly so callers don't assume it works.
    if request.enable_thinking is not None:
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message="enable_thinking is not supported by the native chat backend in v0.1.2",
        )
    # NOTE on temperature / top_p: v0.1.2 ships greedy-only on the
    # runtime side, but the SDK's GenerationRequest defaults
    # temperature=0.7 and top_p=1.0 — meaning every default-shape
    # /v1/chat/completions and Responses call would reject if we
    # passed those defaults through to the runtime. Codex R1 P1.
    # Resolution: NativeChatBackend.generate() does NOT forward
    # temperature/top_p to send_chat — the runtime applies its
    # greedy default. A logger.warning is emitted when the request's
    # values differ from greedy so callers see the coercion
    # without a hard failure on the default product path. Per the
    # cutover spec rules (4): the user listed grammar / tools /
    # top_k / unsupported roles / streaming as the bounded-error
    # cases; temperature/top_p are sampling parameters the cutover
    # explicitly subsets to greedy, not unsupported features.


class NativeChatBackend(InferenceBackend):
    """Hard-cutover chat.completion backend backed by
    ``octomil-runtime v0.1.2+``.

    Caches a ``NativeRuntime`` + warmed ``NativeModel`` per backend
    instance. Each request opens a fresh session against the cached
    model, calls ``send_chat(...)``, drains events, closes session.
    """

    name: str = _BACKEND_NAME
    attention_backend: str = "native"

    def __init__(self, *, model_dir: str | None = None) -> None:
        super().__init__()
        self._model_dir: str | None = model_dir
        self._model_name: str = ""
        self._gguf_path: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None  # NativeModel

    # ------------------------------------------------------------------
    # GGUF resolution — mirrors LlamaCppBackend._resolve_local_gguf_file
    # so PrepareManager-materialized artifacts work identically. The
    # only difference: we also accept a bare ``model_name`` ending in
    # ``.gguf`` for tests / dev tooling. Product flow ALWAYS goes
    # through model_dir; OCTOMIL_LLAMA_CPP_GGUF is NEVER read here.
    # ------------------------------------------------------------------
    def _resolve_gguf_path(self, model_name: str) -> str:
        if self._model_dir:
            sentinel = os.path.join(self._model_dir, "artifact")
            if os.path.isfile(sentinel):
                return sentinel
            if os.path.isdir(self._model_dir):
                for entry in sorted(os.listdir(self._model_dir)):
                    if entry.lower().endswith(".gguf"):
                        return os.path.join(self._model_dir, entry)
        # No injected model_dir — accept bare *.gguf for dev/tests.
        # Product flow shouldn't hit this branch (the PrepareManager
        # always sets model_dir before the engine is constructed).
        if model_name.endswith(".gguf") and os.path.isfile(model_name):
            return model_name
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=(
                f"native chat backend could not resolve a GGUF for model {model_name!r}. "
                f"Expected a PrepareManager-materialized artifact via model_dir; got "
                f"{self._model_dir!r}."
            ),
        )

    def load_model(self, model_name: str) -> None:
        if self._runtime is not None:
            # Idempotent: once a model is warmed on this instance
            # the planner reuses it across requests.
            return
        self._model_name = model_name
        self._gguf_path = self._resolve_gguf_path(model_name)
        logger.info(
            "NativeChatBackend: loading GGUF via octomil-runtime native: %s",
            self._gguf_path,
        )
        try:
            self._runtime = NativeRuntime.open()
            self._model = self._runtime.open_model(model_uri=self._gguf_path)
            self._model.warm()
        except NativeRuntimeError as exc:
            # Roll back any partial state so a retry can re-open cleanly.
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                f"native chat backend failed to load {self._gguf_path}",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

    def warmup(self) -> None:
        """Already-warmed at load_model. Noop."""

    def close(self) -> None:
        """Tear down the cached model + runtime in the right order
        (sessions are per-request and already closed deterministically;
        only the model + runtime live across requests)."""
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001 — best-effort drain
                logger.warning("NativeChatBackend.close: model.close failed", exc_info=True)
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001 — best-effort drain
                logger.warning("NativeChatBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Inference paths
    # ------------------------------------------------------------------
    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        if self._runtime is None or self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeChatBackend.generate called before load_model",
            )
        _gate_unsupported_request_features(request)
        clean_messages = _validate_messages_for_native(request.messages)

        try:
            sess = self._runtime.open_session(
                capability="chat.completion",
                locality="on_device",
                policy_preset="private",
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native chat backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        try:
            t_send = time.monotonic()
            # Codex R1 P1: do NOT forward request.temperature /
            # request.top_p to send_chat — v0.1.2 ships greedy-only
            # and the SDK's GenerationRequest defaults
            # (temperature=0.7, top_p=1.0) would reject every
            # default-shape product call. Runtime applies its
            # greedy default when the options are omitted. We emit
            # a diagnostic warning if the request's values differ
            # so coercion is observable without a hard failure on
            # the default path.
            if request.temperature not in (0.0, 0):
                logger.warning(
                    "NativeChatBackend: v0.1.2 ships greedy-only; " "ignoring request.temperature=%s (treating as 0.0)",
                    request.temperature,
                )
            if request.top_p not in (1.0, 1):
                logger.warning(
                    "NativeChatBackend: v0.1.2 ships greedy-only; " "ignoring request.top_p=%s (treating as 1.0)",
                    request.top_p,
                )
            try:
                sess.send_chat(
                    clean_messages,
                    max_tokens=int(request.max_tokens),
                )
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native chat backend send_chat failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            # Drain events. The runtime emits one TRANSCRIPT_CHUNK per
            # generated token; SESSION_COMPLETED carries the bounded
            # terminal status. ERROR events surface the bounded
            # error_code (we map to OctomilError on terminal).
            assembled: list[str] = []
            n_chunks = 0
            ttfc_ms: float = 0.0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            # Hard wall on session lifetime so a runaway model can't
            # block the request thread forever. 5 minutes is generous
            # for any reasonable chat completion.
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                ev = sess.poll_event(timeout_ms=200)
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_CHUNK:
                    if n_chunks == 0:
                        ttfc_ms = (time.monotonic() - t_send) * 1000.0
                    n_chunks += 1
                    if ev.text:
                        assembled.append(ev.text)
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    saw_error = True
                    # Codex R1 P1: read last_error IMMEDIATELY on
                    # OCT_EVENT_ERROR so a subsequent poll can't
                    # overwrite the diagnostic. The runtime emits
                    # ERROR before SESSION_COMPLETED in the rejected
                    # path; capturing here keeps the message bound
                    # to the originating event.
                    if not error_message:
                        error_message = self._runtime.last_error()
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    # Codex R1 P1: read the TYPED terminal_status
                    # from the event (NativeEvent now exposes it).
                    # An UNSUPPORTED reject from the runtime (e.g.,
                    # non-greedy temperature) maps to UNSUPPORTED_MODALITY,
                    # NOT INVALID_INPUT.
                    terminal_status = int(ev.terminal_status)
                    break
            else:
                # Loop exited via the deadline without seeing
                # SESSION_COMPLETED.
                raise OctomilError(
                    code=OctomilErrorCode.REQUEST_TIMEOUT,
                    message="native chat backend timed out waiting for SESSION_COMPLETED",
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native chat backend reported error during generation",
                    last_error=error_message,
                )

            elapsed = time.monotonic() - t_send
            text = "".join(assembled)
            tps = (n_chunks / elapsed) if elapsed > 0 else 0.0
            metrics = InferenceMetrics(
                ttfc_ms=ttfc_ms,
                prompt_tokens=0,  # not exposed by v0.1.2 envelope
                total_tokens=n_chunks,
                tokens_per_second=tps,
                total_duration_ms=elapsed * 1000.0,
                attention_backend=self.attention_backend,
            )
            return text, metrics
        finally:
            sess.close()

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Streaming chat is the ``chat.stream`` capability — NOT
        implemented in v0.1.2's runtime. The planner MUST NOT route
        streaming requests to native until that capability ships.
        Calling this raises ``UNSUPPORTED_MODALITY`` rather than
        falling back to the Python-local path (no silent fallback)."""
        raise OctomilError(
            code=OctomilErrorCode.UNSUPPORTED_MODALITY,
            message=(
                "Streaming chat (chat.stream) is not implemented in "
                "octomil-runtime v0.1.2. The planner should route "
                "stream=True requests to a streaming-capable engine; "
                "the native chat backend will gain streaming when the "
                "runtime adds the chat.stream capability."
            ),
        )
        yield  # pragma: no cover — async-generator marker

    def list_models(self) -> list[str]:
        """The v0.1.2 native runtime doesn't enumerate models; the
        planner-side catalog is the source of truth. Return the
        currently loaded model only (or empty if not loaded)."""
        return [self._model_name] if self._model_name else []

    def get_verbose_metadata(
        self,
        event_name: str,
        *,
        request: GenerationRequest | None = None,
        metrics: InferenceMetrics | None = None,
    ) -> dict[str, Any]:
        """Surface the cutover identity in verbose telemetry events."""
        meta: dict[str, Any] = {
            "backend": self.name,
            "runtime": "octomil-runtime",
            "engine": "llama_cpp",
            "gguf_path": self._gguf_path,
        }
        if metrics is not None:
            meta["ttfc_ms"] = metrics.ttfc_ms
            meta["total_tokens"] = metrics.total_tokens
            meta["tokens_per_second"] = metrics.tokens_per_second
        return meta
