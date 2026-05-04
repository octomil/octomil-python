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
3. Streaming (chat.stream) is supported as of cutover follow-up #72:
   ``generate_stream`` relays the runtime's per-token TRANSCRIPT_CHUNK
   events as ``GenerationChunk`` instances, with a final
   ``finish_reason="stop"`` marker on SESSION_COMPLETED.
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
    BackendCapabilities,
    GenerationChunk,
    GenerationRequest,
    InferenceBackend,
    InferenceMetrics,
)
from .loader import (
    OCT_EVENT_CACHE_HIT,
    OCT_EVENT_CACHE_MISS,
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
    # Cutover follow-up #71: v0.1.2 native subset is greedy-only,
    # no grammar/tools/streaming. Every reject path raises
    # OctomilError(UNSUPPORTED_MODALITY) — see
    # _gate_unsupported_request_features. The serve layer reads
    # these flags to decide whether to pass grammar through or
    # fall back to system-prompt JSON nudging.
    capabilities = BackendCapabilities(
        grammar_supported=False,
        json_mode_supported=False,
        # Cutover follow-up #72: chat.stream landed. The runtime emits
        # one TRANSCRIPT_CHUNK event per generated token; generate_stream
        # now relays them as ``GenerationChunk`` instances rather than
        # accumulating into a single string.
        streaming_supported=True,
        tools_supported=False,
        attention_backend="native",
    )

    # Cutover follow-up #74: configurable per-request deadline.
    # 5 minutes is the backend-wide default; overridable per
    # construction (test/CI tuning) or per request via
    # ``GenerationRequest.deadline_ms``.
    DEFAULT_DEADLINE_MS: int = 300_000

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_deadline_ms: int | None = None,
    ) -> None:
        super().__init__()
        self._model_dir: str | None = model_dir
        self._model_name: str = ""
        self._gguf_path: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None  # NativeModel
        # Cutover follow-up #74: instance default for the per-request
        # deadline. Tests / CI inject a smaller value to exercise the
        # timeout path without waiting 5 minutes.
        self._default_deadline_ms: int = (
            default_deadline_ms if default_deadline_ms is not None else self.DEFAULT_DEADLINE_MS
        )

    def _resolve_deadline_seconds(self, request: GenerationRequest) -> float:
        """Cutover follow-up #74: per-request deadline resolution.
        Order of precedence:
          1. ``request.deadline_ms`` if set (explicit per-request).
          2. ``self._default_deadline_ms`` (backend-instance default,
             constructor-injectable).
          3. ``DEFAULT_DEADLINE_MS`` (class-level fallback).
        Returns seconds (because ``time.monotonic()`` works in seconds).
        Negative / zero values raise INVALID_INPUT — a 0ms deadline
        is a configuration error, not an instant timeout.
        """
        deadline_ms = request.deadline_ms if request.deadline_ms is not None else self._default_deadline_ms
        if deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"deadline_ms must be > 0; got {deadline_ms}. Use "
                    f"None to fall back to the backend's default "
                    f"({self.DEFAULT_DEADLINE_MS}ms)."
                ),
            )
        return deadline_ms / 1000.0

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
        # Cutover follow-up #74: validate the deadline BEFORE opening
        # a session so deadline_ms<=0 fails fast as INVALID_INPUT
        # without burning a session/model handle.
        deadline_seconds = self._resolve_deadline_seconds(request)

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
                    "NativeChatBackend: v0.1.2 ships greedy-only; ignoring request.temperature=%s (treating as 0.0)",
                    request.temperature,
                )
            if request.top_p not in (1.0, 1):
                logger.warning(
                    "NativeChatBackend: v0.1.2 ships greedy-only; ignoring request.top_p=%s (treating as 1.0)",
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
            # Cutover follow-up #73: also drain CACHE_HIT / CACHE_MISS
            # events and capture session-completed timing fields so
            # InferenceMetrics carries real cache + latency telemetry.
            assembled: list[str] = []
            n_chunks = 0
            ttfc_ms: float = 0.0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            cache_hits = 0
            cache_misses = 0
            cache_saved_tokens = 0
            sc_setup_ms = 0.0
            sc_engine_first_chunk_ms = 0.0
            sc_queued_ms = 0.0
            sc_total_latency_ms = 0.0
            # Cutover follow-up #74: deadline already validated +
            # resolved before open_session above so a bad deadline_ms
            # fails fast without burning a session/model handle.
            deadline = time.monotonic() + deadline_seconds
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
                if ev.type == OCT_EVENT_CACHE_HIT:
                    cache_hits += 1
                    cache_saved_tokens += int(ev.cache_saved_tokens)
                    continue
                if ev.type == OCT_EVENT_CACHE_MISS:
                    cache_misses += 1
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    saw_error = True
                    if not error_message:
                        error_message = self._runtime.last_error()
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    terminal_status = int(ev.terminal_status)
                    sc_setup_ms = float(ev.setup_ms)
                    sc_engine_first_chunk_ms = float(ev.engine_first_chunk_ms)
                    sc_queued_ms = float(ev.queued_ms)
                    sc_total_latency_ms = float(ev.total_latency_ms)
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
            # Cutover follow-up #73: prefer the runtime's reported
            # total_latency_ms when present; fall back to the SDK-
            # measured wall-clock if the runtime didn't populate it
            # (e.g., older runtime, or future event ordering where
            # SESSION_COMPLETED hasn't fired yet — though we'd have
            # raised on deadline above).
            duration_ms = sc_total_latency_ms if sc_total_latency_ms > 0 else (elapsed * 1000.0)
            metrics = InferenceMetrics(
                ttfc_ms=ttfc_ms,
                prompt_tokens=0,  # not exposed by v0.1.2 envelope
                total_tokens=n_chunks,
                tokens_per_second=tps,
                total_duration_ms=duration_ms,
                attention_backend=self.attention_backend,
                cache_hit=cache_hits > 0,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_saved_tokens=cache_saved_tokens,
                queued_ms=sc_queued_ms,
                setup_ms=sc_setup_ms,
                engine_first_chunk_ms=sc_engine_first_chunk_ms,
            )
            return text, metrics
        finally:
            sess.close()

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Cutover follow-up #72: chat.stream capability for the native
        backend. The runtime already emits one TRANSCRIPT_CHUNK event per
        generated token; this method relays them as ``GenerationChunk``
        instances as they arrive (rather than accumulating into a single
        string the way ``generate()`` does).

        Same gating, validation, and terminal-status mapping as
        ``generate()`` — ``request.stream`` is just an output-shape
        choice; the underlying session/poll loop is identical. Yields
        a final empty-text chunk with ``finish_reason="stop"`` once
        SESSION_COMPLETED arrives.
        """
        import asyncio

        # Gate UNSUPPORTED features before checking runtime availability
        # so a grammar/json_mode/tools request gets a 422 even if the
        # backend hasn't been loaded yet — those modalities will never
        # be supported regardless of load state, while RUNTIME_UNAVAILABLE
        # is a transient condition.
        _gate_unsupported_request_features(request)
        clean_messages = _validate_messages_for_native(request.messages)
        # Cutover follow-up #74: deadline-validate before opening
        # a session (same as generate()).
        deadline_seconds = self._resolve_deadline_seconds(request)
        if self._runtime is None or self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeChatBackend.generate_stream called before load_model",
            )

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
                "native chat backend failed to open streaming session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        loop = asyncio.get_event_loop()
        try:
            if request.temperature not in (0.0, 0):
                logger.warning(
                    "NativeChatBackend.generate_stream: v0.1.2 ships greedy-only; ignoring request.temperature=%s",
                    request.temperature,
                )
            if request.top_p not in (1.0, 1):
                logger.warning(
                    "NativeChatBackend.generate_stream: v0.1.2 ships greedy-only; ignoring request.top_p=%s",
                    request.top_p,
                )
            try:
                await loop.run_in_executor(
                    self._executor, lambda: sess.send_chat(clean_messages, max_tokens=int(request.max_tokens))
                )
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native chat backend send_chat failed (stream)",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            # Same drain loop as generate(), but yield each chunk
            # instead of accumulating. SESSION_COMPLETED still carries
            # the bounded terminal_status; an UNSUPPORTED reject on
            # the streaming path raises the same OctomilError.
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            # Cutover follow-up #74: deadline already validated +
            # resolved before open_session above.
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                ev = await loop.run_in_executor(self._executor, lambda: sess.poll_event(timeout_ms=200))
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_CHUNK:
                    if ev.text:
                        yield GenerationChunk(text=ev.text, finish_reason=None)
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    saw_error = True
                    if not error_message:
                        error_message = self._runtime.last_error()
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    terminal_status = int(ev.terminal_status)
                    break
            else:
                raise OctomilError(
                    code=OctomilErrorCode.REQUEST_TIMEOUT,
                    message="native chat backend timed out waiting for SESSION_COMPLETED (stream)",
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native chat backend reported error during streaming generation",
                    last_error=error_message,
                )

            # Final marker chunk so callers can detect terminal cleanly.
            yield GenerationChunk(text="", finish_reason="stop")
        finally:
            # Cutover follow-up #72 (R1 Codex): NativeSession is single-
            # thread-affine per the loader contract. send_chat / poll_event
            # ran on `self._executor` (max_workers=1, so all work serializes
            # on one thread); close() MUST run on the same thread.
            try:
                await loop.run_in_executor(self._executor, sess.close)
            except Exception:  # noqa: BLE001
                # Generator-close paths can run after the loop has been
                # shut down (e.g., test teardown). Fall back to a direct
                # sync close — the session destructor will still cleanly
                # tear down in that path.
                try:
                    sess.close()
                except Exception:  # noqa: BLE001
                    logger.warning("NativeChatBackend.generate_stream: session close failed", exc_info=True)

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
        """Surface the cutover identity + cache/latency telemetry in
        verbose events. Cutover follow-up #73 added the cache_*,
        queued_ms, setup_ms, engine_first_chunk_ms fields — they ride
        along on backend.generate_completed events when the runtime
        emitted CACHE_HIT/MISS or populated SESSION_COMPLETED timing.
        Defaults to 0 when no telemetry was produced (e.g., a
        cold-start request with no prior session warmth)."""
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
            # Cutover follow-up #73: cache + latency telemetry.
            meta["cache_hits"] = metrics.cache_hits
            meta["cache_misses"] = metrics.cache_misses
            meta["cache_saved_tokens"] = metrics.cache_saved_tokens
            meta["queued_ms"] = metrics.queued_ms
            meta["setup_ms"] = metrics.setup_ms
            meta["engine_first_chunk_ms"] = metrics.engine_first_chunk_ms
        return meta
