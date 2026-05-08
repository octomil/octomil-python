"""NativeEmbeddingsBackend — embeddings.text via octomil-runtime v0.1.3+.

Hard-cutover backend for local GGUF embeddings. Runs on the same
``NativeRuntime`` + ``NativeModel`` lifecycle as ``NativeChatBackend``,
but opens sessions with ``capability="embeddings.text"`` instead of
``"chat.completion"`` and drains pooled-vector events instead of
transcript chunks.

Lifecycle:
- One ``NativeEmbeddingsBackend`` instance per (planner-selected)
  embedding model. ``load_model(model_name)`` resolves the GGUF
  path, opens a ``NativeRuntime``, opens + warms the model, caches
  both. The cached model is reused across requests.
- ``embed(inputs)`` opens one session per request, calls
  ``send_embed(...)`` with the wrapped JSON shape, drains the
  event stream, closes the session. Returns an
  ``EmbeddingsResult`` matching the existing SDK shape.
- ``close()`` closes the model + runtime in correct order.

Hard rules (cutover discipline — no silent Python fallback):
1. The runtime's per-context pooling-type gate at session_open
   rejects decoder-only chat GGUFs (``LLAMA_POOLING_TYPE_NONE``)
   AND RANK pooling models (``OCT_EMBED_POOLING_RANK``) with
   ``OCT_STATUS_UNSUPPORTED`` → bounded
   ``OctomilError(UNSUPPORTED_MODALITY)``. Callers MUST propagate
   the error rather than fall back to a Python-local embedder.
2. The runtime advertises ``embeddings.text`` only when llama.cpp
   is built into the dylib on a supported platform. Callers
   should query ``oct_runtime_capabilities()`` (via
   ``NativeRuntime.capabilities()``) before opening a session.
3. Atomic-batch failure: a per-input failure mid-batch causes the
   runtime to emit ``OCT_EVENT_ERROR`` followed by
   ``OCT_EVENT_SESSION_COMPLETED(matching_status)`` and stop;
   subsequent inputs are NOT processed. The SDK surfaces the
   bounded error; partial results are NOT returned (atomic batch
   semantics from the runtime side propagate cleanly through
   this binding).
4. Privacy: the runtime's error messages do not echo input bytes.
   The SDK preserves that contract by surfacing ``last_error``
   verbatim from the runtime.

Bounded-error mapping (runtime → SDK):
- ``OCT_STATUS_NOT_FOUND`` (model file missing) →
  ``MODEL_NOT_FOUND``.
- ``OCT_STATUS_INVALID_INPUT`` →
  ``INVALID_INPUT``.
- ``OCT_STATUS_UNSUPPORTED`` →
  ``UNSUPPORTED_MODALITY`` (decoder-only chat GGUF, RANK
  pooling, or unsupported request shape).
- ``OCT_STATUS_VERSION_MISMATCH`` →
  ``RUNTIME_UNAVAILABLE`` (binding/dylib skew — fix is to
  rebuild the runtime or upgrade the SDK).
- ``OCT_STATUS_BUSY`` →
  ``SERVER_ERROR``.
- Any other / non-OK terminal_status →
  ``INFERENCE_FAILED``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from ...errors import OctomilError, OctomilErrorCode
from .error_mapping import map_oct_status
from .loader import (
    OCT_EVENT_EMBEDDING_VECTOR,
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_OK,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)


_BACKEND_NAME = "native-llama-cpp-embeddings"
_DEFAULT_DEADLINE_MS = 300_000  # 5 minutes — same shape as NativeChatBackend


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Thin wrapper over :func:`octomil.runtime.native.error_mapping.map_oct_status`
    pinned to the embeddings/chat policy: ``OCT_STATUS_UNSUPPORTED``
    → ``UNSUPPORTED_MODALITY`` for request-shape rejects (e.g.
    OCT_EMBED_POOLING_RANK at session_open). v0.1.6 PR1 centralized
    the mapping in ``error_mapping.py``.
    """
    return map_oct_status(
        status,
        last_error,
        message=message,
        default_unsupported_code=OctomilErrorCode.UNSUPPORTED_MODALITY,
    )


@dataclass
class EmbeddingsResult:
    """Result of a native embeddings batch.

    Shape mirrors ``octomil.embeddings.EmbeddingResult`` (the cloud
    embeddings response) so the kernel can return either with the
    same caller-visible interface. Differences:

    - ``embeddings`` is ``list[list[float]]`` in input order with one
      vector per input string. Vectors are L2-normalized fp32 (the
      runtime applies the normalization; ``is_normalized=True`` is
      the only shape v0.1.3 emits).
    - ``model`` is the model identifier (GGUF path or model-id) the
      backend was loaded with.
    - ``usage.prompt_tokens`` is the SUM of ``n_input_tokens`` across
      all returned vectors. ``total_tokens`` equals ``prompt_tokens``
      for embeddings (no completion tokens).
    - ``pooling_type`` is the runtime's ``OCT_EMBED_POOLING_*``
      discriminator (1=MEAN, 2=CLS, 3=LAST). RANK is rejected at
      session_open and never reaches a result.
    - ``n_dim`` is the pooled output dimension. Stable across the
      batch (every vector has the same length).
    """

    embeddings: list[list[float]]
    model: str
    n_dim: int
    pooling_type: int
    is_normalized: bool
    prompt_tokens: int
    total_tokens: int


class NativeEmbeddingsBackend:
    """Hard-cutover embeddings.text backend backed by
    ``octomil-runtime v0.1.3+``.

    Caches a ``NativeRuntime`` + warmed ``NativeModel`` per backend
    instance. Each ``embed(inputs)`` call opens a fresh session
    against the cached model, calls ``send_embed(...)``, drains
    EMBEDDING_VECTOR events, closes the session.
    """

    name: str = _BACKEND_NAME
    DEFAULT_DEADLINE_MS: int = _DEFAULT_DEADLINE_MS

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_deadline_ms: int | None = None,
    ) -> None:
        self._model_dir: str | None = model_dir
        self._model_name: str = ""
        self._gguf_path: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None
        self._default_deadline_ms: int = (
            default_deadline_ms if default_deadline_ms is not None else self.DEFAULT_DEADLINE_MS
        )

    # ------------------------------------------------------------------
    # GGUF resolution — same shape as NativeChatBackend.
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
        if model_name.endswith(".gguf") and os.path.isfile(model_name):
            return model_name
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=(
                f"native embeddings backend could not resolve a GGUF for model {model_name!r}. "
                f"Expected a PrepareManager-materialized artifact via model_dir; got "
                f"{self._model_dir!r}."
            ),
        )

    def load_model(self, model_name: str) -> None:
        if self._runtime is not None:
            return  # idempotent
        self._model_name = model_name
        self._gguf_path = self._resolve_gguf_path(model_name)
        logger.info(
            "NativeEmbeddingsBackend: loading GGUF via octomil-runtime native: %s",
            self._gguf_path,
        )
        try:
            self._runtime = NativeRuntime.open()
            self._model = self._runtime.open_model(model_uri=self._gguf_path)
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                f"native embeddings backend failed to load {self._gguf_path}",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeEmbeddingsBackend.close: model.close failed", exc_info=True)
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeEmbeddingsBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def embed(
        self,
        inputs: str | list[str],
        *,
        deadline_ms: int | None = None,
    ) -> EmbeddingsResult:
        """Run an embeddings.text request against the cached model.

        Parameters
        ----------
        inputs
            Single string OR non-empty list of strings. Empty /
            whitespace-only strings reject INVALID_INPUT (the
            runtime-side validator catches them).
        deadline_ms
            Per-request poll deadline. Falls back to
            ``self._default_deadline_ms`` (5 minutes) when None.
            Negative / zero values raise INVALID_INPUT.

        Returns
        -------
        EmbeddingsResult
            Vectors in input order with L2-normalized fp32 values,
            stable n_dim, total_tokens summed from per-input
            n_input_tokens.

        Raises
        ------
        OctomilError
            ``RUNTIME_UNAVAILABLE`` if ``load_model`` wasn't called.
            ``UNSUPPORTED_MODALITY`` if the model has no embedding
            head (decoder-only chat GGUF) or uses RANK pooling.
            ``INVALID_INPUT`` for malformed input shapes / bad
            deadline / empty strings.
            ``REQUEST_TIMEOUT`` if the deadline elapses before
            SESSION_COMPLETED.
            ``INFERENCE_FAILED`` for runtime-side decode errors.
        """
        if self._runtime is None or self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeEmbeddingsBackend.embed called before load_model",
            )

        # Deadline validation BEFORE opening a session so a bad
        # deadline_ms fails fast without burning a session handle.
        resolved_deadline_ms = deadline_ms if deadline_ms is not None else self._default_deadline_ms
        if resolved_deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"deadline_ms must be > 0; got {resolved_deadline_ms}. Use None to "
                    f"fall back to the backend's default ({self.DEFAULT_DEADLINE_MS}ms)."
                ),
            )

        try:
            sess = self._runtime.open_session(
                capability="embeddings.text",
                locality="on_device",
                policy_preset="private",
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native embeddings backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        try:
            try:
                sess.send_embed(inputs)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native embeddings backend send_embed failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            # Drain events. The runtime emits one EMBEDDING_VECTOR per
            # input in order, then SESSION_COMPLETED(OK). On per-input
            # failure: ERROR + SESSION_COMPLETED with matching status;
            # subsequent inputs NOT processed.
            vectors: list[list[float]] = []
            n_dim = 0
            pooling_type = 0
            is_normalized = False
            total_tokens = 0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""

            deadline_seconds = resolved_deadline_ms / 1000.0
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                ev = sess.poll_event(timeout_ms=200)
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_EMBEDDING_VECTOR:
                    # Sanity: index must match arrival order. The
                    # runtime guarantees in-order emission; mismatch
                    # would be a runtime bug, not caller misuse —
                    # surface as INFERENCE_FAILED rather than silently
                    # tolerate.
                    if ev.index != len(vectors):
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(
                                f"native embeddings: out-of-order EMBEDDING_VECTOR "
                                f"(got index={ev.index}, expected {len(vectors)})"
                            ),
                        )
                    if n_dim == 0:
                        n_dim = ev.n_dim
                        pooling_type = ev.pooling_type
                        is_normalized = ev.is_normalized
                    elif ev.n_dim != n_dim:
                        # Stable n_dim is part of the contract.
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(f"native embeddings: n_dim drift across batch (got {ev.n_dim}, expected {n_dim})"),
                        )
                    vectors.append(list(ev.values))
                    total_tokens += int(ev.n_input_tokens)
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
                    message=(
                        f"native embeddings backend timed out waiting for SESSION_COMPLETED ({resolved_deadline_ms}ms)"
                    ),
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native embeddings backend reported error during generation",
                    last_error=error_message,
                )

            # Atomic batch: terminal OK means we got a vector for every
            # input. Defensive guard — runtime shouldn't violate this.
            expected_count = 1 if isinstance(inputs, str) else len(inputs)
            if len(vectors) != expected_count:
                raise OctomilError(
                    code=OctomilErrorCode.INFERENCE_FAILED,
                    message=(
                        f"native embeddings: vector count mismatch (got {len(vectors)}, expected {expected_count})"
                    ),
                )

            return EmbeddingsResult(
                embeddings=vectors,
                model=self._model_name,
                n_dim=n_dim,
                pooling_type=pooling_type,
                is_normalized=is_normalized,
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            )
        finally:
            sess.close()

    # ------------------------------------------------------------------
    # Capability advertisement helpers (for callers / planners doing
    # capability-honesty checks before construction).
    # ------------------------------------------------------------------
    @staticmethod
    def runtime_advertises_embeddings_text(rt: NativeRuntime) -> bool:
        """Query the runtime's capability list for ``embeddings.text``.

        Use this to gate construction so callers don't open a runtime
        whose dylib was built without llama.cpp (or on an unsupported
        platform). Returns False if the capability is not advertised;
        callers MUST raise UNSUPPORTED_MODALITY rather than fall back
        to a Python-local embedder."""
        try:
            caps = rt.capabilities()
        except Exception:  # noqa: BLE001
            return False
        return "embeddings.text" in caps.supported_capabilities
