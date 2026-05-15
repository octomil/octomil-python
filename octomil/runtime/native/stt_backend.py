"""NativeSttBackend — audio.transcription via octomil-runtime v0.1.5+.

Hard-cutover backend for local whisper.cpp speech-to-text. Replaces
the legacy ``pywhispercpp``-backed ``WhisperCppEngine`` for the
product STT path. Modeled on :mod:`octomil.runtime.native.chat_backend`
and :mod:`octomil.runtime.native.embeddings_backend`.

Lifecycle:
- One ``NativeSttBackend`` instance per (planner-selected) STT engine.
- ``load_model(model_name)`` opens a ``NativeRuntime``, verifies the
  ``audio.transcription`` capability is advertised, and caches the
  runtime. W1 wires ``whisper-tiny`` and ``whisper-base`` only; the
  SDK verifies the actual ``OCTOMIL_WHISPER_BIN`` file matches the
  requested model's size + SHA-256 before model_open.
- ``transcribe(pcm_f32, sample_rate_hz=...)`` opens one session per
  request with ``capability="audio.transcription"``, sends the audio
  view, drains ``OCT_EVENT_TRANSCRIPT_SEGMENT`` events, captures
  ``OCT_EVENT_TRANSCRIPT_FINAL``, then closes the session
  deterministically. The runtime stays warm.
- ``close()`` closes the runtime.

Hard rules (cutover discipline — no silent Python fallback):
1. The caller MUST resolve the model artifact + capability advertisement
   at the planner; this backend is selected only when the runtime has
   already advertised ``audio.transcription``.
2. ``transcribe()`` rejects NaN / Inf / zero-length input via the
   runtime-side validator; the SDK surfaces those as ``INVALID_INPUT``.
3. Bad or wrong-size registered-artifact bytes fail closed as
   ``CHECKSUM_MISMATCH`` before ``oct_model_open``; runtime digest
   rejects with ``last_error`` mentioning ``digest`` are mapped the
   same way.
4. ``OCTOMIL_WHISPER_BIN`` unset / artifact missing → the runtime
   does NOT advertise the capability and ``open_session`` returns
   ``OCT_STATUS_UNSUPPORTED`` — surfaced as ``RUNTIME_UNAVAILABLE``.

Bounded-error mapping (runtime → SDK):
- Missing artifact / file → ``MODEL_NOT_FOUND``.
- Bad digest (last_error contains ``digest``) → ``CHECKSUM_MISMATCH``.
- Unsupported capability / ABI mismatch → ``RUNTIME_UNAVAILABLE``.
- ``OCT_STATUS_INVALID_INPUT`` (NaN / Inf / format) → ``INVALID_INPUT``.
- Caller-driven cancel → ``CANCELLED``.
- Caller-driven timeout → ``REQUEST_TIMEOUT``.
- Other terminal → ``INFERENCE_FAILED``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import stat
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...errors import OctomilError, OctomilErrorCode
from .capabilities import CAPABILITY_AUDIO_TRANSCRIPTION
from .error_mapping import map_oct_status
from .loader import (
    OCT_EVENT_ERROR,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_TRANSCRIPT_FINAL,
    OCT_EVENT_TRANSCRIPT_SEGMENT,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_OK,
    NativeRuntime,
    NativeRuntimeError,
)

logger = logging.getLogger(__name__)


_BACKEND_NAME = "native-whisper-cpp"
_DEFAULT_DEADLINE_MS = 300_000  # 5 minutes — same shape as chat / embeddings.
# whisper.cpp STT is hard-coded to 16kHz mono PCM-f32 for the W1
# tiny/base native path; the runtime rejects anything else with
# INVALID_INPUT. We keep this as a Python-side guard so callers see
# the shape error before the FFI call. SR-mismatch surfacing as
# INVALID_INPUT preserves bounded-rejection behavior.
_WHISPER_SAMPLE_RATE_HZ: int = 16000
# The runtime's whisper.cpp adapter resolves ggml-tiny.bin /
# ggml-base.bin off OCTOMIL_WHISPER_BIN. oct_model_open still
# requires a valid model_uri scheme (absolute path, file://,
# local://) - we hand it the same env value so the URI parses and the
# digest verification has something concrete to point at when
# last_error is rendered.
_WHISPER_BIN_ENV: str = "OCTOMIL_WHISPER_BIN"

# OCT-113 v2: env var the runtime checks to relax its built-in
# digest-registry gate for user-uploaded artifacts. The runtime
# enforces a registry match against shipped canonical Whisper SHAs
# by default; when this flag is set the runtime accepts any
# whisper-shaped artifact whose ``artifact_digest`` matches the
# file's actual SHA-256 (integrity-only check). Required for the
# user-upload load path below. Runtime support: TODO in
# ``octomil-runtime`` — until that ships, this env var is set
# defensively by the SDK and ``load_uploaded_model`` will surface
# the runtime's registry-mismatch error with a clear diagnostic
# pointing operators at the missing runtime support.
_WHISPER_ALLOW_USER_ARTIFACTS_ENV: str = "OCT_WHISPER_ALLOW_USER_ARTIFACTS"


@dataclass(frozen=True)
class _WhisperArtifactSpec:
    model_name: str
    size_name: str
    filename: str
    sha256: str
    size_bytes: int

    @property
    def artifact_digest(self) -> str:
        return f"sha256:{self.sha256}"


_WHISPER_ARTIFACTS: dict[str, _WhisperArtifactSpec] = {
    "whisper-tiny": _WhisperArtifactSpec(
        model_name="whisper-tiny",
        size_name="tiny",
        filename="ggml-tiny.bin",
        sha256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        size_bytes=77_691_713,
    ),
    "whisper-base": _WhisperArtifactSpec(
        model_name="whisper-base",
        size_name="base",
        filename="ggml-base.bin",
        sha256="60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        size_bytes=147_951_465,
    ),
}
_SUPPORTED_MODEL_NAMES: tuple[str, ...] = tuple(_WHISPER_ARTIFACTS)


def is_supported_native_whisper_model(model_name: str) -> bool:
    """Return True for native whisper.cpp sizes backed by runtime rows."""
    return model_name.lower() in _WHISPER_ARTIFACTS


def _sha256_file_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_whisper_artifact_matches_spec(path: str, spec: _WhisperArtifactSpec) -> None:
    """Verify the env artifact matches the requested model identity."""
    try:
        st = os.stat(path)
    except OSError as exc:
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=f"native STT backend: {_WHISPER_BIN_ENV} path not found or unreadable: {path}",
        ) from exc
    if not stat.S_ISREG(st.st_mode):
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=f"native STT backend: {_WHISPER_BIN_ENV} is not a regular file: {path}",
        )
    actual_size = int(st.st_size)
    if actual_size != spec.size_bytes:
        raise OctomilError(
            code=OctomilErrorCode.CHECKSUM_MISMATCH,
            message=(
                f"native STT backend: {spec.model_name} requires {spec.filename} "
                f"size {spec.size_bytes} bytes, but {_WHISPER_BIN_ENV} points at "
                f"{actual_size} bytes"
            ),
        )
    try:
        digest = _sha256_file_hex(path)
    except OSError as exc:
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=f"native STT backend: could not read {_WHISPER_BIN_ENV} artifact: {path}",
        ) from exc
    if digest != spec.sha256:
        raise OctomilError(
            code=OctomilErrorCode.CHECKSUM_MISMATCH,
            message=(
                f"native STT backend: {spec.model_name} requires {spec.filename} "
                f"SHA-256 {spec.sha256}, but {_WHISPER_BIN_ENV} computed {digest}"
            ),
        )


@dataclass
class Segment:
    """One timestamped span emitted as ``OCT_EVENT_TRANSCRIPT_SEGMENT``.

    ``start_ms`` and ``end_ms`` are runtime-monotonic from the start
    of the audio window; ``end_ms >= start_ms``.
    """

    start_ms: int
    end_ms: int
    text: str


@dataclass
class TranscriptionResult:
    """Native-STT transcription result.

    Distinct from :class:`octomil.audio.types.TranscriptionResult` so
    the native backend can carry per-segment timestamps and decoded
    duration without forcing a contract bump on the public audio API.
    The kernel adapter projects this down to the public type.
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    duration_ms: int = 0


def _runtime_status_to_sdk_error(
    status: int,
    message: str,
    *,
    last_error: str = "",
) -> OctomilError:
    """Thin wrapper over :func:`octomil.runtime.native.error_mapping.map_oct_status`
    pinned to the audio-capability ``default_unsupported_code`` policy
    (``RUNTIME_UNAVAILABLE``).

    Kept as a backend-local symbol for back-compat with v0.1.5 tests
    that import it directly. v0.1.6 PR1 centralized the body in
    ``error_mapping.py``; rules + boundary contract live there.
    """
    return map_oct_status(
        status,
        last_error,
        message=message,
        default_unsupported_code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
    )


def _validate_pcm_f32(samples: Sequence[float] | Any, sample_rate_hz: int) -> bytes:
    """Pre-flight the audio buffer against the shape contract.

    The runtime side validates again (NaN/Inf, zero-length, bad sr),
    but a Python-side check produces a precise diagnostic before
    paying session_open + send_audio cost. Returns the buffer as a
    contiguous float32 ``bytes`` so the cffi cast in
    ``NativeSession.send_audio`` reads the right layout.

    Accepts:
      * 1-D numpy float32 array (mono).
      * Python sequence of floats (mono).
      * Raw ``bytes`` already in float32-LE shape.

    Stereo / multichannel and non-16kHz inputs are rejected here;
    the W1 native whisper.cpp path expects 16kHz mono for both tiny
    and base.
    """
    if sample_rate_hz != _WHISPER_SAMPLE_RATE_HZ:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"native STT: sample_rate_hz must be {_WHISPER_SAMPLE_RATE_HZ} "
                f"(native whisper.cpp STT is mono-16kHz-only); got {sample_rate_hz}"
            ),
        )
    if isinstance(samples, (bytes, bytearray, memoryview)):
        buf = bytes(samples)
        if len(buf) == 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native STT: zero-length audio buffer",
            )
        if len(buf) % 4 != 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(f"native STT: PCM-f32 buffer length {len(buf)} is not a multiple of 4 bytes"),
            )
        # Codex R1 nit: bytes inputs previously skipped NaN/Inf
        # checks. The runtime would still reject them with
        # INVALID_INPUT, but doing it Python-side preserves the
        # same diagnostic shape across input types and saves a
        # session round-trip. We reinterpret the buffer as fp32
        # without copying via numpy.frombuffer, then check
        # `isfinite`.
        try:
            import numpy as _np  # type: ignore[import-not-found]
        except ImportError:
            # No numpy — skip the Python-side check; runtime will
            # still reject. Don't fail the call just because numpy
            # isn't installed.
            return buf
        view = _np.frombuffer(buf, dtype=_np.float32)
        if not _np.isfinite(view).all():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="native STT: audio contains NaN or Inf samples",
            )
        return buf
    # numpy / list-of-float path. Avoid a hard numpy dep at module
    # import time; only require it when the caller passes an array-like.
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "native STT: array-like audio input requires numpy; pass bytes (float32-LE) or `pip install numpy`"
            ),
        ) from exc
    arr = _np.asarray(samples, dtype=_np.float32)
    if arr.ndim != 1:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(f"native STT: audio must be 1-D mono PCM-f32; got shape {arr.shape}"),
        )
    if arr.size == 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native STT: zero-length audio buffer",
        )
    # NaN / Inf would also be rejected by the runtime; surfacing
    # Python-side preserves bounded-rejection behavior + saves a
    # session-open round-trip. Use the runtime mapping (INVALID_INPUT)
    # so the error matches the runtime path one-for-one.
    if not _np.isfinite(arr).all():
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message="native STT: audio contains NaN or Inf samples",
        )
    return arr.tobytes()


def _runtime_advertises_audio_transcription(rt: NativeRuntime) -> bool:
    """Capability-honesty check used by the planner before constructing
    a backend. Returns False if the runtime doesn't advertise
    ``audio.transcription`` (capability gate failed: missing
    ``OCTOMIL_WHISPER_BIN`` or digest mismatch). Callers MUST raise
    ``RUNTIME_UNAVAILABLE`` rather than fall back to a Python-local
    transcriber on the product path."""
    try:
        caps = rt.capabilities()
    except Exception:  # noqa: BLE001
        return False
    return CAPABILITY_AUDIO_TRANSCRIPTION in caps.supported_capabilities


# Public alias used by the planner / kernel — picks up the existence
# check without requiring callers to import the leading-underscore
# helper. The leading-underscore form is kept for back-compat with
# the embeddings_backend pattern.
runtime_advertises_audio_transcription = _runtime_advertises_audio_transcription


class NativeSttBackend:
    """Hard-cutover ``audio.transcription`` backend backed by
    ``octomil-runtime v0.1.5+``.

    Caches a ``NativeRuntime`` per backend instance. Each
    ``transcribe(...)`` call opens a fresh session, sends the audio
    view, drains transcript events, closes the session.
    """

    name: str = _BACKEND_NAME
    DEFAULT_DEADLINE_MS: int = _DEFAULT_DEADLINE_MS

    def __init__(
        self,
        *,
        default_deadline_ms: int | None = None,
    ) -> None:
        self._model_name: str = ""
        self._runtime: NativeRuntime | None = None
        self._model: Any | None = None
        self._default_deadline_ms: int = (
            default_deadline_ms if default_deadline_ms is not None else self.DEFAULT_DEADLINE_MS
        )
        # OCT-113 v2 review2 P1 #2: idempotency triple for
        # load_uploaded_model. Same model_name with a different path
        # or sha is a NEW load, not a no-op.
        self._uploaded_path: str | None = None
        self._uploaded_sha: str | None = None
        # OCT-113 v2 review2 P1 #3: snapshot of env vars we mutate
        # so close()/failure paths can restore. None means "we
        # haven't touched env yet". The dict's values capture the
        # original env state — None means "var was unset" so we
        # delete-on-restore rather than setting to empty.
        self._prior_env: dict[str, str | None] | None = None

    def load_model(self, model_name: str) -> None:
        """Open the runtime and verify ``audio.transcription`` is
        advertised. Idempotent — a second call is a no-op.

        W1 hard rule: the SDK accepts exactly the native whisper.cpp
        model names backed by runtime registry rows (``whisper-tiny``
        and ``whisper-base``). Other whisper sizes are NOT supported;
        silently substituting one registered artifact for another
        would be a correctness bug (different model identity means
        different WER / RTF profile).

        Raises
        ------
        OctomilError
            ``UNSUPPORTED_MODALITY`` if the requested model name is
            not one of ``whisper-tiny`` or ``whisper-base``.
            ``RUNTIME_UNAVAILABLE`` if the runtime fails to open or
            does not advertise ``audio.transcription`` (operator
            forgot ``OCTOMIL_WHISPER_BIN`` or the dylib was built
            without ``OCT_ENABLE_ENGINE_WHISPER_CPP=ON``).
            ``CHECKSUM_MISMATCH`` if the artifact's digest doesn't
            match the requested model's runtime-pinned SHA-256.
        """
        # Validate model_name before the idempotent short-circuit so a
        # second load_model call for a different size cannot silently
        # reuse an already-warmed artifact.
        requested_model = model_name.lower()
        spec = _WHISPER_ARTIFACTS.get(requested_model)
        if spec is None:
            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message=(
                    f"native STT backend: model {model_name!r} is not supported by the native runtime. "
                    f"Supported native sizes are {', '.join(_SUPPORTED_MODEL_NAMES)}. "
                    "whisper-small, whisper-medium, and whisper-large-v3 are not enabled."
                ),
            )
        if self._runtime is not None:
            if self._model_name.lower() == requested_model:
                # Same canonical name on a re-load is idempotent.
                return
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"native STT backend already loaded {self._model_name!r}; "
                    f"cannot reuse it for {requested_model!r}"
                ),
            )
        self._model_name = requested_model
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT backend failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            # Dylib resolution failure is a runtime-not-available signal
            # at the SDK layer. Operator points at the wrong file or the
            # cffi extra is missing.
            self._runtime = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"native STT backend: dylib not found ({exc})",
            ) from exc

        if not _runtime_advertises_audio_transcription(self._runtime):
            # Capability not advertised. Two distinct causes (per the
            # runtime contract): (a) OCTOMIL_WHISPER_BIN unset OR
            # capability not built into the dylib → RUNTIME_UNAVAILABLE;
            # (b) env IS set but the artifact's SHA-256 doesn't match
            # the runtime-pinned digest → CHECKSUM_MISMATCH (the
            # runtime writes a "digest mismatch" message into the
            # thread-local last_error when it skips advertising on
            # this branch). The disambiguation matters because (a)
            # is a setup problem the operator fixes by pointing at a
            # real registered ggml artifact, while (b) is an integrity-violation
            # signal that callers route differently (e.g. trigger a
            # re-download). Codex R1 blocker: previously we flattened
            # both to RUNTIME_UNAVAILABLE.
            last_error_lc = (self._runtime.last_error() or "").lower()
            self.close()
            if "digest" in last_error_lc:
                raise OctomilError(
                    code=OctomilErrorCode.CHECKSUM_MISMATCH,
                    message=(
                        "native STT backend: Whisper artifact SHA-256 does not "
                        "match a runtime-registered digest. Re-download the artifact."
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native STT backend: runtime does not advertise "
                    "'audio.transcription'. Check OCTOMIL_WHISPER_BIN "
                    "(must point at a registered ggml-tiny.bin or "
                    "ggml-base.bin artifact) and that the dylib was built with "
                    "OCT_ENABLE_ENGINE_WHISPER_CPP=ON."
                ),
            )

        # Runtime registry admission accepts any registered Whisper row;
        # the SDK request is for a specific identity, so verify the
        # env artifact against `spec` before model_open. We still pass
        # artifact_digest as defense in depth, but the SDK does not rely
        # on runtime-side expected-identity consumption.
        whisper_bin = os.environ.get(_WHISPER_BIN_ENV, "")
        if not whisper_bin:
            self.close()
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"native STT backend: {_WHISPER_BIN_ENV} not set. "
                    f"Point at verified {spec.filename} bytes for {spec.model_name} "
                    f"(SHA-256 {spec.sha256[:12]}...)."
                ),
            )
        try:
            _verify_whisper_artifact_matches_spec(whisper_bin, spec)
        except OctomilError:
            self.close()
            raise
        try:
            self._model = self._runtime.open_model(
                model_uri=whisper_bin,
                artifact_digest=spec.artifact_digest,
                engine_hint="whisper_cpp",
            )
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()
            raise _runtime_status_to_sdk_error(
                exc.status,
                f"native STT backend failed to warm {spec.model_name} model",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        # Codex R1 nit: was logger.info; lowered to debug so the
        # cutover-rule "no new stderr/metric emission" stays clean.
        logger.debug("NativeSttBackend: runtime opened + %s warmed", spec.model_name)

    def load_uploaded_model(
        self,
        *,
        model_name: str,
        artifact_path: str,
        expected_sha256: str,
    ) -> None:
        """OCT-113 v2: load a user-uploaded Whisper artifact by path,
        skipping the canonical ``_WHISPER_ARTIFACTS`` SHA registry.

        Use this entry point when the model was uploaded by an org
        (server-side ``ArtifactPackage.origin_source == "upload"``)
        rather than served from the canonical catalog. The caller
        supplies the artifact's own SHA-256 (typically read from the
        catalog's ``checksum_sha256``); the SDK verifies the file
        actually hashes to that value (integrity check) and asks the
        runtime to open it with that digest.

        Parameters
        ----------
        model_name : str
            User-facing model slug. Used only as a label in logs +
            error messages; not validated against the canonical
            registry.
        artifact_path : str
            Filesystem path to the .gguf / .bin file the runtime
            should open. The SDK reads it to verify the SHA matches
            ``expected_sha256`` before model_open.
        expected_sha256 : str
            64-char hex SHA-256 the caller expects. Anything else
            raises ``CHECKSUM_MISMATCH`` before the runtime is
            invoked.

        Raises
        ------
        OctomilError
            * ``INVALID_INPUT`` when the SDK is already serving a
              different model (caller must close + re-open) or when
              ``expected_sha256`` is malformed.
            * ``MODEL_NOT_FOUND`` when ``artifact_path`` doesn't
              exist or isn't readable.
            * ``CHECKSUM_MISMATCH`` when the file's actual SHA-256
              doesn't match ``expected_sha256``.
            * ``RUNTIME_UNAVAILABLE`` when the runtime dylib or the
              ``audio.transcription`` capability is missing, or when
              the runtime refuses the artifact because user uploads
              aren't enabled in this runtime build. The error message
              names the missing runtime support so operators know to
              upgrade the runtime side.

        Runtime support requirement
        ---------------------------
        The runtime must honor ``OCT_WHISPER_ALLOW_USER_ARTIFACTS=1``
        and skip its built-in digest-registry check, otherwise
        ``oct_runtime_advertise_caps`` will refuse to surface
        ``audio.transcription`` for the user's artifact. When that
        runtime support is missing we raise ``RUNTIME_UNAVAILABLE``
        with a diagnostic pointing operators at the gap rather than
        silently falling back to the canonical pin.
        """
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=("native STT (uploaded): expected_sha256 must be a " "64-char hex string"),
            )

        requested_model = model_name.lower()
        if self._runtime is not None:
            # OCT-113 v2 review2 P1 #2: idempotency keyed on the
            # FULL triple. Same model_name + same path + same sha →
            # genuine no-op. Any change (re-downloaded file, new
            # version pointed at the same slug, different sha) →
            # this is a NEW load and we refuse silent-serve-stale.
            same_name = self._model_name.lower() == requested_model
            same_path = self._uploaded_path == artifact_path
            same_sha = self._uploaded_sha == expected_sha256
            if same_name and same_path and same_sha:
                return  # genuine idempotent re-load
            if same_name and (not same_path or not same_sha):
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=(
                        f"native STT (uploaded): model "
                        f"{requested_model!r} already loaded with a "
                        f"different artifact_path or expected_sha256. "
                        "Call close() before re-loading; the SDK won't "
                        "silently keep serving stale bytes under the "
                        "same slug."
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"native STT backend already loaded "
                    f"{self._model_name!r}; cannot reuse it for "
                    f"uploaded model {requested_model!r}"
                ),
            )

        # Verify the file's actual SHA matches the catalog-supplied
        # checksum. Integrity check: protects against bytes mutating
        # between server-side storage and on-device load.
        try:
            st = os.stat(artifact_path)
        except OSError as exc:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message=(f"native STT (uploaded): artifact path not found " f"or unreadable: {artifact_path}"),
            ) from exc
        if not stat.S_ISREG(st.st_mode):
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message=(f"native STT (uploaded): artifact path is not a " f"regular file: {artifact_path}"),
            )
        try:
            actual_digest = _sha256_file_hex(artifact_path)
        except OSError as exc:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message=(f"native STT (uploaded): could not read artifact " f"for SHA check: {artifact_path}"),
            ) from exc
        if actual_digest != expected_sha256:
            raise OctomilError(
                code=OctomilErrorCode.CHECKSUM_MISMATCH,
                message=(
                    "native STT (uploaded): on-disk artifact "
                    f"SHA-256 ({actual_digest[:12]}…) does not match "
                    f"catalog-supplied checksum ({expected_sha256[:12]}…). "
                    "Re-download the artifact."
                ),
            )

        # OCT-113 v2 review2 P1 #3: capture prior env values BEFORE
        # mutating, so close() and failure paths can restore. We
        # store the original value (or None for "was unset") so
        # restore can delete-on-None rather than re-setting to empty.
        # _set_env_for_uploaded handles the snapshot + set; we restore
        # via _restore_env any time we raise after this point.
        self._set_env_for_uploaded(artifact_path)

        self._model_name = requested_model
        self._uploaded_path = artifact_path
        self._uploaded_sha = expected_sha256
        try:
            self._runtime = NativeRuntime.open()
        except NativeRuntimeError as exc:
            self._restore_env()
            self._model_name = ""
            self._uploaded_path = None
            self._uploaded_sha = None
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT (uploaded): failed to open runtime",
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        except ImportError as exc:
            self._runtime = None
            self._restore_env()
            self._model_name = ""
            self._uploaded_path = None
            self._uploaded_sha = None
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(f"native STT (uploaded): dylib not found ({exc})"),
            ) from exc

        if not _runtime_advertises_audio_transcription(self._runtime):
            last_error_lc = (self._runtime.last_error() or "").lower()
            self.close()  # restores env + clears all state
            # The runtime rejected our artifact. Without user-upload
            # support, this happens because the SHA isn't in the
            # built-in canonical registry. Surface a diagnostic that
            # explicitly names the missing runtime feature.
            if "digest" in last_error_lc:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        "native STT (uploaded): runtime refused the "
                        "artifact's SHA-256 because it isn't in the "
                        "built-in canonical Whisper registry. This "
                        "runtime build does not yet support user-"
                        "uploaded Whisper artifacts; rebuild with "
                        "OCT_WHISPER_ALLOW_USER_ARTIFACTS support."
                    ),
                )
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native STT (uploaded): runtime does not "
                    "advertise 'audio.transcription' after pointing "
                    f"at user artifact {artifact_path!r}."
                ),
            )

        try:
            self._model = self._runtime.open_model(
                model_uri=artifact_path,
                # Pass the file's actual SHA — runtime integrity
                # check now passes by construction.
                artifact_digest=f"sha256:{actual_digest}",
                engine_hint="whisper_cpp",
            )
            self._model.warm()
        except NativeRuntimeError as exc:
            self.close()  # restores env + clears all state
            raise _runtime_status_to_sdk_error(
                exc.status,
                (f"native STT (uploaded): failed to warm " f"{requested_model!r} model"),
                last_error=getattr(exc, "last_error", ""),
            ) from exc
        logger.debug(
            "NativeSttBackend: uploaded model %s warmed (sha %s…)",
            requested_model,
            actual_digest[:12],
        )

    def _set_env_for_uploaded(self, artifact_path: str) -> None:
        """Snapshot prior values then set the user-upload env vars.

        Called by ``load_uploaded_model``. ``close`` (or any failure
        path that calls ``_restore_env`` / ``self.close()``) reverses
        these mutations so subsequent canonical loads or other
        backend instances in the same process don't inherit
        ``OCTOMIL_WHISPER_BIN=<user_path>``.
        """
        if self._prior_env is None:
            self._prior_env = {
                _WHISPER_ALLOW_USER_ARTIFACTS_ENV: os.environ.get(_WHISPER_ALLOW_USER_ARTIFACTS_ENV),
                _WHISPER_BIN_ENV: os.environ.get(_WHISPER_BIN_ENV),
            }
        os.environ[_WHISPER_ALLOW_USER_ARTIFACTS_ENV] = "1"
        os.environ[_WHISPER_BIN_ENV] = artifact_path

    def _restore_env(self) -> None:
        """Reverse the env mutations applied by
        ``_set_env_for_uploaded``. Idempotent — safe to call from
        ``close()`` even when load_uploaded_model wasn't used.
        """
        if self._prior_env is None:
            return
        for key, prior in self._prior_env.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        self._prior_env = None

    def close(self) -> None:
        # Close the model BEFORE the runtime so oct_runtime_close
        # doesn't refuse with BUSY. Mirrors the embeddings backend.
        if self._model is not None:
            try:
                self._model.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeSttBackend.close: model.close failed", exc_info=True)
            self._model = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                logger.warning("NativeSttBackend.close: runtime.close failed", exc_info=True)
            self._runtime = None
        # OCT-113 v2 review2 P1 #3: restore the env vars
        # load_uploaded_model mutated. Subsequent canonical loads or
        # other backend instances in the same process now see the
        # original OCTOMIL_WHISPER_BIN (or no env var at all if it
        # wasn't set pre-load).
        self._restore_env()
        # Reset uploaded-mode bookkeeping so a future load via
        # either load_model or load_uploaded_model starts clean.
        self._model_name = ""
        self._uploaded_path = None
        self._uploaded_sha = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_pcm_f32: Any,
        *,
        sample_rate_hz: int = _WHISPER_SAMPLE_RATE_HZ,
        language: str = "en",
        deadline_ms: int | None = None,
    ) -> TranscriptionResult:
        """Run an ``audio.transcription`` request against the runtime.

        Parameters
        ----------
        audio_pcm_f32
            Mono PCM-f32 audio. ``bytes`` (float32-LE) OR a 1-D
            numpy float32 array OR a list of floats. Stereo /
            multichannel rejects ``INVALID_INPUT``.
        sample_rate_hz
            Hz. Must be 16000 for the native whisper.cpp STT path.
        language
            BCP-47 language hint. Echoed back on the result; the
            runtime is multilingual but v0.1.5 surfaces a single
            language tag per session.
        deadline_ms
            Per-request poll deadline. Falls back to
            ``self._default_deadline_ms`` (5 minutes) when None.

        Returns
        -------
        TranscriptionResult
            ``text`` is the concatenated final transcript; ``segments``
            carry timestamped spans; ``duration_ms`` is the decoded
            audio duration emitted with ``OCT_EVENT_TRANSCRIPT_FINAL``.

        Raises
        ------
        OctomilError
            See module docstring for the bounded error mapping.
        """
        if self._runtime is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="NativeSttBackend.transcribe called before load_model",
            )

        # Deadline validation BEFORE opening a session.
        resolved_deadline_ms = deadline_ms if deadline_ms is not None else self._default_deadline_ms
        if resolved_deadline_ms <= 0:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"deadline_ms must be > 0; got {resolved_deadline_ms}. Use None to "
                    f"fall back to the backend's default ({self.DEFAULT_DEADLINE_MS}ms)."
                ),
            )

        audio_bytes = _validate_pcm_f32(audio_pcm_f32, sample_rate_hz)
        if not math.isfinite(float(sample_rate_hz)):  # belt-and-suspenders
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"native STT: invalid sample_rate_hz {sample_rate_hz!r}",
            )

        if self._model is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "NativeSttBackend.transcribe: model not warmed; load_model() must succeed before transcribe()"
                ),
            )

        try:
            sess = self._runtime.open_session(
                capability=CAPABILITY_AUDIO_TRANSCRIPTION,
                locality="on_device",
                policy_preset="private",
                sample_rate_in=sample_rate_hz,
                model=self._model,
            )
        except NativeRuntimeError as exc:
            raise _runtime_status_to_sdk_error(
                exc.status,
                "native STT backend failed to open session",
                last_error=getattr(exc, "last_error", ""),
            ) from exc

        try:
            try:
                sess.send_audio(audio_bytes, sample_rate=sample_rate_hz, channels=1)
            except NativeRuntimeError as exc:
                raise _runtime_status_to_sdk_error(
                    exc.status,
                    "native STT backend send_audio failed",
                    last_error=getattr(exc, "last_error", ""),
                ) from exc

            segments: list[Segment] = []
            final_text = ""
            duration_ms = 0
            terminal_status: int = OCT_STATUS_OK
            saw_error = False
            error_message = ""
            saw_final = False

            deadline_seconds = resolved_deadline_ms / 1000.0
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                try:
                    ev = sess.poll_event(timeout_ms=200)
                except NativeRuntimeError as exc:
                    raise _runtime_status_to_sdk_error(
                        exc.status,
                        "native STT backend poll_event failed",
                        last_error=getattr(exc, "last_error", ""),
                    ) from exc
                if ev is None or ev.type == OCT_EVENT_NONE:
                    continue
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_SEGMENT:
                    # Defensive: clamp end_ms >= start_ms (the runtime
                    # guarantees this; a violation is a runtime bug
                    # but we don't want to silently emit broken spans).
                    if ev.segment_end_ms < ev.segment_start_ms:
                        raise OctomilError(
                            code=OctomilErrorCode.INFERENCE_FAILED,
                            message=(
                                f"native STT: transcript segment with end_ms < start_ms "
                                f"(start={ev.segment_start_ms}, end={ev.segment_end_ms})"
                            ),
                        )
                    segments.append(
                        Segment(
                            start_ms=int(ev.segment_start_ms),
                            end_ms=int(ev.segment_end_ms),
                            text=ev.text,
                        )
                    )
                    continue
                if ev.type == OCT_EVENT_TRANSCRIPT_FINAL:
                    final_text = ev.text
                    duration_ms = int(ev.final_duration_ms)
                    saw_final = True
                    continue
                if ev.type == OCT_EVENT_ERROR:
                    saw_error = True
                    if not error_message and self._runtime is not None:
                        error_message = self._runtime.last_error()
                    continue
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    terminal_status = int(ev.terminal_status)
                    break
            else:
                raise OctomilError(
                    code=OctomilErrorCode.REQUEST_TIMEOUT,
                    message=(f"native STT backend timed out waiting for SESSION_COMPLETED ({resolved_deadline_ms}ms)"),
                )

            if saw_error or terminal_status != OCT_STATUS_OK:
                raise _runtime_status_to_sdk_error(
                    terminal_status if terminal_status != OCT_STATUS_OK else OCT_STATUS_INVALID_INPUT,
                    "native STT backend reported error during transcription",
                    last_error=error_message,
                )

            if not saw_final:
                raise OctomilError(
                    code=OctomilErrorCode.INFERENCE_FAILED,
                    message=("native STT: SESSION_COMPLETED(OK) without preceding TRANSCRIPT_FINAL"),
                )

            return TranscriptionResult(
                text=final_text,
                segments=segments,
                language=language,
                duration_ms=duration_ms,
            )
        finally:
            sess.close()


# Alias so callers can choose either name. Keeps the spec's
# `NativeTranscriptionBackend` available without forcing a second
# class definition.
NativeTranscriptionBackend = NativeSttBackend


__all__ = [
    "NativeSttBackend",
    "NativeTranscriptionBackend",
    "Segment",
    "TranscriptionResult",
    "runtime_advertises_audio_transcription",
]
