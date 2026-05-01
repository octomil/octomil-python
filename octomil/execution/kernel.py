"""Shared execution kernel -- the single execution path for all Octomil surfaces.

Encapsulates:
  - model resolution
  - serving-policy evaluation
  - planner-driven runtime selection
  - local runtime resolution
  - cloud runtime resolution
  - fallback decisions
  - structured response generation
  - post-execution benchmark upload
  - telemetry

Does NOT require an HTTP server process.

Extracted sub-modules (prefer importing from them directly):
  - route_metadata_mapper: RouteMetadata types + builder
  - planner_resolution: planner call wrappers + candidate helpers
  - cloud_dispatch: cloud URL/key helpers
  - benchmark_upload: background benchmark telemetry upload
  - attempt_execution: locality decision logic + policy resolution
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from octomil._generated.message_role import MessageRole
from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CAPABILITY_TTS,
    CloudProfile,
    InlinePolicy,
    LoadedConfigSet,
    RequestOverrides,
    ResolvedExecutionDefaults,
    load_standalone_config,
    resolve_capability_defaults,
)
from octomil.errors import OctomilError, OctomilErrorCode

# --- Re-exports from extracted modules (backward compatibility) ---
from octomil.execution.attempt_execution import (  # noqa: F401
    _cloud_available,
    _inline_to_routing_policy,
    _resolve_localities,
    _resolve_routing_policy,
    _select_locality_for_capability,
)
from octomil.execution.benchmark_upload import (  # noqa: F401
    _sanitize_benchmark_payload,
    _upload_benchmark_async,
)
from octomil.execution.cloud_dispatch import (  # noqa: F401
    _cloud_api_key,
    _openai_base_url,
    _platform_api_base_url,
)
from octomil.execution.planner_resolution import (  # noqa: F401
    _PLANNER_CAPABILITY_MAP,
    _candidate_fallback_allowed,
    _candidate_to_selection,
    _is_synthetic_cloud_fallback,
    _resolve_planner_selection,
    _routing_policy_for_candidate,
    _runtime_model_for_selection,
    _selection_candidate_dicts,
)
from octomil.execution.route_metadata_mapper import (  # noqa: F401
    ArtifactCache,
    FallbackInfo,
    PlannerInfo,
    RouteArtifact,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteModelResolved,
    RouteReason,
    _route_metadata_from_selection,
)
from octomil.runtime.core.router import LOCALITY_CLOUD, LOCALITY_ON_DEVICE, RouterModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
)

logger = logging.getLogger(__name__)

# Capabilities whose adapters actually consume the prepared ``artifact_dir``
# today. A capability only enters this set once its dispatch path threads
# the ``model_dir`` from PrepareOutcome into the backend. Anything below
# is a false-success risk: prepare downloads bytes the next inference call
# ignores, then cold-starts through the engine's own lookup.
#
# Wiring history:
#   - tts             (PR 4 + 6 + 7)  — SherpaTtsEngine.create_backend(model_dir)
#   - transcription   (PR 10a)        — _WhisperBackend honors injected model_dir
#                                       and skips pywhispercpp's HF download path.
# Wiring backlog (intentionally NOT in the supported set):
#   - chat (kernel)   — MLXBackend and LlamaCppBackend accept ``model_dir``
#                       (PR 10c backend threading), but the capability is
#                       still gated because:
#                         (a) the public ``client.responses.create`` facade
#                             goes through ``OctomilResponses``, which does
#                             not thread ``model_dir`` into its engine
#                             ``create_backend`` calls — flipping ``chat``
#                             on the kernel would mean ``client.prepare``
#                             succeeds but the next ``client.responses.
#                             create`` cold-loads anyway;
#                         (b) PrepareManager materializes only single-file
#                             artifacts (``<dir>/artifact`` sentinel) and
#                             has no snapshot/manifest support, so MLX
#                             loads from a prepared dir don't work for
#                             real model shapes that mlx_lm requires
#                             (config.json + tokenizer + safetensors).
#                       ``chat`` and ``responses`` flip to wired once
#                       OctomilResponses goes through the kernel (or
#                       threads model_dir itself) AND PrepareManager grows
#                       snapshot materialization proven by an MLX e2e test.
#   - responses        — same gate as chat (dispatches through chat).
#   - embedding        — thread model_dir into the local embeddings backend.
_PREPAREABLE_CAPABILITIES = frozenset({CAPABILITY_TTS, CAPABILITY_TRANSCRIPTION})

# Capabilities whose dispatch path can construct + cache a backend
# ahead of first inference — i.e., where ``client.warmup()`` produces
# real first-call savings instead of a false-success. A capability
# enters this set only when both:
#   (a) it's already in ``_PREPAREABLE_CAPABILITIES`` (prepare bytes
#       on disk first), AND
#   (b) the kernel's local resolver consults the warmup cache before
#       calling ``engine.create_backend`` again.
# Today: tts + transcription. Chat / responses join once their
# OctomilResponses bypass and MLX snapshot materialization gates
# clear (same blockers as prepare).
_WARMUPABLE_CAPABILITIES = frozenset({CAPABILITY_TTS, CAPABILITY_TRANSCRIPTION})


# ---------------------------------------------------------------------------
# Result types (ExecutionResult, StreamChunk, ChatRoutingDecision stay here)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Unified result from any execution kernel call."""

    id: str = ""
    model: str = ""
    capability: str = ""
    locality: str = ""  # "on_device" | "cloud"
    fallback_used: bool = False
    output_text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    # Embeddings-specific
    embeddings: Optional[list[list[float]]] = None
    dimensions: Optional[int] = None
    # Transcription-specific
    segments: Optional[list[dict[str, Any]]] = None
    # Raw data for --json
    raw: Optional[dict[str, Any]] = None
    # Route metadata from planner
    route: Optional[RouteMetadata] = None


@dataclass
class StreamChunk:
    """A single chunk during streaming."""

    delta: str = ""
    done: bool = False
    result: Optional[ExecutionResult] = None


@dataclass
class WarmupOutcome:
    """Result of :meth:`ExecutionKernel.warmup`.

    ``client.warmup(model, capability)`` runs prepare (so the artifact
    bytes are on disk), constructs the local backend with the prepared
    ``model_dir``, calls ``backend.load_model``, and caches the
    instance so the next inference call reuses it. The outcome reports
    everything a caller needs to confirm warmup happened *and* drove
    real work — first-call latency savings only show up when
    ``backend_loaded`` is True.

    Fields
    ~~~~~~
    - ``capability`` / ``model``: the warmed cell.
    - ``prepare_outcome``: the underlying :class:`PrepareOutcome`. The
      same fields the prepare CLI prints (artifact_id, artifact_dir,
      cached, files map).
    - ``backend_loaded``: True when ``engine.create_backend`` +
      ``backend.load_model`` both succeeded; False when warmup ran
      but a downstream backend constructor refused (engine missing on
      this host, runtime imports failed, etc.) — prepare bytes are
      still on disk, just not loaded.
    - ``latency_ms``: wall time for the full prepare + load loop.
    """

    capability: str
    model: str
    prepare_outcome: Any
    backend_loaded: bool
    latency_ms: float


@dataclass
class ChatRoutingDecision:
    """Routing decision for a chat request — used by serve to dispatch to the right backend.

    Does not create runtimes or execute inference.  Tells the caller which
    locality to try first, whether fallback is available, and which cloud
    profile to use.
    """

    model: str
    primary_locality: str  # "on_device" | "cloud"
    fallback_locality: Optional[str] = None  # "on_device" | "cloud" | None
    cloud_profile: Optional[CloudProfile] = None
    policy_preset: Optional[str] = None
    inline_policy: Optional[InlinePolicy] = None


def _internal_locality_for_attempt(attempt: Any) -> str:
    if attempt is not None and getattr(attempt, "locality", None) == "cloud":
        return LOCALITY_CLOUD
    return LOCALITY_ON_DEVICE


# ---------------------------------------------------------------------------
# Streaming TTS helpers — keep the kernel.synthesize_speech_stream body small
# by factoring locality-specific producers here.
# ---------------------------------------------------------------------------


def _backend_synthesize_kwargs(backend: Any, resolved_speaker: Any) -> dict[str, Any]:
    """Engine-facing kwargs derived from a :class:`ResolvedTtsSpeaker`.

    Backends that opt into speaker-aware synthesis declare a
    ``speaker_kwargs`` attribute or accept ``speaker_profile=``. We
    detect that by inspection so the call site can stay uniform across
    engine families: legacy Kokoro/Piper consume only ``voice``, while
    Pocket consumes ``speaker_profile``.
    """
    accepts_speaker = bool(getattr(backend, "accepts_speaker_profile", False))
    if accepts_speaker and resolved_speaker is not None:
        return {"speaker_profile": resolved_speaker}
    return {}


def _call_backend_synthesize(
    backend: Any,
    text: str,
    resolved_speaker: Any,
    speed: float,
) -> dict[str, Any]:
    """Synchronous bridge to ``backend.synthesize`` that respects the resolver.

    Used by ``synthesize_speech`` under ``asyncio.to_thread``. Pulls the
    native voice off the resolved profile and adds ``speaker_profile``
    when the backend opts in. Keeps the kernel call site terse.
    """
    voice = resolved_speaker.native_voice if resolved_speaker is not None else None
    extra = _backend_synthesize_kwargs(backend, resolved_speaker)
    return backend.synthesize(text, voice, speed, **extra)


def _build_local_realtime_stream(
    *,
    backend: Any,
    text: str,
    voice: Optional[str] = None,
    resolved_speaker: Any = None,
    speed: float,
    runtime_model: str,
    policy_preset: Optional[str],
    fallback_used: bool,
    sdk_t0: float,
) -> Any:
    """Wrap a backend's ``synthesize_stream`` in a typed event stream.

    ``sdk_t0`` is the monotonic timestamp captured at the SDK call
    boundary (``FacadeSpeech.stream`` entry). All customer-visible
    metrics measure FROM that timestamp so callers see the real
    end-to-end latency they experience, not just the engine's
    self-reported portion.
    """
    from octomil.audio.streaming import (
        SAMPLE_FORMAT_PCM_S16LE,
        SpeechAudioChunk,
        SpeechStream,
        SpeechStreamCompleted,
        SpeechStreamStarted,
        TtsStreamingCapability,
    )

    # Voice resolution must run synchronously here so an unsupported
    # explicit voice raises BEFORE we hand the caller a stream object
    # whose producer would emit SpeechStreamStarted ahead of the real
    # error. Backends without validate_voice (test fakes) fall back to
    # advisory metadata only.
    sample_rate = int(getattr(backend, "_sample_rate", 24000) or 24000)
    # Effective voice the backend should use for native-voice
    # validation. Prefer the resolver's native_voice when supplied
    # (covers logical-speaker -> native_voice planner mappings) and
    # fall back to the legacy ``voice=`` kwarg for callers that
    # haven't migrated.
    effective_voice = (
        getattr(resolved_speaker, "native_voice", None) if resolved_speaker is not None else None
    ) or voice
    validate_voice = getattr(backend, "validate_voice", None)
    if callable(validate_voice):
        _sid_unused, resolved_voice = validate_voice(effective_voice)
    else:
        resolved_voice = (effective_voice or getattr(backend, "_default_voice", "") or None) or None

    # Honest capability advertisement: ask the backend what it will
    # actually do for THIS input. Backends without
    # streaming_capability default to final_chunk so the SDK never
    # over-promises.
    streaming_capability_fn = getattr(backend, "streaming_capability", None)
    if callable(streaming_capability_fn):
        advertised = streaming_capability_fn(text)
    else:
        advertised = TtsStreamingCapability.final_only(verified=False)

    extra_kwargs = _backend_synthesize_kwargs(backend, resolved_speaker)
    inner = backend.synthesize_stream(text, effective_voice, speed, **extra_kwargs)

    async def producer():
        # Setup window: SDK call entry -> SpeechStreamStarted emitted.
        # Includes routing, voice validation, backend acquisition, and
        # any future scheduler-queue time.
        setup_ms = (time.monotonic() - sdk_t0) * 1000.0

        engine_t0: Optional[float] = None
        engine_first_chunk_ms: Optional[float] = None
        e2e_first_chunk_ms: Optional[float] = None
        sample_index = 0
        observed_chunks = 0

        yield SpeechStreamStarted(
            model=runtime_model,
            voice=resolved_voice,
            sample_rate=sample_rate,
            channels=1,
            sample_format=SAMPLE_FORMAT_PCM_S16LE,
            streaming_capability=advertised,
            locality="on_device",
            engine="sherpa-onnx",
        )

        # Mark engine_t0 immediately AFTER Started is emitted — i.e.
        # right before we start awaiting backend chunks. This is the
        # boundary the engine's own TTFB should be measured from, so
        # callers can attribute the SDK setup cost vs. the engine
        # synthesis cost separately.
        engine_t0 = time.monotonic()
        try:
            async for raw in inner:
                pcm: bytes = raw["pcm_s16le"]
                n = int(raw.get("num_samples") or (len(pcm) // 2))
                sample_index += n
                if engine_first_chunk_ms is None:
                    now = time.monotonic()
                    engine_first_chunk_ms = (now - engine_t0) * 1000.0
                    e2e_first_chunk_ms = (now - sdk_t0) * 1000.0
                observed_chunks += 1
                yield SpeechAudioChunk(
                    data=pcm,
                    sample_index=sample_index,
                    timestamp_ms=int(round(1000 * sample_index / sample_rate)) if sample_rate else 0,
                    is_final=False,
                )
        finally:
            inner_close = getattr(inner, "aclose", None)
            if inner_close is not None:
                try:
                    await inner_close()
                except Exception:
                    pass

        total_latency_ms = (time.monotonic() - sdk_t0) * 1000.0
        duration_ms = int(round(1000 * sample_index / sample_rate)) if sample_rate else 0

        # Verification: the advertised capability claims a cadence;
        # confirm the actual run delivered it. ``sentence_chunk`` /
        # ``progressive`` advertised + only one chunk observed
        # downgrades to ``final_chunk`` with verified=False so callers
        # know the engine over-promised on this input.
        observed_capability = _verify_capability(advertised, observed_chunks=observed_chunks)

        yield SpeechStreamCompleted(
            duration_ms=duration_ms,
            total_samples=sample_index,
            sample_rate=sample_rate,
            channels=1,
            sample_format=SAMPLE_FORMAT_PCM_S16LE,
            streaming_capability=observed_capability,
            setup_ms=setup_ms,
            engine_first_chunk_ms=engine_first_chunk_ms,
            e2e_first_chunk_ms=e2e_first_chunk_ms,
            total_latency_ms=total_latency_ms,
            observed_chunks=observed_chunks,
            capability_verified=observed_capability.verified,
        )

    async def _on_cancel() -> None:
        inner_close = getattr(inner, "aclose", None)
        if inner_close is not None:
            try:
                await inner_close()
            except Exception:
                pass

    return SpeechStream(producer(), on_cancel=_on_cancel)


def _backend_can_stream(backend: Any) -> bool:
    """Return True iff ``backend`` exposes the streaming protocol.

    Replaces the legacy ``backend.supports_streaming: bool`` flag.
    The contract is now: a streaming backend implements
    ``synthesize_stream(text, voice, speed)`` and may advertise a
    :class:`TtsStreamingCapability` via ``streaming_capability(text)``
    (defaulting to ``final_only`` when absent). Backends that only
    implement ``synthesize`` are not streaming backends.
    """
    return callable(getattr(backend, "synthesize_stream", None))


def _verify_capability(advertised: Any, *, observed_chunks: int) -> Any:
    """Return the actually-observed capability after a stream completes.

    If the engine advertised ``sentence_chunk`` or ``progressive`` but
    only delivered one chunk, the run was effectively ``final_chunk``;
    verify=False on the returned capability so callers can stop
    trusting the advertised label for this (model, input) shape.
    Single-chunk runs that *advertised* final_chunk pass verification.
    """
    from octomil.audio.streaming import TtsStreamingCapability, TtsStreamingMode

    if advertised.mode == TtsStreamingMode.FINAL_CHUNK:
        # Final-chunk advertised: any chunk count <= 1 is correct.
        return TtsStreamingCapability.final_only(verified=observed_chunks <= 1)
    if observed_chunks > 1:
        # Sub-utterance cadence delivered as advertised.
        return advertised.__class__(
            mode=advertised.mode,
            granularity=advertised.granularity,
            verified=True,
        )
    # Advertised better than delivered → downgrade and flag.
    return TtsStreamingCapability.final_only(verified=False)


def _build_cloud_final_chunk_stream(
    *,
    kernel: "ExecutionKernel",
    cloud_model: str,
    text: str,
    voice: Optional[str],
    speed: float,
    profile: CloudProfile,
    response_format: str,
    runtime_model: str,
    policy_preset: Optional[str],
    fallback_used: bool,
    sdk_t0: float,
) -> Any:
    """Wrap a non-streaming cloud TTS call as a single-chunk event stream.

    Always advertises ``final_chunk`` — the hosted endpoint returns
    one WAV per request, no per-sentence cadence. The SDK strips the
    RIFF header and emits the PCM body for ``response_format=
    pcm_s16le`` parity with the local stream. Metrics measure from
    the SDK call boundary (``sdk_t0``).
    """
    from octomil.audio.streaming import (
        SAMPLE_FORMAT_PCM_S16LE,
        SpeechAudioChunk,
        SpeechStream,
        SpeechStreamCompleted,
        SpeechStreamStarted,
        TtsStreamingCapability,
    )

    advertised = TtsStreamingCapability.final_only(verified=False)

    async def producer():
        cloud_result = await kernel._cloud_synthesize_speech(cloud_model, text, voice, "wav", speed, profile)
        wav_bytes = cloud_result["audio_bytes"]
        sample_rate = int(cloud_result.get("sample_rate") or 24000)
        pcm_body = _strip_wav_header(wav_bytes)
        if response_format == "wav":
            data = wav_bytes
        else:
            data = pcm_body
        total_samples = len(pcm_body) // 2 if pcm_body else 0
        duration_ms = int(cloud_result.get("duration_ms") or 0) or (
            int(round(1000 * total_samples / sample_rate)) if sample_rate else 0
        )

        # Setup spans the entire cloud round-trip — by the time we
        # have audio bytes back from the provider, "Started" emission
        # is the same instant as "first chunk". That's honest, but
        # also makes setup_ms ~= total_latency_ms here, which is
        # exactly the signal callers need to see that final_chunk
        # mode does NOT give them TTFB benefit.
        setup_ms = (time.monotonic() - sdk_t0) * 1000.0
        sample_format = SAMPLE_FORMAT_PCM_S16LE if response_format != "wav" else "wav"

        yield SpeechStreamStarted(
            model=runtime_model,
            voice=voice,
            sample_rate=sample_rate,
            channels=1,
            sample_format=sample_format,
            streaming_capability=advertised,
            locality="cloud",
            engine=None,
        )
        e2e_first_chunk_ms: Optional[float] = None
        observed_chunks = 0
        if data:
            now = time.monotonic()
            e2e_first_chunk_ms = (now - sdk_t0) * 1000.0
            observed_chunks = 1
            yield SpeechAudioChunk(
                data=data,
                sample_index=total_samples,
                timestamp_ms=duration_ms,
                is_final=True,
            )
        total_latency_ms = (time.monotonic() - sdk_t0) * 1000.0
        observed_capability = _verify_capability(advertised, observed_chunks=observed_chunks)
        yield SpeechStreamCompleted(
            duration_ms=duration_ms,
            total_samples=total_samples,
            sample_rate=sample_rate,
            channels=1,
            sample_format=sample_format,
            streaming_capability=observed_capability,
            setup_ms=setup_ms,
            engine_first_chunk_ms=None,  # cloud branch has no in-engine start signal
            e2e_first_chunk_ms=e2e_first_chunk_ms,
            total_latency_ms=total_latency_ms,
            observed_chunks=observed_chunks,
            capability_verified=observed_capability.verified,
        )

    return SpeechStream(producer())


def _strip_wav_header(wav_bytes: bytes) -> bytes:
    """Return PCM body of a RIFF/WAVE byte string, or the input unchanged
    if it doesn't start with a parseable RIFF header.

    Cheap shim — avoids depending on :mod:`wave` for the same reason
    :func:`octomil.audio.streaming.pcm_s16le_to_wav_bytes` is hand-rolled.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes
    # Walk the chunks until we find ``data``.
    idx = 12
    while idx + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[idx : idx + 4]
        chunk_size = int.from_bytes(wav_bytes[idx + 4 : idx + 8], "little")
        if chunk_id == b"data":
            return wav_bytes[idx + 8 : idx + 8 + chunk_size]
        idx += 8 + chunk_size
    return wav_bytes


# ---------------------------------------------------------------------------
# Execution Kernel
# ---------------------------------------------------------------------------


class ExecutionKernel:
    """Shared execution kernel for all Octomil surfaces.

    Usage::

        kernel = ExecutionKernel()
        result = await kernel.create_response("Hello!", model="gemma3-1b")
    """

    def __init__(
        self,
        *,
        config_set: Optional[LoadedConfigSet] = None,
        start_dir: Optional[Path] = None,
        prepare_manager: Optional[Any] = None,
    ) -> None:
        self._config_set = config_set or load_standalone_config(start_dir)
        # Optional caller-supplied PrepareManager. Tests and embedded callers
        # inject one to control where artifacts land and stub the downloader;
        # production code lazily constructs a default-rooted manager on first
        # prepare call.
        self._prepare_manager: Optional[Any] = prepare_manager
        # PR 11: warmup cache. ``client.warmup(model, capability)`` runs
        # prepare and then constructs + load_model's the local backend
        # so the next inference dispatch can reuse it without paying
        # cold-start latency. Keyed by ``_warmup_cache_key`` —
        # ``(capability, runtime_model, digest, format, quantization)``
        # — so distinct artifact identities (different versions, MLX
        # snapshot vs GGUF, etc.) sharing the same runtime model don't
        # alias. The local resolvers
        # (``_resolve_local_tts_backend`` /
        # ``_resolve_local_transcription_backend``) call
        # ``_lookup_warmed_backend`` which composes the same key
        # against the *current* planner selection. Reset by
        # ``release_warmed_backends`` (test helper).
        self._warmed_backends: dict[tuple, Any] = {}

    @property
    def config_set(self) -> LoadedConfigSet:
        return self._config_set

    def resolve_chat_defaults(
        self,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ResolvedExecutionDefaults:
        """Resolve chat defaults without executing inference."""
        return self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)

    # ------------------------------------------------------------------
    # Chat routing decision (for serve integration)
    # ------------------------------------------------------------------

    def resolve_chat_routing(
        self,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        local_available: bool,
        cloud_available: Optional[bool] = None,
    ) -> ChatRoutingDecision:
        """Resolve routing without executing inference.

        Returns a ``ChatRoutingDecision`` that tells the caller which
        backend locality to use as primary and which (if any) as fallback.
        Serve uses this to dispatch to its own ``InferenceBackend`` instances.
        """
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        routing_policy = _resolve_routing_policy(defaults)

        # Determine cloud availability from config when caller does not know
        if cloud_available is None:
            cloud_available = _cloud_available(defaults)

        primary, fallback = _resolve_localities(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
        )

        return ChatRoutingDecision(
            model=effective_model,
            primary_locality=primary,
            fallback_locality=fallback,
            cloud_profile=defaults.cloud_profile,
            policy_preset=defaults.policy_preset,
            inline_policy=defaults.inline_policy,
        )

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    async def create_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Create a one-shot response. The primary execution entrypoint."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"

        # Planner-driven routing
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # PR B: explicit local-only / private policy forces a
        # cloud_only-rejected candidate list so a planner outage can't
        # leak the request to cloud even if cloud creds are present.
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        # PR 10c: drop *individually* unpreparable local sdk_runtime
        # candidates (synthetic plan: missing digest/url, traversal,
        # etc.) so the runner doesn't burn an attempt cold-loading
        # through the engine's own download path. Other local
        # candidates in the same plan survive.
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
        # PR B: when local-only forcing emptied the candidate list
        # (planner gave only cloud, no local) and the caller insisted
        # on local, surface an actionable error rather than a generic
        # "no runtime available".
        if _is_local_only_policy(policy) and not candidates:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"local_runtime_unavailable: caller passed policy={policy!r} but "
                    f"no local candidate is available for {effective_model!r}. "
                    "Install the relevant local engine extras or relax the policy."
                ),
            )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part(prompt)],
                ),
            ],
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=False)

        # Cache prepared dirs by artifact_id within this call so a
        # local-then-local retry (rare but possible under fallback)
        # doesn't re-download. Prepare runs INSIDE the local attempt:
        # cloud-first plans never touch local prepare.
        prepared_dirs: dict[tuple, Optional[str]] = {}

        async def _execute_candidate(candidate: dict[str, Any]) -> RuntimeResponse:
            candidate_selection = _candidate_to_selection(selection, candidate)
            local_dir: Optional[str] = None
            if candidate.get("locality") == "local":
                planner_candidate = runner_to_original.get(id(candidate))
                local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
            router = await self._build_router(
                effective_model,
                CAPABILITY_CHAT,
                defaults,
                planner_selection=candidate_selection,
                prepared_model_dir=local_dir,
                cloud_execution_model=_execution_model_for_cloud_dispatch(
                    requested_model=model,
                    effective_model=effective_model,
                    planner_model=planner_model,
                    app=app,
                ),
            )
            return await router.run(request, policy=_routing_policy_for_candidate(candidate))

        t0 = time.monotonic()
        attempt_loop = await runner.run_with_inference(
            candidates,
            execute_candidate=_execute_candidate,
        )
        if not attempt_loop.succeeded:
            if attempt_loop.error is not None:
                raise attempt_loop.error
            raise RuntimeError("No runtime available")

        response = attempt_loop.value
        if not isinstance(response, RuntimeResponse):
            raise RuntimeError("Runtime returned an invalid response.")
        latency_ms = (time.monotonic() - t0) * 1000

        locality = _internal_locality_for_attempt(attempt_loop.selected_attempt)
        is_fallback = attempt_loop.fallback_used
        route = _route_metadata_from_selection(
            selection,
            locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
            attempt_loop=attempt_loop,
        )
        usage = _extract_usage(response)

        # Post-execution benchmark upload for local execution
        if locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                tokens_per_second=usage.get("output_tokens", 0) / max(latency_ms / 1000, 0.001),
                latency_ms=latency_ms,
            )

        return ExecutionResult(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            model=effective_model,
            capability=CAPABILITY_CHAT,
            locality=locality,
            fallback_used=is_fallback,
            output_text=response.text,
            usage=usage,
            route=route,
        )

    async def stream_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response token by token."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )
        routing_policy = _resolve_routing_policy(defaults)
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
            if not candidates:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_runtime_unavailable: caller passed policy={policy!r} but "
                        f"no local candidate is available for {effective_model!r}."
                    ),
                )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part(prompt)],
                ),
            ],
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=True)

        prepared_dirs: dict[tuple, Optional[str]] = {}

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                local_dir: Optional[str] = None
                if candidate.get("locality") == "local":
                    planner_candidate = runner_to_original.get(id(candidate))
                    local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
                    prepared_model_dir=local_dir,
                    cloud_execution_model=_execution_model_for_cloud_dispatch(
                        requested_model=model,
                        effective_model=effective_model,
                        planner_model=planner_model,
                        app=app,
                    ),
                )
                async for chunk in router.stream(request, policy=_routing_policy_for_candidate(candidate)):
                    text = chunk.text or ""
                    if text:
                        first_token_emitted = True
                    collected_text += text
                    yield StreamChunk(delta=text)
                selected_locality = LOCALITY_CLOUD if candidate.get("locality") == "cloud" else LOCALITY_ON_DEVICE
                is_fallback = idx > 0
                break
            except Exception as exc:
                last_error = exc
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No runtime available")

        latency_ms = (time.monotonic() - t0) * 1000

        route = _route_metadata_from_selection(
            selection,
            selected_locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
        )

        if selected_locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                latency_ms=latency_ms,
            )

        yield StreamChunk(
            delta="",
            done=True,
            result=ExecutionResult(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                model=effective_model,
                capability=CAPABILITY_CHAT,
                locality=selected_locality,
                fallback_used=is_fallback,
                output_text=collected_text,
                route=route,
            ),
        )

    async def stream_chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a multi-turn chat request through the shared router."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
            if not candidates:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_runtime_unavailable: caller passed policy={policy!r} but "
                        f"no local candidate is available for {effective_model!r}."
                    ),
                )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=_chat_messages_to_runtime_messages(messages),
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=True)

        prepared_dirs: dict[tuple, Optional[str]] = {}

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                local_dir: Optional[str] = None
                if candidate.get("locality") == "local":
                    planner_candidate = runner_to_original.get(id(candidate))
                    local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
                    prepared_model_dir=local_dir,
                    cloud_execution_model=_execution_model_for_cloud_dispatch(
                        requested_model=model,
                        effective_model=effective_model,
                        planner_model=planner_model,
                        app=app,
                    ),
                )
                async for chunk in router.stream(request, policy=_routing_policy_for_candidate(candidate)):
                    text = chunk.text or ""
                    if text:
                        first_token_emitted = True
                    collected_text += text
                    yield StreamChunk(delta=text)
                selected_locality = LOCALITY_CLOUD if candidate.get("locality") == "cloud" else LOCALITY_ON_DEVICE
                is_fallback = idx > 0
                break
            except Exception as exc:
                last_error = exc
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No runtime available")

        latency_ms = (time.monotonic() - t0) * 1000

        route = _route_metadata_from_selection(
            selection,
            selected_locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
        )

        if selected_locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                latency_ms=latency_ms,
            )

        yield StreamChunk(
            delta="",
            done=True,
            result=ExecutionResult(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                model=effective_model,
                capability=CAPABILITY_CHAT,
                locality=selected_locality,
                fallback_used=is_fallback,
                output_text=collected_text,
                route=route,
            ),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        inputs: list[str],
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ExecutionResult:
        """Generate embeddings for one or more text inputs."""
        defaults = self._resolve(CAPABILITY_EMBEDDING, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_EMBEDDING)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(
            effective_model=effective_model, app=app, capability=CAPABILITY_EMBEDDING
        )
        selection = _resolve_planner_selection(planner_model, CAPABILITY_EMBEDDING, policy_preset)

        # PR B follow-up: apply the same app-ref refusal gate as
        # TTS / transcription / chat. Embeddings was the lone surface
        # where ``model='nomic-...', app='private-app', policy=None``
        # plus a planner outage could still silently route to cloud.
        # The refusal is the same shape: with no explicit policy AND
        # an app-scoped request AND no planner selection, raise so
        # the caller picks ``local_only`` / ``cloud_first`` or fixes
        # planner connectivity.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)

        local_available = self._can_local(effective_model, CAPABILITY_EMBEDDING)
        cloud_available = _cloud_available(defaults)
        # Same explicit local-only / private cloud-disable as the
        # other dispatchers: even if cloud creds are present in env,
        # ``policy='local_only'`` blocks promotion to cloud.
        if _is_local_only_policy(policy):
            cloud_available = False
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_EMBEDDING,
        )

        route = _route_metadata_from_selection(
            selection, locality, is_fallback, model_name=effective_model, capability=CAPABILITY_EMBEDDING
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            # PR B follow-up: when ``app=`` was explicit (or ``model``
            # was already an app ref), cloud dispatch must run under
            # the app identity, not the resolved underlying model.
            # Otherwise the planner sees ``@app/foo/embeddings`` but
            # the hosted call goes out under ``nomic-...`` and the
            # server can't apply the app's quota / policy / billing.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=model,
                effective_model=effective_model,
                planner_model=planner_model,
                app=app,
            )
            result = await self._cloud_embed(inputs, cloud_model, defaults.cloud_profile, is_fallback)
            result.route = route
            return result

        t0 = time.monotonic()
        result = await self._local_embed(inputs, effective_model, is_fallback)
        latency_ms = (time.monotonic() - t0) * 1000
        result.route = route

        _upload_benchmark_async(
            model=effective_model,
            capability=CAPABILITY_EMBEDDING,
            engine=route.execution.engine if route.execution else None,
            policy_preset=policy_preset,
            latency_ms=latency_ms,
        )

        return result

    async def _cloud_embed(
        self,
        inputs: list[str],
        model: str,
        profile: CloudProfile,
        fallback_used: bool = False,
    ) -> ExecutionResult:
        """Dispatch embeddings to cloud."""
        from octomil.embeddings import embed

        api_key = os.environ.get(profile.api_key_env, "")
        base_url = _platform_api_base_url(profile)
        if not api_key:
            raise RuntimeError(
                f"Cloud embedding requires {profile.api_key_env} to be set.\n\n"
                f"Export {profile.api_key_env} or configure a cloud profile."
            )

        result = embed(
            server_url=base_url,
            api_key=api_key,
            model_id=model,
            input=inputs if len(inputs) > 1 else inputs[0],
        )

        dims = len(result.embeddings[0]) if result.embeddings else 0
        return ExecutionResult(
            id=f"emb_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_EMBEDDING,
            locality=LOCALITY_CLOUD,
            fallback_used=fallback_used,
            embeddings=result.embeddings,
            dimensions=dims,
            usage={
                "input_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )

    async def _local_embed(self, inputs: list[str], model: str, fallback_used: bool = False) -> ExecutionResult:
        """Dispatch embeddings to a local runtime."""
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()
        runtime = registry.resolve(model)
        if runtime is None:
            raise RuntimeError(f"No local runtime found for embedding model '{model}'.")

        embed_fn = getattr(runtime, "embed", None) or getattr(runtime, "create_embeddings", None)
        if embed_fn is None:
            raise RuntimeError(f"Local runtime for embedding model '{model}' does not expose an embedding interface.")

        maybe_result = embed_fn(inputs)
        if hasattr(maybe_result, "__await__"):
            maybe_result = await maybe_result

        if hasattr(maybe_result, "embeddings"):
            all_vectors = maybe_result.embeddings
            usage_obj = getattr(maybe_result, "usage", None)
            total_tokens = getattr(usage_obj, "total_tokens", 0) if usage_obj is not None else 0
        else:
            all_vectors = maybe_result
            total_tokens = sum(len(text.split()) for text in inputs)

        if not isinstance(all_vectors, list):
            raise RuntimeError(f"Local embedding runtime for '{model}' returned an invalid embedding result.")

        dims = len(all_vectors[0]) if all_vectors and all_vectors[0] else 0
        return ExecutionResult(
            id=f"emb_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_EMBEDDING,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=fallback_used,
            embeddings=all_vectors,
            dimensions=dims,
            usage={"input_tokens": total_tokens, "total_tokens": total_tokens},
        )

    # ------------------------------------------------------------------
    # Audio Transcription
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ExecutionResult:
        """Transcribe audio to text."""
        defaults = self._resolve(CAPABILITY_TRANSCRIPTION, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TRANSCRIPTION)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(
            effective_model=effective_model, app=app, capability=CAPABILITY_TRANSCRIPTION
        )
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TRANSCRIPTION, policy_preset)

        # PR B: same app-ref refusal + local-only forcing as TTS.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # Routing is satisfied if a backend is already staged OR if the
        # planner emitted a preparable sdk_runtime candidate the kernel
        # can materialize. BUT if the planner emits an *unpreparable*
        # prepare_required=True candidate (synthetic plan: no digest/url,
        # traversal, etc.), the SDK cannot honor the plan even if a
        # backend happens to be importable on disk — that staged-backend
        # case used to slip past via the `or` short-circuit and crash in
        # prepare(). The unpreparable-candidate veto fires first.
        if self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._has_local_transcription_backend(effective_model) or (
                self._can_prepare_local_transcription(effective_model, selection)
            )
        cloud_available = _cloud_available(defaults)
        if _is_local_only_policy(policy):
            cloud_available = False
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_TRANSCRIPTION,
        )

        route = _route_metadata_from_selection(
            selection, locality, is_fallback, model_name=effective_model, capability=CAPABILITY_TRANSCRIPTION
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            # PR B follow-up: send the app ref to cloud when ``app=``
            # was explicit so the server keeps app-scoping. Local
            # ``effective_model`` keeps driving local backend
            # resolution below.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=model,
                effective_model=effective_model,
                planner_model=planner_model,
                app=app,
            )
            result = await self._cloud_transcribe(
                audio_data, cloud_model, defaults.cloud_profile, language, is_fallback
            )
            result.route = route
            return result

        # PR 10: prepare transcription artifact (whisper.cpp model file)
        # before backend load, mirroring the TTS path. The downstream
        # _resolve_local_transcription_backend threads model_dir into
        # engine.create_backend so whisper.cpp loads the prepared file
        # instead of triggering pywhispercpp's own download path.
        prepared_dir = self._prepare_local_transcription_artifact(selection)
        # PR 11 follow-up: thread the planner candidate through to the
        # backend resolver so the warmup-cache lookup uses the current
        # request's artifact identity, not a re-planned one.
        local_candidate = _local_sdk_runtime_candidate(selection)

        t0 = time.monotonic()
        result = await self._local_transcribe(
            audio_data,
            effective_model,
            language,
            is_fallback,
            prepared_model_dir=prepared_dir,
            planner_candidate=local_candidate,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        result.route = route

        _upload_benchmark_async(
            model=effective_model,
            capability=CAPABILITY_TRANSCRIPTION,
            engine=route.execution.engine if route.execution else None,
            policy_preset=policy_preset,
            latency_ms=latency_ms,
        )

        return result

    async def _cloud_transcribe(
        self,
        audio_data: bytes,
        model: str,
        profile: CloudProfile,
        language: Optional[str],
        fallback_used: bool = False,
    ) -> ExecutionResult:
        """Dispatch audio transcription to a hosted OpenAI-compatible endpoint."""
        import httpx

        api_key = os.environ.get(profile.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Cloud transcription requires {profile.api_key_env} to be set.\n\n"
                f"Export {profile.api_key_env} or configure a cloud profile."
            )

        data = {"model": model}
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(base_url=_openai_base_url(profile), headers=headers, timeout=120.0) as client:
            response = await client.post(
                "/audio/transcriptions",
                data=data,
                files={"file": ("audio.wav", audio_data, "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()

        return ExecutionResult(
            id=f"txn_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_TRANSCRIPTION,
            locality=LOCALITY_CLOUD,
            fallback_used=fallback_used,
            output_text=payload.get("text", ""),
            segments=payload.get("segments"),
            raw=payload,
        )

    async def _local_transcribe(
        self,
        audio_data: bytes,
        model: str,
        language: Optional[str],
        fallback_used: bool = False,
        *,
        prepared_model_dir: Optional[str] = None,
        planner_candidate: Optional[Any] = None,
    ) -> ExecutionResult:
        """Dispatch audio transcription to a local Whisper-compatible backend.

        ``prepared_model_dir`` is the directory PrepareManager materialized
        the artifact under. When set, the resolver passes it as
        ``model_dir`` to the engine so whisper.cpp loads the prepared file
        instead of going through pywhispercpp's own download path.

        ``planner_candidate`` (when known) is threaded through so the
        warmup-cache lookup uses the *current* request's artifact
        identity instead of re-planning from ``model``.
        """
        backend = self._resolve_local_transcription_backend(
            model,
            prepared_model_dir=prepared_model_dir,
            planner_candidate=planner_candidate,
        )
        if backend is None:
            raise RuntimeError(f"No local transcription runtime found for model '{model}'.")

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(fd, audio_data)
            os.close(fd)
            result = await asyncio.to_thread(backend.transcribe, tmp_path)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        segments = result.get("segments") if isinstance(result, dict) else None
        return ExecutionResult(
            id=f"txn_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_TRANSCRIPTION,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=fallback_used,
            output_text=text,
            segments=segments,
            raw=result if isinstance(result, dict) else None,
        )

    # ------------------------------------------------------------------
    # Audio Speech (TTS)
    # ------------------------------------------------------------------

    async def synthesize_speech(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        speaker: Optional[str] = None,
        response_format: str = "wav",
        speed: float = 1.0,
        app: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> Any:
        """Routed TTS synthesis. Returns a ``SpeechResponse``.

        Resolves the model ref through the same planner pipeline as the
        other capabilities. Local execution dispatches to ``SherpaTtsEngine``;
        cloud execution wraps the existing
        ``octomil.hosted.HostedSpeech.create`` so the kernel does not
        duplicate hosted request shaping or error handling.
        """
        from octomil.audio.speech import SpeechResponse, SpeechRoute

        if not input or not input.strip():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a non-empty string.",
            )

        requested_model = model
        defaults = self._resolve(CAPABILITY_TTS, model=requested_model, policy=policy, app=app)
        effective_model = defaults.model or requested_model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TTS)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_TTS)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TTS, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)

        planner_policy = _tts_policy_from_selection(selection)
        if planner_policy:
            policy_preset = planner_policy
            defaults.policy_preset = planner_policy
            defaults.inline_policy = None

        # PR B: refuse to silently route an @app/... ref to cloud when
        # planner resolution failed AND the caller didn't pass an
        # explicit policy. Without this, an outage of the planner +
        # an app whose intended policy is "private" lets the SDK
        # cold-route to cloud, surfacing a confusing 403 telling the
        # caller to use a local SDK that they're already using.
        # When the caller explicitly passed ``policy='local_only'``
        # or ``policy='private'``, they've expressed their intent
        # directly — honour it by forcing cloud_available=False
        # below so a planner outage cannot leak a local-only request
        # to the network.
        _enforce_app_ref_routing_policy(
            requested_model=requested_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
            resolved_policy_preset=defaults.policy_preset,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # Local routing is satisfied if a prepared artifact dir is
        # already on disk AND the runtime can load it AND the
        # planner did not select a *different* artifact identity, OR
        # the planner emitted a preparable sdk_runtime candidate
        # the kernel can materialize.
        #
        # The prepared cache may short-circuit dispatch in three
        # well-defined cases — anything else means the planner has
        # picked an artifact and we must defer to it, even if the
        # static-recipe cache happens to be on disk for the same
        # runtime model:
        #
        #   a) No local sdk_runtime candidate at all (planner
        #      offline, returned only cloud, etc.).
        #   b) Candidate is structurally unpreparable
        #      (``can_prepare`` rejects it: synthetic, no urls,
        #      no digest, traversal). The planner's intent cannot
        #      be honored, but a usable prepared artifact already
        #      exists; that artifact wins instead of failing closed.
        #   c) Candidate's artifact identity (artifact_id + digest)
        #      matches the static recipe's. The planner-selected
        #      artifact IS the static-recipe artifact, just already
        #      on disk; no re-download needed.
        #
        # The new P1 reviewer reproducer: planner emits a
        # *legitimate* candidate for ``kokoro-82m`` with
        # ``artifact_id='private-kokoro-v2'``, a real digest, and a
        # working download_url. None of (a)-(c) hold; we must run
        # the planner's prepare and load *that* artifact, NOT the
        # static-recipe cache that happens to share ``runtime_model``.
        #
        # Cache-without-runtime is NOT local availability either:
        # under ``local_first`` we'd otherwise commit to local and
        # raise "could not load sherpa backend" instead of cleanly
        # falling back to cloud (PR D P2).
        local_candidate = _local_sdk_runtime_candidate(selection)
        # PR D round 4: an explicit ``app=`` kwarg is the same
        # scope as an ``@app/...`` model ref — PR B's
        # ``_planner_model_for_request`` synthesizes
        # ``@app/<app>/tts`` for planner resolution either way.
        # Treating only the model-ref form as app-scoped let
        # ``speech.create(model='kokoro-82m', app='tts-tester')``
        # silently fall through to the public static cache when
        # the planner returned an echo-only candidate, hiding the
        # same Task #51 server bug the model-ref form refuses.
        app_scoped = bool(app) or _is_app_ref(requested_model or "")
        prepared_cache_dir: Optional[str] = None
        if self._sherpa_tts_runtime_loadable(runtime_model) and self._prepared_cache_may_short_circuit(
            local_candidate, runtime_model, CAPABILITY_TTS, app_scoped=app_scoped
        ):
            prepared_cache_dir = self._prepared_local_artifact_dir(CAPABILITY_TTS, runtime_model)

        if prepared_cache_dir is not None:
            local_available = True
        elif self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._can_prepare_local_tts(runtime_model, selection)
        cloud_available = _cloud_available(defaults)
        # PR B: explicit private / local_only policies force
        # cloud_available=False so a planner outage cannot leak the
        # request to a hosted backend even if cloud creds happen to
        # be present in the environment.
        if _is_local_only_policy(policy):
            cloud_available = False

        # Policy gating mirrors transcription. local_only never falls back.
        try:
            locality, is_fallback = _select_locality_for_capability(
                routing_policy,
                local_available=local_available,
                cloud_available=cloud_available,
                capability=CAPABILITY_TTS,
            )
        except RuntimeError as exc:
            if "local" in str(exc).lower():
                raise self._tts_local_unavailable_error(
                    runtime_model,
                    requested_model=requested_model,
                    app=app,
                    local_candidate=local_candidate,
                    app_scoped=app_scoped,
                ) from exc
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=str(exc),
            ) from exc

        if locality == LOCALITY_ON_DEVICE and not local_available:
            # Don't try to load — fail closed with the canonical error.
            raise self._tts_local_unavailable_error(
                runtime_model,
                requested_model=requested_model,
                app=app,
                local_candidate=local_candidate,
                app_scoped=app_scoped,
            )

        speech_route = SpeechRoute(
            locality="on_device" if locality == LOCALITY_ON_DEVICE else "cloud",
            engine="sherpa-onnx" if locality == LOCALITY_ON_DEVICE else None,
            policy=policy_preset,
            fallback_used=is_fallback,
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None, "cloud locality requires cloud profile"
            # PR B follow-up: when ``app=`` is explicit, send the
            # synthesized ``@app/<app>/tts`` ref to cloud so the
            # server applies the app's quota / policy / billing.
            # The pre-existing ``@app/...`` pass-through behaviour
            # for callers who put the ref in ``model=`` keeps working.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=requested_model,
                effective_model=runtime_model,
                planner_model=planner_model,
                app=app,
            )
            t0 = time.monotonic()
            cloud_result = await self._cloud_synthesize_speech(
                cloud_model,
                input,
                voice,
                response_format,
                speed,
                defaults.cloud_profile,
            )
            return SpeechResponse(
                audio_bytes=cloud_result["audio_bytes"],
                content_type=cloud_result["content_type"],
                format=cloud_result.get("format") or response_format,
                model=cloud_result.get("model") or runtime_model,
                provider=cloud_result.get("provider"),
                voice=voice,
                sample_rate=cloud_result.get("sample_rate"),
                duration_ms=cloud_result.get("duration_ms"),
                latency_ms=(time.monotonic() - t0) * 1000.0,
                route=speech_route,
                billed_units=cloud_result.get("billed_units"),
                unit_kind=cloud_result.get("unit_kind"),
            )

        # Local branch — sherpa-onnx.
        if response_format.lower() != "wav":
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"format_not_supported_for_local_tts: local sherpa-onnx "
                    f"returns WAV. Got response_format='{response_format}'. "
                    "Cloud-routed apps can request other formats; local apps "
                    "should request 'wav' until local transcoding ships."
                ),
            )

        from octomil.execution.tts_speaker_resolver import resolve_tts_speaker

        resolved_speaker = resolve_tts_speaker(
            speaker=speaker,
            voice=voice,
            selection=selection,
            is_app_ref=app_scoped,
        )
        # The native_voice on the resolved profile is what voice
        # validation checks against the engine's catalog. For pure
        # reference-audio profiles (PocketTTS) ``native_voice`` is
        # ``None`` and ``_validate_local_voice`` is a no-op — the
        # backend's reference-validation path takes over later.
        self._validate_local_voice(
            runtime_model,
            resolved_speaker.native_voice,
            selection=selection,
            prepared_cache_dir=prepared_cache_dir,
        )
        # PR D: when a prepared static-recipe cache is what made
        # local routing available, use it directly and SKIP the
        # planner-driven prepare. Otherwise a synthetic / broken
        # planner candidate (e.g. ``artifact_id='private-kokoro-v2'``
        # with no ``download_urls``) would still hit
        # ``_prepare_local_tts_artifact(selection)`` →
        # ``manager.prepare(candidate)`` and surface a
        # ``no download_urls`` error even though the cache was
        # already usable. The reorder above admitted the route
        # because of the cache; honor that here.
        if prepared_cache_dir is not None:
            prepared_model_dir: Optional[str] = prepared_cache_dir
        else:
            prepared_model_dir = self._prepare_local_tts_artifact(selection)
        # PR 11 follow-up: thread the planner candidate (already
        # resolved above for the cache short-circuit decision) so
        # the warmup-cache lookup uses the *current* artifact
        # identity. Without this, a v1 backend cached earlier in the
        # process could shadow a freshly prepared v2 ``model_dir``.
        load_error: list[str] = []
        backend = self._resolve_local_tts_backend(
            runtime_model,
            prepared_model_dir=prepared_model_dir,
            planner_candidate=local_candidate,
            load_error=load_error,
        )
        if backend is None:
            # Three distinct failure modes; the old single message
            # made debugging Eternum / Ren'Py / sandboxed Python
            # impossible because all three looked identical.
            reason = load_error[0] if load_error else "unknown"
            if reason.startswith("sherpa_import:"):
                # (1) The sherpa-onnx Python wheel isn't importable
                # in this interpreter. Most common in stripped
                # embedded Pythons (Ren'Py, PyInstaller without
                # the runtime extra).
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_tts_runtime_unavailable: sherpa_onnx is not importable in this Python "
                        f"interpreter ({reason}). Install the ``[tts]`` extra (``pip install octomil[tts]``) "
                        f"OR run on a Python that has ``_sherpa_onnx`` available — embedded interpreters "
                        f"(Ren'Py, stripped PyInstaller bundles) often ship without it."
                    ),
                )
            if not prepared_model_dir:
                # (2) Sherpa imported, but no prepared artifact dir
                # was found. Caller never ran ``client.prepare(...)``
                # and the planner emitted no preparable candidate.
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_tts_runtime_unavailable: no prepared artifact dir on disk for model "
                        f"{runtime_model!r}. Call ``client.prepare(model={runtime_model!r}, capability='tts')`` "
                        f"first (it downloads + extracts the canonical Kokoro layout), or pass a "
                        f"planner-emitted candidate carrying download_urls + digest."
                    ),
                )
            # (3) Sherpa imports AND a prepared dir exists, but the
            # backend rejected it — likely a missing required file,
            # corrupted extraction, or a model the runtime doesn't
            # support yet. Surface the actual exception so the user
            # can see what's wrong.
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"local_tts_runtime_unavailable: prepared artifact dir {prepared_model_dir!r} "
                    f"exists but the sherpa backend failed to load model {runtime_model!r} from it "
                    f"({reason}). The directory may be missing required files (model.onnx / "
                    f"voices.bin / tokens.txt / espeak-ng-data/), corrupted, or the runtime version "
                    f"may not support this model. Check the directory layout and the underlying error."
                ),
            )

        t0 = time.monotonic()
        # Backends that opt into speaker-aware synthesis accept a
        # ``speaker_profile=`` kwarg; legacy backends (Kokoro pre-Pocket)
        # ignore it and consume only ``voice``. Hand the engine the
        # native_voice the resolver picked so the call site stays
        # uniform across engine families.
        local_result = await asyncio.to_thread(
            _call_backend_synthesize,
            backend,
            input,
            resolved_speaker,
            speed,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Local execution: never write cloud_usage_logs / increment cloud quotas.
        # Route telemetry only.
        return SpeechResponse(
            audio_bytes=local_result["audio_bytes"],
            content_type=local_result.get("content_type", "audio/wav"),
            format=local_result.get("format", "wav"),
            model=runtime_model,
            provider=None,
            voice=resolved_speaker.native_voice or local_result.get("voice"),
            sample_rate=local_result.get("sample_rate"),
            duration_ms=local_result.get("duration_ms"),
            latency_ms=latency_ms,
            route=speech_route,
            billed_units=None,
            unit_kind=None,
        )

    async def list_speech_voices(
        self,
        *,
        model: str,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> Any:
        # ``speaker=`` is intentionally NOT a parameter here:
        # ``voices.list`` enumerates the catalog, it does not select
        # an entry. Callers wanting to validate a speaker should
        # check ``catalog.get(speaker_id)`` and inspect the returned
        # :class:`VoiceInfo`. The closure-of-loop guarantee is that
        # every ``id`` in the returned list is accepted by
        # ``speech.create(speaker=id)`` / ``speech.stream(speaker=id)``.
        """Return the ordered voice catalog for ``model`` under the
        active routing policy.

        Powers ``client.audio.voices.list``. Walks the same routing
        pipeline as :meth:`synthesize_speech` so the locality
        decision matches what synthesis would do; the returned
        catalog is therefore the *exact* set of voices a caller can
        legally pass to ``speech.create`` / ``speech.stream`` right
        now.

        Resolution order (local locality):

          - Prepared artifact dir on disk → read ``voices.txt``
            from the prepared dir (authoritative). This wins for
            both static-recipe and planner-selected artifacts.
          - Planner-selected non-static candidate, not yet on disk
            → defer: artifact identity is owned by the planner, the
            SDK's static-recipe preview would advertise the wrong
            catalog. Return ``locality='on_device', source='planner_pending'``
            with empty voices so callers know to ``client.prepare(...)``
            and re-list.
          - Static recipe IS the chosen artifact → preview the
            recipe's ``voice_manifest`` so the UI can list voices
            without forcing a download.

        Cloud locality returns ``locality='cloud', source='hosted'``
        with an empty list — provider catalogs are out of SDK scope.

        Policy / routing errors propagate as ``OctomilError`` so the
        listing API surfaces the same failures synthesis would.
        """
        from octomil.audio.speech import VoiceCatalog, VoiceInfo
        from octomil.runtime.engines.sherpa import is_sherpa_tts_model, resolve_voice_catalog
        from octomil.runtime.lifecycle.static_recipes import get_static_recipe

        requested_model = model
        defaults = self._resolve(CAPABILITY_TTS, model=requested_model, policy=policy, app=app)
        effective_model = defaults.model or requested_model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TTS)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_TTS)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TTS, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)

        planner_policy = _tts_policy_from_selection(selection)
        if planner_policy:
            policy_preset = planner_policy
            defaults.policy_preset = planner_policy
            defaults.inline_policy = None

        # Same app-ref policy gate ``synthesize_speech`` uses — a
        # listing API that hides this would let UIs preview a public
        # catalog for a request synthesis would refuse.
        _enforce_app_ref_routing_policy(
            requested_model=requested_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
            resolved_policy_preset=defaults.policy_preset,
        )

        local_candidate = _local_sdk_runtime_candidate(selection)
        app_scoped = bool(app) or _is_app_ref(requested_model or "")
        prepared_cache_dir: Optional[str] = None
        if self._sherpa_tts_runtime_loadable(runtime_model) and self._prepared_cache_may_short_circuit(
            local_candidate, runtime_model, CAPABILITY_TTS, app_scoped=app_scoped
        ):
            prepared_cache_dir = self._prepared_local_artifact_dir(CAPABILITY_TTS, runtime_model)

        if prepared_cache_dir is not None:
            local_available = True
        elif self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._can_prepare_local_tts(runtime_model, selection)

        cloud_available = _cloud_available(defaults)
        if _is_local_only_policy(policy):
            cloud_available = False

        # Locality decision: surface RuntimeError as the SAME
        # OctomilError synthesize_speech raises, so the two APIs
        # cannot disagree about whether the request is routable.
        try:
            routing_policy = _resolve_routing_policy(defaults)
            locality, _is_fallback = _select_locality_for_capability(
                routing_policy,
                local_available=local_available,
                cloud_available=cloud_available,
                capability=CAPABILITY_TTS,
            )
        except RuntimeError as exc:
            if "local" in str(exc).lower():
                raise self._tts_local_unavailable_error(
                    runtime_model,
                    requested_model=requested_model,
                    app=app,
                    local_candidate=local_candidate,
                    app_scoped=app_scoped,
                ) from exc
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=str(exc),
            ) from exc

        if locality == LOCALITY_ON_DEVICE and not local_available:
            raise self._tts_local_unavailable_error(
                runtime_model,
                requested_model=requested_model,
                app=app,
                local_candidate=local_candidate,
                app_scoped=app_scoped,
            )

        if locality == LOCALITY_ON_DEVICE:
            from octomil.execution.tts_speaker_resolver import list_logical_speakers

            recipe = (
                get_static_recipe(runtime_model.lower(), CAPABILITY_TTS) if is_sherpa_tts_model(runtime_model) else None
            )

            # Logical speakers from the planner ride alongside the
            # native catalog. For app refs whose planner publishes a
            # speaker map, the catalog returned to ``voices.list``
            # *prepends* logical speakers so the UI can list them as
            # the canonical speaker ids.
            logical_speakers = list_logical_speakers(selection)

            # P1 fix: only feed the SDK's static recipe manifest to
            # the resolver when the static recipe IS the chosen
            # artifact. For a planner-selected non-static artifact
            # (e.g. private-kokoro-v2 under runtime_model
            # kokoro-82m), advertising the public v1.0 catalog +
            # public digest would lie about what synthesis is going
            # to use.
            static_recipe_active = self._static_recipe_is_active(recipe, selection, prepared_cache_dir)
            static_manifest = (
                recipe.materialization.voice_manifest if (recipe is not None and static_recipe_active) else ()
            )
            static_version = (
                recipe.materialization.artifact_version if (recipe is not None and static_recipe_active) else ""
            )

            resolved = resolve_voice_catalog(
                runtime_model.lower(),
                prepared_model_dir=prepared_cache_dir,
                static_recipe_manifest=static_manifest,
                static_recipe_artifact_version=static_version or "",
            )

            # P1 fix: when the planner picked a non-static artifact
            # AND it isn't on disk yet, the SDK has no authoritative
            # source for the catalog. Tell the caller to prepare —
            # do NOT advertise the public recipe.
            if not resolved.voices and not static_recipe_active and prepared_cache_dir is None:
                return VoiceCatalog(
                    model=runtime_model,
                    locality="on_device",
                    source="planner_pending",
                    voices=(),
                    artifact_id=getattr(getattr(local_candidate, "artifact", None), "artifact_id", None),
                    artifact_version=None,
                    digest=getattr(getattr(local_candidate, "artifact", None), "digest", None),
                    default_voice=None,
                    sample_rate=None,
                )

            from octomil.runtime.engines.sherpa.engine import _default_voice as _engine_default_voice

            # P2 fix: only mark a default when it's actually present
            # in the resolved catalog. Otherwise fall back to the
            # bundle's first speaker so default_voice and the flagged
            # VoiceInfo always agree — synthesis with voice=None will
            # use that exact id.
            model_default = (_engine_default_voice(runtime_model) or "").strip()
            resolved_lower = {name.lower() for name in resolved.voices}
            if model_default and model_default.lower() in resolved_lower:
                effective_default = model_default
            elif resolved.voices:
                effective_default = resolved.voices[0]
            else:
                effective_default = ""

            native_source = resolved.source or ("static_recipe" if static_recipe_active else "planner_pending")
            native_voices = tuple(
                VoiceInfo(
                    id=name,
                    sid=idx,
                    default=bool(effective_default) and name.lower() == effective_default.lower(),
                    source=native_source,
                    speaker=None,
                    native_voice=name,
                    requires_reference=False,
                )
                for idx, name in enumerate(resolved.voices)
            )

            # Logical speakers from the planner sit at the top of the
            # catalog when present — they're the canonical caller-
            # facing ids for app refs. Resolution rule from
            # tts_speaker_resolver: a speaker with ``reference_audio``
            # requires reference at synthesis time; one with only
            # ``native_voice`` is a label alias for the underlying
            # engine voice.
            logical_voice_infos: list[VoiceInfo] = []
            for entry in logical_speakers:
                speaker_id = entry["speaker_id"]
                native_voice = entry.get("native_voice")
                # When a logical speaker maps to a native voice that's
                # already in the catalog, surface the underlying sid so
                # callers that want to skip the speaker indirection
                # have it. Otherwise (pure reference-audio profile)
                # ``sid`` stays ``None``.
                sid = None
                if native_voice:
                    for idx, native_name in enumerate(resolved.voices):
                        if native_name.lower() == native_voice.lower():
                            sid = idx
                            break
                logical_voice_infos.append(
                    VoiceInfo(
                        id=speaker_id,
                        sid=sid,
                        default=False,
                        source="planner_profile",
                        speaker=speaker_id,
                        native_voice=native_voice,
                        requires_reference=bool(entry.get("reference_audio")),
                    )
                )

            voices: tuple[VoiceInfo, ...] = tuple(logical_voice_infos) + native_voices

            # Identity comes from the actually-chosen artifact, NOT
            # the static recipe, when those differ.
            cand_artifact = getattr(local_candidate, "artifact", None) if local_candidate is not None else None
            artifact_id: Optional[str]
            digest: Optional[str]
            if static_recipe_active and recipe is not None:
                artifact_id = recipe.model_id
                digest = recipe.files[0].digest if recipe.files else None
            else:
                artifact_id = getattr(cand_artifact, "artifact_id", None) if cand_artifact is not None else None
                digest = getattr(cand_artifact, "digest", None) if cand_artifact is not None else None

            artifact_version = resolved.artifact_version or static_version or None

            # Catalog source reflects what the *primary* entries are.
            # When the planner publishes logical speakers and the
            # native catalog is empty (e.g. PocketTTS bundle whose
            # voice manifest is the speaker list itself), advertise
            # ``planner_profile`` so the UI doesn't claim
            # ``voices_txt`` or ``static_recipe`` provenance for
            # entries that didn't come from there.
            catalog_source = "planner_profile" if (logical_voice_infos and not native_voices) else native_source

            return VoiceCatalog(
                model=runtime_model,
                locality="on_device",
                source=catalog_source,
                voices=voices,
                artifact_id=artifact_id,
                artifact_version=artifact_version,
                digest=digest,
                default_voice=effective_default or None,
                sample_rate=24000 if voices else None,
            )

        # Cloud locality. The SDK doesn't ship a curated hosted
        # catalog table — return an empty list with provenance so
        # the UI can render "ask the provider" rather than a stale
        # hardcoded set. Callers wanting the full hosted catalog
        # should hit the provider's own endpoint or wait for the
        # server-side ``/v1/audio/voices`` route.
        return VoiceCatalog(
            model=runtime_model,
            locality="cloud",
            source="hosted",
            voices=(),
            artifact_id=None,
            artifact_version=None,
            digest=None,
            default_voice=None,
            sample_rate=None,
        )

    async def synthesize_speech_stream(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        speaker: Optional[str] = None,
        response_format: str = "pcm_s16le",
        speed: float = 1.0,
        app: Optional[str] = None,
        policy: Optional[str] = None,
        sdk_t0: Optional[float] = None,
    ) -> Any:
        """Streaming TTS. Returns a :class:`SpeechStream`.

        Resolution / routing / voice validation mirror
        :meth:`synthesize_speech`. The local branch advertises a
        :class:`TtsStreamingCapability` (sentence_chunk for
        multi-sentence input, final_chunk for single-sentence) and
        the completion event verifies actual cadence vs.
        advertised. The cloud branch always advertises ``final_chunk``.

        ``response_format`` accepts ``pcm_s16le`` (canonical low-latency)
        or ``wav`` (final-only WAV bytes). ``opus`` is reserved and
        currently raises ``invalid_input``.

        ``sdk_t0`` is the monotonic timestamp the caller captured at
        the SDK call boundary (``FacadeSpeech.stream`` entry).
        Threaded through so honest end-to-end TTFB metrics can attribute
        the kernel's resolution / scheduler-queue time to the customer-
        visible setup window. When omitted, defaults to "now" — useful
        for tests but loses the e2e attribution.
        """
        from octomil.audio.streaming import (
            SAMPLE_FORMAT_PCM_S16LE,
            SUPPORTED_STREAM_FORMATS,
        )

        if sdk_t0 is None:
            sdk_t0 = time.monotonic()

        if not input or not input.strip():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a non-empty string.",
            )
        normalized_format = (response_format or SAMPLE_FORMAT_PCM_S16LE).lower()
        if normalized_format not in SUPPORTED_STREAM_FORMATS:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"unsupported_stream_format: '{response_format}' is not a supported "
                    f"streaming response_format. Use one of: {', '.join(SUPPORTED_STREAM_FORMATS)}."
                ),
            )

        # Resolve route the same way the non-streaming path does.
        requested_model = model
        defaults = self._resolve(CAPABILITY_TTS, model=requested_model, policy=policy, app=app)
        effective_model = defaults.model or requested_model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TTS)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_TTS)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TTS, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)

        planner_policy = _tts_policy_from_selection(selection)
        if planner_policy:
            policy_preset = planner_policy
            defaults.policy_preset = planner_policy
            defaults.inline_policy = None

        _enforce_app_ref_routing_policy(
            requested_model=requested_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
            resolved_policy_preset=defaults.policy_preset,
        )

        routing_policy = _resolve_routing_policy(defaults)
        local_candidate = _local_sdk_runtime_candidate(selection)
        app_scoped = bool(app) or _is_app_ref(requested_model or "")
        prepared_cache_dir: Optional[str] = None
        if self._sherpa_tts_runtime_loadable(runtime_model) and self._prepared_cache_may_short_circuit(
            local_candidate, runtime_model, CAPABILITY_TTS, app_scoped=app_scoped
        ):
            prepared_cache_dir = self._prepared_local_artifact_dir(CAPABILITY_TTS, runtime_model)

        if prepared_cache_dir is not None:
            local_available = True
        elif self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._can_prepare_local_tts(runtime_model, selection)
        cloud_available = _cloud_available(defaults)
        if _is_local_only_policy(policy):
            cloud_available = False

        try:
            locality, is_fallback = _select_locality_for_capability(
                routing_policy,
                local_available=local_available,
                cloud_available=cloud_available,
                capability=CAPABILITY_TTS,
            )
        except RuntimeError as exc:
            if "local" in str(exc).lower():
                raise self._tts_local_unavailable_error(
                    runtime_model,
                    requested_model=requested_model,
                    app=app,
                    local_candidate=local_candidate,
                    app_scoped=app_scoped,
                ) from exc
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=str(exc),
            ) from exc

        if locality == LOCALITY_ON_DEVICE and not local_available:
            raise self._tts_local_unavailable_error(
                runtime_model,
                requested_model=requested_model,
                app=app,
                local_candidate=local_candidate,
                app_scoped=app_scoped,
            )

        # Voice / speaker resolution and validation must happen before
        # any synthesis kicks off so the consumer never sees a
        # SpeechStreamStarted for an unsupported request. Mirrors the
        # non-streaming path's signature: ``selection`` +
        # ``prepared_cache_dir`` together tell the validator whether
        # the request will run the static recipe (preflight enforces)
        # or a planner-private artifact (defer to backend's voices.txt
        # check post-prepare).
        from octomil.execution.tts_speaker_resolver import resolve_tts_speaker

        resolved_speaker = resolve_tts_speaker(
            speaker=speaker,
            voice=voice,
            selection=selection,
            is_app_ref=app_scoped,
        )
        if locality == LOCALITY_ON_DEVICE:
            # Pure reference-audio profiles (PocketTTS) leave
            # ``native_voice`` ``None`` and skip the catalog check;
            # the backend's reference-validation path enforces them.
            self._validate_local_voice(
                runtime_model,
                resolved_speaker.native_voice,
                selection=selection,
                prepared_cache_dir=prepared_cache_dir,
            )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None, "cloud locality requires cloud profile"
            cloud_profile = defaults.cloud_profile
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=requested_model,
                effective_model=runtime_model,
                planner_model=planner_model,
                app=app,
            )
            return _build_cloud_final_chunk_stream(
                kernel=self,
                cloud_model=cloud_model,
                text=input,
                voice=voice,
                speed=speed,
                profile=cloud_profile,
                response_format=normalized_format,
                runtime_model=runtime_model,
                policy_preset=policy_preset,
                fallback_used=is_fallback,
                sdk_t0=sdk_t0,
            )

        # Local realtime stream.
        if prepared_cache_dir is not None:
            prepared_model_dir: Optional[str] = prepared_cache_dir
        else:
            prepared_model_dir = self._prepare_local_tts_artifact(selection)

        load_error: list[str] = []
        backend = self._resolve_local_tts_backend(
            runtime_model,
            prepared_model_dir=prepared_model_dir,
            planner_candidate=local_candidate,
            load_error=load_error,
        )
        if backend is None or not _backend_can_stream(backend):
            # Fall back through the same unavailable-backend chain the
            # non-streaming path uses; map "loaded but doesn't stream"
            # to runtime_unavailable so callers can ask capabilities()
            # before retrying with create(). A backend "can stream"
            # iff it exposes synthesize_stream AND advertises a
            # capability — bool flags are no longer the contract.
            reason = load_error[0] if load_error else "backend_does_not_support_streaming"
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"local_tts_streaming_unavailable: backend for {runtime_model!r} "
                    f"({reason}). Use client.audio.speech.create(...) for non-streaming "
                    f"output, or check engine capabilities."
                ),
            )

        return _build_local_realtime_stream(
            backend=backend,
            text=input,
            voice=voice,
            resolved_speaker=resolved_speaker,
            speed=speed,
            runtime_model=runtime_model,
            policy_preset=policy_preset,
            fallback_used=is_fallback,
            sdk_t0=sdk_t0,
        )

    async def _cloud_synthesize_speech(
        self,
        model: str,
        text: str,
        voice: Optional[str],
        response_format: str,
        speed: float,
        profile: CloudProfile,
    ) -> dict[str, Any]:
        """Wrap ``octomil.hosted.HostedSpeech.create`` for the cloud branch.

        Internal-only use of ``HostedClient``; never expose this in
        quickstarts. The kernel reuses the hosted speech transport so we
        don't duplicate request shaping or error handling.
        """
        from octomil.hosted.client import HostedClient

        api_key = os.environ.get(profile.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Cloud TTS requires {profile.api_key_env} to be set.")
        base_url = _openai_base_url(profile)
        client = HostedClient(api_key=api_key, base_url=base_url)
        # HostedSpeech.create is sync today; thread it.
        resp = await asyncio.to_thread(
            client.audio.speech.create,
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )
        return {
            "audio_bytes": resp.audio_bytes,
            "content_type": resp.content_type,
            "format": response_format,
            "model": getattr(resp, "model", None),
            "provider": resp.provider,
            "billed_units": resp.billed_units,
            "unit_kind": resp.unit_kind,
        }

    def _has_local_tts_backend(self, model: str) -> bool:
        """Return True iff the local TTS path can serve ``model`` *now*.

        After the PR D cutover, "ready" means BOTH:

          1. A complete prepared layout under PrepareManager's
             artifact cache (``<cache>/artifacts/<key>``) for the
             static recipe — the on-disk shape ``client.prepare(
             model, capability='tts')`` produces. The legacy
             ``OCTOMIL_SHERPA_MODELS_DIR`` / ``~/.octomil/models/sherpa``
             path was removed; this is the only on-disk source.
          2. The sherpa-onnx runtime is importable and the model id
             is recognized. Cache-without-runtime cannot be loaded;
             admitting it as local availability would let
             ``local_first`` commit to local and raise "could not
             load sherpa backend" instead of falling back to cloud.
        """
        if not self._sherpa_tts_runtime_loadable(model):
            return False
        return self._prepared_local_artifact_dir(CAPABILITY_TTS, model) is not None

    def _prepared_cache_may_short_circuit(
        self,
        local_candidate: Optional[Any],
        model: str,
        capability: str,
        *,
        app_scoped: bool,
    ) -> bool:
        """Decide whether the static-recipe prepared cache may
        short-circuit the planner candidate's ``prepare()`` call.

        Reviewer history:

        - Round 2 fixed cache-shadowing of a *preparable* candidate
          with a different identity by introducing
          :meth:`_candidate_matches_static_recipe`.
        - Round 4 (this gate): the previous "candidate is
          unpreparable → cache wins" escape hatch was too broad. A
          planner candidate with ``artifact_id='private-kokoro-v2'``
          and missing ``download_urls`` IS naming a different
          artifact (a private fork); silently substituting the
          public Kokoro cache hides the planner/server bug AND
          serves the wrong bytes. App-scoped requests
          (``@app/<slug>/...``) are even more dangerous: the user
          asked for the app's artifact, not the public one.

        After round 4, the cache may short-circuit the planner only
        in three narrow cases:

          a) Identity match: candidate's ``artifact_id`` AND
             ``digest`` equal the static recipe's. The planner-
             selected artifact IS the recipe's artifact, already on
             disk. Bit-identical reuse.
          b) The request is *direct* (``app_scoped=False``) AND no
             local candidate was emitted. Planner is offline /
             returned only cloud. The user typed
             ``model='kokoro-82m'`` and we fall back to the public
             static recipe — exactly what they asked for.
          c) The request is *direct* AND the candidate carries no
             meaningful artifact identity — i.e. no ``digest`` AND
             ``artifact_id`` is missing OR equals the runtime model.
             The planner echoed the model name back without
             committing to a specific artifact version; treat as
             silent and fall through to the public cache.

        ``app_scoped`` is True for BOTH ``model='@app/<slug>/<cap>'``
        and ``model='kokoro-82m', app='<slug>'`` — PR B's
        ``_planner_model_for_request`` synthesizes the same planner
        ref in either case, and the user expressed an app-scoped
        intent both times.

        App-scoped requests are deliberately excluded from (b)/(c):
        a missing or echo-only candidate for an app ref means the
        planner could not resolve the app, and the right behavior
        is to surface that error (Task #51) rather than substitute
        the public artifact. Identity match (a) still applies —
        if the planner explicitly selected the recipe-shaped
        artifact for an app, the cache is the right bytes.
        """
        # (a) Identity match — works for both app-scoped and direct.
        if local_candidate is not None and self._candidate_matches_static_recipe(local_candidate, model, capability):
            return True

        # App-scoped requests: only identity match (a) admits the
        # cache. Anything else surfaces the planner error.
        # ``app_scoped`` is true for BOTH ``model='@app/...'`` and
        # ``model='kokoro-82m', app='tts-tester'`` — PR B's
        # ``_planner_model_for_request`` synthesizes the same
        # ``@app/<slug>/<capability>`` planner ref either way.
        if app_scoped:
            return False

        # Direct request:
        # (b) no candidate at all.
        if local_candidate is None:
            return True
        # (c) candidate carries no meaningful artifact identity.
        if not self._candidate_has_meaningful_identity(local_candidate, model):
            return True
        return False

    @staticmethod
    def _candidate_has_meaningful_identity(candidate: Any, runtime_model: str) -> bool:
        """Return True iff ``candidate`` names a *specific* artifact
        beyond echoing the runtime model.

        Meaningful when EITHER:

          - ``artifact.digest`` is present (planner committed to a
            specific bytes version), OR
          - ``artifact.artifact_id`` is set AND differs from
            ``runtime_model`` (planner named something other than
            the model).

        ``artifact_id`` equal to ``runtime_model`` with no digest is
        treated as "the planner echoed the model name" — not a
        commitment to a particular artifact. Empty / missing
        ``artifact_id`` and ``digest`` is the synthetic-no-info case.
        """
        artifact = getattr(candidate, "artifact", None)
        if artifact is None:
            return False
        digest = getattr(artifact, "digest", None)
        if digest:
            return True
        artifact_id = getattr(artifact, "artifact_id", None)
        if not artifact_id:
            return False
        return artifact_id != runtime_model

    @staticmethod
    def _candidate_matches_static_recipe(candidate: Any, model: str, capability: str) -> bool:
        """Return True iff ``candidate``'s artifact identity matches
        the static recipe registered for ``(model, capability)``.

        "Matches" means same ``artifact_id`` AND same ``digest`` —
        a candidate with the recipe's id but a different digest is
        a *different* artifact (e.g. a private re-cut) and must not
        be served from the static-recipe cache.
        """
        try:
            from octomil.runtime.lifecycle.static_recipes import get_static_recipe
        except Exception:
            return False
        recipe = get_static_recipe(model, capability)
        if recipe is None:
            return False
        artifact = getattr(candidate, "artifact", None)
        if artifact is None:
            return False
        candidate_id = getattr(artifact, "artifact_id", None) or getattr(artifact, "model_id", None)
        candidate_digest = getattr(artifact, "digest", None)
        if not candidate_id or not candidate_digest:
            return False
        if candidate_id != recipe.model_id:
            return False
        if not recipe.files:
            return False
        return candidate_digest == recipe.files[0].digest

    def _tts_local_unavailable_error(
        self,
        runtime_model: str,
        *,
        requested_model: Optional[str],
        app: Optional[str],
        local_candidate: Optional[Any],
        app_scoped: bool,
    ) -> OctomilError:
        """Pick the right ``RUNTIME_UNAVAILABLE`` error for a failed
        local TTS attempt.

        Reviewer P2 (round 5): the generic
        :func:`_local_tts_runtime_unavailable` message tells the
        user to install ``octomil[tts]`` and run
        ``octomil prepare kokoro-82m`` — the right remediation for a
        clean-device first run on a direct request, but actively
        misleading for an app-scoped request that was refused
        because the planner returned a synthetic / echo-only
        candidate. Both ``[tts]`` AND ``prepare`` may already be
        green; the actual problem is server-side (Task #51).

        Branch on ``app_scoped`` AND "candidate exists but didn't
        give us a usable artifact" (no candidate / unpreparable /
        no meaningful identity / mismatched). When that holds, raise
        a planner-config error that names the app and points at
        server config. Otherwise fall through to the canonical
        clean-device message.
        """
        if app_scoped and self._app_planner_returned_unusable_candidate(local_candidate, runtime_model):
            slug = app or self._app_slug_from_model_ref(requested_model) or "<app>"
            requested = requested_model or f"@app/{slug}/tts"
            return OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"local_tts_app_planner_unresolved: app {slug!r} "
                    f"(requested as {requested!r}) routed to runtime model "
                    f"{runtime_model!r}, but the runtime planner returned no "
                    f"usable artifact identity (missing download_urls / digest, "
                    f"or echoed the model name without committing to a version). "
                    f"This is a planner / app-config issue, not a missing local "
                    f"install — the SDK refused to silently substitute the "
                    f"public {runtime_model!r} static-recipe cache because that "
                    f"could ship the wrong bytes for {slug!r}. Verify the app's "
                    f"runtime configuration in the dashboard, OCTOMIL_SERVER_KEY "
                    f"auth, and that the app's planner emits a complete "
                    f"download_urls / digest for its TTS artifact."
                ),
            )
        return _local_tts_runtime_unavailable(runtime_model)

    def _app_planner_returned_unusable_candidate(
        self,
        local_candidate: Optional[Any],
        runtime_model: str,
    ) -> bool:
        """Return True iff an app-scoped request's planner candidate
        is in a "can't honor this" state: no candidate at all, no
        meaningful artifact identity (echo-only / empty), or
        otherwise unpreparable. These are the shapes the
        cache-short-circuit gate refuses for app-scoped requests, so
        the resulting failure should attribute to the planner / app
        config, not a missing local install.
        """
        if local_candidate is None:
            return True
        if not self._candidate_has_meaningful_identity(local_candidate, runtime_model):
            return True
        # Meaningful identity present but missing urls/digest etc.
        # ``_local_candidate_is_unpreparable`` consults
        # ``PrepareManager.can_prepare`` for the precise structural
        # rules; routing already invoked it.
        try:
            from octomil.runtime.lifecycle.prepare_manager import PrepareManager
        except Exception:
            return True
        manager = getattr(self, "_prepare_manager", None) or PrepareManager()
        try:
            return not manager.can_prepare(local_candidate)
        except Exception:
            return True

    @staticmethod
    def _app_slug_from_model_ref(model: Optional[str]) -> Optional[str]:
        """Extract ``<slug>`` from ``@app/<slug>/<capability>``."""
        if not model or not isinstance(model, str) or not model.startswith("@app/"):
            return None
        parts = model.split("/", 3)
        if len(parts) < 3:
            return None
        slug = parts[1]
        return slug or None

    @staticmethod
    def _sherpa_tts_runtime_loadable(model: str) -> bool:
        """Return True iff sherpa-onnx is importable AND ``model`` is a
        recognized Sherpa TTS id.

        Pure runtime-availability check — does NOT touch disk. Pairs
        with :meth:`_prepared_local_artifact_dir` to gate
        :meth:`_has_local_tts_backend`.

        Issue E: when ``octomil.runtime.engines.sherpa`` itself fails
        to import (e.g. stripped CPython 3.9 without ``audioop``,
        which the stdlib ``wave`` module pulls in), the previous
        bare ``except Exception: return False`` silently swallowed
        the real cause and surfaced as a vague
        ``local_tts_runtime_unavailable``. Log the underlying
        exception at WARNING so users on embedded interpreters can
        diagnose without diving into SDK source. Returning False is
        still correct — the runtime IS unloadable.
        """
        try:
            from octomil.runtime.engines.sherpa import is_sherpa_tts_runtime_available
        except Exception as exc:
            logger.warning(
                "Sherpa TTS engine module failed to import (%r). The runtime is unavailable in this Python "
                "interpreter — most often this is a stripped embedded build (Ren'Py / PyInstaller) missing a "
                "stdlib transitive dep (e.g. ``audioop`` on CPython 3.9). Install the ``[tts]`` extra or run on "
                "a Python that ships the missing module.",
                exc,
            )
            return False
        try:
            return is_sherpa_tts_runtime_available(model)
        except Exception as exc:
            logger.warning("is_sherpa_tts_runtime_available(%r) raised %r", model, exc)
            return False

    def _prepared_local_artifact_dir(self, capability: str, model: str) -> Optional[str]:
        """Return ``<cache>/artifacts/<key>`` when ``(model, capability)``
        already has a complete prepared layout on disk, else ``None``.

        Generic across capabilities: looks up the static recipe for
        ``(model, capability)``, derives the same cache key
        :class:`PrepareManager` writes to via ``artifact_dir_for``, and
        runs the recipe's :class:`Materializer` idempotently. When the
        layout was completed by an earlier ``client.prepare(...)`` call,
        the materializer is a marker-presence check — no I/O beyond a
        few ``stat`` calls; nothing is downloaded. A partially-extracted
        layout (marker missing but archive still on disk) is re-extracted
        from the local archive — still no network.

        Used by ``_has_local_tts_backend`` to count a prepared cache as
        local availability, and by ``synthesize_speech`` to thread the
        prepared dir into ``SherpaTtsEngine.create_backend(model_dir=...)``
        when the planner emitted no candidate (offline / planner outage /
        purely-static SDK use). Without this, ``client.prepare`` could
        succeed but ``speech.create`` would still fall through to the
        legacy staging path and raise ``local_tts_runtime_unavailable``,
        which was the bug PR D fixes.
        """
        try:
            from octomil.runtime.lifecycle.materialization import Materializer
            from octomil.runtime.lifecycle.prepare_manager import PrepareManager
            from octomil.runtime.lifecycle.static_recipes import get_static_recipe
        except Exception:
            return None

        recipe = get_static_recipe(model, capability)
        if recipe is None:
            return None

        # ``__new__``-constructed kernels in tests skip ``__init__``,
        # so ``_prepare_manager`` may not exist as an attribute. Treat
        # a missing attribute the same as the no-injection case.
        manager = getattr(self, "_prepare_manager", None) or PrepareManager()
        # Tests inject a stripped-down PrepareManager stub that lacks
        # ``artifact_dir_for``; treat any failure as "no prepared dir"
        # and let the rest of the route-selection chain decide.
        try:
            artifact_dir = manager.artifact_dir_for(recipe.model_id)
        except Exception:
            return None
        if not artifact_dir.is_dir():
            return None

        try:
            Materializer().materialize(artifact_dir, recipe.materialization)
        except Exception:
            return None
        return str(artifact_dir)

    def _local_candidate_is_unpreparable(self, selection: Optional[Any]) -> bool:
        """Return True iff the planner's local sdk_runtime candidate is
        ``prepare_required=True`` AND ``PrepareManager.can_prepare`` rejects
        it (synthetic plan: missing digest/url, traversal, dot path, etc.).

        This is the *hard veto* the reviewer flagged. ``_has_local_*_backend``
        only checks whether a runtime is importable / files are staged;
        when the planner says "prepare this artifact" but the metadata is
        unpreparable, the SDK cannot honor that plan. Letting an
        already-staged or runtime-importable model win local routing in
        that case ignores the planner's intent and crashes at first
        prepare(). When this veto fires, the kernel must mark local
        unavailable regardless of staging so local_first falls back to
        cloud.

        ``prepare_required=False`` candidates are never blocking — those
        engines manage their own bytes (e.g. ollama). Same when the
        planner emits no local candidate at all.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return False
        if not getattr(candidate, "prepare_required", False):
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return not manager.can_prepare(candidate)

    def _can_prepare_local_transcription(self, model: str, selection: Optional[Any]) -> bool:
        """Mirror of ``_can_prepare_local_tts`` for the transcription path.

        Returns True only when a transcription backend is importable for
        ``model`` AND ``PrepareManager.can_prepare`` says the planner
        candidate has enough metadata to actually succeed. Without this
        dry-run, ``local_first`` could commit to local on a synthetic
        candidate (prepare_required=True with no digest/url) and fail in
        prepare instead of falling back to cloud.

        We don't have a single runtime-availability helper for
        transcription engines (whisper.cpp, sherpa-ASR, etc. each detect
        themselves through the engine registry), so the import check is
        the registry walk inside ``_resolve_local_transcription_backend``
        — we simply run that resolver against an empty prepared dir and
        check whether *any* engine claims support. The resolver only
        returns when the backend exposes ``transcribe``; that's the same
        runtime check the no-prepare path uses.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None or not getattr(candidate, "prepare_required", False):
            return False

        # Runtime availability: at least one engine in the registry must
        # produce a transcription backend for ``model`` (any model_dir).
        try:
            backend = self._resolve_local_transcription_backend(model)
            if backend is None:
                return False
        except Exception:
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return manager.can_prepare(candidate)

    def _can_prepare_local_tts(self, model: str, selection: Optional[Any]) -> bool:
        """Return True iff sherpa-onnx is importable, the model id is known,
        and ``PrepareManager.can_prepare`` confirms the planner candidate
        has enough metadata to succeed (digest + download_urls + a
        non-disabled policy + at most one required file).

        ``prepare_required`` alone is not enough: the server can emit
        synthetic prepare metadata with no urls/digest, in which case
        committing to local routing would fail at first prepare instead of
        falling back to cloud. This dry-run inspection is pure — it never
        touches disk or network.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None or not getattr(candidate, "prepare_required", False):
            return False
        try:
            from octomil.runtime.engines.sherpa import is_sherpa_tts_runtime_available

            if not is_sherpa_tts_runtime_available(model):
                return False
        except Exception:
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return manager.can_prepare(candidate)

    def _select_prepare_candidate(
        self,
        *,
        effective_model: str,
        capability: str,
        planner_candidate: Optional[Any],
        app_scoped: bool,
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Decide which candidate ``prepare()`` / ``warmup()`` should
        hand to :class:`PrepareManager`, and which (if any) static
        recipe drives post-download materialization.

        Architectural rule:

          - **PrepareManager** prepares exactly the artifact it is
            handed. It does NOT guess a static recipe by artifact id
            because it lacks request scope (direct vs app-scoped).
          - **The kernel** owns the substitution decision because it
            sees the request shape. It must NOT silently substitute
            the public static recipe for an app-scoped or
            mismatched-identity request.
          - When the *planner* explicitly emits ``source='static_recipe',
            recipe_id=…`` PrepareManager's
            :func:`_expand_static_recipe_source` expands it — that is
            *not* a kernel substitution; it's the planner's
            explicit choice.

        Substitution gate (returns the planner candidate as-is unless
        all conditions hold):

          a) The request is **not app-scoped** (``app_scoped=False``).
          b) The planner candidate is missing OR carries no
             meaningful artifact identity (echo-only, no digest).
          c) A static recipe is registered for ``(effective_model,
             capability)``.

        When all three hold, substitute the static recipe candidate.
        When the planner candidate's identity *matches* the static
        recipe (same artifact_id + digest), the planner candidate is
        already the recipe — return it unchanged.

        Returns ``(candidate, used_static_recipe)``. ``candidate`` is
        the candidate to hand to PrepareManager. ``used_static_recipe``
        is the recipe whose ``MaterializationPlan`` should run
        post-download — set when this kernel performed the
        substitution, ``None`` otherwise (PrepareManager handles
        materialization itself when the planner emitted
        ``source='static_recipe'``).
        """
        from octomil.runtime.lifecycle.static_recipes import (
            get_static_recipe,
            static_recipe_candidate,
        )

        # Honor an explicit planner ``source='static_recipe'`` —
        # PrepareManager expands it. Treat the candidate as
        # ground truth and let the manager handle materialization.
        if planner_candidate is not None:
            artifact = getattr(planner_candidate, "artifact", None)
            if artifact is not None and getattr(artifact, "source", None) == "static_recipe":
                return planner_candidate, None

        # App-scoped requests: never substitute the public static
        # recipe. Hand the planner candidate (or None) through; if
        # it's unusable, the dispatch chain raises the actionable
        # planner-config error instead of serving the wrong bytes.
        if app_scoped:
            return planner_candidate, None

        # Direct request:
        if planner_candidate is None:
            # No candidate → fall back to the static recipe (gate (a)
            # + (b) + (c) all hold).
            recipe = get_static_recipe(effective_model, capability)
            if recipe is None:
                return None, None
            return static_recipe_candidate(effective_model, capability), recipe

        # Direct + planner candidate present.
        if self._candidate_matches_static_recipe(planner_candidate, effective_model, capability):
            # Identity match — the planner already named the recipe.
            # Use the planner candidate; the recipe materialization
            # is conditional on whether the candidate carries the
            # static_recipe discriminator (handled by PrepareManager
            # if so, by the dispatch path's static-recipe cache check
            # otherwise).
            return planner_candidate, None
        if not self._candidate_has_meaningful_identity(planner_candidate, effective_model):
            # Echo-only candidate with no meaningful identity. The
            # planner echoed the model name without committing to an
            # artifact (server bug Task #51 / offline / sandboxed).
            # For *direct* requests this is exactly the case the
            # static recipe was designed to cover.
            recipe = get_static_recipe(effective_model, capability)
            if recipe is None:
                return planner_candidate, None  # let PrepareManager surface its rejection
            return static_recipe_candidate(effective_model, capability), recipe
        # Meaningful identity, but mismatches the static recipe.
        # That is a different artifact (private re-cut). Honor the
        # planner candidate and let PrepareManager prepare it (or
        # reject if the metadata is incomplete).
        return planner_candidate, None

    @staticmethod
    def _no_prepare_candidate_error(
        *,
        stage: str,
        effective_model: str,
        capability: str,
        planner_candidate: Optional[Any],
        app_scoped: bool,
    ) -> OctomilError:
        """Construct the actionable error for the no-candidate cases
        ``_select_prepare_candidate`` couldn't resolve. Distinguishes
        the app-scoped vs. direct paths so users see the right next
        step (set OCTOMIL_SERVER_KEY for app refs, switch model for
        direct refs)."""
        if app_scoped:
            return OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"{stage}: app-scoped request for model={effective_model!r} "
                    f"capability={capability!r} got no usable planner candidate, and the SDK "
                    f"refuses to substitute the public static recipe for app/private artifacts. "
                    f"Set OCTOMIL_SERVER_KEY (the planner needs auth to resolve the app's "
                    f"intended artifact) or remove the app= scope for a direct public request."
                ),
            )
        if planner_candidate is None:
            return OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"{stage}: planner returned no local sdk_runtime candidate for "
                    f"model={effective_model!r} capability={capability!r} and the SDK has "
                    f"no static offline recipe for it. Either the model is cloud-only, "
                    f"the planner is offline, or this is a private artifact that requires "
                    f"OCTOMIL_SERVER_KEY auth. Set OCTOMIL_SERVER_KEY or use a model with "
                    f"a static recipe (e.g. 'kokoro-82m')."
                ),
            )
        # Echo-only / no-recipe direct case: hand back to the
        # caller to let PrepareManager surface its own error.
        return OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=(
                f"{stage}: planner candidate for model={effective_model!r} capability={capability!r} "
                f"carries no usable artifact identity and no static recipe is registered. "
                f"Have the planner emit download_urls + digest, or use a built-in recipe model."
            ),
        )

    def prepare(
        self,
        *,
        model: str,
        capability: str = "tts",
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> Any:
        """Resolve a planner candidate for ``model`` and pre-warm its artifact.

        Public, caller-driven equivalent of the implicit prepare path. Calls
        :meth:`PrepareManager.prepare` with ``mode=PrepareMode.EXPLICIT`` so
        candidates whose ``prepare_policy='explicit_only'`` succeed when
        invoked through this method.

        Supported capabilities: ``"tts"`` and ``"transcription"`` today.
        Both have dispatch paths that thread the prepared ``artifact_dir``
        into their backend:

        - ``tts`` -> ``SherpaTtsEngine.create_backend(model_dir=...)``.
        - ``transcription`` -> ``_WhisperBackend`` honors injected
          ``model_dir`` and prefers PrepareManager's ``<dir>/artifact``
          sentinel before falling back to ``.bin`` / ``.gguf`` / ``.ggml``.

        Embedding and chat (and responses, which routes through chat)
        will be added one at a time as their adapters learn to accept a
        ``model_dir`` kwarg — exposing ``prepare`` for them now would be
        a false success: the bytes would land on disk and the next
        inference call would still cold-start through the engine's own
        lookup. PR 10c added the kernel-side threading for chat into
        ``MLXBackend`` and ``LlamaCppBackend``, but the capability
        remains gated until (a) the public ``client.responses.create``
        facade goes through the kernel (or threads ``model_dir``
        itself), and (b) PrepareManager grows snapshot/manifest support
        for multi-file MLX artifacts. The current set is the source of
        truth; the lifecycle support fixture in octomil-contracts cites
        the e2e test that proves each cell's claim.

        Returns a :class:`PrepareOutcome`. Raises :class:`OctomilError` if
        the capability is not yet wired, the planner emits no preparable
        local candidate, or :class:`PrepareManager.prepare` rejects the
        metadata.
        """
        if capability not in _PREPAREABLE_CAPABILITIES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"client.prepare() does not yet support capability {capability!r}. "
                    f"Supported today: {sorted(_PREPAREABLE_CAPABILITIES)}. "
                    f"Other capabilities will be added once their backends thread the "
                    f"prepared model_dir into dispatch."
                ),
            )

        defaults = self._resolve(capability, model=model, policy=policy, app=app)
        effective_model = defaults.model or model
        if not effective_model:
            raise _no_model_error(capability)

        policy_preset = defaults.policy_preset or "local_first"
        selection = _resolve_planner_selection(effective_model, capability, policy_preset)
        planner_candidate = _local_sdk_runtime_candidate(selection)
        app_scoped = bool(app) or _is_app_ref(model or "")
        candidate, used_static_recipe = self._select_prepare_candidate(
            effective_model=effective_model,
            capability=capability,
            planner_candidate=planner_candidate,
            app_scoped=app_scoped,
        )
        if candidate is None:
            raise self._no_prepare_candidate_error(
                stage="prepare",
                effective_model=effective_model,
                capability=capability,
                planner_candidate=planner_candidate,
                app_scoped=app_scoped,
            )

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate, mode=PrepareMode.EXPLICIT)

        # PR C: post-prepare materialization. The kernel does NOT
        # know archive shapes / extension matchers / Kokoro layouts —
        # it just hands the recipe's MaterializationPlan to the
        # generic Materializer, which handles archive extraction,
        # safety filtering, idempotency, and required-output
        # verification. ``kind='none'`` plans (single-file backends
        # like whisper.cpp) are a no-op aside from layout
        # validation.
        if used_static_recipe is not None:
            from octomil.runtime.lifecycle.materialization import Materializer

            Materializer().materialize(outcome.artifact_dir, used_static_recipe.materialization)

        return outcome

    def warmup(
        self,
        *,
        model: str,
        capability: str = "tts",
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> WarmupOutcome:
        """Run ``prepare`` and load the local backend so first-call is hot.

        Strict superset of :meth:`prepare`: bytes on disk *plus* the
        engine constructed and ``backend.load_model`` complete, with
        the loaded instance cached on the kernel so the next dispatch
        call reuses it. Currently supports the same capabilities as
        ``prepare`` (tts + transcription) — chat / responses inherit
        the same gates.

        Failure modes
        ~~~~~~~~~~~~~
        - Capability not in :data:`_WARMUPABLE_CAPABILITIES` →
          :class:`OctomilError` ``INVALID_INPUT`` (same actionable
          message as ``prepare``).
        - PrepareManager rejects the candidate or download fails →
          original error propagates unchanged.
        - Backend construction / load fails → returns
          ``WarmupOutcome(backend_loaded=False)`` so the prepare half
          isn't lost. Caller decides whether that's fatal; subsequent
          inference will fall through the cold path naturally.

        Returns a :class:`WarmupOutcome` with the underlying
        :class:`PrepareOutcome`, ``backend_loaded`` flag, and
        ``latency_ms`` wall time.
        """
        if capability not in _WARMUPABLE_CAPABILITIES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"client.warmup() does not yet support capability {capability!r}. "
                    f"Supported today: {sorted(_WARMUPABLE_CAPABILITIES)}. "
                    f"Other capabilities will be added once their backends thread the "
                    f"prepared model_dir into dispatch and the kernel can cache the "
                    f"loaded instance."
                ),
            )

        t0 = time.monotonic()

        # We need the planner selection up front for two reasons:
        #   1. ``_runtime_model_for_selection`` resolves ``@app/...``
        #      refs to the concrete model id (e.g. ``kokoro-en-v0_19``);
        #      that's the key the inference dispatch path uses, so the
        #      cache MUST be keyed under it (not the request alias).
        #   2. The same selection feeds the cache's artifact-identity
        #      tuple (digest/format/quantization) so a second
        #      ``warmup`` for the same runtime model but a different
        #      artifact version doesn't reuse the stale cached backend.
        defaults = self._resolve(capability, model=model, policy=policy, app=app)
        effective_model = defaults.model or model
        if not effective_model:
            raise _no_model_error(capability)
        policy_preset = defaults.policy_preset or "local_first"
        selection = _resolve_planner_selection(effective_model, capability, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)
        planner_candidate = _local_sdk_runtime_candidate(selection)
        app_scoped = bool(app) or _is_app_ref(model or "")
        candidate, used_static_recipe = self._select_prepare_candidate(
            effective_model=effective_model,
            capability=capability,
            planner_candidate=planner_candidate,
            app_scoped=app_scoped,
        )
        if candidate is None:
            raise self._no_prepare_candidate_error(
                stage="warmup",
                effective_model=effective_model,
                capability=capability,
                planner_candidate=planner_candidate,
                app_scoped=app_scoped,
            )

        # Run prepare via PrepareManager directly (mode=EXPLICIT) so
        # candidates with ``prepare_policy='explicit_only'`` succeed,
        # mirroring ``self.prepare``. Doing it here (instead of
        # delegating to ``self.prepare``) lets us reuse the same
        # ``selection`` / ``candidate`` for both prepare and the cache
        # key — no second planner round-trip, no risk of resolving a
        # different artifact for the cache than the one prepare ran on.
        from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

        manager = self._prepare_manager or PrepareManager()
        prepare_outcome = manager.prepare(candidate, mode=PrepareMode.EXPLICIT)
        # Mirror ``prepare()``: when the kernel substituted a static
        # recipe, run its MaterializationPlan after the download.
        # When the planner emitted ``source='static_recipe'``,
        # PrepareManager already ran materialization itself.
        if used_static_recipe is not None:
            from octomil.runtime.lifecycle.materialization import Materializer

            Materializer().materialize(prepare_outcome.artifact_dir, used_static_recipe.materialization)
        artifact_dir = str(prepare_outcome.artifact_dir)
        cache_key = self._warmup_cache_key(capability, runtime_model, candidate)

        backend_loaded = False
        if capability == CAPABILITY_TTS:
            backend = self._resolve_local_tts_backend(runtime_model, prepared_model_dir=artifact_dir)
            if backend is not None:
                self._warmed_backends[cache_key] = backend
                backend_loaded = True
        elif capability == CAPABILITY_TRANSCRIPTION:
            backend = self._resolve_local_transcription_backend(runtime_model, prepared_model_dir=artifact_dir)
            if backend is not None:
                # ``_resolve_local_transcription_backend`` only calls
                # ``engine.create_backend``; ``_WhisperBackend`` (and
                # most ASR backends) keep ``load_model`` lazy so the
                # actual model parse + memory allocation happens on
                # the first ``transcribe`` call. Force load_model now
                # so the warmup contract is real: by the time the
                # outcome is returned, the next ``transcribe_audio``
                # dispatch reuses an *already-loaded* instance.
                load_fn = getattr(backend, "load_model", None)
                if callable(load_fn):
                    try:
                        load_fn(runtime_model)
                    except Exception:
                        # Backend constructor returned but load_model
                        # raised (corrupt artifact, missing native
                        # lib, etc.) — keep prepare bytes on disk and
                        # report load_skipped, same shape as a failed
                        # ``create_backend``.
                        backend = None
                if backend is not None:
                    self._warmed_backends[cache_key] = backend
                    backend_loaded = True

        latency_ms = (time.monotonic() - t0) * 1000.0
        return WarmupOutcome(
            capability=capability,
            model=runtime_model,
            prepare_outcome=prepare_outcome,
            backend_loaded=backend_loaded,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _warmup_cache_key(
        capability: str,
        runtime_model: str,
        candidate: Optional[Any],
    ) -> tuple[str, str, Optional[str], Optional[str], Optional[str]]:
        """Compose a cache key that distinguishes artifact identity.

        Two private apps or two artifact versions can resolve to the
        same ``runtime_model`` but represent different artifacts
        (different digests, formats, quantizations). Keying solely by
        ``(capability, runtime_model)`` would let warmup cache the
        first artifact and then serve a *stale* backend for a later
        warmup of the same model with different bytes.

        ``digest`` is the strongest distinguishing field; ``format``
        and ``quantization`` cover cases where the planner omits the
        digest but emits separate engine artifacts (MLX safetensors
        vs GGUF, q4 vs q4_k_m, etc.). All three may be None when the
        candidate has no artifact metadata; in that case the key
        degenerates to ``(capability, runtime_model, None, None,
        None)`` and behaves like the original (capability, model)
        cache.
        """
        artifact = getattr(candidate, "artifact", None) if candidate is not None else None
        return (
            capability,
            runtime_model,
            getattr(artifact, "digest", None) if artifact is not None else None,
            getattr(artifact, "format", None) if artifact is not None else None,
            getattr(artifact, "quantization", None) if artifact is not None else None,
        )

    def _lookup_warmed_backend(
        self,
        capability: str,
        runtime_model: str,
        *,
        candidate: Optional[Any] = None,
    ) -> Optional[Any]:
        """Look up a cached warmed backend for the *current* request.

        The local resolvers call this at the top of their resolution
        path. The lookup composes the same cache key the warmup
        writer used: digest/format/quantization off the local
        candidate so two warmups for the same runtime model but
        different artifact identities don't collide.

        Reviewer P1 (post PR 11)
        ~~~~~~~~~~~~~~~~~~~~~~~~
        The earlier shape of this helper had a tail loop that
        returned *any* cached backend matching ``(capability,
        runtime_model)`` when the precise key missed. That defeated
        the artifact-identity fix: with v1 warmed and the current
        request resolving v2, the resolver could return the v1
        backend before the freshly-prepared v2 ``model_dir`` ever
        loaded. The current rule is:

          - When a current candidate is known (caller threaded one in
            OR the planner returned one), use the *exact* key only.
            An exact-key miss is a real cache miss; the cold path
            then loads the v2 bytes and warmup writes them under the
            v2 key. The stale v1 entry isn't used.
          - When NO current candidate is available (offline planner,
            synthetic policy candidate, deliberately minimal test
            setup), fall back to the runtime-model-only scan. A
            single warmup against a runtime model still serves the
            common case where there's only one artifact identity in
            play.

        Callers should thread ``candidate`` (the already-resolved
        planner candidate from the request currently in flight) when
        they have it — that avoids a second planner round-trip and
        guarantees the lookup uses the same artifact identity the
        request will actually run against.
        """
        # Step 1: prefer the caller-supplied candidate (zero extra
        # planner work; same identity the request will run against).
        if candidate is None:
            try:
                selection = _resolve_planner_selection(runtime_model, capability, "local_first")
                candidate = _local_sdk_runtime_candidate(selection)
            except Exception:
                candidate = None

        # Step 2: when an artifact identity exists for this request,
        # the cache hit MUST be against that identity. An exact-key
        # miss is a real miss. (If we fell through to the model-only
        # scan here, a stale v1 backend would shadow the freshly
        # prepared v2 bytes — the precise bug the reviewer flagged.)
        if candidate is not None:
            key = self._warmup_cache_key(capability, runtime_model, candidate)
            return self._warmed_backends.get(key)

        # Step 3: no artifact identity available for this request.
        # The model-only scan is the legitimate fallback shape: it
        # serves the common case where exactly one warmup happened
        # against the runtime model and no planner is in play
        # (offline / Ren'Py / synthetic tests).
        for stored_key, backend in self._warmed_backends.items():
            if len(stored_key) >= 2 and stored_key[0] == capability and stored_key[1] == runtime_model:
                return backend
        return None

    def release_warmed_backends(self) -> None:
        """Drop every cached warmed backend.

        Used by tests and long-running embedded callers that want to
        free GPU memory between phases. The next inference dispatch
        rebuilds through the normal cold path. Idempotent.
        """
        self._warmed_backends.clear()

    def _prepare_local_transcription_artifact(self, selection: Optional[Any]) -> Optional[str]:
        """Run PrepareManager for the local transcription candidate, if any.

        Symmetric with ``_prepare_local_tts_artifact``: returns the
        prepared ``artifact_dir`` as a string when the planner emits a
        local sdk_runtime candidate with prepare_required=True; returns
        None otherwise so the existing whisper.cpp / sherpa-ASR fallback
        path runs unchanged.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return None
        if not getattr(candidate, "prepare_required", False):
            return None

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        return str(outcome.artifact_dir)

    def _prepare_local_chat_artifact_cached(
        self,
        candidate: Optional[Any],
        cache: dict[tuple, Optional[str]],
    ) -> Optional[str]:
        """Lazy, per-call prepare for *this specific* local candidate.

        Called from inside the local branch of the candidate runner
        with the planner candidate that's about to be attempted, so
        cloud-first plans (and plans whose primary cloud candidate
        succeeds) never trigger prepare. The cache de-dupes redundant
        prepares for the same artifact shape within a single
        ``create_response`` call.

        Cache key
        ~~~~~~~~~
        ``(artifact_id, digest, format, quantization)``. Earlier shapes
        keyed only on ``artifact_id`` / ``model_id``, which let two
        local candidates that shared the logical id but carried
        different digests or formats reuse the first candidate's
        prepared directory — for example, an MLX snapshot candidate
        and a GGUF candidate that both quote ``model_id="gemma3-1b"``
        because ``artifact_id`` was omitted. The second candidate would
        receive a ``model_dir`` containing the wrong bytes. Including
        digest / format / quantization in the key ensures the cache
        only short-circuits when the planner is truly asking for the
        same verified artifact.

        Returns ``None`` (cold-load through the engine's own resolution
        path) when:
          - the candidate is missing or not a local sdk_runtime shape,
          - ``prepare_required`` is False (engine-managed, e.g. ollama),
          - the candidate has no artifact id (synthetic policy
            candidate built without planner metadata).
        ``manager.prepare`` exceptions surface unchanged so the runner
        records the failure and falls back to the next candidate — same
        contract as the TTS / transcription paths.
        """
        if candidate is None:
            return None
        if getattr(candidate, "locality", None) != "local":
            return None
        delivery_mode = getattr(candidate, "delivery_mode", None) or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            return None
        if not getattr(candidate, "prepare_required", False):
            return None
        artifact = getattr(candidate, "artifact", None)
        artifact_id = getattr(artifact, "artifact_id", None) or getattr(artifact, "model_id", None)
        if not artifact_id:
            return None
        cache_key = (
            artifact_id,
            getattr(artifact, "digest", None),
            getattr(artifact, "format", None),
            getattr(artifact, "quantization", None),
        )
        if cache_key in cache:
            return cache[cache_key]

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        prepared = str(outcome.artifact_dir)
        cache[cache_key] = prepared
        return prepared

    def _prepare_local_tts_artifact(self, selection: Optional[Any]) -> Optional[str]:
        """Run :class:`PrepareManager` for the local TTS candidate, if any.

        Returns the prepared ``artifact_dir`` as a string when the planner
        marks the local candidate as ``delivery_mode='sdk_runtime'`` with
        ``prepare_required=True``. Returns ``None`` when no planner-driven
        preparation is required — the dispatch path then consults
        :meth:`_prepared_local_artifact_dir` for a static-recipe prepared
        cache, which after the PR D cutover is the only on-disk source.

        Errors from PrepareManager (policy violations, contract violations,
        download failures) are surfaced unchanged so users get the actionable
        messages the manager constructs.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return None
        if not getattr(candidate, "prepare_required", False):
            return None

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        return str(outcome.artifact_dir)

    def _resolve_local_tts_backend(
        self,
        model: str,
        *,
        prepared_model_dir: Optional[str] = None,
        planner_candidate: Optional[Any] = None,
        load_error: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Resolve a local Sherpa TTS backend; ``None`` on any failure.

        ``load_error`` is an optional out-list the caller can pass in;
        when the backend fails to instantiate / load, the helper
        appends the underlying exception's repr so the dispatch path
        can surface a specific failure mode (sherpa import vs.
        backend load) rather than the old vague
        ``local_tts_runtime_unavailable``.
        """
        # PR 11: prefer the warmup cache. When the caller invoked
        # ``client.warmup(model, capability='tts')`` earlier in this
        # session, the loaded SherpaTts backend is already in
        # ``_warmed_backends`` and the next ``synthesize_speech`` call
        # reuses it without paying ``engine.create_backend`` +
        # ``backend.load_model`` again.
        #
        # The lookup composes the artifact-identity cache key
        # (digest/format/quantization) against the *current* request:
        # callers that already resolved a planner selection thread
        # ``planner_candidate`` so we don't re-plan and we use the
        # exact identity the request will run against. An exact-key
        # miss with a current candidate is a real cache miss — it
        # never falls back to a stale backend cached under a
        # different artifact identity.
        cached = self._lookup_warmed_backend(CAPABILITY_TTS, model, candidate=planner_candidate)
        if cached is not None:
            return cached
        # Two failure surfaces to distinguish:
        #   1. ``import SherpaTtsEngine`` raises — the runtime extra
        #      isn't installed (or ``_sherpa_onnx`` is missing on
        #      stripped embedded Pythons).
        #   2. The engine instantiates but ``create_backend`` /
        #      ``load_model`` fails — the prepared dir is broken
        #      (wrong layout, missing files) or the runtime can't
        #      load the bytes.
        # Both used to raise a single vague
        # ``local_tts_runtime_unavailable`` error; the dispatch
        # path now keys on ``load_error`` to pick a specific message.
        try:
            from octomil.runtime.engines.sherpa import SherpaTtsEngine
        except Exception as exc:
            if load_error is not None:
                load_error.append(f"sherpa_import: {exc!r}")
            return None
        try:
            engine = SherpaTtsEngine()
            backend_kwargs: dict[str, Any] = {}
            if prepared_model_dir:
                # PrepareManager materialized the artifact at this path;
                # load the sherpa-onnx model from there. After the PR D
                # cutover this is the *only* supported source — the
                # backend's ``_resolve_model_dir`` raises when no
                # ``model_dir`` is injected.
                backend_kwargs["model_dir"] = prepared_model_dir
            backend = engine.create_backend(model, **backend_kwargs)
            backend.load_model(model)
            return backend
        except Exception as exc:
            if load_error is not None:
                load_error.append(f"backend_load: {exc!r}")
            return None

    def _validate_local_voice(
        self,
        model: str,
        voice: Optional[str],
        *,
        selection: Optional[Any] = None,
        prepared_cache_dir: Optional[str] = None,
    ) -> None:
        """Pre-flight voice check for local TTS.

        Two artifact-identity-aware code paths:

          1. **Static-recipe artifact** — the same bundle the SDK
             would download for this model id. ``voice_manifest`` on
             the recipe is the authoritative catalog; reject unknown
             voices BEFORE prepare so callers don't pay for a
             download just to learn the voice was wrong.

          2. **Planner-selected (non-static) artifact** — the
             planner picked something the SDK doesn't know the
             catalog of (a private app's custom Kokoro bundle, a
             newer release, etc.). Skip preflight and let the
             backend's own ``voices.txt``-based check surface
             mismatches after prepare. The cost: a ~megabytes-to-
             gigabytes download before the error appears, but
             that's the only safe choice because the SDK's static
             manifest WOULD reject voices the planner artifact
             actually supports.

        The static path is taken when the kernel is going to use
        the static recipe for this request:

          - ``prepared_cache_dir`` is set (static cache short-circuit
            already on disk), OR
          - ``selection`` is None (no planner candidate; static
            recipe is the only fallback), OR
          - ``selection``'s artifact identity matches the static
            recipe (planner returned the same bundle).

        Errors carry both ``voice_not_supported_for_model`` (the new,
        model-id-aware tag) and ``voice_not_supported_for_locality``
        (legacy substring) so existing string assertions still match.
        For cloud, voice mismatches surface post-dispatch via provider 4xx.
        """
        if not voice:
            return
        from octomil.runtime.engines.sherpa import is_sherpa_tts_model, resolve_voice_catalog
        from octomil.runtime.lifecycle.static_recipes import get_static_recipe

        if not is_sherpa_tts_model(model):
            return

        recipe = get_static_recipe(model.lower(), CAPABILITY_TTS)
        if recipe is None:
            return
        manifest = recipe.materialization.voice_manifest
        if not manifest:
            return

        if not self._static_recipe_is_active(recipe, selection, prepared_cache_dir):
            # Planner-selected artifact with a different identity.
            # Defer to the backend, which reads voices.txt from the
            # actually-prepared artifact directory.
            return

        # Use the shared resolver so this preflight, the engine's
        # ``_voice_to_sid``, and ``list_speech_voices`` all walk the
        # same code path. ``prepared_cache_dir`` is preferred when
        # set so a stale recipe-time manifest can't reject voices
        # the on-disk artifact actually supports.
        resolved = resolve_voice_catalog(
            model.lower(),
            prepared_model_dir=prepared_cache_dir,
            static_recipe_manifest=manifest,
            static_recipe_artifact_version=recipe.materialization.artifact_version or "",
        )
        catalog_voices = resolved.voices or manifest

        if voice.strip().lower() not in {name.lower() for name in catalog_voices}:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"voice_not_supported_for_model: voice {voice!r} is not in the "
                    f"speaker catalog for model {model!r}. Supported voices: "
                    f"{', '.join(catalog_voices)}. (voice_not_supported_for_locality)"
                ),
            )

    @staticmethod
    def _static_recipe_is_active(
        recipe: Any,
        selection: Any,
        prepared_cache_dir: Optional[str],
    ) -> bool:
        """Decide whether the static recipe will actually serve this
        request. See ``_validate_local_voice`` for the rationale.
        """
        if recipe is None:
            return False
        if prepared_cache_dir is not None:
            # Static cache short-circuit already on disk.
            return True
        if selection is None:
            # No planner candidate; static recipe is the fallback.
            return True
        # Planner returned a candidate. Compare artifact identity
        # (digest is the strongest signal; falls back to artifact_id).
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return True
        cand_artifact = getattr(candidate, "artifact", None)
        if cand_artifact is None:
            return True
        recipe_digest = recipe.files[0].digest if recipe.files else None
        cand_digest = getattr(cand_artifact, "digest", None)
        if recipe_digest and cand_digest:
            return recipe_digest == cand_digest
        recipe_id = recipe.model_id
        cand_id = getattr(cand_artifact, "artifact_id", None)
        return cand_id == recipe_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(
        self,
        capability: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ResolvedExecutionDefaults:
        overrides = RequestOverrides(model=model, policy=policy, app_slug=app)
        return resolve_capability_defaults(capability, overrides, self._config_set)

    def _can_local(self, model: str, capability: str) -> bool:
        """Check if a local runtime is available for the given model."""
        if capability == CAPABILITY_TRANSCRIPTION:
            return self._has_local_transcription_backend(model)

        try:
            from octomil.runtime.core.registry import ModelRuntimeRegistry

            registry = ModelRuntimeRegistry.shared()
            runtime = registry.resolve(model)
            if runtime is None:
                return False
            if capability == CAPABILITY_EMBEDDING:
                return hasattr(runtime, "embed") or hasattr(runtime, "create_embeddings")
            return True
        except Exception:
            return False

    def _has_local_transcription_backend(self, model: str) -> bool:
        return self._resolve_local_transcription_backend(model) is not None

    def _resolve_local_transcription_backend(
        self,
        model: str,
        *,
        prepared_model_dir: Optional[str] = None,
        planner_candidate: Optional[Any] = None,
    ) -> Optional[Any]:
        """Return a transcription backend for ``model``.

        ``prepared_model_dir`` is threaded into ``engine.create_backend``
        as ``model_dir=...`` so the backend loads from the directory
        PrepareManager just materialized. The whisper.cpp backend honors
        this and skips its own pywhispercpp download path; engines that
        don't recognize the kwarg ignore it (the registry's create_backend
        accepts ``**kwargs``).

        ``planner_candidate``, when supplied, is the candidate the
        request will run against. The warmup-cache lookup uses its
        artifact identity so a stale backend cached under a different
        digest/format never shadows the freshly-prepared bytes.
        """
        # PR 11: warmup cache hit short-circuits the registry walk.
        # Reviewer P1: pass the caller's planner candidate so the
        # cache lookup uses the *current* request's artifact identity
        # rather than re-planning from ``runtime_model``. An
        # exact-identity miss is a real miss — no fall-through to a
        # stale (capability, model)-only entry from a prior version.
        cached = self._lookup_warmed_backend(CAPABILITY_TRANSCRIPTION, model, candidate=planner_candidate)
        if cached is not None:
            return cached
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            backend_kwargs: dict[str, Any] = {}
            if prepared_model_dir:
                backend_kwargs["model_dir"] = prepared_model_dir
            for detection in registry.detect_all(model):
                if not detection.available:
                    continue
                backend = detection.engine.create_backend(model, **backend_kwargs)
                if hasattr(backend, "transcribe"):
                    return backend
        except Exception:
            return None
        return None

    async def _build_router(
        self,
        model: str,
        capability: str,
        defaults: ResolvedExecutionDefaults,
        *,
        planner_selection: Optional[Any] = None,
        prepared_model_dir: Optional[str] = None,
        cloud_execution_model: Optional[str] = None,
    ) -> RouterModelRuntime:
        """Build a RouterModelRuntime for the given model and capability.

        When a planner_selection is provided and recommends a specific engine,
        the local factory tries that engine first before falling back to the
        default registry resolution.

        ``prepared_model_dir`` is threaded into ``engine.create_backend``
        as ``model_dir=...`` for engines that consume prepared bytes
        (mlx-lm, llama.cpp, sherpa-onnx, whisper). Engines that do not
        recognize the kwarg ignore it (each ``create_backend`` accepts
        ``**kwargs``). When ``None`` the engine's own model resolution
        path runs unchanged.

        ``cloud_execution_model`` is the identity to send to hosted
        inference. PR B follow-up: when the caller passed ``app=`` (or
        the request was an ``@app/...`` ref), cloud dispatch must run
        under the app identity even though local backend resolution
        uses the resolved runtime model. ``None`` (default) keeps the
        legacy behaviour of running cloud under ``model``.
        """
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()

        # If planner says cloud, force cloud by returning None from local factory
        planner_forces_cloud = planner_selection is not None and planner_selection.locality == "cloud"
        planner_engine = (
            planner_selection.engine
            if planner_selection is not None and planner_selection.locality == "local"
            else None
        )
        backend_kwargs: dict[str, Any] = {}
        if prepared_model_dir:
            backend_kwargs["model_dir"] = prepared_model_dir

        def local_factory(hint: str):
            if planner_forces_cloud:
                return None

            # If planner recommends a specific local engine, try it first
            if planner_engine:
                try:
                    from octomil.runtime.engines import get_registry as get_engine_registry

                    engine_registry = get_engine_registry()
                    engine = engine_registry.get_engine(planner_engine)
                    if engine is not None and engine.detect() and engine.name != "echo":
                        from octomil.runtime.core.adapter import InferenceBackendAdapter
                        from octomil.runtime.core.engine_bridge import _infer_tool_call_tier
                        from octomil.runtime.core.types import RuntimeCapabilities

                        backend = engine.create_backend(model, **backend_kwargs)
                        return InferenceBackendAdapter(
                            backend=backend,
                            model_name=model,
                            capabilities=RuntimeCapabilities(
                                tool_call_tier=_infer_tool_call_tier(model),
                                supports_streaming=True,
                            ),
                        )
                except Exception:
                    logger.debug(
                        "Planner-recommended engine %s failed, falling back to registry",
                        planner_engine,
                        exc_info=True,
                    )

            resolved = registry.resolve(model)
            # Never let echo leak into user-facing execution paths
            if resolved is not None:
                backend = getattr(resolved, "_backend", None)
                backend_cls = type(backend).__name__ if backend else ""
                if backend_cls == "EchoBackend":
                    logger.debug("Rejecting echo backend from user-facing execution path")
                    return None
            return resolved

        def cloud_factory(hint: str):
            if defaults.cloud_profile is None:
                return None
            try:
                from octomil.runtime.core.cloud_runtime import CloudModelRuntime

                api_key = os.environ.get(defaults.cloud_profile.api_key_env, "")
                if not api_key:
                    return None
                # When the caller explicitly app-scoped the request,
                # send the synthesized ``@app/<app>/<cap>`` identity
                # to the hosted endpoint so server-side routing /
                # quota / billing match the planner's app
                # resolution. Falls back to the resolved ``model``
                # for non-app callers (legacy behaviour).
                cloud_model = cloud_execution_model or model
                return CloudModelRuntime(
                    base_url=_openai_base_url(defaults.cloud_profile),
                    api_key=api_key,
                    model=cloud_model,
                )
            except Exception:
                return None

        return RouterModelRuntime(
            local_factory=local_factory,
            cloud_factory=cloud_factory,
        )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _runner_dict_to_planner_candidate(
    selection: Optional[Any],
    candidates: list[dict[str, Any]],
) -> dict[int, Any]:
    """Map each runner-dict in ``candidates`` to its planner candidate.

    ``_selection_candidate_dicts`` produces its dicts in 1:1 order with
    ``selection.candidates`` whenever the planner emitted a non-empty
    list, so we pair by position. The dict identity (``id(c)``) is the
    map key — that survives the post-filter list as long as the dicts
    aren't copied (which they aren't: filter just removes entries).

    Dicts that have no original (the policy-synthesized fallback path
    when the planner emitted nothing) are not mapped; ``cache.get``
    returns ``None`` for those, which the prepare helper treats as
    no-op.
    """
    originals = list(getattr(selection, "candidates", None) or [])
    mapping: dict[int, Any] = {}
    for idx, dict_candidate in enumerate(candidates):
        if idx < len(originals):
            mapping[id(dict_candidate)] = originals[idx]
    return mapping


def _filter_unpreparable_local_candidates(
    kernel: "ExecutionKernel",
    selection: Optional[Any],
    candidates: list[dict[str, Any]],
    *,
    fallback_allowed: bool,
) -> list[dict[str, Any]]:
    """Drop *only* the unpreparable local candidates from the runner list.

    The earlier shape of this helper looked at ``_local_candidate_is_unpreparable``
    (which inspects the *first* local sdk_runtime candidate of the
    selection) and then removed *every* local candidate. That was wrong
    in two ways:

      1. Multi-local plans like ``[bad local mlx, good local llama, cloud]``
         lost the valid llama fallback. The bad mlx candidate poisoned
         the entire local lane.
      2. Plans with ``fallback_allowed=False`` like
         ``[bad local, cloud]`` could be silently promoted to cloud:
         the local primary disappeared and cloud — which the planner
         only emitted as a non-fallback option — became the first
         remaining candidate.

    The fix evaluates each local sdk_runtime candidate independently
    against ``PrepareManager.can_prepare`` and drops only the
    individually unpreparable ones. When a *primary* candidate (index
    0) is unpreparable AND ``fallback_allowed`` is False, the helper
    returns an empty list so the runner surfaces the planner's "no
    runnable candidate" error instead of laundering the request to
    cloud.

    Cloud candidates and engine-managed (``prepare_required=False``)
    local candidates pass through untouched.
    """
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    manager = kernel._prepare_manager or PrepareManager()
    # ``_selection_candidate_dicts`` produces its list 1:1 in order with
    # ``selection.candidates`` whenever the planner emitted a non-empty
    # list, so we can pair them by position. When the dicts came from
    # the policy synthesis path (no planner candidates) the originals
    # list is shorter / empty and the corresponding entries pair to
    # ``None`` — those candidates have no prepare metadata to validate
    # so they pass through unchanged.
    originals = list(getattr(selection, "candidates", None) or [])
    paired: list[tuple[dict[str, Any], Optional[Any]]] = []
    for idx, dict_candidate in enumerate(candidates):
        original = originals[idx] if idx < len(originals) else None
        paired.append((dict_candidate, original))

    def _is_unpreparable_local(candidate_dict: dict[str, Any], original: Optional[Any]) -> bool:
        if candidate_dict.get("locality") != "local":
            return False
        delivery_mode = candidate_dict.get("delivery_mode") or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            return False
        if original is None:
            # No matching planner plan object — synthetic policy
            # candidate, no prepare metadata to validate.
            return False
        if not getattr(original, "prepare_required", False):
            return False
        try:
            return not manager.can_prepare(original)
        except Exception:
            return True

    primary_unpreparable = bool(paired) and _is_unpreparable_local(*paired[0])
    filtered = [c for c, original in paired if not _is_unpreparable_local(c, original)]

    if primary_unpreparable and not fallback_allowed:
        # Planner forbids fallback and the primary is unrunnable.
        # Raising an actionable OctomilError here is better than
        # returning ``[]`` and letting the runner emit a generic
        # "No runtime available": the latter loses the rejected
        # primary's identity (engine, artifact_id, reason), which
        # is exactly the diagnostic the caller needs to fix their
        # plan or relax fallback_allowed.
        primary_dict, primary_original = paired[0]
        engine = primary_dict.get("engine") or getattr(primary_original, "engine", None) or "?"
        artifact = getattr(primary_original, "artifact", None)
        artifact_id = getattr(artifact, "artifact_id", None) or getattr(artifact, "model_id", None) or "<unknown>"
        reason = primary_dict.get("reason") or getattr(primary_original, "reason", "") or ""
        raise OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=(
                f"local_runtime_unavailable: planner emitted a local sdk_runtime candidate "
                f"(engine={engine!r}, artifact={artifact_id!r}) but PrepareManager.can_prepare "
                f"rejected the metadata (synthetic plan: missing digest/url, traversal, disabled "
                f"policy, etc.) and fallback_allowed=False blocks promotion to cloud. "
                f"{('reason=' + reason + '. ') if reason else ''}"
                f"Fix the planner candidate or relax fallback_allowed to admit cloud."
            ),
        )
    return filtered


def _local_sdk_runtime_candidate(selection: Optional[Any]) -> Optional[Any]:
    """Return the first ``locality='local', delivery_mode='sdk_runtime'``
    candidate in ``selection.candidates``, or ``None``.

    Cloud and external-endpoint candidates are skipped; PrepareManager only
    handles SDK-runtime artifacts. The caller still has to check
    ``prepare_required`` before deciding whether to invoke prepare at all.
    """
    if selection is None:
        return None
    candidates = getattr(selection, "candidates", None) or []
    for candidate in candidates:
        if getattr(candidate, "locality", None) != "local":
            continue
        delivery_mode = getattr(candidate, "delivery_mode", None) or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            continue
        return candidate
    return None


def _tts_policy_from_selection(selection: Any) -> Optional[str]:
    if selection is None:
        return None

    for attr in ("app_resolution", "resolution"):
        resolved = getattr(selection, attr, None)
        raw = getattr(resolved, "routing_policy", None)
        if raw:
            policy = getattr(raw, "value", raw)
            policy = str(policy).strip().lower()
            if policy in {
                "private",
                "local_only",
                "local_first",
                "cloud_first",
                "cloud_only",
                "performance_first",
            }:
                return policy
            if policy == "auto":
                return "auto"
    return None


_LOCAL_ONLY_POLICY_NAMES = frozenset({"private", "local_only"})


def _is_local_only_policy(policy: Optional[str]) -> bool:
    """Return True iff ``policy`` is an explicit local-only preset.

    Used by the chat / TTS / transcription dispatchers to force
    ``cloud_available=False`` when the caller has expressed a hard
    requirement that the request must not leave the device. Both
    ``"private"`` and ``"local_only"`` qualify; the SDK treats
    "private" as the privacy-strict superset of "local_only" (no
    benchmark uploads either).
    """
    if not policy:
        return False
    return policy.strip().lower() in _LOCAL_ONLY_POLICY_NAMES


def _execution_model_for_cloud_dispatch(
    *,
    requested_model: Optional[str],
    effective_model: str,
    planner_model: str,
    app: Optional[str],
) -> str:
    """Return the model id cloud dispatch should send under.

    Reviewer P1 (post PR #454): the planner now sees
    ``@app/<app>/<capability>`` when ``app=`` is explicit, but each
    capability's cloud branch was passing the *resolved* runtime
    model (e.g. ``kokoro-en-v0_19``) to the hosted endpoint. That
    means the server can't apply the app's quota / routing policy /
    billing the planner just blessed — the request reaches hosted
    inference under a different identity than the one routing was
    decided against.

    Resolution rule:

      - ``requested_model`` is already an ``@app/...`` ref → send
        the request as-is so the server keeps app-scoping.
      - ``requested_model`` is concrete BUT ``app=`` was explicit
        → send the synthesized ``@app/<app>/<capability>`` ref
        (the same one the planner saw) so cloud dispatch agrees
        with the routing decision.
      - otherwise → send the concrete ``effective_model``. Same
        behaviour as before for non-app callers.
    """
    if requested_model is not None and isinstance(requested_model, str) and requested_model.startswith("@app/"):
        return requested_model
    if app and planner_model.startswith("@app/"):
        return planner_model
    return effective_model


def _planner_visible_capability(capability: str) -> str:
    """Map an internal capability constant to the planner-visible name.

    The internal ``CAPABILITY_CHAT`` constant is ``"chat"``, but the
    planner endpoint and server-side routing model both speak of
    that public surface as ``"responses"``.
    ``RuntimePlanner.resolve`` already runs incoming capability
    strings through ``_PLANNER_CAPABILITY_MAP`` for its outbound
    request — but when the SDK *synthesizes* an app ref like
    ``@app/<app>/<capability>`` locally, that synthesized string IS
    the canonical app ref the server keys quotas / billing / app
    metadata against. If we synthesized ``@app/eternum/chat`` and
    the server only knows ``@app/eternum/responses``, the app
    resolution would fall apart even though the planner endpoint
    received ``capability="responses"``.

    Aligning both halves on the public name keeps the planner's
    parsed capability and the server-visible app ref in agreement.
    Other capabilities (tts, transcription, embedding) round-trip
    unchanged because the planner map only rewrites
    ``chat → responses``.
    """

    return _PLANNER_CAPABILITY_MAP.get(capability, capability)


def _planner_model_for_request(
    *,
    effective_model: str,
    app: Optional[str],
    capability: str,
) -> str:
    """Synthesize an ``@app/<slug>/<canonical-capability>`` ref when
    ``app=`` is set.

    Reviewer P1 on PR #454: the public facade exposes ``app=`` so a
    caller can do ``client.audio.speech.create(model='kokoro-82m',
    app='eternum')``. Previously that ``app`` value reached
    ``ResolvedExecutionDefaults`` but never propagated into the
    planner — ``RuntimePlanner.resolve()`` only sets ``app_slug`` by
    parsing an ``@app/...`` model ref. The result was that the
    request bypassed the app's routing policy entirely AND fell
    through the app-ref refusal gate (``requested_model`` was a
    concrete model id, not an ``@app/`` string).

    The synthesis uses the *planner-visible* capability name (so
    chat → ``@app/<app>/responses``, not ``@app/<app>/chat``) so the
    same string the planner endpoint receives also identifies the
    app on the server side for cloud dispatch. Otherwise
    ``RuntimePlanner.resolve`` would parse our synthesized
    ``@app/eternum/chat`` and the cloud branch would dispatch under
    a capability the server doesn't recognize.
    """
    if not app:
        return effective_model
    if isinstance(effective_model, str) and effective_model.startswith("@app/"):
        return effective_model
    canonical_capability = _planner_visible_capability(capability)
    return f"@app/{app}/{canonical_capability}"


def _enforce_app_ref_routing_policy(
    *,
    requested_model: str,
    selection: Optional[Any],
    explicit_policy: Optional[str],
    explicit_app: Optional[str] = None,
    resolved_policy_preset: Optional[str] = None,
) -> None:
    """Refuse to silently cloud-route an ``@app/...`` ref on planner outage.

    Reviewer P1: when the caller asks for an app ref like
    ``@app/eternum/tts`` and the planner is unreachable (auth
    misconfig, network partition, Ren'Py / sandboxed embed without
    an HTTP transport), the SDK previously fell through to the
    routing-policy default (typically ``local_first``), which can
    silently land the request on a hosted provider. That's the worst
    DX shape: a developer who deliberately chose a private app gets
    a 403 from cloud telling them to use a local SDK that they're
    already using.

    This helper preserves caller intent:

      - When ``explicit_policy`` is ``"private"`` / ``"local_only"``
        — caller has expressed a hard local requirement. We let the
        request proceed; ``cloud_available`` is forced False
        downstream by :func:`_is_local_only_policy` so the planner
        outage cannot leak it to cloud.
      - When ``explicit_policy`` is some other value (``cloud_first``,
        ``performance_first``, etc.) — caller has accepted that cloud
        is allowed. We let the request proceed unchanged.
      - When ``explicit_policy`` is ``None`` AND the request is an
        app ref AND the planner returned no selection — the SDK
        cannot know the app's intended routing policy. Raising here
        is better than guessing: caller should either fix the
        planner / auth or pass an explicit ``policy=`` so we know
        what they want.

    Non-app refs (concrete model ids like ``kokoro-en-v0_19``) skip
    this check entirely; the caller has already named the artifact
    they want and there's no app-side policy to consult.
    """
    if explicit_policy is not None:
        return
    # A locally-resolved policy preset (e.g. from ``OctomilConfig``
    # or octomil.toml) carries the same caller intent as an explicit
    # ``policy=`` argument — a developer who set ``local_only`` in
    # config has expressed the hard local requirement just as
    # firmly. Without this short-circuit, app refs blow up with
    # "Could not resolve app routing policy" even when the local
    # config plainly says local-only.
    if resolved_policy_preset and resolved_policy_preset != "local_first":
        return
    is_app_scoped = (isinstance(requested_model, str) and requested_model.startswith("@app/")) or bool(explicit_app)
    if not is_app_scoped:
        return
    if selection is not None:
        # Planner answered. Even if it returned a synthetic cloud
        # fallback, the request has a planner-blessed path.
        return
    descriptor = requested_model
    if explicit_app and isinstance(requested_model, str) and not requested_model.startswith("@app/"):
        descriptor = f"{requested_model!r} app={explicit_app!r}"
    raise OctomilError(
        code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        message=(
            f"Could not resolve app routing policy for {descriptor}; "
            "the runtime planner is unavailable (network, auth, or planner-import "
            "failure). The SDK refuses to silently fall back to cloud for "
            "app-scoped requests because that surfaces a confusing 403 to "
            "local-first callers. Pass policy='local_only' to force local, "
            "policy='cloud_first' to allow cloud, or fix planner connectivity / "
            "OCTOMIL_SERVER_KEY auth."
        ),
    )


def _is_app_ref(model: str) -> bool:
    """Return True for ``@app/<slug>/<capability>`` model refs.

    Used by the prepared-cache short-circuit gate to refuse cache
    substitution when the request was app-scoped — silently serving
    the public static-recipe artifact in place of a private app
    artifact would hide the planner/server config error and could
    serve the wrong bytes.
    """
    return isinstance(model, str) and model.startswith("@app/")


def _local_tts_runtime_unavailable(model: str) -> OctomilError:
    return OctomilError(
        code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        message=(
            "local_tts_runtime_unavailable: sherpa-onnx is not installed "
            "or no prepared artifact dir exists for "
            f"'{model}'. Install sherpa-onnx (`pip install octomil[tts]`) "
            f"and run `octomil prepare {model} --capability tts` "
            "(or `client.prepare(model='" + model + "', capability='tts')`) "
            "before invoking speech.create. The legacy "
            "OCTOMIL_SHERPA_MODELS_DIR / ~/.octomil/models/sherpa staging "
            "path was removed in 4.11.0; PrepareManager's artifact cache "
            "is the only supported source."
        ),
    )


def _no_model_error(capability: str) -> RuntimeError:
    return RuntimeError(
        f"No default model configured for {capability}.\n\n"
        f"Pass --model, run `octomil models`, or add .octomil.toml:\n\n"
        f"[capabilities.{capability}]\n"
        f'model = "your-model-name"'
    )


def _extract_usage(response: RuntimeResponse) -> dict[str, int]:
    usage: dict[str, int] = {}
    if hasattr(response, "usage") and response.usage:
        u = response.usage
        if hasattr(u, "input_tokens"):
            usage["input_tokens"] = u.input_tokens
        if hasattr(u, "output_tokens"):
            usage["output_tokens"] = u.output_tokens
        if hasattr(u, "total_tokens"):
            usage["total_tokens"] = u.total_tokens
        elif "input_tokens" in usage and "output_tokens" in usage:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _chat_messages_to_runtime_messages(messages: list[dict[str, str]]) -> list[RuntimeMessage]:
    runtime_messages: list[RuntimeMessage] = []
    for message in messages:
        raw_role = message.get("role", MessageRole.USER.value)
        try:
            role = MessageRole(raw_role)
        except ValueError:
            role = MessageRole.USER
        runtime_messages.append(
            RuntimeMessage(
                role=role,
                parts=[RuntimeContentPart.text_part(message.get("content", ""))],
            )
        )
    return runtime_messages
