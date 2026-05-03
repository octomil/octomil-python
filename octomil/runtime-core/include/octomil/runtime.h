/*
 * octomil/runtime.h — Layer 2a native runtime core C ABI
 *
 * This header is the contract between the Octomil native runtime
 * core (liboctomil-runtime) and every SDK binding above it (Python
 * cffi, iOS Swift framework, Android Kotlin JNI, Node N-API, browser
 * WASM).
 *
 * Per the engineering-debate consensus on
 * `strategy/runtime-architecture-v2.md` (R3, three rounds), this
 * header is the slice-1 deliverable. Slice 2 lands a working
 * `audio.realtime.session` implementation against this surface
 * (Moshi-on-MLX on macOS via Kyutai's official MLX path); slice 3
 * makes the Python SDK the first binding; slice 4 builds the iOS
 * framework around the same dylib.
 *
 * Architectural rules this header enforces (per the consensus):
 *
 *   1. Layer 2a is in-process — no network, no IPC, no Python
 *      interpreter on the latency path. The runtime never opens a
 *      socket. Callbacks for telemetry stay in-process.
 *   2. Layer 2b (planner, prep, policy, audit) stays in
 *      language-side code. It does NOT cross this ABI on day one;
 *      its outputs flow as primitive strings and structs.
 *   3. Same-runtime-everywhere = same implementation, same contract.
 *      Different binaries per (platform, arch). Capability
 *      verification on `oct_session_open` is how we honor "recipe ≠
 *      availability."
 *   4. Same scheduler implementation, NOT shared state across
 *      processes. Each `oct_runtime_t` owns its own scheduler.
 *
 * Versioning:
 *
 *   - `OCT_RUNTIME_ABI_VERSION` is the SemVer of this header. Bumped
 *     on any breaking change.
 *   - Versioned config structs (`version` field on
 *     `oct_runtime_config_t`, `oct_session_config_t`,
 *     `oct_event_t`) decouple binding compile-time version from
 *     runtime build version. A binding compiled against version N
 *     can talk to a runtime built for version N+k as long as N's
 *     fields are still recognized; otherwise
 *     `OCT_STATUS_VERSION_MISMATCH`.
 *   - Minor versions are additive (new fields after the version
 *     field on existing structs; new event types via tag
 *     extension). Major versions ship as side-by-side dylibs;
 *     bindings link against one major.
 *
 * This is an EXPLICIT-EXTERNAL-LINKAGE C header. Every public
 * symbol is `OCT_API`-decorated for cross-platform export visibility
 * (gcc `__attribute__((visibility("default")))`, MSVC
 * `__declspec(dllexport)`).
 */

#ifndef OCTOMIL_RUNTIME_H
#define OCTOMIL_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------- *
 * ABI version                                                         *
 * ------------------------------------------------------------------- */

/* Bumped on breaking changes. Bindings inspect this at runtime via
 * `oct_runtime_abi_version()` to fail-fast on incompatible dylibs. */
#define OCT_RUNTIME_ABI_VERSION_MAJOR 0
#define OCT_RUNTIME_ABI_VERSION_MINOR 1
#define OCT_RUNTIME_ABI_VERSION_PATCH 0

/* Versions of versioned config structs. Bumped lockstep with the
 * struct's field set. Bindings set these on every config they pass;
 * the runtime fails OCT_STATUS_VERSION_MISMATCH on unknown values. */
#define OCT_RUNTIME_CONFIG_VERSION   1
#define OCT_SESSION_CONFIG_VERSION   1
#define OCT_EVENT_VERSION            1
#define OCT_CAPABILITIES_VERSION     1

/* ------------------------------------------------------------------- *
 * Export visibility                                                   *
 * ------------------------------------------------------------------- */

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef OCTOMIL_RUNTIME_BUILDING
#    define OCT_API __declspec(dllexport)
#  else
#    define OCT_API __declspec(dllimport)
#  endif
#elif __GNUC__ >= 4
#  define OCT_API __attribute__((visibility("default")))
#else
#  define OCT_API
#endif

/* ------------------------------------------------------------------- *
 * Forward declarations                                                *
 * ------------------------------------------------------------------- *
 * Codex R1 fix on the strategy doc: oct_telemetry_sink_fn references *
 * oct_event_t, which is defined later. Forward-declare here so the   *
 * function-pointer typedef can name the struct without ordering      *
 * pain.                                                              *
 * ------------------------------------------------------------------- */

typedef struct oct_runtime  oct_runtime_t;
typedef struct oct_session  oct_session_t;
typedef struct oct_event    oct_event_t;

/* ------------------------------------------------------------------- *
 * Status enum                                                         *
 * ------------------------------------------------------------------- */

typedef enum {
    OCT_STATUS_OK                = 0,
    OCT_STATUS_INVALID_INPUT     = 1,  /* malformed config, NULL out, etc. */
    OCT_STATUS_UNSUPPORTED       = 2,  /* capability / locality / engine not built into this dylib OR not loadable on this device */
    OCT_STATUS_NOT_FOUND         = 3,  /* model_uri / artifact missing on disk */
    OCT_STATUS_BUSY              = 4,  /* input queue full (realtime backpressure) — bindings drop or retry */
    OCT_STATUS_TIMEOUT           = 5,  /* poll_event timeout — out->type == OCT_EVENT_NONE */
    OCT_STATUS_CANCELLED         = 6,  /* session was cancelled; subsequent polls also return CANCELLED */
    OCT_STATUS_INTERNAL          = 7,  /* runtime invariant violated — diagnostic via oct_runtime_last_error */
    OCT_STATUS_VERSION_MISMATCH  = 8,  /* config.version unknown to this runtime build */
} oct_status_t;

/* ------------------------------------------------------------------- *
 * Capability discovery                                                *
 * ------------------------------------------------------------------- *
 * Surfaces "recipe ≠ availability." A recipe in octomil-contracts is *
 * a declaration; whether it works on this build of liboctomil-       *
 * runtime depends on the engine adapters compiled in AND the device  *
 * having the required RAM / accelerator / format support.            *
 *                                                                    *
 * Ownership rule (cross-SDK contract): the runtime allocates the     *
 * string arrays and the strings they point to. They are valid until  *
 * either oct_runtime_capabilities_free(out) is called, or            *
 * oct_runtime_close is called on the parent runtime, whichever       *
 * comes first. Bindings MUST NOT free individual strings; use        *
 * oct_runtime_capabilities_free.                                     *
 *                                                                    *
 * Lengths: each `const char**` is null-terminated (sentinel = NULL). *
 * A single sentinel scan computes count when bindings need it.       *
 * ------------------------------------------------------------------- */

typedef struct {
    uint32_t      version;                  /* OCT_CAPABILITIES_VERSION */
    const char**  supported_engines;        /* null-terminated; runtime-owned */
    const char**  supported_capabilities;   /* null-terminated; runtime-owned */
    const char**  supported_archs;          /* null-terminated; runtime-owned */
    uint64_t      ram_total_bytes;
    uint64_t      ram_available_bytes;
    bool          has_apple_silicon;
    bool          has_cuda;
    bool          has_metal;
} oct_capabilities_t;

/* ------------------------------------------------------------------- *
 * Telemetry sink                                                      *
 * ------------------------------------------------------------------- *
 * The runtime emits structured events (oct_event_t) to the binding   *
 * via this synchronous callback. The binding decides where the       *
 * events go (network export, on-disk audit log, in-memory ring,      *
 * drop). The runtime NEVER opens a socket — telemetry transport is   *
 * Layer 2b's responsibility per the architecture doc's "Layer 2a is  *
 * in-process; no network, no IPC" rule.                              *
 *                                                                    *
 * REENTRANCY: the callback is invoked synchronously on the producing *
 * thread. Bindings whose telemetry path has any latency (e.g.        *
 * forwards over the network) MUST hand off to a different thread     *
 * inside the callback, OR risk blocking the audio thread. The        *
 * runtime cannot enforce this without losing the no-network          *
 * invariant.                                                          *
 *                                                                    *
 * LIFETIME: the `event` pointer is borrowed; valid only for the      *
 * duration of the callback. Bindings that need durable storage copy  *
 * the relevant fields before returning.                              *
 * ------------------------------------------------------------------- */

typedef void (*oct_telemetry_sink_fn)(
    const oct_event_t* event,    /* borrowed; lifetime = call duration */
    void* user_data
);

/* ------------------------------------------------------------------- *
 * Runtime lifecycle                                                   *
 * ------------------------------------------------------------------- */

typedef struct {
    uint32_t version;                          /* OCT_RUNTIME_CONFIG_VERSION */
    const char* artifact_root;                 /* where prepared artifacts live on disk */
    oct_telemetry_sink_fn telemetry_sink;      /* optional; NULL = drop telemetry */
    void*       telemetry_user_data;           /* passed back to sink callback */
    uint32_t    max_sessions;                  /* hard cap; 0 = unbounded */
} oct_runtime_config_t;

/*
 * Open returns oct_status_t; the handle is an out-parameter. Status-
 * returning open lets failure modes (version, capability, RAM,
 * internal) surface as structured oct_status_t rather than
 * collapsing into a NULL handle.
 *
 * Edge case: if `out == NULL`, returns OCT_STATUS_INVALID_INPUT.
 * On any non-OK return, `*out` is set to NULL — bindings can rely
 * on this for error-path cleanup without a separate sentinel check.
 */
OCT_API oct_status_t oct_runtime_open(
    const oct_runtime_config_t* config,
    oct_runtime_t** out
);

OCT_API void oct_runtime_close(oct_runtime_t* runtime);

OCT_API oct_status_t oct_runtime_capabilities(
    oct_runtime_t* runtime,
    oct_capabilities_t* out
);

OCT_API void oct_runtime_capabilities_free(oct_capabilities_t* caps);

/*
 * ABI version inspection — bindings call this immediately after
 * dlopen / LoadLibrary to fail-fast on incompatible dylibs. The
 * returned version reflects the dylib's compiled-in version, not
 * any header consumed at compile time.
 */
OCT_API uint32_t oct_runtime_abi_version_major(void);
OCT_API uint32_t oct_runtime_abi_version_minor(void);
OCT_API uint32_t oct_runtime_abi_version_patch(void);

/* ------------------------------------------------------------------- *
 * Diagnostic strings                                                  *
 * ------------------------------------------------------------------- *
 * THREAD-LOCAL per-thread last-error message. Useful for surfacing   *
 * "RAM too low (4GB available, 6GB required)" vs "engine not in this *
 * build" without inflating the status enum. Optional for bindings;   *
 * recommended.                                                        *
 *                                                                    *
 * THREAD-LOCALITY: the last-error string is per-thread and valid     *
 * until the next runtime call on that thread, OR the next            *
 * oct_runtime_last_error / oct_last_thread_error call.               *
 *                                                                    *
 * Two variants:                                                       *
 *   - oct_runtime_last_error: scoped to a specific runtime handle.   *
 *     Use this when the prior call had a runtime handle in scope.    *
 *   - oct_last_thread_error: no handle required. Use this when the   *
 *     prior call was oct_runtime_open and failed (no handle yet).    *
 *                                                                    *
 * Both write up to `buflen-1` bytes plus a trailing NUL into `buf`.  *
 * Returns the number of bytes written (excluding NUL). Negative on   *
 * error (NULL buf, buflen == 0).                                     *
 * ------------------------------------------------------------------- */

OCT_API int oct_runtime_last_error(
    oct_runtime_t* runtime,
    char* buf,
    size_t buflen
);

OCT_API int oct_last_thread_error(
    char* buf,
    size_t buflen
);

/* ------------------------------------------------------------------- *
 * Session config                                                      *
 * ------------------------------------------------------------------- *
 * INVARIANTS the bindings should encode:                              *
 *                                                                    *
 *  - All input structs MUST be zero-initialized by the caller before *
 *    populating. The runtime treats unset fields as 0 / NULL         *
 *    deliberately. The `version` field is always non-zero, so        *
 *    version-mismatch detection works against zeroed structs.        *
 *  - NO CALLBACK INTO HOST CODE on session_open. All data needed for *
 *    synthesis flows through this struct (or via                     *
 *    oct_session_send_audio / send_text after open). Callbacks-      *
 *    into-host across the C ABI are the worst kind of complexity;   *
 *    speaker resolution, voice profile lookup, etc. happen in        *
 *    Layer 2b BEFORE oct_session_open is called.                     *
 * ------------------------------------------------------------------- */

/* Priority enum — mirrors octomil-python's TtsRequestPriority. */
#define OCT_PRIORITY_SPECULATIVE  0
#define OCT_PRIORITY_PREFETCH     1
#define OCT_PRIORITY_FOREGROUND   2

typedef struct {
    uint32_t version;                /* OCT_SESSION_CONFIG_VERSION */
    const char* model_uri;           /* "@app/eternum/realtime" | "kokoro-82m" | ... */
    const char* capability;          /* "realtime" | "tts" | "stt" | "chat" | "embed" */
    /*
     * Locality the control plane resolved to. INFORMATIONAL ONLY.
     * The runtime acts on "on_device" only; any other value returns
     * OCT_STATUS_UNSUPPORTED. The runtime MUST NOT initiate cloud
     * fallback or make a network call from this field. Cloud routing
     * is the SDK / control plane's responsibility above the ABI.
     */
    const char* locality;            /* "on_device" only — anything else => UNSUPPORTED */
    /*
     * Policy preset the control plane resolved. INFORMATIONAL ONLY.
     * The runtime carries this in events for observability and
     * audit correlation; it does NOT enforce policy (no cloud-
     * fallback decisions, no quota checks, no auth). Policy
     * enforcement is Layer 2b's job.
     */
    const char* policy_preset;       /* "private" | "local_first" | ... */
    const char* speaker_id;          /* optional; NULL ok */
    uint32_t    sample_rate_in;      /* 0 = engine preferred input rate */
    uint32_t    sample_rate_out;     /* 0 = engine preferred output rate */
    uint32_t    priority;            /* OCT_PRIORITY_* */
    void*       user_data;           /* opaque, returned on events */
} oct_session_config_t;

OCT_API oct_status_t oct_session_open(
    oct_runtime_t* runtime,
    const oct_session_config_t* config,
    oct_session_t** out
);

OCT_API void oct_session_close(oct_session_t* session);

/* ------------------------------------------------------------------- *
 * Threading model                                                     *
 * ------------------------------------------------------------------- *
 * `oct_runtime_t` is THREAD-SAFE for concurrent oct_session_open and *
 * oct_runtime_capabilities calls. Bindings may share one runtime    *
 * instance across threads.                                            *
 *                                                                    *
 * `oct_session_t` is SINGLE-THREAD-AFFINE within one logical         *
 * pipeline: send_audio + send_text + poll_event may NOT be called    *
 * concurrently against the same session. Bindings that want a        *
 * separate "send" thread and "receive" thread MUST serialise via    *
 * their own lock; the runtime does not lock per-session for them.    *
 * Rationale: the engine's KV cache and audio buffer state are not    *
 * safe under concurrent mutation, and adding internal locking would *
 * impose latency on the read path that single-threaded callers (the *
 * majority) should not pay.                                           *
 *                                                                    *
 * `oct_session_cancel` is the ONE EXCEPTION: it is safe to call     *
 * from any thread at any time, against an in-flight session. Atomic *
 * flag flip; the producer thread observes at chunk boundaries.       *
 * Subsequent poll_event calls on the cancelled session deliver a    *
 * SESSION_COMPLETED event with terminal_status=CANCELLED, then      *
 * return OCT_STATUS_CANCELLED.                                       *
 *                                                                    *
 * `poll_event` ordering: events are delivered in the order the      *
 * runtime emitted them. There is no batching; one event per call.   *
 * ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- *
 * Bidirectional I/O                                                   *
 * ------------------------------------------------------------------- *
 * AUDIO VIEW LIFETIME: caller-owned, valid only for the duration of  *
 * the oct_session_send_audio call. The runtime copies internally if  *
 * it needs to retain. Same posture for oct_session_send_text's       *
 * `utf8` argument.                                                    *
 * ------------------------------------------------------------------- */

typedef struct {
    const float* samples;            /* borrowed; lifetime = call duration */
    uint32_t     n_samples;
    uint32_t     sample_rate;
    uint32_t     channels;           /* 1 = mono (canonical) */
} oct_audio_view_t;

OCT_API oct_status_t oct_session_send_audio(
    oct_session_t* session,
    const oct_audio_view_t* audio
);

/*
 * `utf8` is caller-owned and valid only for the duration of the
 * call. Same model as oct_audio_view_t.samples.
 */
OCT_API oct_status_t oct_session_send_text(
    oct_session_t* session,
    const char* utf8
);

/* ------------------------------------------------------------------- *
 * Event envelope                                                      *
 * ------------------------------------------------------------------- *
 * Single tagged-union event envelope. New event types extend the    *
 * tag; old consumers ignore unknown tags via the OCT_EVENT_NONE     *
 * fallback. Inner struct lifetime: ALL pointer fields inside        *
 * `oct_event_t.data.*` are runtime-owned and valid from the return  *
 * of oct_session_poll_event UNTIL the NEXT call to                  *
 * oct_session_poll_event on the same session, OR oct_session_close *
 * on the session, whichever comes first. Bindings that need durable *
 * storage MUST copy the bytes / strings into binding-owned storage *
 * before issuing another poll. The runtime does NOT defensively     *
 * copy because most bindings consume payload synchronously in the  *
 * poll loop and a defensive copy is wasted work for the common     *
 * case.                                                               *
 * ------------------------------------------------------------------- */

typedef enum {
    OCT_EVENT_NONE                  = 0,  /* timeout placeholder; out->type set to this when poll_event returns OCT_STATUS_TIMEOUT */
    OCT_EVENT_SESSION_STARTED       = 1,
    OCT_EVENT_AUDIO_CHUNK           = 2,
    OCT_EVENT_TRANSCRIPT_CHUNK      = 3,
    OCT_EVENT_USER_SPEECH_DETECTED  = 4,
    OCT_EVENT_TURN_ENDED            = 5,
    OCT_EVENT_CAPABILITY_VERIFIED   = 6,
    OCT_EVENT_ERROR                 = 7,
    OCT_EVENT_SESSION_COMPLETED     = 8,
    OCT_EVENT_INPUT_DROPPED         = 9,  /* realtime backpressure; see strategy/realtime-architecture.md */
} oct_event_type_t;

struct oct_event {
    uint32_t           version;       /* OCT_EVENT_VERSION */
    oct_event_type_t   type;
    uint64_t           monotonic_ns;
    void*              user_data;     /* echoed from oct_session_config_t.user_data */
    union {
        struct {
            const uint8_t* pcm;        /* runtime-owned; lifetime per the rule above */
            uint32_t       n_bytes;
            uint32_t       sample_rate;
            bool           is_final;
        } audio_chunk;
        struct {
            const char* utf8;          /* runtime-owned */
            uint32_t    n_bytes;
        } transcript_chunk;
        struct {
            const char* code;          /* runtime-owned */
            const char* message;       /* runtime-owned */
        } error;
        struct {
            const char* engine;
            const char* model_digest;
            const char* locality;
            const char* streaming_mode;
            const char* runtime_build_tag;
        } session_started;
        struct {
            float        setup_ms;
            float        engine_first_chunk_ms;
            float        e2e_first_chunk_ms;
            float        total_latency_ms;
            float        queued_ms;
            uint32_t     observed_chunks;
            bool         capability_verified;
            oct_status_t terminal_status;
        } session_completed;
        /* INPUT_DROPPED — realtime backpressure / queue overflow */
        struct {
            uint32_t     n_samples_dropped;
            uint32_t     sample_rate;
            const char*  reason;        /* runtime-owned. "queue_full" | "session_busy" | "engine_lagging" */
            uint64_t     dropped_at_ns; /* monotonic timestamp of the drop */
        } input_dropped;
    } data;
};

/*
 * poll_event blocks up to timeout_ms for the next event. Returns:
 *   - OCT_STATUS_OK with `out->type` set to the actual event type.
 *   - OCT_STATUS_TIMEOUT with `out->type = OCT_EVENT_NONE` if no
 *     event is ready within timeout_ms.
 *   - OCT_STATUS_CANCELLED if the session has been cancelled and
 *     all queued events have been drained (after the
 *     SESSION_COMPLETED event with terminal_status=CANCELLED).
 *   - OCT_STATUS_INVALID_INPUT if `session` or `out` is NULL.
 *
 * Single-thread-affine — see threading model above.
 */
OCT_API oct_status_t oct_session_poll_event(
    oct_session_t* session,
    oct_event_t* out,
    uint32_t timeout_ms
);

/*
 * Safe to call from any thread, any time. Atomic flag flip; the
 * producer thread observes at chunk boundaries. Subsequent
 * poll_event calls on the cancelled session deliver a
 * SESSION_COMPLETED event with terminal_status=CANCELLED, then
 * return OCT_STATUS_CANCELLED.
 */
OCT_API oct_status_t oct_session_cancel(oct_session_t* session);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* OCTOMIL_RUNTIME_H */
