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
#define OCT_RUNTIME_ABI_VERSION_MINOR 5  /* v0.4 step 2 (PE review octomil-workspace#27 §1.5/§1.6/§1.9.5): +operational envelope on oct_event_t (request_id/route_id/trace_id/engine_version/adapter_version/accelerator/artifact_digest/cache_was_hit — APPENDED after union per the slice-2A append-only rule); +error_code field on OCT_EVENT_ERROR (APPENDED inside the inner struct); +10 runtime-scope event types (MODEL_LOADED/EVICTED, CACHE_HIT/MISS, QUEUED, PREEMPTED, MEMORY_PRESSURE, THERMAL_STATE, WATCHDOG_TIMEOUT, METRIC); +oct_session_config_t v2 with appended correlation IDs (request_id/route_id/trace_id/kv_prefix_key). Additive only; reads stay 0.3/0.4-compat. Future v0.4 steps: session-scope events (TEXT_DELTA, EMBEDDING_VECTOR, VAD_SEGMENT, etc.); oct_index_t; generic send_json/blob/control; capability descriptors; installed_artifact introspection. */
#define OCT_RUNTIME_ABI_VERSION_PATCH 0

/* Versions of versioned config structs. Bumped lockstep with the
 * struct's field set. Bindings set these on every config they pass;
 * the runtime fails OCT_STATUS_VERSION_MISMATCH on unknown values.
 *
 * IMPORTANT (Codex R1): version alone is INSUFFICIENT for output
 * structs (`oct_event_t`, `oct_capabilities_t`). The runtime cannot
 * know how much memory an older binding allocated; a v2 runtime
 * emitting v2-shaped events would write past the end of a v1
 * binding's allocated buffer. Output structs MUST also carry a
 * `size_t size` field that the binding sets to `sizeof(struct)`
 * before the call. The runtime writes only up to `size` bytes.
 * Input config structs (`oct_runtime_config_t`, `oct_session_config_t`)
 * are read-only from the runtime's perspective, so version alone
 * suffices there. */
#define OCT_RUNTIME_CONFIG_VERSION   1
/* v0.4 step 2 bump 1→2: appended request_id, route_id, trace_id,
 * kv_prefix_key (caller-owned correlation IDs; runtime echoes on
 * every event). v0.3 bindings setting version=1 stay compatible
 * because the new fields are read past the v=1 cutoff and treated
 * as NULL/empty. */
#define OCT_SESSION_CONFIG_VERSION   2
/* v0.4 step 2 bump 1→2: appended operational envelope after the
 * union. Versioned-output `size` handshake makes v0.3/v0.4-step-1
 * bindings (which set out->size = older sizeof) invisible to the
 * envelope writes — runtime stops at out->size. */
#define OCT_EVENT_VERSION            2
#define OCT_CAPABILITIES_VERSION     1

/* v0.4 (PE review octomil-workspace#27 §1.1) — model lifecycle handle.
 * Caller-owned config struct read once at open; version-only versioning
 * per the slice-2A input-config rule. */
#define OCT_MODEL_CONFIG_VERSION     1

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
typedef struct oct_model    oct_model_t;   /* v0.4 — see §1 below */

/* ------------------------------------------------------------------- *
 * Status enum                                                         *
 * ------------------------------------------------------------------- */

/*
 * Status code. Codex R2 fix: typedef'd to `uint32_t` rather than a C
 * enum because C enum size is implementation-defined (same class of
 * ABI layout risk as bare `bool`). Fixed-width integer + named
 * constants is the FFI-portable shape: bindings see a known-width
 * field and codegen targets (Swift, Kotlin, Rust bindgen, Python
 * ctypes) all agree on layout.
 */
typedef uint32_t oct_status_t;
#define OCT_STATUS_OK                ((oct_status_t)0)
#define OCT_STATUS_INVALID_INPUT     ((oct_status_t)1)  /* malformed config, NULL out, etc. */
#define OCT_STATUS_UNSUPPORTED       ((oct_status_t)2)  /* capability / locality / engine not built into this dylib OR not loadable on this device */
#define OCT_STATUS_NOT_FOUND         ((oct_status_t)3)  /* model_uri / artifact missing on disk */
#define OCT_STATUS_BUSY              ((oct_status_t)4)  /* input queue full (realtime backpressure) — bindings drop or retry */
#define OCT_STATUS_TIMEOUT           ((oct_status_t)5)  /* poll_event timeout — out->type == OCT_EVENT_NONE */
#define OCT_STATUS_CANCELLED         ((oct_status_t)6)  /* session was cancelled; subsequent polls also return CANCELLED */
#define OCT_STATUS_INTERNAL          ((oct_status_t)7)  /* runtime invariant violated — diagnostic via oct_runtime_last_error */
#define OCT_STATUS_VERSION_MISMATCH  ((oct_status_t)8)  /* config.version unknown to this runtime build */

/* ------------------------------------------------------------------- *
 * v0.4 — Bounded error taxonomy                                       *
 * ------------------------------------------------------------------- *
 * Closed enum of error codes that travel ASYNCHRONOUSLY in            *
 * OCT_EVENT_ERROR.error_code (NEW field appended after slice-2A's     *
 * free-form `code` and `message` strings — see future v0.4 step that *
 * adds the operational envelope). Distinct from oct_status_t (sync   *
 * return codes); this is the bounded-cardinality form for telemetry  *
 * labels.                                                             *
 *                                                                    *
 * uint32_t typedef (NOT C enum — implementation-defined size).       *
 * Numeric assignments are STABLE FOREVER; mirrors                    *
 * octomil-contracts/fixtures/runtime_error_code/canonical_error_codes.json
 * Schema validation: octomil-contracts/schemas/core/runtime_error_code.json *
 * ------------------------------------------------------------------- */

typedef uint32_t oct_error_code_t;
#define OCT_ERR_OK                          ((oct_error_code_t)0)         /* sentinel */
#define OCT_ERR_MODEL_LOAD_FAILED           ((oct_error_code_t)1)
#define OCT_ERR_ARTIFACT_DIGEST_MISMATCH    ((oct_error_code_t)2)
#define OCT_ERR_ENGINE_INIT_FAILED          ((oct_error_code_t)3)
#define OCT_ERR_RAM_INSUFFICIENT            ((oct_error_code_t)4)
#define OCT_ERR_ACCELERATOR_UNAVAILABLE     ((oct_error_code_t)5)
#define OCT_ERR_INPUT_OUT_OF_RANGE          ((oct_error_code_t)6)
#define OCT_ERR_INPUT_FORMAT_UNSUPPORTED    ((oct_error_code_t)7)
#define OCT_ERR_TIMEOUT                     ((oct_error_code_t)8)
#define OCT_ERR_PREEMPTED                   ((oct_error_code_t)9)
#define OCT_ERR_QUOTA_EXCEEDED              ((oct_error_code_t)10)
#define OCT_ERR_INTERNAL                    ((oct_error_code_t)11)
#define OCT_ERR_UNKNOWN                     ((oct_error_code_t)0xFFFFFFFFu) /* forward-compat sentinel — UINT32_MAX */

/* ------------------------------------------------------------------- *
 * v0.4 — Canonical capability name constants                          *
 * ------------------------------------------------------------------- *
 * Mirror octomil-contracts/schemas/core/runtime_capability.json       *
 * exactly. SDK bindings reference these constants instead of the     *
 * literal strings to catch drift at build time.                       *
 * ------------------------------------------------------------------- */

#define OCT_CAPABILITY_AUDIO_REALTIME_SESSION   "audio.realtime.session"
#define OCT_CAPABILITY_AUDIO_STT_BATCH          "audio.stt.batch"
#define OCT_CAPABILITY_AUDIO_STT_STREAM         "audio.stt.stream"
#define OCT_CAPABILITY_AUDIO_TRANSCRIPTION      "audio.transcription"
#define OCT_CAPABILITY_AUDIO_TTS_BATCH          "audio.tts.batch"
#define OCT_CAPABILITY_AUDIO_TTS_STREAM         "audio.tts.stream"
#define OCT_CAPABILITY_CHAT_COMPLETION          "chat.completion"
#define OCT_CAPABILITY_CHAT_STREAM              "chat.stream"
/* v0.4 additions — strict-reject still applies; runtime advertises iff implemented. */
#define OCT_CAPABILITY_AUDIO_DIARIZATION        "audio.diarization"
#define OCT_CAPABILITY_AUDIO_SPEAKER_EMBEDDING  "audio.speaker.embedding"
#define OCT_CAPABILITY_AUDIO_VAD                "audio.vad"
#define OCT_CAPABILITY_EMBEDDINGS_IMAGE         "embeddings.image"
#define OCT_CAPABILITY_EMBEDDINGS_TEXT          "embeddings.text"
#define OCT_CAPABILITY_INDEX_VECTOR_QUERY       "index.vector.query"

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
    /*
     * Caller MUST set this to `sizeof(oct_capabilities_t)` before
     * calling `oct_runtime_capabilities`. The runtime writes only
     * up to `size` bytes; bindings compiled against an older version
     * see only the fields up to their version's struct size. New
     * fields are added at the END only — never in the middle.
     */
    size_t        size;
    /*
     * Null-terminated array of strings. Runtime-owned; valid until
     * `oct_runtime_capabilities_free(out)` OR `oct_runtime_close`.
     * EMPTY-LIST CONVENTION: an empty supported set is a non-NULL
     * pointer to an array of length 1 whose only element is NULL
     * (the sentinel). Bindings can iterate without a NULL check on
     * the outer pointer. The outer pointer is NULL only if the
     * runtime failed to populate the field; treat as
     * OCT_STATUS_INTERNAL.
     */
    const char**  supported_engines;
    const char**  supported_capabilities;
    const char**  supported_archs;          /* values: "darwin-arm64", "darwin-x86_64", "linux-amd64", "linux-arm64", "windows-amd64", "wasm32" */
    uint64_t      ram_total_bytes;
    uint64_t      ram_available_bytes;
    /*
     * Codex R1: prefer fixed-width integer fields over `bool` in
     * public ABI structs because bool has implementation-defined
     * size on some platforms. Use uint8_t (0 = false, non-zero =
     * true) for ABI-stable layout. The runtime asserts size at
     * impl time via static_assert.
     */
    uint8_t       has_apple_silicon;
    uint8_t       has_cuda;
    uint8_t       has_metal;
    uint8_t       _reserved0;               /* padding; always 0 */
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
 * SELF-REENTRANCY (callback calling back into runtime APIs):         *
 * forbidden in v1. The callback MUST NOT call oct_session_*,         *
 * oct_runtime_capabilities, or oct_runtime_close on a runtime/      *
 * session that's currently producing the event. Doing so deadlocks  *
 * (single-thread-affine session) or violates the no-recursion       *
 * invariant. Bindings that need to drive the runtime in response to *
 * telemetry signals MUST queue the work and execute it from a       *
 * different thread.                                                   *
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
    /*
     * STRING LIFETIME (Codex R1): all `const char*` fields in this
     * config are caller-owned. The runtime COPIES the strings during
     * `oct_runtime_open`; the caller is free to free / mutate the
     * underlying buffers immediately after `oct_runtime_open`
     * returns. This is the only safe rule for FFI bindings (Python
     * `str.encode("utf-8")` returns a temporary; Swift `String`
     * may move buffers between calls).
     */
    const char* artifact_root;                 /* where prepared artifacts live on disk; copied at open */
    oct_telemetry_sink_fn telemetry_sink;      /* optional; NULL = drop telemetry */
    void*       telemetry_user_data;           /* passed back to sink callback verbatim */
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

/*
 * Close the runtime. Behavior with live state:
 *   - Live sessions are CANCELLED and closed implicitly. Bindings
 *     should call oct_session_close on every open session BEFORE
 *     calling oct_runtime_close to ensure clean drain; the implicit
 *     close after runtime_close is best-effort.
 *   - v0.4: Live models (oct_model_t) are CLOSED implicitly under the
 *     SAME contract as sessions. Bindings should call oct_model_close
 *     on every open model BEFORE calling oct_runtime_close; the
 *     implicit close after runtime_close is best-effort. Engine
 *     adapters that hold expensive resources (mmap'd weights, KV
 *     buffers, accelerator contexts) MUST tolerate the implicit
 *     cleanup path — runtime_close is also the cleanup-of-last-resort
 *     for processes terminating uncleanly. (Codex R2 fix: previously
 *     this docstring only mentioned sessions; v0.4 model lifecycle
 *     extends the rule.)
 *   - Outstanding `oct_capabilities_t` allocations from
 *     `oct_runtime_capabilities` become INVALID. Bindings MUST call
 *     `oct_runtime_capabilities_free` on every retained
 *     `oct_capabilities_t` BEFORE oct_runtime_close.
 *   - The runtime drains pending telemetry sink callbacks before
 *     returning. If the callback is currently executing on another
 *     thread (telemetry-from-the-runtime-thread case), runtime_close
 *     blocks until it returns.
 *   - Calling oct_runtime_close twice on the same handle is
 *     undefined behavior. Bindings MUST set the handle to NULL after
 *     close (the helper macro OCT_CLOSE_RUNTIME below does this).
 *   - Safe to call from any thread.
 */
OCT_API void oct_runtime_close(oct_runtime_t* runtime);

/* Convenience macro: close + null in one shot. Idempotent. */
#define OCT_CLOSE_RUNTIME(rt_ptr)  do { \
    if ((rt_ptr) != NULL) { oct_runtime_close(rt_ptr); (rt_ptr) = NULL; } \
} while (0)

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

/*
 * Packed accessor for bindings that want a single comparable value.
 * Layout: (major << 32) | (minor << 16) | patch. Useful for "is the
 * dylib >= 0.2.0?" checks: `oct_runtime_abi_version_packed() >=
 * OCT_PACK_VERSION(0, 2, 0)`.
 */
#define OCT_PACK_VERSION(maj, min, pat) \
    (((uint64_t)(maj) << 32) | ((uint64_t)(min) << 16) | (uint64_t)(pat))

OCT_API uint64_t oct_runtime_abi_version_packed(void);

/*
 * ABI struct-layout introspection. Returns sizeof(struct) as
 * computed by the C compiler that built the dylib. Bindings call
 * these to verify their own (cffi cdef / Swift / JNI) struct
 * declarations don't drift from the header's definition. Codex R1
 * fix on the Python cffi binding: ABI mode does NOT catch
 * struct-layout mismatch at parse time; runtime crash on first
 * non-version field read is the only signal without these.
 */
OCT_API size_t oct_runtime_config_size(void);
OCT_API size_t oct_capabilities_size(void);

/*
 * Slice 2A: introspection for the session-level structs that bindings
 * marshal across the ABI. Same purpose as the runtime/capabilities
 * variants — pin struct-layout parity between the cdef / Swift / JNI
 * declaration and the C compiler's view, so cdef drift fails the
 * binding's regression suite immediately rather than producing a
 * runtime crash on the first non-version field touch.
 */
OCT_API size_t oct_session_config_size(void);
OCT_API size_t oct_audio_view_size(void);
OCT_API size_t oct_event_size(void);

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

/* Priority — mirrors octomil-python's TtsRequestPriority. Codex R2
 * fix: typedef'd to `uint32_t` rather than a C enum because C enum
 * size is implementation-defined. Fixed-width integer + named
 * constants is the FFI-portable shape. Bindings that want switch
 * exhaustiveness can build their own enum on top (Swift `enum`,
 * Rust enum) — they just don't cross the ABI as enums. */
typedef uint32_t oct_priority_t;
#define OCT_PRIORITY_SPECULATIVE  ((oct_priority_t)0)
#define OCT_PRIORITY_PREFETCH     ((oct_priority_t)1)
#define OCT_PRIORITY_FOREGROUND   ((oct_priority_t)2)

typedef struct {
    uint32_t version;                /* OCT_SESSION_CONFIG_VERSION */
    /*
     * STRING LIFETIME: same rule as oct_runtime_config_t — all
     * `const char*` fields are caller-owned and the runtime COPIES
     * them during `oct_session_open`. Caller is free to free /
     * mutate buffers immediately after open returns.
     */
    const char* model_uri;           /* "@app/eternum/realtime" | "kokoro-82m" | ...; copied at open */
    /*
     * Canonical capability identifier — MUST be one of the strings
     * defined by `octomil-contracts/schemas/core/runtime_capability.json`
     * (mirrored in Python as `octomil.runtime.native.capabilities.RUNTIME_CAPABILITIES`):
     *
     *   "audio.realtime.session"
     *   "audio.tts.stream" | "audio.tts.batch"
     *   "audio.stt.stream" | "audio.stt.batch"
     *   "audio.transcription"
     *   "chat.completion" | "chat.stream"
     *   v0.4 additions:
     *   "audio.diarization" | "audio.speaker.embedding" | "audio.vad"
     *   "embeddings.text" | "embeddings.image"
     *   "index.vector.query"
     *
     * The runtime applies the strict-reject rule on this field: any
     * value not in the canonical enum returns OCT_STATUS_UNSUPPORTED.
     * v0.4 (octomil-contracts#99): `embeddings.text` IS a canonical
     * capability — Cactus-parity + native ONNX embed is 2-5× faster
     * than Python sentence-transformers. The v0.3 "intentionally
     * absent" rule was reversed in the v0.4 PE review consensus
     * (octomil-workspace#27). Strict-reject still applies on requests;
     * runtime advertises iff the engine adapter ships. Copied at open.
     */
    const char* capability;
    /*
     * Locality the control plane resolved to. INFORMATIONAL ONLY.
     * The runtime acts on "on_device" only; any other value returns
     * OCT_STATUS_UNSUPPORTED. The runtime MUST NOT initiate cloud
     * fallback or make a network call from this field. Cloud routing
     * is the SDK / control plane's responsibility above the ABI.
     */
    const char* locality;            /* "on_device" only — anything else => UNSUPPORTED; copied at open */
    /*
     * Policy preset the control plane resolved. INFORMATIONAL ONLY.
     * The runtime carries this in events for observability and
     * audit correlation; it does NOT enforce policy (no cloud-
     * fallback decisions, no quota checks, no auth). Policy
     * enforcement is Layer 2b's job.
     */
    const char* policy_preset;       /* "private" | "local_first" | ...; copied at open */
    const char* speaker_id;          /* optional; NULL ok; copied at open */
    uint32_t    sample_rate_in;      /* 0 = engine preferred input rate */
    uint32_t    sample_rate_out;     /* 0 = engine preferred output rate */
    oct_priority_t priority;         /* OCT_PRIORITY_*; codegen-friendlier than bare uint32_t */
    void*       user_data;           /* opaque, echoed verbatim on every event */

    /* ──────── v0.4 step 2 — session_config v=2 appended fields ────
     * Correlation IDs set by Layer 2b at oct_session_open; runtime
     * echoes on every event from this session. Caller-owned strings
     * copied at open per the slice-2A STRING LIFETIME contract.
     * NULL = "no correlation for this slot"; runtime echoes empty
     * string ("") on events.
     *
     * v=1 bindings stop at user_data; runtime treats their config
     * as if NULL was passed for these fields. v=2 bindings populate
     * them.
     *
     * LENGTH LIMITS (PE review octomil-workspace#27 §1.9.5):
     *   - kv_prefix_key  ≤ 256 B UTF-8
     *   - request_id     ≤ 128 B
     *   - route_id       ≤ 128 B
     *   - trace_id       ≤ 128 B
     * ASCII-printable characters only, no whitespace, no control
     * chars. Out-of-bounds returns OCT_STATUS_INVALID_INPUT.
     * ─────────────────────────────────────────────────────────── */
    const char* request_id;          /* per-session correlation; NULL ok */
    const char* route_id;            /* set by Layer 2b at session_open; NULL ok */
    const char* trace_id;            /* W3C-compatible if present; NULL ok */
    const char* kv_prefix_key;       /* KV-prefix cache key (system prompt + tool schemas); NULL = no prefix-cache lookup */
} oct_session_config_t;
/*
 * RESERVED FIELDS POLICY (Codex R3): all `_reserved*` fields in
 * public structs MUST be zero-initialized by the caller. The
 * runtime returns OCT_STATUS_INVALID_INPUT if any `_reserved*`
 * field is non-zero on input. This guards future extension bits
 * from being silently ignored by older runtimes.
 */

/*
 * Open returns oct_status_t; same edge-case contract as
 * oct_runtime_open (Codex R1):
 *   - If `out == NULL`, returns OCT_STATUS_INVALID_INPUT.
 *   - On any non-OK return, `*out` is set to NULL.
 *   - Failure modes: OCT_STATUS_VERSION_MISMATCH (config.version
 *     unknown), OCT_STATUS_UNSUPPORTED (capability/locality/engine
 *     not loadable), OCT_STATUS_NOT_FOUND (model_uri or artifact
 *     missing on disk), OCT_STATUS_INVALID_INPUT (bad config),
 *     OCT_STATUS_INTERNAL (runtime invariant violated; check
 *     oct_runtime_last_error or oct_last_thread_error).
 *   - Configuration strings are copied during open; caller may free
 *     them immediately after this call returns.
 */
OCT_API oct_status_t oct_session_open(
    oct_runtime_t* runtime,
    const oct_session_config_t* config,
    oct_session_t** out
);

/*
 * Closes the session, draining any queued events. Safe to call from
 * any thread, but MUST NOT race oct_session_send_audio /
 * oct_session_send_text / oct_session_poll_event on the same
 * session — those are single-thread-affine. The closing thread is
 * the affine thread; coordinate with the producer.
 */
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

/*
 * Caller-owned view over an audio buffer. Codex R2 fix: tightened
 * field semantics to remove ambiguity that would let runtime and
 * binding disagree about buffer size and read past the end.
 *
 * LAYOUT:
 *   - samples: pointer to a contiguous float32 buffer; range
 *     [-1.0, 1.0] (clamped at the runtime; out-of-range values are
 *     not undefined behavior, but quality degrades).
 *   - n_frames: number of FRAMES (NOT total float elements). One
 *     frame == one sample per channel. Total float element count
 *     is `n_frames * channels`. Earlier draft used `n_samples`
 *     which was ambiguous (frames vs total elements); renamed.
 *   - sample_rate: Hz. 0 is INVALID; runtime returns
 *     OCT_STATUS_INVALID_INPUT.
 *   - channels: number of channels. 1 = mono (canonical for
 *     speech), 2 = stereo. Multichannel buffers MUST be
 *     INTERLEAVED (LRLRLR…), never planar. The runtime does NOT
 *     accept planar input; bindings must interleave first.
 *
 * Endian: float32 native to the host architecture. Cross-platform
 * audio bindings (iOS AVAudioEngine, Android AudioRecord, ALSA
 * snd_pcm_format_t) use native float on every supported target.
 */
typedef struct {
    const float* samples;            /* borrowed; lifetime = call duration only */
    uint32_t     n_frames;           /* frames per channel; total float count = n_frames * channels */
    uint32_t     sample_rate;        /* Hz; 0 => OCT_STATUS_INVALID_INPUT */
    uint16_t     channels;           /* 1 = mono (canonical), 2 = stereo interleaved */
    uint16_t     _reserved0;         /* padding; always 0 */
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

/*
 * Event type discriminator. Codex R2 fix: typedef'd to `uint32_t`
 * rather than a C enum for the same FFI-portability reason as
 * oct_status_t and oct_priority_t. New types append; closed-list
 * forward-compat per `schema_version` bump.
 */
typedef uint32_t oct_event_type_t;
#define OCT_EVENT_NONE                  ((oct_event_type_t)0)  /* timeout placeholder; out->type set to this when poll_event returns OCT_STATUS_TIMEOUT */
#define OCT_EVENT_SESSION_STARTED       ((oct_event_type_t)1)
#define OCT_EVENT_AUDIO_CHUNK           ((oct_event_type_t)2)
#define OCT_EVENT_TRANSCRIPT_CHUNK      ((oct_event_type_t)3)
#define OCT_EVENT_USER_SPEECH_DETECTED  ((oct_event_type_t)4)
#define OCT_EVENT_TURN_ENDED            ((oct_event_type_t)5)
#define OCT_EVENT_CAPABILITY_VERIFIED   ((oct_event_type_t)6)
#define OCT_EVENT_ERROR                 ((oct_event_type_t)7)
#define OCT_EVENT_SESSION_COMPLETED     ((oct_event_type_t)8)
#define OCT_EVENT_INPUT_DROPPED         ((oct_event_type_t)9)  /* realtime backpressure; see strategy/realtime-architecture.md */

/* v0.4 step 2 — runtime-scope events (delivered via the
 * oct_telemetry_sink_fn callback, NOT oct_session_poll_event).
 * Empty correlation envelope strings — these are TRULY runtime-
 * scoped, no session in scope. Bindings forward to traces/audit
 * log per the slice-2A telemetry-sink reentrancy rules. */
#define OCT_EVENT_MODEL_LOADED          ((oct_event_type_t)10)  /* engine + model_id + artifact_digest + load_ms + warm_ms + policy_preset + user_data + source */
#define OCT_EVENT_MODEL_EVICTED         ((oct_event_type_t)11)  /* engine + model_id + artifact_digest + freed_bytes + reason ∈ {memory_pressure, ttl, manual} + user_data */
#define OCT_EVENT_CACHE_HIT             ((oct_event_type_t)12)  /* layer ∈ {kv-prefix, phoneme, voice, phrase, route} + saved_tokens */
#define OCT_EVENT_CACHE_MISS            ((oct_event_type_t)13)  /* same payload as CACHE_HIT (saved_tokens=0); separate type for filterability */
#define OCT_EVENT_QUEUED                ((oct_event_type_t)14)  /* queue_position + queue_depth */
#define OCT_EVENT_PREEMPTED             ((oct_event_type_t)15)  /* preempted_by_priority + reason ∈ closed enum */
#define OCT_EVENT_MEMORY_PRESSURE       ((oct_event_type_t)16)  /* ram_available_bytes + severity ∈ {0=warn, 1=critical} */
#define OCT_EVENT_THERMAL_STATE         ((oct_event_type_t)17)  /* state ∈ {0=nominal, 1=fair, 2=serious, 3=critical} */
#define OCT_EVENT_WATCHDOG_TIMEOUT      ((oct_event_type_t)18)  /* timeout_ms + phase ∈ closed enum */
#define OCT_EVENT_METRIC                ((oct_event_type_t)19)  /* name from runtime_metric.json closed enum + value (double). Free-form names forbidden by-construction. */

/*
 * Sample format codes for audio_chunk payload (Codex R1 — pcm +
 * n_bytes + sample_rate is insufficient; bindings need format /
 * channels / interleaving / endian / range to decode). Values
 * are stable across ABI versions; new formats append.
 */
#define OCT_SAMPLE_FORMAT_PCM_S16LE  1   /* int16 little-endian, range [-32768, 32767], interleaved if channels > 1 */
#define OCT_SAMPLE_FORMAT_PCM_F32LE  2   /* float32 little-endian, range [-1.0, 1.0], interleaved if channels > 1 */

struct oct_event {
    uint32_t           version;       /* OCT_EVENT_VERSION */
    /*
     * Caller MUST set `size = sizeof(oct_event_t)` before passing
     * `out` to oct_session_poll_event. The runtime writes only up to
     * `size` bytes; bindings compiled against an older version see
     * fields up to their version's struct size. New event types and
     * new fields append at the end; never insert in the middle.
     */
    size_t             size;
    oct_event_type_t   type;
    uint64_t           monotonic_ns;
    void*              user_data;     /* echoed VERBATIM from the oct_session_config_t.user_data of the session that produced this event */
    union {
        struct {
            const uint8_t* pcm;             /* runtime-owned; lifetime per the rule above */
            uint32_t       n_bytes;
            uint32_t       sample_rate;     /* Hz */
            uint32_t       sample_format;   /* OCT_SAMPLE_FORMAT_PCM_S16LE or OCT_SAMPLE_FORMAT_PCM_F32LE */
            uint16_t       channels;        /* 1 = mono (canonical), 2 = stereo interleaved */
            uint8_t        is_final;        /* uint8 instead of bool for ABI portability */
            uint8_t        _reserved0;      /* padding; always 0 */
        } audio_chunk;
        struct {
            const char* utf8;          /* runtime-owned */
            uint32_t    n_bytes;
        } transcript_chunk;
        struct {
            const char* code;          /* runtime-owned (slice-2A free-form string; kept for human context) */
            const char* message;       /* runtime-owned */
            /* v0.4 step 2 APPENDED — bounded enum form for telemetry
             * labels. Drawn from runtime_error_code.json. v0.3/0.4-step-1
             * bindings allocating the smaller struct never read this
             * field; runtime respects out->size. Unknown values map to
             * OCT_ERR_UNKNOWN at receive time per the forward-compat
             * sentinel rule. */
            oct_error_code_t error_code;
            uint32_t         _reserved0;     /* padding; always 0 */
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
            uint8_t      capability_verified;     /* uint8 instead of bool for ABI portability */
            uint8_t      _reserved0;              /* padding; always 0 */
            uint16_t     _reserved1;              /* padding; always 0 */
            oct_status_t terminal_status;
        } session_completed;
        /* INPUT_DROPPED — realtime backpressure / queue overflow */
        struct {
            uint32_t     n_frames_dropped;  /* frames per channel (NOT total float elements); mirrors oct_audio_view_t.n_frames */
            uint32_t     sample_rate;
            uint16_t     channels;          /* channels of the dropped audio */
            uint16_t     _reserved0;        /* padding; always 0 */
            const char*  reason;            /* runtime-owned. "queue_full" | "session_busy" | "engine_lagging" */
            uint64_t     dropped_at_ns;     /* monotonic timestamp of the drop */
        } input_dropped;

        /* ──────── v0.4 step 2 — runtime-scope event payloads ──────── *
         * All inner pointer fields are runtime-owned static strings
         * drawn from closed enums declared in this header / contracts.
         * Lifetime: callback duration only (slice-2A telemetry-sink
         * reentrancy rules). Bindings copy if they need durability.
         * ─────────────────────────────────────────────────────────── */
        struct {
            const char* engine;             /* runtime-owned: "moshi-mlx@<ver>" | "llama.cpp@<ver>" | ... */
            const char* model_id;           /* runtime-owned */
            const char* artifact_digest;    /* sha256 of prepared artifact */
            uint64_t    load_ms;
            uint64_t    warm_ms;
            const char* policy_preset;      /* echoed from oct_model_config_t.policy_preset */
            void*       config_user_data;   /* echoed from oct_model_config_t.user_data */
            const char* source;             /* closed enum: "bench-cache-recommended" | "engine-hint" | "auto" */
        } model_loaded;
        struct {
            const char* engine;
            const char* model_id;
            const char* artifact_digest;
            uint64_t    freed_bytes;
            const char* reason;             /* closed enum: "memory_pressure" | "ttl" | "manual" */
            void*       config_user_data;
        } model_evicted;
        struct {
            const char* layer;              /* closed enum: "kv-prefix" | "phoneme" | "voice" | "phrase" | "route" */
            uint32_t    saved_tokens;       /* 0 for CACHE_MISS */
            uint32_t    _reserved0;
        } cache;
        struct {
            uint32_t    queue_position;
            uint32_t    queue_depth;
        } queued;
        struct {
            uint32_t    preempted_by_priority;     /* OCT_PRIORITY_* */
            uint32_t    _reserved0;
            const char* reason;             /* runtime-owned closed enum */
        } preempted;
        struct {
            uint64_t    ram_available_bytes;
            uint8_t     severity;           /* 0 = warn, 1 = critical */
            uint8_t     _reserved0;
            uint16_t    _reserved1;
            uint32_t    _reserved2;
        } memory_pressure;
        struct {
            uint8_t     state;              /* 0=nominal, 1=fair, 2=serious, 3=critical */
            uint8_t     _reserved0;
            uint16_t    _reserved1;
            uint32_t    _reserved2;
        } thermal_state;
        struct {
            uint32_t    timeout_ms;
            uint32_t    _reserved0;
            const char* phase;              /* runtime-owned closed enum: "load" | "warm" | "first_audio" | "session_step" */
        } watchdog_timeout;
        struct {
            const char* name;               /* runtime-owned; MUST be a value from runtime_metric.json closed enum */
            double      value;
        } metric;
    } data;

    /* ──────── v0.4 step 2 — operational envelope (APPENDED) ─────────
     * Per ABI v0.4 PE review (octomil-workspace#27 §1.6 + R1 fix):
     * envelope APPENDS after the union, NOT before, so v0.3 / v0.4
     * step 1 bindings reading event payloads at the same offsets
     * stay compatible. The runtime writes envelope fields ONLY when
     * out->size is large enough to cover them (versioned-output
     * size handshake from slice-2A).
     *
     * Runtime-owned strings, lifetime = until next poll (for
     * session-scope events) or callback duration (for runtime-scope
     * events). The runtime ALWAYS writes non-NULL pointers; when
     * `oct_session_config_t` passed NULL for a correlation slot
     * (or the event is runtime-scope, no session in scope), the
     * runtime echoes an EMPTY STRING ("") rather than NULL. Bindings
     * can `strlen()` safely without a NULL check.
     *
     * The runtime NEVER mints correlation IDs — Layer 2b sets them
     * at session_open and the runtime echoes. Empty strings on
     * runtime-scope events are TRULY runtime-scoped (no session).
     * ─────────────────────────────────────────────────────────── */
    const char*        request_id;          /* per-session correlation; "" for runtime-scope */
    const char*        route_id;            /* set by Layer 2b at session_open; "" for runtime-scope */
    const char*        trace_id;            /* W3C-compatible if present; "" for runtime-scope */
    const char*        engine_version;      /* e.g. "moshi-mlx@0.2.6"; "" if not engine-attributable */
    const char*        adapter_version;     /* runtime adapter SHA; "" if not adapter-attributable */
    const char*        accelerator;         /* "metal" | "cuda" | "cpu" | "ane" | "" */
    const char*        artifact_digest;     /* sha256 of model artifact; "" if not artifact-attributable */
    uint8_t            cache_was_hit;       /* 0/1 — was this fed by cache? */
    uint8_t            _reserved0;
    uint16_t           _reserved1;
    uint32_t           _reserved2;
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
 *
 * Return values (Codex R1 — document the space explicitly):
 *   - OCT_STATUS_OK: cancellation requested; the session will
 *     transition to cancelled state at the next chunk boundary.
 *   - OCT_STATUS_INVALID_INPUT: session == NULL.
 *   - OCT_STATUS_CANCELLED: session was already cancelled. Idempotent
 *     to call cancel multiple times; the second+ calls return this
 *     code rather than OCT_STATUS_OK.
 */
OCT_API oct_status_t oct_session_cancel(oct_session_t* session);

/*
 * send_audio semantics (Codex R1 — atomic per-call):
 *   - Either the entire `audio_view_t` is consumed (returns
 *     OCT_STATUS_OK), or none of it is (returns OCT_STATUS_BUSY).
 *     The runtime never partial-consumes a buffer. Bindings facing
 *     OCT_STATUS_BUSY can either drop the chunk OR retry after
 *     yielding (drop is the iOS / Android default for realtime
 *     because the audio I/O thread cannot block; retry is the
 *     Python default).
 *   - Returns OCT_STATUS_INVALID_INPUT if session or audio is NULL,
 *     or if audio->channels == 0, n_frames == 0, sample_rate == 0,
 *     or samples == NULL.
 *   - Returns OCT_STATUS_CANCELLED if the session has been cancelled.
 *
 * Documented here rather than at the function comment to centralize
 * the contract.
 */

/* ------------------------------------------------------------------- *
 * v0.4 — Model lifecycle                                              *
 * ------------------------------------------------------------------- *
 * Per the ABI v0.4 PE review consensus (octomil-workspace#27 §1.1):  *
 * `oct_model_t` is the warm-handle abstraction the runtime needs for *
 * pool-keyed caching, eviction, and signed-manifest identity. A      *
 * session may open against a model_uri (slice-2A behavior preserved) *
 * or against a pre-warmed `oct_model_t*` (v0.4, set via              *
 * `oct_session_config_t.model` — APPENDED in a future v0.4 step).    *
 *                                                                    *
 * v0.4 step 1: stubs only. Every entry returns OCT_STATUS_UNSUPPORTED *
 * with a descriptive last_error. The Slice 2C Moshi adapter and      *
 * future engine adapters fill these in.                               *
 * ------------------------------------------------------------------- */

typedef struct {
    uint32_t    version;                   /* OCT_MODEL_CONFIG_VERSION */
    /* LOCAL URIs ONLY — Layer 2a does NOT resolve `@app/...` refs.
     * By the time the runtime sees a URI, Layer 2b has resolved it
     * to a local path / digest. */
    const char* model_uri;                 /* "kyutai/moshiko-mlx-q4@<digest>" | "local:///abs/path/..." */
    const char* artifact_digest;           /* sha256 of the prepared artifact;
                                              runtime rejects on mismatch */
    const char* engine_hint;               /* optional: "mlx" | "llama.cpp" |
                                              "sherpa-onnx" | "" (auto) */
    const char* policy_preset;             /* informational; carried on events */
    uint32_t    accelerator_pref;          /* OCT_ACCEL_* */
    uint64_t    ram_budget_bytes;          /* 0 = no cap */
    void*       user_data;                 /* echoed verbatim on model events */
} oct_model_config_t;

/* Accelerator preference — fixed-width int + named constants per the
 * slice-2A typed-int FFI portability rule. */
typedef uint32_t oct_accelerator_pref_t;
#define OCT_ACCEL_AUTO    ((oct_accelerator_pref_t)0)
#define OCT_ACCEL_METAL   ((oct_accelerator_pref_t)1)
#define OCT_ACCEL_CUDA    ((oct_accelerator_pref_t)2)
#define OCT_ACCEL_CPU     ((oct_accelerator_pref_t)3)
#define OCT_ACCEL_ANE     ((oct_accelerator_pref_t)4)

/*
 * Open a warm model handle.
 *   - If `out == NULL`: OCT_STATUS_INVALID_INPUT.
 *   - On any non-OK return, *out is set to NULL.
 *   - v0.4 step 1: stub returns OCT_STATUS_UNSUPPORTED.
 *   - Real implementations: validate config.version; mmap or load
 *     weights; verify artifact_digest; call engine adapter init;
 *     emit OCT_EVENT_MODEL_LOADED via telemetry sink (future v0.4).
 *
 * Strings are caller-owned and copied at open per the slice-2A
 * STRING LIFETIME contract.
 */
OCT_API oct_status_t oct_model_open(
    oct_runtime_t* runtime,
    const oct_model_config_t* config,
    oct_model_t** out_model
);

/*
 * Idempotent. Loads weights into memory, builds KV/cache scaffolding,
 * runs one-token / silence-frame warmup. Optional but how the runtime
 * hits warm-open SLOs without paying cold-open every time. Stub
 * returns UNSUPPORTED.
 */
OCT_API oct_status_t oct_model_warm(oct_model_t* model);

/*
 * Eviction — explicit. Returns OCT_STATUS_BUSY if any session still
 * references the model (advisory; scheduler honors at next idle).
 * Bindings that need synchronous eviction call `oct_session_close`
 * on each session first. Stub returns UNSUPPORTED.
 */
OCT_API oct_status_t oct_model_evict(oct_model_t* model);

/*
 * Close the model handle. void return — slice-2A close-style.
 * Cancellation cascades to live sessions via the v0.4 step where
 * `oct_session_config_t.model` is appended; for v0.4 step 1 (no
 * session→model wiring yet) close is a simple delete.
 *
 * Post-close use of `oct_model_t*` is UB at the C ABI per the
 * slice-2A close-style precedent. Bindings track validity via a
 * WeakSet + invalidate-flag pattern (see octomil-python's
 * NativeModel wrapper).
 */
OCT_API void oct_model_close(oct_model_t* model);

/* Sizeof introspection — same role as oct_runtime_config_size /
 * oct_session_config_size. Bindings call this to verify cffi cdef /
 * Swift / JNI struct declarations don't drift from runtime.h. */
OCT_API size_t oct_model_config_size(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* OCTOMIL_RUNTIME_H */
