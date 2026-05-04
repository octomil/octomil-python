/*
 * runtime.cpp — stub implementation of include/octomil/runtime.h
 *
 * Per the slice-2 build-system PR scope: every OCT_API entry point is
 * implemented as a stub that returns OCT_STATUS_UNSUPPORTED with a
 * descriptive last-error string. The stubs let downstream bindings
 * (Python cffi in slice 3, Swift in slice 4) start integrating
 * against a real liboctomil-runtime binary while the actual session
 * adapter is being filled in.
 *
 * The ONLY behaviors implemented for real here:
 *   * oct_runtime_abi_version_{major,minor,patch,packed}() — return
 *     the constants from the header so bindings can verify they're
 *     linked against a compatible build.
 *   * oct_runtime_open() — allocates a tiny opaque struct so the
 *     handle is non-NULL, but rejects any non-zero version field
 *     and any non-default config to make the stub status visible.
 *   * oct_runtime_close() — frees the allocation. Idempotent
 *     against NULL.
 *   * oct_runtime_last_error / oct_last_thread_error — read back
 *     the last status string set by the most recent stub call.
 *
 * Everything else returns OCT_STATUS_UNSUPPORTED. This is intentional
 * — it lets the smoke test verify the symbols are exported and the
 * version handshake works without needing to drag in MLX, Moshi, or
 * any of the engine-side code. Slice-2 implementation replaces these
 * stubs file-by-file.
 */

#include "octomil/runtime.h"

#include "adapters/adapter_base.h"
#if defined(OCT_ENABLE_ENGINE_MOSHI_RS)
#include "adapters/moshi_rs/moshi_rs_adapter.h"
#endif

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

/* Per-runtime + per-thread last-error storage. Bindings can call
 * oct_runtime_last_error after a runtime_t handle is obtained, or
 * oct_last_thread_error before runtime_open succeeds. */
std::mutex& thread_error_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::thread::id, std::string>& thread_error_map() {
    static std::unordered_map<std::thread::id, std::string> m;
    return m;
}

void set_thread_error(const char* msg) {
    std::lock_guard<std::mutex> lock(thread_error_mutex());
    thread_error_map()[std::this_thread::get_id()] = msg ? msg : "";
}

int copy_error(const std::string& src, char* buf, size_t buflen) {
    if (buf == nullptr || buflen == 0) {
        return -1;
    }
    const size_t n = src.size() < (buflen - 1) ? src.size() : (buflen - 1);
    std::memcpy(buf, src.data(), n);
    buf[n] = '\0';
    return static_cast<int>(n);
}

}  // namespace

/* Opaque handle types. Bindings only see pointers to these. */
struct oct_runtime {
    uint32_t version;        /* echo of config.version */
    std::string last_error;  /* human-readable diag for last failed call */
    std::mutex error_mutex;  /* guards last_error */

    /* Capability snapshot, populated at runtime-open from the
     * adapter registry. The C ABI vends `const char**` lists; we
     * keep the strings owned by the runtime and a parallel pointer
     * array whose elements are stable for the runtime's lifetime. */
    std::vector<std::string> capability_strings;
    std::vector<std::string> engine_strings;
    std::vector<const char*> capability_ptrs;  /* NULL-terminated */
    std::vector<const char*> engine_ptrs;      /* NULL-terminated */

    void set_error(const std::string& msg) {
        std::lock_guard<std::mutex> lock(error_mutex);
        last_error = msg;
    }
};

struct oct_session {
    /* Slice-2 stub: never actually constructed. Reserved for the
     * real implementation. */
    int reserved;
};

extern "C" {

/* -------------------------------------------------------------------------
 * Version inspection — implemented for real so bindings can check
 * that a loaded dylib is a compatible build.
 * ------------------------------------------------------------------------- */

OCT_API uint32_t oct_runtime_abi_version_major(void) {
    return OCT_RUNTIME_ABI_VERSION_MAJOR;
}

OCT_API uint32_t oct_runtime_abi_version_minor(void) {
    return OCT_RUNTIME_ABI_VERSION_MINOR;
}

OCT_API uint32_t oct_runtime_abi_version_patch(void) {
    /* Header doesn't yet define a patch macro; v0.1.0 = 0. Bumped
     * when a patch-level header change ships. */
    return 0u;
}

OCT_API uint64_t oct_runtime_abi_version_packed(void) {
    return (static_cast<uint64_t>(oct_runtime_abi_version_major()) << 32) |
           (static_cast<uint64_t>(oct_runtime_abi_version_minor()) << 16) |
           (static_cast<uint64_t>(oct_runtime_abi_version_patch()));
}

/* ABI struct-layout introspection — returns sizeof as computed by
 * the C compiler that built this TU. Bindings call these to verify
 * their own (cffi cdef / Swift / JNI) struct declarations don't
 * drift from the header. */
OCT_API size_t oct_runtime_config_size(void) {
    return sizeof(oct_runtime_config_t);
}

OCT_API size_t oct_capabilities_size(void) {
    return sizeof(oct_capabilities_t);
}

/* Slice 2A — session-level struct-layout introspection. */
OCT_API size_t oct_session_config_size(void) {
    return sizeof(oct_session_config_t);
}

OCT_API size_t oct_audio_view_size(void) {
    return sizeof(oct_audio_view_t);
}

OCT_API size_t oct_event_size(void) {
    return sizeof(oct_event_t);
}

/* -------------------------------------------------------------------------
 * Runtime lifecycle
 * ------------------------------------------------------------------------- */

OCT_API oct_status_t oct_runtime_open(
    const oct_runtime_config_t* config,
    oct_runtime_t** out
) {
    if (out == nullptr) {
        set_thread_error("oct_runtime_open: out parameter is NULL");
        return OCT_STATUS_INVALID_INPUT;
    }
    *out = nullptr;
    if (config == nullptr) {
        set_thread_error("oct_runtime_open: config is NULL");
        return OCT_STATUS_INVALID_INPUT;
    }
    if (config->version != 1u) {
        set_thread_error("oct_runtime_open: config.version must be 1");
        return OCT_STATUS_VERSION_MISMATCH;
    }

    auto* rt = new (std::nothrow) oct_runtime{};
    if (rt == nullptr) {
        set_thread_error("oct_runtime_open: allocation failure");
        return OCT_STATUS_INTERNAL;
    }
    rt->version = config->version;
    rt->last_error.clear();

    /* Adapter registration. Each adapter constructor performs its
     * own platform + artifact gates and contributes a capability
     * IFF it can actually serve a session right now. The registry
     * filters by `is_loadable_now()` — see adapters/adapter_base.cpp.
     *
     * Slice 2C: the Moshi-rs adapter's `is_loadable_now()` returns
     * false in scaffolding state, so `audio.realtime.session` does
     * NOT appear in `oct_runtime_capabilities` even though the
     * adapter object code is linked. The follow-up commit flips the
     * flag once the Rust shim wires the real moshi-core streaming
     * pipeline. */
#if defined(OCT_ENABLE_ENGINE_MOSHI_RS)
    octomil::adapters::moshi_rs::register_with_runtime();
#endif

    /* Snapshot the registry into the runtime. Per-runtime ownership
     * of the strings means a future per-runtime capability override
     * (e.g., disable an engine for a specific runtime) is local to
     * this struct, not a global mutation. */
    rt->capability_strings = octomil::adapters::CapabilityRegistry::instance().snapshot_capabilities();
    rt->engine_strings = octomil::adapters::CapabilityRegistry::instance().snapshot_engines();
    rt->capability_ptrs.clear();
    for (const auto& s : rt->capability_strings) {
        rt->capability_ptrs.push_back(s.c_str());
    }
    rt->capability_ptrs.push_back(nullptr);
    rt->engine_ptrs.clear();
    for (const auto& s : rt->engine_strings) {
        rt->engine_ptrs.push_back(s.c_str());
    }
    rt->engine_ptrs.push_back(nullptr);

    *out = rt;
    return OCT_STATUS_OK;
}

OCT_API void oct_runtime_close(oct_runtime_t* runtime) {
    if (runtime == nullptr) {
        return;
    }
    delete runtime;
}

/* Sentinel "empty list" — non-NULL pointer to a single-element array
 * whose element is NULL. Per the header's empty-list convention so
 * bindings can iterate without an outer NULL check. The struct field
 * is `const char**` so the inner element type is `const char*`; we
 * declare the storage with that exact type. Lifetime is the dylib's;
 * `oct_runtime_capabilities_free` does not free it. */
static const char* k_empty_string_array[1] = {nullptr};

OCT_API oct_status_t oct_runtime_capabilities(
    oct_runtime_t* runtime,
    oct_capabilities_t* out
) {
    if (runtime == nullptr || out == nullptr) {
        if (runtime != nullptr) {
            runtime->set_error("oct_runtime_capabilities: out is NULL");
        } else {
            set_thread_error("oct_runtime_capabilities: runtime is NULL");
        }
        return OCT_STATUS_INVALID_INPUT;
    }
    /* Honor the versioned-output-struct contract:
     *   - Caller sets out->size = sizeof(oct_capabilities_t) before
     *     the call. We refuse anything smaller than the version+size
     *     header (the binding compiled against a pre-v1 header).
     *   - We write at most out->size bytes (no overrun on older
     *     bindings).
     * Codex R1 fix: previous stub did memset(out, 0, sizeof(*out))
     * which violated both invariants. */
    const size_t header_min = offsetof(oct_capabilities_t, supported_engines);
    if (out->size < header_min) {
        runtime->set_error(
            "oct_runtime_capabilities: out->size smaller than minimum required header"
        );
        return OCT_STATUS_INVALID_INPUT;
    }
    /* Snapshot the caller's size BEFORE we overwrite it. Codex R3 fix:
     * a smaller binding passes out->size = (its sizeof); we must
     * preserve that so capabilities_free knows how many bytes to
     * touch. Otherwise the staged struct's size = sizeof(*staged)
     * would clobber the caller's smaller value, and a subsequent
     * capabilities_free could memset past the caller's allocation. */
    const size_t caller_size = out->size;

    /* Stage to a local fully-populated struct, then copy caller_size
     * bytes — that way bindings compiled against v1 see all fields,
     * and a hypothetical pre-v1 binding sees only its slice. The
     * staged size matches the CALLER's view, not ours, so the free
     * path knows where to stop. */
    oct_capabilities_t staged{};
    staged.version = OCT_CAPABILITIES_VERSION;
    staged.size = caller_size;
    /* Empty-list convention: non-NULL pointer to a single-element NULL-
     * terminated array. Bindings iterate without an outer NULL check.
     * Slice 2C: we vend the per-runtime snapshot taken at open
     * time. The pointer array is owned by `oct_runtime`; lifetime
     * is the runtime's. Falls back to `k_empty_string_array` only
     * when the snapshot is empty (e.g., no adapters registered). */
    staged.supported_engines = runtime->engine_ptrs.empty()
        ? k_empty_string_array
        : runtime->engine_ptrs.data();
    staged.supported_capabilities = runtime->capability_ptrs.empty()
        ? k_empty_string_array
        : runtime->capability_ptrs.data();
    staged.supported_archs = k_empty_string_array;
    staged.ram_total_bytes = 0;
    staged.ram_available_bytes = 0;
    staged.has_apple_silicon = 0;
    staged.has_cuda = 0;
    staged.has_metal = 0;
    staged._reserved0 = 0;

    const size_t copy_n = caller_size < sizeof(staged) ? caller_size : sizeof(staged);
    std::memcpy(out, &staged, copy_n);
    return OCT_STATUS_OK;
}

OCT_API void oct_runtime_capabilities_free(oct_capabilities_t* caps) {
    if (caps == nullptr) {
        return;
    }
    /* Slice-2 stub: the const char** fields point at the static
     * k_empty_string_array (lifetime is the dylib's, never freed).
     * We zero only up to caps->size bytes so a binding that
     * allocated a smaller struct doesn't get its tail clobbered.
     * Real implementation will additionally free runtime-allocated
     * string lists. Codex R2 nit. */
    const size_t buf_size = caps->size;
    if (buf_size > 0 && buf_size <= sizeof(*caps)) {
        std::memset(caps, 0, buf_size);
    }
}

OCT_API int oct_runtime_last_error(
    oct_runtime_t* runtime,
    char* buf,
    size_t buflen
) {
    if (runtime == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(runtime->error_mutex);
    return copy_error(runtime->last_error, buf, buflen);
}

OCT_API int oct_last_thread_error(char* buf, size_t buflen) {
    std::string msg;
    {
        std::lock_guard<std::mutex> lock(thread_error_mutex());
        auto it = thread_error_map().find(std::this_thread::get_id());
        if (it != thread_error_map().end()) {
            msg = it->second;
        }
    }
    return copy_error(msg, buf, buflen);
}

/* -------------------------------------------------------------------------
 * Session lifecycle — slice-2 stubs. Every entry point returns
 * OCT_STATUS_UNSUPPORTED with a descriptive runtime->last_error.
 * ------------------------------------------------------------------------- */

OCT_API oct_status_t oct_session_open(
    oct_runtime_t* runtime,
    const oct_session_config_t* config,
    oct_session_t** out
) {
    /* Codex R1 fix: header invariant requires INVALID_INPUT when
     * out == NULL. Previous stub fell through to UNSUPPORTED, which
     * a binding might attempt to recover from by issuing oct_session_close
     * on an uninitialized handle. */
    if (out == nullptr) {
        if (runtime != nullptr) {
            runtime->set_error("oct_session_open: out parameter is NULL");
        } else {
            set_thread_error("oct_session_open: out parameter is NULL");
        }
        return OCT_STATUS_INVALID_INPUT;
    }
    *out = nullptr;
    if (runtime == nullptr) {
        set_thread_error("oct_session_open: runtime is NULL");
        return OCT_STATUS_INVALID_INPUT;
    }
    (void)config;
    runtime->set_error(
        "oct_session_open: not implemented in slice-2 build (stub returns "
        "OCT_STATUS_UNSUPPORTED until the Moshi-on-MLX adapter lands)"
    );
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API void oct_session_close(oct_session_t* session) {
    /* Stub never produces a session — nothing to close. The header's
     * contract says close-of-NULL is fine. */
    if (session != nullptr) {
        delete session;
    }
}

OCT_API oct_status_t oct_session_send_audio(
    oct_session_t* session,
    const oct_audio_view_t* audio
) {
    (void)session;
    (void)audio;
    set_thread_error("oct_session_send_audio: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_session_send_text(
    oct_session_t* session,
    const char* text
) {
    (void)session;
    (void)text;
    set_thread_error("oct_session_send_text: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_session_poll_event(
    oct_session_t* session,
    oct_event_t* out,
    uint32_t timeout_ms
) {
    (void)session;
    (void)timeout_ms;
    /* Respect out->size to avoid overrunning an older binding's
     * event buffer. The fields the stub writes are version + size +
     * type; the buffer must have room for all three.
     *
     * Codex R2 fix: previous patch did `memset(out, 0, out->size)`
     * (which zeroed out->size itself) followed by a self-assign that
     * left out->size = 0. The caller's versioned handshake was
     * defeated. Now we snapshot out->size first, clear via memset
     * up to that limit, then restore size and write the remaining
     * fields only if the buffer covers them. */
    if (out != nullptr) {
        const size_t buf_size = out->size;
        const size_t need_through_type = offsetof(oct_event_t, type) + sizeof(out->type);
        if (buf_size >= need_through_type && buf_size <= sizeof(*out)) {
            std::memset(out, 0, buf_size);
            out->version = OCT_EVENT_VERSION;
            out->size = buf_size;
            out->type = OCT_EVENT_NONE;
            /* v0.4 step 2: if the buffer covers the operational
             * envelope (v0.4-step-2 size), populate envelope strings
             * with empty string sentinels. Per the contract, runtime
             * NEVER writes NULL to envelope fields — bindings can
             * strlen() safely. */
            const size_t need_envelope =
                offsetof(oct_event_t, request_id) + sizeof(const char*) * 7 + sizeof(uint8_t) * 4;
            if (buf_size >= need_envelope) {
                out->request_id = "";
                out->route_id = "";
                out->trace_id = "";
                out->engine_version = "";
                out->adapter_version = "";
                out->accelerator = "";
                out->artifact_digest = "";
                out->cache_was_hit = 0;
            }
        }
        /* If buf_size is too small for version/size/type or wildly
         * larger than sizeof(*out), the binding violated the contract;
         * we leave the buffer untouched. */
    }
    set_thread_error("oct_session_poll_event: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_session_cancel(oct_session_t* session) {
    (void)session;
    set_thread_error("oct_session_cancel: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

/* ==========================================================================
 * v0.4 step 1 — Model lifecycle stubs.
 * Every entry point returns OCT_STATUS_UNSUPPORTED with a descriptive
 * last_error. Engine adapters (Slice 2C Moshi/MLX, future llama.cpp /
 * sherpa-onnx / whisper.cpp / ONNX) will fill these in.
 * ========================================================================== */

struct oct_model {
    /* v0.4 step 1: never actually constructed. Reserved for the real
     * implementation. */
    int reserved;
};

OCT_API size_t oct_model_config_size(void) {
    return sizeof(oct_model_config_t);
}

OCT_API oct_status_t oct_model_open(
    oct_runtime_t* runtime,
    const oct_model_config_t* config,
    oct_model_t** out_model
) {
    /* Slice-2A invariant preserved: NULL-out → INVALID_INPUT before
     * any other check, so binding cleanup paths can rely on
     * `*out == NULL` after a non-OK return. */
    if (out_model == nullptr) {
        if (runtime != nullptr) {
            runtime->set_error("oct_model_open: out_model parameter is NULL");
        } else {
            set_thread_error("oct_model_open: out_model parameter is NULL");
        }
        return OCT_STATUS_INVALID_INPUT;
    }
    *out_model = nullptr;
    if (runtime == nullptr) {
        set_thread_error("oct_model_open: runtime is NULL");
        return OCT_STATUS_INVALID_INPUT;
    }
    if (config == nullptr) {
        runtime->set_error("oct_model_open: config is NULL");
        return OCT_STATUS_INVALID_INPUT;
    }
    if (config->version != OCT_MODEL_CONFIG_VERSION) {
        runtime->set_error("oct_model_open: config.version must be 1");
        return OCT_STATUS_VERSION_MISMATCH;
    }
    runtime->set_error(
        "oct_model_open: not implemented in v0.4 step 1 stub (engine adapters "
        "land in Slice 2C and following slices). Returns OCT_STATUS_UNSUPPORTED."
    );
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_model_warm(oct_model_t* model) {
    (void)model;
    set_thread_error("oct_model_warm: not implemented in v0.4 step 1 stub");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_model_evict(oct_model_t* model) {
    (void)model;
    set_thread_error("oct_model_evict: not implemented in v0.4 step 1 stub");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API void oct_model_close(oct_model_t* model) {
    /* Stub never produces a model — nothing to close. The header's
     * close-of-NULL contract is preserved. */
    if (model != nullptr) {
        delete model;
    }
}

}  // extern "C"
