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

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>

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
    *out = rt;
    return OCT_STATUS_OK;
}

OCT_API void oct_runtime_close(oct_runtime_t* runtime) {
    if (runtime == nullptr) {
        return;
    }
    delete runtime;
}

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
    /* Slice-2 stub: empty capability list. The smoke test asserts on
     * this so bindings learn early that the stub doesn't claim any
     * capabilities. */
    std::memset(out, 0, sizeof(*out));
    return OCT_STATUS_OK;
}

OCT_API void oct_runtime_capabilities_free(oct_capabilities_t* caps) {
    if (caps == nullptr) {
        return;
    }
    /* No-op; slice-2 stub allocates nothing inside oct_capabilities_t. */
    std::memset(caps, 0, sizeof(*caps));
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
    if (out != nullptr) {
        *out = nullptr;
    }
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
    if (out != nullptr) {
        std::memset(out, 0, sizeof(*out));
    }
    set_thread_error("oct_session_poll_event: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

OCT_API oct_status_t oct_session_cancel(oct_session_t* session) {
    (void)session;
    set_thread_error("oct_session_cancel: not implemented in slice-2 build");
    return OCT_STATUS_UNSUPPORTED;
}

}  // extern "C"
