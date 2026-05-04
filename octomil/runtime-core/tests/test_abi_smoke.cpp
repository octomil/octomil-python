/*
 * test_abi_smoke.cpp — slice-2 ABI smoke test
 *
 * Verifies that:
 *   * Every OCT_API symbol is exported and callable.
 *   * Version-inspection functions return the constants from runtime.h.
 *   * `oct_runtime_open` accepts a v1 config and rejects v0/v2.
 *   * Stub session entry points return OCT_STATUS_UNSUPPORTED with a
 *     reachable last-error message.
 *
 * Pass = exit code 0. Fail = exit code 1 with a stderr description.
 *
 * No external test framework: this binary runs under CTest as a
 * standalone executable. Keeping it framework-free avoids dragging
 * a test dependency into the slice-2 build.
 */

#include "octomil/runtime.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

namespace {

#define EXPECT(cond, msg)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__,   \
                         msg);                                              \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

void test_version_inspection() {
    EXPECT(oct_runtime_abi_version_major() == OCT_RUNTIME_ABI_VERSION_MAJOR,
           "abi_version_major mismatch");
    EXPECT(oct_runtime_abi_version_minor() == OCT_RUNTIME_ABI_VERSION_MINOR,
           "abi_version_minor mismatch");
    /* Patch is 0 in this slice-2 stub. */
    EXPECT(oct_runtime_abi_version_patch() == 0u, "abi_version_patch should be 0");

    const uint64_t packed = oct_runtime_abi_version_packed();
    const uint32_t major = static_cast<uint32_t>((packed >> 32) & 0xFFFFFFFFu);
    const uint32_t minor = static_cast<uint32_t>((packed >> 16) & 0xFFFFu);
    const uint32_t patch = static_cast<uint32_t>(packed & 0xFFFFu);
    EXPECT(major == OCT_RUNTIME_ABI_VERSION_MAJOR, "packed major");
    EXPECT(minor == OCT_RUNTIME_ABI_VERSION_MINOR, "packed minor");
    EXPECT(patch == 0u, "packed patch");
}

void test_runtime_open_invalid_inputs() {
    /* NULL out → INVALID_INPUT. */
    oct_runtime_config_t cfg = {};
    cfg.version = 1;
    oct_status_t st = oct_runtime_open(&cfg, nullptr);
    EXPECT(st == OCT_STATUS_INVALID_INPUT, "open with NULL out should fail");

    /* NULL config → INVALID_INPUT, out cleared. */
    oct_runtime_t* rt = reinterpret_cast<oct_runtime_t*>(0xdeadbeef);
    st = oct_runtime_open(nullptr, &rt);
    EXPECT(st == OCT_STATUS_INVALID_INPUT, "open with NULL config should fail");
    EXPECT(rt == nullptr, "out should be NULL after invalid open");

    /* Wrong version → VERSION_MISMATCH. */
    cfg.version = 0;
    st = oct_runtime_open(&cfg, &rt);
    EXPECT(st == OCT_STATUS_VERSION_MISMATCH, "v0 config should mismatch");
    cfg.version = 2;
    st = oct_runtime_open(&cfg, &rt);
    EXPECT(st == OCT_STATUS_VERSION_MISMATCH, "v2 config should mismatch");
}

void test_runtime_open_v1_succeeds_then_close() {
    oct_runtime_config_t cfg = {};
    cfg.version = 1;
    oct_runtime_t* rt = nullptr;
    oct_status_t st = oct_runtime_open(&cfg, &rt);
    EXPECT(st == OCT_STATUS_OK, "v1 config should succeed");
    EXPECT(rt != nullptr, "out should be non-NULL on success");

    /* Capabilities returns OK with the empty-list sentinel arrays
     * AND respects the versioned-output-struct contract:
     *   - Caller sets out->size before the call.
     *   - Runtime writes <= out->size bytes.
     *   - Empty list = non-NULL pointer to a length-1 array of NULL. */
    oct_capabilities_t caps = {};
    caps.size = sizeof(caps);
    st = oct_runtime_capabilities(rt, &caps);
    EXPECT(st == OCT_STATUS_OK, "capabilities should return OK");
    EXPECT(caps.version == OCT_CAPABILITIES_VERSION,
           "capabilities version should be OCT_CAPABILITIES_VERSION");
    EXPECT(caps.size == sizeof(caps),
           "capabilities size should round-trip");
    EXPECT(caps.supported_engines != nullptr,
           "supported_engines must be non-NULL (empty-list sentinel)");
    EXPECT(caps.supported_engines[0] == nullptr,
           "supported_engines must be empty-sentinel-terminated");
    EXPECT(caps.supported_capabilities != nullptr &&
           caps.supported_capabilities[0] == nullptr,
           "supported_capabilities empty sentinel");
    EXPECT(caps.supported_archs != nullptr && caps.supported_archs[0] == nullptr,
           "supported_archs empty sentinel");
    oct_runtime_capabilities_free(&caps);

    /* Versioned-output-struct: out->size = 0 violates the contract. */
    oct_capabilities_t small_caps = {};
    small_caps.size = 0;
    st = oct_runtime_capabilities(rt, &small_caps);
    EXPECT(st == OCT_STATUS_INVALID_INPUT,
           "capabilities should reject out->size = 0");

    /* Small-buffer capabilities canary (Codex R3): a binding that
     * passes a smaller out->size must see size preserved AND
     * capabilities_free must not write past size. */
    {
        const size_t small_size = offsetof(oct_capabilities_t, supported_engines);
        unsigned char canary[sizeof(oct_capabilities_t) + 16] = {};
        for (size_t i = 0; i < sizeof(canary); ++i) canary[i] = 0xBB;
        oct_capabilities_t* small_view = reinterpret_cast<oct_capabilities_t*>(canary);
        small_view->size = small_size;
        st = oct_runtime_capabilities(rt, small_view);
        EXPECT(st == OCT_STATUS_OK, "small-buffer capabilities should succeed");
        EXPECT(small_view->size == small_size,
               "small-buffer size must round-trip (Codex R3: was clobbered)");
        EXPECT(small_view->version == OCT_CAPABILITIES_VERSION, "small-buffer version set");
        for (size_t i = small_size; i < sizeof(canary); ++i) {
            EXPECT(canary[i] == 0xBB, "capabilities must not write past out->size");
        }
        oct_runtime_capabilities_free(small_view);
        for (size_t i = small_size; i < sizeof(canary); ++i) {
            EXPECT(canary[i] == 0xBB, "capabilities_free must not write past caps->size");
        }
    }

    /* Session_open rejects NULL out per the header invariant. */
    oct_session_config_t bad_cfg = {};
    bad_cfg.version = 1;
    st = oct_session_open(rt, &bad_cfg, nullptr);
    EXPECT(st == OCT_STATUS_INVALID_INPUT,
           "session_open with NULL out should return INVALID_INPUT");

    /* Stub session_open returns UNSUPPORTED. Slice 2A: also assert
     * the canonical capability field (strict-reject case under the
     * stub returns UNSUPPORTED uniformly — no fake handle). */
    oct_session_config_t sess_cfg = {};
    sess_cfg.version = 1;
    sess_cfg.capability = "audio.realtime.session";
    sess_cfg.locality = "on_device";
    oct_session_t* sess = reinterpret_cast<oct_session_t*>(0xbadc0de);
    st = oct_session_open(rt, &sess_cfg, &sess);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "stub session_open should return UNSUPPORTED");
    EXPECT(sess == nullptr, "session out should be NULL on failure");

    /* Slice 2A: session-level introspection sizes are non-zero and
     * stable under the C compiler. The values themselves don't matter
     * here — they're checked for parity in the Python-side test. */
    EXPECT(oct_session_config_size() > 0, "session_config_size > 0");
    EXPECT(oct_audio_view_size() > 0, "audio_view_size > 0");
    EXPECT(oct_event_size() > 0, "event_size > 0");
    EXPECT(oct_session_config_size() == sizeof(oct_session_config_t),
           "session_config_size matches sizeof in this TU");
    EXPECT(oct_audio_view_size() == sizeof(oct_audio_view_t),
           "audio_view_size matches sizeof in this TU");
    EXPECT(oct_event_size() == sizeof(oct_event_t),
           "event_size matches sizeof in this TU");

    /* v0.4 step 1: model_config_size introspection (no last_error mutation). */
    EXPECT(oct_model_config_size() > 0, "model_config_size > 0");
    EXPECT(oct_model_config_size() == sizeof(oct_model_config_t),
           "model_config_size matches sizeof in this TU");

    /* v0.4 error code constants. */
    EXPECT(OCT_ERR_OK == 0u, "OCT_ERR_OK == 0");
    EXPECT(OCT_ERR_UNKNOWN == 0xFFFFFFFFu, "OCT_ERR_UNKNOWN == UINT32_MAX");
    EXPECT(OCT_ERR_INTERNAL == 11u, "OCT_ERR_INTERNAL stable assignment");

    /* v0.4 capability constants — string-equality smoke. */
    EXPECT(std::string(OCT_CAPABILITY_AUDIO_REALTIME_SESSION) == "audio.realtime.session",
           "OCT_CAPABILITY_AUDIO_REALTIME_SESSION literal");
    EXPECT(std::string(OCT_CAPABILITY_EMBEDDINGS_TEXT) == "embeddings.text",
           "OCT_CAPABILITY_EMBEDDINGS_TEXT literal");
    EXPECT(std::string(OCT_CAPABILITY_INDEX_VECTOR_QUERY) == "index.vector.query",
           "OCT_CAPABILITY_INDEX_VECTOR_QUERY literal");

    /* The runtime's last_error reflects the last failed call. */
    char errbuf[256] = {};
    int n = oct_runtime_last_error(rt, errbuf, sizeof(errbuf));
    EXPECT(n > 0, "last_error should have a message after failed session_open");
    const std::string err(errbuf);
    EXPECT(err.find("session_open") != std::string::npos,
           "last_error should mention session_open");

    /* v0.4 step 1: model lifecycle stubs run AFTER the session_open
     * last_error check (since these mutate runtime->last_error). */
    {
        oct_model_config_t mcfg = {};
        mcfg.version = OCT_MODEL_CONFIG_VERSION;
        oct_status_t st_m = oct_model_open(rt, &mcfg, nullptr);
        EXPECT(st_m == OCT_STATUS_INVALID_INPUT,
               "model_open with NULL out should return INVALID_INPUT");
    }
    {
        oct_model_config_t mcfg = {};
        mcfg.version = OCT_MODEL_CONFIG_VERSION;
        oct_model_t* mh = reinterpret_cast<oct_model_t*>(0xb16b00b5);
        oct_status_t st_m = oct_model_open(rt, &mcfg, &mh);
        EXPECT(st_m == OCT_STATUS_UNSUPPORTED, "stub model_open returns UNSUPPORTED");
        EXPECT(mh == nullptr, "model out should be NULL on failure");
    }
    EXPECT(oct_model_warm(nullptr) == OCT_STATUS_UNSUPPORTED, "model_warm stub");
    EXPECT(oct_model_evict(nullptr) == OCT_STATUS_UNSUPPORTED, "model_evict stub");
    oct_model_close(nullptr);  /* close-of-NULL is no-op */
    /* v0.4 Codex R2: oct_runtime_close documents implicit cleanup
     * of live models (parallel to sessions). Stub never produces a
     * model handle so there's nothing concrete to verify here, but
     * the contract is documented in runtime.h's oct_runtime_close
     * docstring. The Python-side regression test in
     * test_runtime_native_loader.py exercises the binding-side
     * close-before-invalidate ordering with a fake model handle. */
    /* model_open with bogus version → VERSION_MISMATCH. */
    {
        oct_model_config_t mcfg_bad = {};
        mcfg_bad.version = 0xFFu;
        oct_model_t* mh = nullptr;
        EXPECT(oct_model_open(rt, &mcfg_bad, &mh) == OCT_STATUS_VERSION_MISMATCH,
               "model_open rejects unknown config.version");
        EXPECT(mh == nullptr, "model out cleared on version mismatch");
    }

    /* close + idempotent NULL-close. */
    OCT_CLOSE_RUNTIME(rt);
    EXPECT(rt == nullptr, "OCT_CLOSE_RUNTIME should null the handle");
    OCT_CLOSE_RUNTIME(rt);  /* second close on already-NULL is a no-op */
}

void test_thread_error_buffer() {
    /* Force a thread-error path: NULL out on runtime_open writes a
     * thread-scoped error string. */
    oct_runtime_config_t cfg = {};
    cfg.version = 1;
    (void)oct_runtime_open(&cfg, nullptr);
    char buf[256] = {};
    int n = oct_last_thread_error(buf, sizeof(buf));
    EXPECT(n > 0, "last_thread_error should be set");
    const std::string msg(buf);
    EXPECT(msg.find("out parameter") != std::string::npos,
           "thread error should describe the NULL out path");
}

void test_session_stub_returns_unsupported() {
    /* All session entry points without a runtime: NULL session pointer
     * should still return UNSUPPORTED (with thread-error set). */
    oct_status_t st = oct_session_send_audio(nullptr, nullptr);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "send_audio stub");
    st = oct_session_send_text(nullptr, "hi");
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "send_text stub");
    oct_event_t ev = {};
    ev.size = sizeof(ev);
    st = oct_session_poll_event(nullptr, &ev, 0);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "poll_event stub");
    /* Stub clears the event safely (respects out->size). */
    EXPECT(ev.version == OCT_EVENT_VERSION, "poll_event sets version on clear");
    EXPECT(ev.size == sizeof(ev), "poll_event preserves out->size (was zeroed by memset; restored)");
    EXPECT(ev.type == OCT_EVENT_NONE, "poll_event sets type=NONE");

    /* Small-buffer event: out->size only large enough to cover the
     * header up through `type`. Stub must not write past out->size. */
    {
        const size_t small_size = offsetof(oct_event_t, type) + sizeof(ev.type);
        unsigned char canary[sizeof(ev) + 16] = {};
        for (size_t i = 0; i < sizeof(canary); ++i) canary[i] = 0xAA;
        oct_event_t* small_ev = reinterpret_cast<oct_event_t*>(canary);
        small_ev->size = small_size;
        st = oct_session_poll_event(nullptr, small_ev, 0);
        EXPECT(st == OCT_STATUS_UNSUPPORTED, "small-buffer poll_event still UNSUPPORTED");
        EXPECT(small_ev->version == OCT_EVENT_VERSION, "small-buffer version set");
        EXPECT(small_ev->size == small_size, "small-buffer size preserved");
        /* Bytes past small_size MUST still be the canary 0xAA. */
        for (size_t i = small_size; i < sizeof(canary); ++i) {
            EXPECT(canary[i] == 0xAA, "small-buffer poll_event must not overrun");
        }
    }
    st = oct_session_cancel(nullptr);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "cancel stub");
    /* close-of-NULL is a no-op (header contract). */
    oct_session_close(nullptr);
}

}  // namespace

int main() {
    test_version_inspection();
    test_runtime_open_invalid_inputs();
    test_runtime_open_v1_succeeds_then_close();
    test_thread_error_buffer();
    test_session_stub_returns_unsupported();
    std::printf("PASS: ABI smoke test (%d functions exercised)\n", 16);
    return 0;
}
