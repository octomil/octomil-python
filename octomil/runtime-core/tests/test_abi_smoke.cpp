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

    /* Capabilities returns OK with an empty descriptor. */
    oct_capabilities_t caps;
    st = oct_runtime_capabilities(rt, &caps);
    EXPECT(st == OCT_STATUS_OK, "capabilities should return OK");
    oct_runtime_capabilities_free(&caps);

    /* Stub session_open returns UNSUPPORTED. */
    oct_session_config_t sess_cfg = {};
    sess_cfg.version = 1;
    oct_session_t* sess = reinterpret_cast<oct_session_t*>(0xbadc0de);
    st = oct_session_open(rt, &sess_cfg, &sess);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "stub session_open should return UNSUPPORTED");
    EXPECT(sess == nullptr, "session out should be NULL on failure");

    /* The runtime's last_error reflects the last failed call. */
    char errbuf[256] = {};
    int n = oct_runtime_last_error(rt, errbuf, sizeof(errbuf));
    EXPECT(n > 0, "last_error should have a message after failed session_open");
    const std::string err(errbuf);
    EXPECT(err.find("session_open") != std::string::npos,
           "last_error should mention session_open");

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
    st = oct_session_poll_event(nullptr, &ev, 0);
    EXPECT(st == OCT_STATUS_UNSUPPORTED, "poll_event stub");
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
