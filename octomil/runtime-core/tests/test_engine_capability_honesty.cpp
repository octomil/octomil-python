/*
 * Slice 2C capability-honesty test.
 *
 * Goals:
 *   1. Confirm `liboctomil-runtime` opens cleanly with the moshi_rs
 *      adapter linked.
 *   2. Confirm that, in scaffolding state, `audio.realtime.session`
 *      does NOT appear in `oct_runtime_capabilities`. The adapter's
 *      `is_loadable_now()` returns false; the registry MUST drop it.
 *   3. Confirm that opening a session for `audio.realtime.session`
 *      via `oct_session_open` returns `OCT_STATUS_UNSUPPORTED` (the
 *      slice-2A behavior is unchanged).
 *
 * The test is gated on OCT_ENABLE_ENGINE_MOSHI_RS=ON in CMakeLists.txt
 * so it only runs on darwin-arm64. When the follow-up commit flips
 * `is_loadable_now()` to true (and wires real inference), this test
 * will need to be extended (or split into "no-artifact" and "artifact-
 * present" variants).
 */

#include "octomil/runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool list_contains(const char* const* list, const char* needle) {
    if (list == nullptr) {
        return false;
    }
    for (size_t i = 0; list[i] != nullptr; ++i) {
        if (std::strcmp(list[i], needle) == 0) {
            return true;
        }
    }
    return false;
}

#define EXPECT_EQ(a, b) do { \
    auto _aa = (a); auto _bb = (b); \
    if (!(_aa == _bb)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s == %s (got %lld vs %lld)\n", \
                     __FILE__, __LINE__, #a, #b, (long long)_aa, (long long)_bb); \
        std::exit(1); \
    } \
} while (0)

#define EXPECT_FALSE(x) do { \
    if ((x)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s should be false\n", __FILE__, __LINE__, #x); \
        std::exit(1); \
    } \
} while (0)

}  // namespace

int main() {
    oct_runtime_config_t cfg{};
    cfg.version = 1;

    oct_runtime_t* rt = nullptr;
    oct_status_t s = oct_runtime_open(&cfg, &rt);
    EXPECT_EQ(s, OCT_STATUS_OK);

    oct_capabilities_t caps{};
    caps.size = sizeof(caps);
    s = oct_runtime_capabilities(rt, &caps);
    EXPECT_EQ(s, OCT_STATUS_OK);

    // Scaffolding-state invariant: `audio.realtime.session` MUST NOT
    // appear in the supported_capabilities list. Even though the
    // moshi_rs adapter object code is linked into this binary, its
    // `is_loadable_now()` returns false because the Rust shim has
    // not yet wired the real moshi-core streaming pipeline.
    EXPECT_FALSE(list_contains(caps.supported_capabilities, "audio.realtime.session"));

    // Symmetric check: moshi_rs MUST NOT appear in supported_engines
    // either, since the registry drops engines that contributed no
    // loadable capabilities. (If/when the follow-up commit flips
    // `is_loadable_now()`, this assertion will be inverted.)
    EXPECT_FALSE(list_contains(caps.supported_engines, "moshi_rs"));

    oct_runtime_capabilities_free(&caps);

    // The slice-2A behavior is preserved: `oct_session_open` still
    // returns OCT_STATUS_UNSUPPORTED for `audio.realtime.session`
    // because the adapter does not yet vend a session object.
    oct_session_config_t sc{};
    sc.version = OCT_SESSION_CONFIG_VERSION;
    sc.capability = "audio.realtime.session";
    sc.locality = "on_device";
    sc.policy_preset = "private";
    oct_session_t* sess = nullptr;
    s = oct_session_open(rt, &sc, &sess);
    EXPECT_EQ(s, OCT_STATUS_UNSUPPORTED);

    oct_runtime_close(rt);
    std::fprintf(stdout, "engine_capability_honesty: OK\n");
    return 0;
}
