/*
 * Moshi-rs engine adapter implementation.
 *
 * Slice 2C scope:
 *   * Platform gate (darwin-arm64) — adapter is not constructed
 *     elsewhere via #ifdef in CMake.
 *   * Capability honesty in scaffolding state: `is_loadable_now()`
 *     returns false because the Rust shim's session worker does
 *     not yet wire the real moshi-core streaming pipeline. Capability
 *     `audio.realtime.session` therefore does NOT appear in
 *     `oct_runtime_capabilities`.
 *   * Follow-up commit will (a) wire `lm_generate_multistream::State`
 *     in the Rust shim, (b) flip this flag, (c) add the artifact-
 *     path config plumbing from the runtime config to the shim.
 *
 * Why this adapter never advertises false-positives:
 *   * `register_with_runtime()` calls `register_adapter` on the
 *     CapabilityRegistry, which has a hard `is_loadable_now()` gate.
 *   * That gate currently always returns false on this build.
 *   * Therefore `oct_runtime_capabilities.supported_capabilities`
 *     does NOT include "audio.realtime.session" in this slice.
 */

#include "moshi_rs_adapter.h"

#include <cstring>
#include <utility>

#if defined(OCT_HAVE_MOSHI_RS_SHIM)
extern "C" {
#include "octomil_moshi_rs.h"
}
#endif

namespace octomil::adapters::moshi_rs {

namespace {

constexpr bool is_supported_platform() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    return true;
#else
    return false;
#endif
}

}  /* namespace */

MoshiRsAdapter::MoshiRsAdapter() {
    if (!is_supported_platform()) {
        loadable_ = false;
        load_status_reason_ =
            "moshi_rs: unsupported platform (Slice 2C is darwin-arm64 only)";
        return;
    }
#if !defined(OCT_HAVE_MOSHI_RS_SHIM)
    loadable_ = false;
    load_status_reason_ =
        "moshi_rs: Rust shim not linked into this build (CMake "
        "OCT_ENABLE_ENGINE_MOSHI_RS=OFF)";
    return;
#else
    /*
     * Scaffolding-state honesty. The Rust shim symbols are linked,
     * but the shim's session worker emits no real audio yet (see
     * engines/moshi-rs/src/lib.rs `session_worker`). Until the
     * follow-up commit wires `moshi::lm::load_lm_model` +
     * `moshi::mimi::load` + `lm_generate_multistream::State`, the
     * adapter MUST refuse to advertise the capability — even
     * though the static-link surface and platform gate are both
     * happy.
     */
    loadable_ = false;
    load_status_reason_ = std::string(
        "moshi_rs: scaffolding only — Rust shim linked (")
        + octomil_moshi_rs_version()
        + ") but real inference pipeline not yet wired (Slice 2C follow-up)";
    return;
#endif
}

void register_with_runtime() {
    static MoshiRsAdapter adapter;  /* lifetime: process */
    CapabilityRegistry::instance().register_adapter(adapter);
}

}  /* namespace octomil::adapters::moshi_rs */
