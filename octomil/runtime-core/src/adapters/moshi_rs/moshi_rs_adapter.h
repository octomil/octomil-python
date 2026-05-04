/*
 * Moshi-rs engine adapter — C++ wrapper around the Rust shim that
 * wraps upstream `moshi-core` (kyutai-labs/moshi crate `moshi`)
 * with candle-metal on Apple Silicon.
 *
 * This adapter implements `audio.realtime.session` for darwin-arm64.
 * The capability is advertised IFF `is_loadable_now()` returns true,
 * which in Slice 2C scaffolding state always returns false (see
 * README.md and load_status_reason()).
 *
 * Why moshi-rs not moshi-mlx:
 *   * Slice-2B probe ran moshi_mlx (Python+MLX). GREEN.
 *   * For Layer 2a (no Python in realtime path), we need a native
 *     streaming path. moshi-mlx has no C++ counterpart.
 *   * `moshi-core` is the upstream Rust streaming Moshi
 *     implementation built on candle-rs with a `metal` feature for
 *     Apple Silicon. It is the canonical native path.
 *   * The Rust shim (engines/moshi-rs/) wraps it in a small
 *     extern-"C" surface that this adapter calls.
 */
#ifndef OCTOMIL_INTERNAL_MOSHI_RS_ADAPTER_H
#define OCTOMIL_INTERNAL_MOSHI_RS_ADAPTER_H

#include "../adapter_base.h"

#include <string>
#include <vector>

namespace octomil::adapters::moshi_rs {

class MoshiRsAdapter final : public IEngineAdapter {
  public:
    /** Construct from the runtime config. The constructor performs
     *  the platform gate and, if applicable, calls into the Rust
     *  shim's check_artifacts. Construction never throws. Failures
     *  surface as `is_loadable_now() == false` plus a non-empty
     *  `load_status_reason()`. */
    MoshiRsAdapter();

    const char* engine_id() const noexcept override { return "moshi_rs"; }

    std::vector<std::string> declared_capabilities() const override {
        return {std::string("audio.realtime.session")};
    }

    bool is_loadable_now() const override { return loadable_; }
    std::string load_status_reason() const override { return load_status_reason_; }

  private:
    bool loadable_ = false;
    std::string load_status_reason_;
};

/** Construct + register the singleton adapter on a freshly opened
 *  runtime. Idempotent. The capability registry filters by
 *  `is_loadable_now()`, so calling this on a host where Moshi can't
 *  load is safe (capability simply doesn't appear). */
void register_with_runtime();

}  /* namespace octomil::adapters::moshi_rs */

#endif  /* OCTOMIL_INTERNAL_MOSHI_RS_ADAPTER_H */
