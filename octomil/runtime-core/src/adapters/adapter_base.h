/*
 * Internal adapter base — Layer 2a engine plumbing.
 *
 * Defines the minimal interface every engine adapter implements, plus
 * the runtime-side capability registry. NOT exported as part of the
 * public Octomil ABI; bindings see only the C ABI in
 * include/octomil/runtime.h.
 *
 * Slice 2C scope:
 *   * IEngineAdapter interface declared.
 *   * Capability registry collects (name, owning adapter) pairs from
 *     adapters constructed at runtime-open. oct_runtime_capabilities
 *     reads from the registry.
 *   * Capability honesty rule: an adapter's name appears in the
 *     registry only if its `is_loadable_now()` method returns true.
 *     For the moshi_rs adapter that means: darwin-arm64 + Rust shim
 *     linked + artifact paths configured + on-disk presence verified.
 *     The scaffolding-state moshi_rs adapter intentionally returns
 *     false so the capability is NOT advertised even though the
 *     adapter object code is present.
 */
#ifndef OCTOMIL_INTERNAL_ADAPTER_BASE_H
#define OCTOMIL_INTERNAL_ADAPTER_BASE_H

#include <memory>
#include <string>
#include <vector>

namespace octomil::adapters {

class IEngineAdapter {
  public:
    virtual ~IEngineAdapter() = default;

    /** Stable engine identifier (e.g., "moshi_rs"). Used as a
     *  key in the engine registry; never user-facing. */
    virtual const char* engine_id() const noexcept = 0;

    /** Capability names this adapter implements (e.g.,
     *  {"audio.realtime.session"}). Returned regardless of
     *  loadability — the registry filters by is_loadable_now. */
    virtual std::vector<std::string> declared_capabilities() const = 0;

    /** True iff this adapter can actually serve a session right now
     *  on this host. Slice 2C interpretation:
     *    - platform gate (darwin-arm64),
     *    - underlying engine objects are linked into the binary,
     *    - artifacts present + verified by upstream path,
     *    - the streaming path produces real audio (NOT scaffolding
     *      mode).
     *  Capability honesty is enforced against this single method. */
    virtual bool is_loadable_now() const = 0;

    /** Human-readable last reason `is_loadable_now()` returned
     *  false. Empty when loadable. Used for last-error diagnostics. */
    virtual std::string load_status_reason() const = 0;
};

/* Slice 2C registry: adapters registered at runtime-open populate
 * this static list. oct_runtime_capabilities walks it and emits the
 * union of declared capabilities, filtered by is_loadable_now.
 *
 * One global registry is sufficient because the runtime's capability
 * surface is per-process — a future per-runtime capability override
 * would graduate this to a member of `oct_runtime`.
 */
struct CapabilityEntry {
    std::string name;
    const IEngineAdapter* owner;  /* non-owning; adapter outlives the entry */
};

class CapabilityRegistry {
  public:
    static CapabilityRegistry& instance();

    /** Register an adapter's loadable capabilities. Idempotent on
     *  duplicate (name, owner) pairs. */
    void register_adapter(const IEngineAdapter& adapter);

    /** Snapshot the current set of capability names. Used by
     *  `oct_runtime_capabilities` to populate the supported list. */
    std::vector<std::string> snapshot_capabilities() const;

    /** Snapshot the set of engine ids that contributed at least one
     *  loadable capability. Used by `oct_runtime_capabilities` for
     *  `supported_engines`. */
    std::vector<std::string> snapshot_engines() const;

    /** Tests-only: clear the registry. */
    void reset_for_tests();

  private:
    CapabilityRegistry() = default;
    mutable std::vector<CapabilityEntry> entries_;
    mutable std::vector<std::string> engine_ids_;
};

}  /* namespace octomil::adapters */

#endif  /* OCTOMIL_INTERNAL_ADAPTER_BASE_H */
