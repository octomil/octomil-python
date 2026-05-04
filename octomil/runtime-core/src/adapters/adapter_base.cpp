/*
 * Capability registry implementation.
 *
 * Threadsafe-by-construction is overkill: registry is populated at
 * `oct_runtime_open` from the calling thread before any other
 * runtime entry points run. Snapshot accessors are read-only after
 * registration. A future change that lets adapters register at any
 * point in a runtime's lifetime needs to add a mutex.
 */
#include "adapter_base.h"

#include <algorithm>

namespace octomil::adapters {

CapabilityRegistry& CapabilityRegistry::instance() {
    static CapabilityRegistry r;
    return r;
}

void CapabilityRegistry::register_adapter(const IEngineAdapter& adapter) {
    if (!adapter.is_loadable_now()) {
        /* Capability honesty: an adapter that cannot serve a session
         * right now does NOT contribute capabilities. The caller
         * (runtime_open) records the reason in last-error for
         * diagnostics. */
        return;
    }
    const std::string engine_id = adapter.engine_id();
    if (std::find(engine_ids_.begin(), engine_ids_.end(), engine_id) == engine_ids_.end()) {
        engine_ids_.push_back(engine_id);
    }
    for (const auto& cap : adapter.declared_capabilities()) {
        const bool already_present = std::any_of(
            entries_.begin(), entries_.end(),
            [&](const CapabilityEntry& e) {
                return e.name == cap && e.owner == &adapter;
            }
        );
        if (!already_present) {
            entries_.push_back(CapabilityEntry{cap, &adapter});
        }
    }
}

std::vector<std::string> CapabilityRegistry::snapshot_capabilities() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& e : entries_) {
        if (std::find(out.begin(), out.end(), e.name) == out.end()) {
            out.push_back(e.name);
        }
    }
    return out;
}

std::vector<std::string> CapabilityRegistry::snapshot_engines() const {
    return engine_ids_;
}

void CapabilityRegistry::reset_for_tests() {
    entries_.clear();
    engine_ids_.clear();
}

}  /* namespace octomil::adapters */
