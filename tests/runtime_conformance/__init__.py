"""Capability-aware conformance harness — slice 3 PR2.

Tests parametrize on backend (`python` / `native` / `both`) with
the `@requires_capability(CAP_NAME)` marker enforcing ownership.

  * `python` backend: skip with reason `no-python-oracle` if the
    capability is NOT in `PYTHON_ORACLE_CAPABILITIES`. Native-first
    surfaces (e.g. `audio.realtime.session`) skip on Python by
    design.
  * `native` backend: skip with reason `runtime_capabilities` if
    the live runtime does NOT advertise the capability. Until
    slice 2-proper lands a real session adapter, every capability-
    gated test on the native backend skips by design.

Cannot mistake the slice-2 stub for a working runtime.
"""
