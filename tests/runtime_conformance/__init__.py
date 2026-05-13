"""Capability-aware conformance harness.

Tests parametrize on backend (`python` / `native` / `both`) with
the `@requires_capability(CAP_NAME)` marker enforcing the current
native capability status partition.

  * `python` backend: skip with reason `no-python-oracle` if the
    capability is `BLOCKED_WITH_PROOF`.
  * `native` backend: skip with reason `runtime_capabilities` if
    the live runtime does NOT advertise the capability. This is
    expected for conditional capabilities whose gates are not met and
    for blocked capabilities that current runtimes must not advertise.

Cannot mistake a legal enum name for a live runtime feature.
"""
