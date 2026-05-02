"""Runtime selection bench — v0.5 Python-owned cache R/W skeleton.

This package implements the v0.5 cache layer of the runtime selection
bench (see ``strategy/runtime-selection-bench.md`` in the contracts
repo for the full architecture). v0.5 is a transitional Python-owned
implementation; v1 moves the same on-disk format into the native
runtime core (``liboctomil-runtime``) once the runtime extraction
lands. Both versions write against the same schemas in
``octomil-contracts/schemas/core/runtime_bench_*.json``.

This first PR ships **cache R/W only**:

  * ``CacheKey`` dataclass mirroring ``runtime_bench_cache_key.json``.
  * ``CacheStore`` — read / write / list / clear, with the on-disk
    layout the strategy doc defined: SHA256-of-canonical-JSON leaf
    filenames, atomic writes (temp-file-plus-rename), advisory file
    locking around writes, partial-read rejection, schema-version-
    mismatch as cache miss not crash, ``incomplete=true`` ignored
    on lookup, index sidecar rebuildable from leaves.
  * ``HardwareFingerprint`` — captures the device descriptor and
    produces the truncated path component the strategy doc requires.

What is NOT in this PR (intentionally — see the strategy doc's
PR sequencing):

  * No benchmark runner. Cache R/W has no opinion on what writes to
    the cache; the bench harness comes in PR B.
  * No dispatch integration. Nothing in the kernel reads this cache
    yet; that's PR C.
  * No CLI. ``octomil bench`` lands in PR D.
  * No telemetry. v0.5 is local-cache only.

Hard cutover policy applies (per the strategy doc): when v1 takes
over the writer role, the v0.5 cache directory at
``runtime_bench_v0_5/`` is read for one release window and then
removed. No aliases, no shims.
"""

from __future__ import annotations

from octomil.runtime.bench.cache import (
    CACHE_DIR_NAME,
    CACHE_SCHEMA_VERSION,
    CacheKey,
    CacheStore,
    DispatchShape,
    HardwareFingerprint,
    Result,
    Winner,
)

__all__ = [
    "CacheKey",
    "CacheStore",
    "DispatchShape",
    "HardwareFingerprint",
    "Result",
    "Winner",
    "CACHE_SCHEMA_VERSION",
    "CACHE_DIR_NAME",
]
