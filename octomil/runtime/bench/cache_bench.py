"""Cache bench harness — v0.1.11 Lane H skeleton.

Measures cold (fresh-process) vs warm (same-process, repeated) cache
lookup latency, hit/miss ratio, and byte overhead. Writes a
``cache_bench_proof`` JSON artifact conforming to
``schemas/core/cache_bench_proof.json`` in octomil-contracts.

Anti-pattern prevention:
  This harness MUST NOT fall back to first-request synthesis when the
  staged fixture set is absent. If absent, emit a proof artifact with
  ``skipped=true`` and ``skip_reason="staged_artifact_absent"`` and
  return without running any measurement. This mirrors the v0.1.9
  Lane 5 no-first-request-surprise rule.

Phase 1 stubs:
  ``StubCacheAdapter`` always returns a miss with ~0 ms latency.
  Lane B/C/etc replace it by passing a real ``CacheBenchAdapter``.

Proof artifact path convention (mirrors runtime_bench cache writer):
  ``<output_dir>/<cache_id>_proof.json``

Privacy:
  Proof artifacts MUST NOT contain input contents. Only digests are
  allowed in ``input_digest`` and ``staged_artifact_ref``. The
  ``_StagedFixtureSet.input_digest`` field must be computed over the
  fixture set metadata, not over the raw inputs.

Metric cross-ref (Lane A canonical telemetry enum, contracts runtime_metric.json):
  cold_p50_ms / cold_p95_ms = p50/p95 of cache.lookup_ms (gauge) samples during cold phase
  warm_p50_ms / warm_p95_ms = p50/p95 of cache.lookup_ms (gauge) samples during warm phase
  hit_ratio                 = cache.hit_total / (cache.hit_total + cache.miss_total) over bench window
These aggregate fields are local to cache_bench_proof.json — NOT metric enum names.

# TODO(lane-b/c): replace StubCacheAdapter with real cache adapters.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Protocol — real cache adapters implement this
# ---------------------------------------------------------------------------


class CacheBenchAdapter(Protocol):
    """Minimal interface for a cache adapter under bench.

    Lane B/C implement this for their specific caches. The harness
    is deliberately decoupled from any specific cache implementation.
    """

    def populate(
        self,
        fixture_dir: Path,
    ) -> tuple[int, int]:
        """Load entries from the staged fixture set.

        Returns (entry_count, bytes_overhead).
        bytes_overhead is the overhead of the cache structure, NOT the
        payload size.

        Raises ValueError if the fixture set is malformed.
        When wired (Lane B/C): emit cache.lookup_ms gauge samples plus
        cache.hit_total / cache.miss_total counter increments per outcome.
        """
        ...

    def cold_lookup(self, key: str) -> tuple[bool, float]:
        """Simulate a cold-path cache lookup.

        Returns (hit, latency_ms). Cold = no warm state in the adapter.
        When wired (Lane B/C): emit one cache.lookup_ms gauge sample plus
        a cache.miss_total / cache.hit_total increment per outcome.
        """
        ...

    def warm_lookup(self, key: str) -> tuple[bool, float]:
        """Simulate a warm-path cache lookup.

        Returns (hit, latency_ms). Warm = same-process, repeated call.
        When wired (Lane B/C): emit one cache.lookup_ms gauge sample plus
        a cache.miss_total / cache.hit_total increment per outcome.
        """
        ...


# ---------------------------------------------------------------------------
# Stub adapter (Phase 1)
# ---------------------------------------------------------------------------


class StubCacheAdapter:
    """Phase 1 stub — always miss, zero latency.

    Replace with real adapters in Lane B/C.
    # TODO(lane-b/c): delete this class when real adapters land.
    """

    def populate(self, fixture_dir: Path) -> tuple[int, int]:
        # Stub: nothing to load. Return 0 entries.
        return 0, 0

    def cold_lookup(self, key: str) -> tuple[bool, float]:
        t0 = time.monotonic()
        # Stub: no actual cache; always miss.
        latency = (time.monotonic() - t0) * 1000.0
        return False, latency

    def warm_lookup(self, key: str) -> tuple[bool, float]:
        t0 = time.monotonic()
        latency = (time.monotonic() - t0) * 1000.0
        return False, latency


# ---------------------------------------------------------------------------
# Staged fixture set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedFixtureSet:
    """Reference to the staged fixture set used for the bench.

    The fixture_dir is the on-disk location. input_digest is the
    SHA256 of the fixture set index (metadata only — NOT the raw
    input content, per privacy rule).
    """

    fixture_dir: Path
    cache_id: str
    capability: str

    def input_digest(self) -> str:
        """SHA256 of the fixture set metadata (not raw inputs).

        Codex B6: hashes the fixture index/content fingerprint, not just
        the directory path. Two different fixture sets staged at the
        same path now produce different digests, so changing fixtures
        is detectable by downstream tooling and the proof artifact
        cannot be silently reused across fixture revisions.

        Privacy: this digest is computed over file paths, sizes, and
        sha256 digests of file contents — never over raw input bytes
        embedded in the artifact (the resulting fingerprint is small
        and fixed-size regardless of fixture size).
        """
        h = hashlib.sha256()
        h.update(self.cache_id.encode())
        h.update(b"|")
        h.update(self.capability.encode())
        h.update(b"|")

        # If the fixture dir is missing, fall back to the path-only
        # mode (digest still reflects cache_id + capability + path).
        # The skip path emits skipped=true, so the digest is only an
        # advisory signal in that branch.
        if not self.fixture_dir.exists():
            h.update(b"missing|")
            h.update(str(self.fixture_dir).encode())
            return "sha256:" + h.hexdigest()

        # Walk the fixture dir deterministically: sorted relative paths,
        # for each file include relpath, byte size, and sha256(content).
        # Skip symlinks pointing outside the fixture dir defensively.
        fixture_root = self.fixture_dir.resolve()
        entries: list[tuple[str, int, str]] = []
        for path in sorted(fixture_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(fixture_root).as_posix()
            except ValueError:
                # Path escaped the fixture root via symlink; skip.
                continue
            try:
                size = path.stat().st_size
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                # Unreadable file; record path only with sentinel size.
                size = -1
                file_hash = "unreadable"
            entries.append((rel, size, file_hash))

        for rel, size, file_hash in entries:
            h.update(rel.encode())
            h.update(b"|")
            h.update(str(size).encode())
            h.update(b"|")
            h.update(file_hash.encode())
            h.update(b"\n")
        return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Proof artifact
# ---------------------------------------------------------------------------


# Allowed skip reasons — mirrors the schema closed enum.
SKIP_STAGED_ARTIFACT_ABSENT = "staged_artifact_absent"
SKIP_POLICY_DISABLED = "policy_disabled"
SKIP_CAPABILITY_UNSUPPORTED = "capability_unsupported"

_ALLOWED_SKIP_REASONS = frozenset({SKIP_STAGED_ARTIFACT_ABSENT, SKIP_POLICY_DISABLED, SKIP_CAPABILITY_UNSUPPORTED})

# Allowed parity_status values.
PARITY_OK = "parity_ok"
PARITY_DRIFT = "parity_drift"
PARITY_NA = "n/a"

# Stub runtime / model digest (Lane B/C will wire real values).
_STUB_DIGEST = "sha256:" + "0" * 64


@dataclass
class CacheBenchProof:
    """In-memory representation of the cache_bench_proof artifact.

    Mirrors schemas/core/cache_bench_proof.json. Serialised to JSON by
    ``to_dict()``. Validated by ``validate_cache_bench_proof.validate_proof()``
    from octomil-contracts before writing to disk.
    """

    cache_id: str
    capability: str
    measured_at: str  # ISO 8601 UTC
    skipped: bool
    runtime_digest: str
    model_digest: str
    adapter_version: str
    staged_artifact_ref: str
    skip_reason: Optional[str] = None
    # Metrics — null when skipped=True
    cold_p50_ms: Optional[float] = None
    cold_p95_ms: Optional[float] = None
    warm_p50_ms: Optional[float] = None
    warm_p95_ms: Optional[float] = None
    hit_ratio: Optional[float] = None
    entries: Optional[int] = None
    bytes_overhead: Optional[int] = None
    parity_status: str = PARITY_NA
    input_digest: Optional[str] = None
    # Writer block
    writer_process_kind: str = "python_sdk"
    writer_runtime_build_tag: str = "octomil-python:unknown"
    writer_pid: Optional[int] = None

    def __post_init__(self) -> None:
        if self.skipped and not self.skip_reason:
            raise ValueError(
                "CacheBenchProof: skipped=True requires skip_reason. "
                "NEVER emit a skipped proof without a reason — silent "
                "skips are indistinguishable from measurement failures."
            )
        if self.skip_reason and self.skip_reason not in _ALLOWED_SKIP_REASONS:
            raise ValueError(
                f"CacheBenchProof: skip_reason={self.skip_reason!r} is not a "
                f"valid enum value. Allowed: {sorted(_ALLOWED_SKIP_REASONS)}"
            )
        if not self.skipped:
            for field_name in ("cold_p50_ms", "cold_p95_ms", "warm_p50_ms", "warm_p95_ms", "hit_ratio"):
                val = getattr(self, field_name)
                if val is None:
                    raise ValueError(f"CacheBenchProof: skipped=False requires {field_name} to be set.")
                if not math.isfinite(val):  # type: ignore[arg-type]
                    raise ValueError(
                        f"CacheBenchProof: {field_name}={val!r} is not finite. "
                        "Bench metrics must be finite — a non-finite value "
                        "indicates a measurement error in the harness."
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical proof artifact dict."""
        d: dict[str, Any] = {
            "$schema_version": 1,
            "schema_version": 1,
            "cache_id": self.cache_id,
            "capability": self.capability,
            "measured_at": self.measured_at,
            "skipped": self.skipped,
            "runtime_digest": self.runtime_digest,
            "model_digest": self.model_digest,
            "adapter_version": self.adapter_version,
            "staged_artifact_ref": self.staged_artifact_ref,
            # Aggregate fields below are local to cache_bench_proof.json; per-sample
            # raw metrics come from cache.lookup_ms / cache.hit_total / cache.miss_total
            # in Lane A's canonical enum (contracts runtime_metric.json).
            "cold_p50_ms": self.cold_p50_ms,
            "cold_p95_ms": self.cold_p95_ms,
            "warm_p50_ms": self.warm_p50_ms,
            "warm_p95_ms": self.warm_p95_ms,
            "hit_ratio": self.hit_ratio,
            "entries": self.entries,
            "bytes_overhead": self.bytes_overhead,
            "parity_status": self.parity_status,
            "writer": {
                "process_kind": self.writer_process_kind,
                "runtime_build_tag": self.writer_runtime_build_tag,
                "pid": self.writer_pid,
            },
        }
        if self.skip_reason is not None:
            d["skip_reason"] = self.skip_reason
        if self.input_digest is not None:
            d["input_digest"] = self.input_digest
        # Remove null pid from writer to keep it clean
        if d["writer"]["pid"] is None:
            del d["writer"]["pid"]
        return d


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. Mirrors octomil/benchmark/runner.py."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


@dataclass
class CacheBenchConfig:
    """Configuration for a single cache bench run."""

    cache_id: str
    capability: str
    fixture_dir: Optional[Path]  # None → staged_artifact_absent
    output_dir: Path
    adapter_version: str = "0.1.11-stub"
    runtime_digest: str = _STUB_DIGEST
    model_digest: str = _STUB_DIGEST
    n_cold_iters: int = 10
    n_warm_iters: int = 50
    n_hit_iters: int = 20
    process_kind: str = "python_sdk"


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _writer_tag() -> str:
    try:
        from octomil import __version__ as v  # type: ignore[attr-defined]

        return f"octomil-python:{v}"
    except ImportError:
        return "octomil-python:unknown"


def run_cache_bench(
    config: CacheBenchConfig,
    adapter: Optional[CacheBenchAdapter] = None,
) -> CacheBenchProof:
    """Run one cache bench cycle and return the proof artifact.

    Anti-pattern guard: if ``config.fixture_dir`` is None or does not
    exist, returns a skipped proof with
    ``skip_reason="staged_artifact_absent"`` WITHOUT running any
    measurement. NEVER synthesise first-request measurements as a
    fallback.

    ``adapter`` defaults to ``StubCacheAdapter`` (Phase 1). Lane B/C
    pass a real adapter here.
    """
    staged_artifact_ref = str(config.fixture_dir) if config.fixture_dir else "/nonexistent"
    base_proof_kwargs: dict[str, Any] = {
        "cache_id": config.cache_id,
        "capability": config.capability,
        "measured_at": _utc_iso(),
        "runtime_digest": config.runtime_digest,
        "model_digest": config.model_digest,
        "adapter_version": config.adapter_version,
        "staged_artifact_ref": staged_artifact_ref,
        "writer_process_kind": config.process_kind,
        "writer_runtime_build_tag": _writer_tag(),
        "writer_pid": os.getpid(),
    }

    # ── Anti-pattern guard ──────────────────────────────────────────────
    # Absent fixture dir → skip, never live-measure.
    if config.fixture_dir is None or not config.fixture_dir.is_dir():
        proof = CacheBenchProof(
            **base_proof_kwargs,
            skipped=True,
            skip_reason=SKIP_STAGED_ARTIFACT_ABSENT,
        )
        _write_proof(proof, config.output_dir)
        return proof

    # ── Measurement path ────────────────────────────────────────────────
    if adapter is None:
        # Phase 1: StubCacheAdapter. TODO(lane-b/c): replace.
        adapter = StubCacheAdapter()

    fixture_set = StagedFixtureSet(
        fixture_dir=config.fixture_dir,
        cache_id=config.cache_id,
        capability=config.capability,
    )

    entries, bytes_overhead = adapter.populate(config.fixture_dir)

    # Cold phase.
    cold_samples: list[float] = []
    for i in range(config.n_cold_iters):
        _hit, latency_ms = adapter.cold_lookup(f"fixture_key_{i}")
        cold_samples.append(latency_ms)

    # Warm phase.
    warm_samples: list[float] = []
    for _ in range(config.n_warm_iters):
        _hit, latency_ms = adapter.warm_lookup("fixture_key_0")  # same key → hit path
        warm_samples.append(latency_ms)

    # Hit/miss ratio.
    hit_count = 0
    for i in range(config.n_hit_iters):
        key = "fixture_key_0" if i % 2 == 0 else f"miss_key_{i}"
        hit, _latency_ms = adapter.warm_lookup(key)
        if hit:
            hit_count += 1

    proof = CacheBenchProof(
        **base_proof_kwargs,
        skipped=False,
        cold_p50_ms=round(_percentile(cold_samples, 50), 4),
        cold_p95_ms=round(_percentile(cold_samples, 95), 4),
        warm_p50_ms=round(_percentile(warm_samples, 50), 4),
        warm_p95_ms=round(_percentile(warm_samples, 95), 4),
        hit_ratio=hit_count / max(config.n_hit_iters, 1),
        entries=entries,
        bytes_overhead=bytes_overhead,
        parity_status=PARITY_NA,
        input_digest=fixture_set.input_digest(),
    )
    _write_proof(proof, config.output_dir)
    return proof


def _write_proof(proof: CacheBenchProof, output_dir: Path) -> Path:
    """Write the proof artifact to ``output_dir/<cache_id>_proof.json``.

    Creates ``output_dir`` if it does not exist. Returns the written path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{proof.cache_id}_proof.json"
    doc = proof.to_dict()
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CacheBenchAdapter",
    "CacheBenchConfig",
    "CacheBenchProof",
    "PARITY_DRIFT",
    "PARITY_NA",
    "PARITY_OK",
    "SKIP_CAPABILITY_UNSUPPORTED",
    "SKIP_POLICY_DISABLED",
    "SKIP_STAGED_ARTIFACT_ABSENT",
    "StagedFixtureSet",
    "StubCacheAdapter",
    "run_cache_bench",
]
