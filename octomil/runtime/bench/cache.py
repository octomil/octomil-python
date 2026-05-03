"""On-disk cache for runtime selection bench results (v0.5).

Implements the contract from
``octomil-contracts/schemas/core/runtime_bench_*.json`` and the
architectural rules from ``strategy/runtime-selection-bench.md``:

  * Path-level cache key: ``<root>/<hardware_fingerprint_short>/
    <model_id>/<sha256(canonical_json(cache_key))>.json``. Path
    encodes the FULL structured key, not just model_digest, so two
    cache entries with the same model + different dispatch_shape
    don't collide (the reviewer P1 from the strategy review).
  * Canonical-JSON serialization: sorted keys, no whitespace, UTF-8.
    Cross-SDK contract — every binding MUST produce the same hash
    for the same key or cross-SDK reads diverge.
  * Atomic writes via temp-file-plus-rename.
  * Advisory file lock around writes (``fcntl.flock`` POSIX,
    ``msvcrt`` Windows). A second writer that fails to acquire
    the lock skips its bench and waits for the first writer's
    result by polling the file with bounded backoff.
  * Partial-read rejection: a JSON parse error or schema-version
    mismatch returns "no cache entry" (cache miss), never raises.
  * ``incomplete=true`` results are ignored on lookup. Operators can
    promote partials via the future CLI; until then a partial entry
    on disk is dispatch-invisible.
  * Index sidecar (``<model_dir>/index.json``) is rebuildable from
    leaves. Loss is non-fatal.

This module is **v0.5**: it owns reads and writes today. v1 (Layer
2a native runtime core) takes over the writer role with read-
compatible files. The cache directory namespace is
``runtime_bench_v0_5/`` so v1 can ignore stale entries cleanly
during the one-release migration window.

This module ships in PR A — cache R/W only. The bench harness
that produces ``Result`` values arrives in PR B.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants pinned in lockstep with the contract schemas.
# Bump in lockstep with schemas/core/runtime_bench_result.json.
# ---------------------------------------------------------------------------

#: On-disk schema version. Mirrors ``$schema_version`` / ``schema_version``
#: in the contract; readers treat any other value as a cache miss; writers
#: refuse to overwrite a file whose schema_version is newer than this.
CACHE_SCHEMA_VERSION = 1

#: Cache root directory name under ``$XDG_CACHE_HOME/octomil/``. Namespaced
#: with ``_v0_5`` so v1 (Layer 2a) can ignore stale entries cleanly when it
#: takes over the writer role. Hard cutover policy: the next major version
#: switches the directory name and removes the v0.5 reader after one
#: release window — no permanent dual-read path.
CACHE_DIR_NAME = "runtime_bench_v0_5"

#: Truncate the hardware fingerprint to this many hex chars for the path
#: component. Full fingerprint is recorded inside each result file for
#: reader-side equality checks.
_HARDWARE_FINGERPRINT_PATH_PREFIX_LEN = 16

#: How long to wait for another writer's bench when our own attempt to
#: acquire the file lock fails. Polled at the interval below; bounded so
#: a stuck writer doesn't block this process indefinitely.
_PEER_WRITE_WAIT_TIMEOUT_S = 60.0
_PEER_WRITE_POLL_INTERVAL_S = 0.5

#: Allowed capability values, kept in sync with the contract enum.
_CAPABILITIES = frozenset(("tts", "transcription", "chat", "embeddings", "realtime"))

#: Allowed quantization_preference values, kept in sync with the contract enum.
_QUANTIZATION_PREFERENCES = frozenset(("fp32", "fp16", "bf16", "int8", "int4", "auto"))

#: model_digest must be sha256:<64 hex>.
_MODEL_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}")

#: candidate_set_version + reference_workload_version are semver-shaped.
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+(\.[0-9]+)?")


def _sanitize_path_component(component: str) -> str:
    """Make ``component`` safe to use as a single filesystem path
    segment.

    The cache layout uses ``model_id`` as a directory name. Reviewer P1
    from the engineering-debate session: raw ``model_id`` joined into
    ``Path()`` has two real failure modes —

      * Hugging Face-style ids like ``"Qwen/Qwen3-0.6B"`` nest into
        sub-directories. ``list_models()`` then returns ``["Qwen"]``,
        not ``["Qwen/Qwen3-0.6B"]``; CLI / observability surfaces lie
        about what's cached.
      * Absolute paths or ``..`` components escape ``cache_root``
        entirely. ``Path("a") / "/tmp/foo"`` is ``Path("/tmp/foo")``
        per pathlib semantics (absolute RHS replaces LHS). A planner-
        emitted or attacker-crafted ``model_id`` like ``/tmp/foo`` or
        ``../../../etc/passwd`` writes outside the cache namespace.

    Sanitization rule: URL-encode (RFC 3986 reserved + path-unsafe
    chars). The encoded form is reversible (debugging friendly), keeps
    the human-readable model_id as a single path component, and is
    safe for every supported filesystem because URL-encoding the
    reserved set leaves only ``[A-Za-z0-9._~%-]``. The reverse
    transform isn't load-bearing for cache identity — the index
    sidecar carries the original ``model_id`` in the cache_key body.

    Empty input is a programming error caught at ``CacheKey``
    construction; this helper returns ``""`` defensively.
    """
    if not component:
        return ""
    # urllib.parse.quote with safe="" encodes EVERYTHING except the
    # unreserved set [A-Za-z0-9._~-]. Forward slashes, backslashes,
    # NUL bytes, and absolute-path roots all get percent-encoded into
    # a single-segment string. Hyphens / underscores / tildes stay
    # readable so ``"kokoro-en-v0_19"`` doesn't get mangled.
    encoded = urllib.parse.quote(component, safe="")
    # The ``.`` is in the unreserved set so urllib.parse.quote won't
    # encode it. That leaves ``"."`` and ``".."`` as path-traversal
    # holes that survive encoding. Defensively re-encode ``.`` so the
    # encoded form NEVER equals "." or ".." or starts with "." (hidden
    # file). Cheap; only fires on the few literal-dot edge cases.
    if encoded in (".", "..") or encoded.startswith("."):
        encoded = encoded.replace(".", "%2E")
    return encoded


# ---------------------------------------------------------------------------
# Dataclasses mirroring the contract schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchShape:
    """Capability-specific dispatch parameters that change which engine
    config is optimal. Embedded in ``CacheKey``.

    Stored as a free-form ``dict`` because the schema is ``oneOf`` over
    five capability-specific shapes (``runtime_bench_dispatch_shape.json``).
    Validation against the per-capability subschema lives in PR B's bench
    harness when we have a real workload to validate; this PR is cache
    R/W only and treats the shape as an opaque hashable payload that
    round-trips through canonical JSON.
    """

    fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # Re-sort to a deterministic shape so subsequent canonical-JSON
        # passes don't re-shuffle.
        return _sort_keys_recursive(self.fields)


@dataclass(frozen=True)
class CacheKey:
    """Structured cache key. Canonical-JSON of this object is hashed to
    produce the leaf filename.

    Mirrors ``schemas/core/runtime_bench_cache_key.json``. Ordering and
    field names match the schema verbatim; cross-SDK readers depend on
    the canonical JSON byte sequence being identical.
    """

    capability: str
    model_id: str
    model_digest: str
    quantization_preference: str
    candidate_set_version: str
    reference_workload_version: str
    dispatch_shape: DispatchShape

    def __post_init__(self) -> None:
        if self.capability not in _CAPABILITIES:
            raise ValueError(f"capability must be one of {sorted(_CAPABILITIES)}; got {self.capability!r}")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        # Reviewer DiD: prefix+length only would let "sha256:" + "z"*64
        # pass Python validation but fail schema validation. Tighten
        # to the exact hex shape the schema requires.
        if not _MODEL_DIGEST_RE.fullmatch(self.model_digest):
            raise ValueError(f"model_digest must be 'sha256:<64 lowercase hex>' shape; got {self.model_digest!r}")
        if self.quantization_preference not in _QUANTIZATION_PREFERENCES:
            raise ValueError(
                f"quantization_preference must be one of {sorted(_QUANTIZATION_PREFERENCES)}; "
                f"got {self.quantization_preference!r}"
            )
        if not _VERSION_RE.fullmatch(self.candidate_set_version):
            raise ValueError(
                f"candidate_set_version must be semver-shaped (X.Y or X.Y.Z); " f"got {self.candidate_set_version!r}"
            )
        if not _VERSION_RE.fullmatch(self.reference_workload_version):
            raise ValueError(
                f"reference_workload_version must be semver-shaped (X.Y or X.Y.Z); "
                f"got {self.reference_workload_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the schema-canonical dict form. Ordering is alphabetical
        at every level — ``canonical_json`` re-sorts unconditionally so
        callers who muck with the order don't break the hash."""
        return {
            "candidate_set_version": self.candidate_set_version,
            "capability": self.capability,
            "dispatch_shape": self.dispatch_shape.to_dict(),
            "model_digest": self.model_digest,
            "model_id": self.model_id,
            "quantization_preference": self.quantization_preference,
            "reference_workload_version": self.reference_workload_version,
        }

    def canonical_json(self) -> bytes:
        """Canonical-JSON byte sequence used for the SHA256 leaf-name hash.

        Cross-SDK contract: sorted keys at every nesting level, no
        whitespace, UTF-8, no trailing newline. Every SDK binding MUST
        implement this identically or cross-SDK cache reads diverge.
        """
        return _canonical_json_bytes(self.to_dict())

    def leaf_filename(self) -> str:
        """``<sha256-hex>.json`` of the canonical-JSON. The cache leaf
        filename used at the on-disk path. Strategy doc P1 fix: full
        cache_key, not just model_digest, encoded in the path."""
        return hashlib.sha256(self.canonical_json()).hexdigest() + ".json"


@dataclass(frozen=True)
class HardwareFingerprint:
    """Device fingerprint that scopes a cache. Matches the
    ``hardware_fingerprint`` SHA256 over the canonical-JSON of the
    ``hardware_descriptor`` fields.

    The strategy doc lists machine + processor + cpu_count + ram_gb +
    os_version + runtime_build_tag. v0.5 fills these from
    ``platform`` + a best-effort RAM probe + the SDK version; the
    truncated digest forms the directory component on disk.
    """

    machine: str
    processor: str
    cpu_count: int
    ram_gb: int
    os_version: str
    runtime_build_tag: str

    def descriptor_dict(self) -> dict[str, Any]:
        """The fields that go into the result file's
        ``hardware_descriptor`` block."""
        return {
            "cpu_count": self.cpu_count,
            "machine": self.machine,
            "os_version": self.os_version,
            "processor": self.processor,
            "ram_gb": self.ram_gb,
        }

    def fingerprint_dict(self) -> dict[str, Any]:
        """The exact tuple that's hashed. ``runtime_build_tag`` is
        included here (not in ``descriptor_dict``) because it's a
        cache-key-relevant field, not a human-readable descriptor."""
        return {
            "cpu_count": self.cpu_count,
            "machine": self.machine,
            "os_version": self.os_version,
            "processor": self.processor,
            "ram_gb": self.ram_gb,
            "runtime_build_tag": self.runtime_build_tag,
        }

    def full_digest(self) -> str:
        """``sha256:<64 hex>`` over the canonical-JSON fingerprint dict.
        Stored in the result file's ``hardware_fingerprint`` field for
        reader-side equality checks."""
        return "sha256:" + hashlib.sha256(_canonical_json_bytes(self.fingerprint_dict())).hexdigest()

    def path_component(self) -> str:
        """Truncated digest used as the on-disk directory name. Two
        devices with the same fingerprint share a directory tree
        (useful for fleet pre-warming)."""
        digest = self.full_digest()
        # Strip the "sha256:" prefix; truncate the hex tail.
        hex_part = digest.split(":", 1)[1]
        return hex_part[:_HARDWARE_FINGERPRINT_PATH_PREFIX_LEN]

    @classmethod
    def detect(cls, *, runtime_build_tag: str) -> "HardwareFingerprint":
        """Best-effort runtime probe.

        Designed to never raise — a probe failure on any field returns
        a sentinel that still produces a valid (if less-discriminating)
        fingerprint. Degraded fingerprints just mean two genuinely
        different devices may share a cache slot, which is a perf
        regression at worst, never a correctness regression (the
        result file's full descriptor is checked at read time).
        """
        try:
            machine = platform.machine() or "unknown"
        except Exception:  # pragma: no cover — platform should always work
            machine = "unknown"
        try:
            processor = platform.processor() or platform.machine() or "unknown"
        except Exception:  # pragma: no cover
            processor = "unknown"
        cpu_count = os.cpu_count() or 1
        ram_gb = _detect_ram_gb_rounded()
        os_version = _detect_os_version()
        return cls(
            machine=machine,
            processor=processor,
            cpu_count=int(cpu_count),
            ram_gb=int(ram_gb),
            os_version=os_version,
            runtime_build_tag=runtime_build_tag,
        )


@dataclass(frozen=True)
class Winner:
    """Committed-winner payload. Mirrors ``measured_candidate`` in the
    contract schema. Free-form ``config``/``quality_metrics`` because
    those are engine-specific."""

    engine: str
    config: dict[str, Any]
    score: float
    provider: Optional[str] = None
    first_chunk_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "engine": self.engine,
            "config": _sort_keys_recursive(self.config),
            "score": self.score,
        }
        if self.provider is not None:
            out["provider"] = self.provider
        if self.first_chunk_ms is not None:
            out["first_chunk_ms"] = self.first_chunk_ms
        if self.total_latency_ms is not None:
            out["total_latency_ms"] = self.total_latency_ms
        if self.quality_metrics:
            out["quality_metrics"] = _sort_keys_recursive(self.quality_metrics)
        return out


@dataclass(frozen=True)
class Result:
    """On-disk cache result. Mirrors ``runtime_bench_result.json``.

    Commit invariant (enforced at write time, reflected by the
    ``incomplete`` field):

      - ``incomplete=False`` REQUIRES ``winner is not None``.
      - ``incomplete=True`` REQUIRES ``winner is None``.

    Lookup ignores ``incomplete=True`` entries. Operators promote
    partials via the future CLI (``octomil bench accept-partial``);
    until then they're dispatch-invisible.
    """

    cache_key: CacheKey
    hardware_fingerprint: str
    hardware_descriptor: dict[str, Any]
    writer_runtime_build_tag: str
    writer_process_kind: str = "python_sdk"
    writer_pid: Optional[int] = None
    incomplete: bool = False
    winner: Optional[Winner] = None
    runners_up: tuple[Winner, ...] = ()
    partial_observations: tuple[Winner, ...] = ()
    disqualified: tuple[dict[str, Any], ...] = ()
    confidence: str = "high"
    created_at: Optional[str] = None  # filled at write time when None

    def __post_init__(self) -> None:
        if self.incomplete and self.winner is not None:
            raise ValueError("Result.incomplete=True forbids a winner; partial_observations is the right field")
        if not self.incomplete and self.winner is None:
            raise ValueError("Result.incomplete=False requires a winner; mark incomplete=True or supply one")
        if self.confidence not in ("high", "low"):
            raise ValueError(f"confidence must be 'high' or 'low'; got {self.confidence!r}")

    def to_dict(self, *, created_at: str) -> dict[str, Any]:
        """Schema-canonical dict for on-disk serialization. ``created_at``
        is supplied by the writer so a single ``time.time()`` call
        timestamps every Result written in one bench cycle."""
        out: dict[str, Any] = {
            "$schema_version": CACHE_SCHEMA_VERSION,
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": self.cache_key.to_dict(),
            "confidence": self.confidence,
            "created_at": created_at,
            "disqualified": [_sort_keys_recursive(d) for d in self.disqualified],
            "hardware_descriptor": _sort_keys_recursive(self.hardware_descriptor),
            "hardware_fingerprint": self.hardware_fingerprint,
            "incomplete": self.incomplete,
            "writer": _writer_dict(self.writer_runtime_build_tag, self.writer_process_kind, self.writer_pid),
        }
        if self.winner is not None:
            out["winner"] = self.winner.to_dict()
        if self.runners_up:
            out["runners_up"] = [w.to_dict() for w in self.runners_up]
        if self.partial_observations:
            out["partial_observations"] = [w.to_dict() for w in self.partial_observations]
        return out


# ---------------------------------------------------------------------------
# Cross-SDK canonical-JSON serialization
# ---------------------------------------------------------------------------


def _sort_keys_recursive(value: Any) -> Any:
    """Recursively sort dict keys at every nesting level. Lists keep
    their element order. Used by canonical-JSON serialization and by
    the ``to_dict`` methods so a single normalization pass produces
    a stable shape."""
    if isinstance(value, dict):
        return {k: _sort_keys_recursive(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_sort_keys_recursive(v) for v in value]
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    """Cross-SDK canonical-JSON: sorted keys, no whitespace, UTF-8,
    no trailing newline. The byte sequence MUST be identical across
    every SDK binding for the same payload — that's the cross-SDK
    cache contract.

    ``ensure_ascii=False`` so non-ASCII strings (model ids in CJK,
    voice family names with diacritics) hash byte-identically to the
    UTF-8 source text, not to escape sequences. The
    ``runtime_bench_cache_key.json`` ``$comment`` block pins this
    rule on the contract side.
    """
    return json.dumps(
        _sort_keys_recursive(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Hardware probe helpers
# ---------------------------------------------------------------------------


def _detect_ram_gb_rounded() -> int:
    """Best-effort total-RAM probe rounded to the nearest 2GB. The
    rounding is deliberate: 16 vs 17 GB doesn't change cache identity,
    but 16 vs 32 does. Strategy doc explicitly calls for tier-bucketing.

    Returns at least ``1`` even if probing fails — the schema requires
    ``ram_gb >= 1`` (DiD from prior debate session). Sandboxed CPython
    environments where both sysconf and sysctl fail otherwise produce
    a Result that fails its own contract; floor at 1 keeps the cache
    write valid even when the fingerprint is degraded.
    """
    bytes_total: Optional[int] = None
    # Linux + most Unixen
    try:
        bytes_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        bytes_total = None
    if bytes_total is None:
        # macOS + BSD
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                bytes_total = int(out.stdout.strip())
        except Exception:  # pragma: no cover — platform-dependent probe
            bytes_total = None
    if bytes_total is None or bytes_total <= 0:
        # DiD: schema requires ram_gb >= 1; degraded fingerprint
        # should still write a valid Result.
        return 1
    gb = bytes_total // (1024 * 1024 * 1024)
    # Round to nearest 2GB — bucketing per the strategy doc.
    # Floor at 1 to satisfy the schema even on tiny/sandboxed boxes.
    return max(1, ((gb + 1) // 2) * 2)


def _detect_os_version() -> str:
    """OS major.minor descriptor. Strategy doc's invalidation rule is
    "OS major version change clears the cache" — we record enough
    granularity that readers can decide what counts as a major bump.

    macOS: ``"macOS <release>"`` (e.g. ``macOS 15.1``).
    Linux: distro + version when readable via ``platform.freedesktop_os_release``.
    Other: bare ``platform.system() + platform.release()``.
    """
    try:
        if sys.platform == "darwin":
            try:
                ver, _, _ = platform.mac_ver()
                if ver:
                    return f"macOS {ver}"
            except Exception:  # pragma: no cover
                pass
        if sys.platform.startswith("linux"):
            try:
                # Python 3.10+: structured os-release reader.
                info = platform.freedesktop_os_release()  # type: ignore[attr-defined]
                name = info.get("NAME") or info.get("ID") or "Linux"
                ver = info.get("VERSION_ID") or info.get("VERSION") or ""
                return f"{name} {ver}".strip()
            except (AttributeError, OSError):
                pass
        return f"{platform.system()} {platform.release()}".strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _writer_dict(runtime_build_tag: str, process_kind: str, pid: Optional[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "process_kind": process_kind,
        "runtime_build_tag": runtime_build_tag,
    }
    if pid is not None:
        out["pid"] = pid
    return out


# ---------------------------------------------------------------------------
# Atomic write + advisory file-lock primitives
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, contents: str) -> None:
    """Atomic UTF-8 write via temp-file-plus-rename.

    Concurrency rule from the strategy doc: writers serialize to
    ``<path>.<pid>.<ns>.tmp`` and ``os.replace`` atomically. POSIX
    rename is atomic; ``os.replace`` is documented atomic on Windows
    when source and dest are on the same filesystem. Readers see
    either the old file or the new file, never a half-written
    intermediate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(contents)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@contextlib.contextmanager
def _try_writer_lock(lock_path: Path, *, blocking: bool = False) -> Iterator[bool]:
    """Acquire an advisory exclusive lock on ``lock_path``.

    Two modes:

      * ``blocking=False`` (default) — non-blocking. Yields ``True``
        when the lock was acquired (caller is the designated writer),
        ``False`` otherwise (some other process is benching this same
        key — caller should skip its own bench and poll for the
        result file).
      * ``blocking=True`` — wait until the lock is acquired. Always
        yields ``True``. Used for short-held locks where a peer would
        finish quickly anyway (e.g. the per-model index lock that
        only wraps a read-modify-write transaction).

    Lock is released on context exit regardless.

    POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``.
    Both are advisory + OS-level + process-level, so two SDK
    bindings on the same machine (Python + iOS app) coordinate
    correctly. Same-process-multi-thread coordination needs an
    in-process mutex on top — out of scope for this PR; the bench
    runner in PR B owns thread-safety.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        if sys.platform == "win32":  # pragma: no cover — POSIX is the test platform
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(fh.fileno(), mode, 1)
                acquired = True
            except OSError:
                acquired = False
            try:
                yield acquired
            finally:
                if acquired:
                    with contextlib.suppress(OSError):
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(fh.fileno(), flags)
                acquired = True
            except OSError:
                acquired = False
            try:
                yield acquired
            finally:
                if acquired:
                    with contextlib.suppress(OSError):
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Cache root + path resolution
# ---------------------------------------------------------------------------


def default_cache_root(*, env: Optional[dict[str, str]] = None) -> Path:
    """Resolve the cache root the strategy doc pins.

    Layout: ``$XDG_CACHE_HOME/octomil/runtime_bench_v0_5/`` or
    ``$HOME/.cache/octomil/runtime_bench_v0_5/`` when
    ``XDG_CACHE_HOME`` is unset. The ``v0_5`` namespace lives in
    :data:`CACHE_DIR_NAME` so v1 can ignore stale entries cleanly
    when Layer 2a takes over.

    ``env`` is injectable for tests; defaults to ``os.environ``.
    """
    e = os.environ if env is None else env
    xdg = e.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path(e.get("HOME", os.path.expanduser("~"))) / ".cache"
    return base / "octomil" / CACHE_DIR_NAME


# ---------------------------------------------------------------------------
# CacheStore — the public surface
# ---------------------------------------------------------------------------


class CacheStore:
    """Read / write / list / clear API over the on-disk runtime bench
    cache.

    The store is **stateless** beyond the cache root — every call
    re-reads from disk so cross-process changes are visible immediately.
    Concurrent access is OS-coordinated (advisory file locks during
    writes, atomic temp-file-plus-rename publishing).

    Typical use (in PR B's bench runner):

    .. code-block:: python

        store = CacheStore.default(runtime_build_tag="octomil-python:4.16.1")
        result = store.get(cache_key, hardware)  # → Result | None
        if result is None:
            # bench harness measures, then calls store.put(...)
            ...
    """

    def __init__(
        self,
        *,
        cache_root: Path,
        hardware: HardwareFingerprint,
    ) -> None:
        self._cache_root = cache_root
        self._hardware = hardware

    # --------- Construction helpers ---------

    @classmethod
    def default(cls, *, runtime_build_tag: str) -> "CacheStore":
        """Construct against the default cache root and a freshly-detected
        hardware fingerprint."""
        return cls(
            cache_root=default_cache_root(),
            hardware=HardwareFingerprint.detect(runtime_build_tag=runtime_build_tag),
        )

    # --------- Path resolution ---------

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    @property
    def hardware(self) -> HardwareFingerprint:
        return self._hardware

    def _hardware_dir(self) -> Path:
        return self._cache_root / self._hardware.path_component()

    def _model_dir(self, cache_key: CacheKey) -> Path:
        # Sanitize model_id before using as a path component. Reviewer
        # P1 from the engineering-debate session: raw model_id allows
        # HF-style nesting ("Qwen/Qwen3-0.6B") AND path-traversal
        # escape ("/tmp/foo" or "..").
        return self._model_dir_for_id(cache_key.model_id)

    def _model_dir_for_id(self, model_id: str) -> Path:
        return self._hardware_dir() / _sanitize_path_component(model_id)

    def _leaf_path(self, cache_key: CacheKey) -> Path:
        return self._model_dir(cache_key) / cache_key.leaf_filename()

    def _index_path(self, cache_key: CacheKey) -> Path:
        return self._model_dir(cache_key) / "index.json"

    def _index_path_for_id(self, model_id: str) -> Path:
        return self._model_dir_for_id(model_id) / "index.json"

    def _index_lock_path(self, cache_key: CacheKey) -> Path:
        # Per-model lock for the index read-modify-write transaction.
        # Reviewer DiD: the per-leaf write lock doesn't serialize
        # concurrent puts under the SAME model — both writers would
        # happily read-modify-write the same index.json and lose one
        # entry. Separate lock, held only during index update.
        return self._model_dir(cache_key) / "index.json.lock"

    def _lock_path(self, cache_key: CacheKey) -> Path:
        return self._model_dir(cache_key) / (cache_key.leaf_filename() + ".lock")

    # --------- Read ---------

    def get(self, cache_key: CacheKey) -> Optional[Result]:
        """Return the cached result for ``cache_key`` or ``None`` on miss.

        Cache miss conditions (all return ``None`` rather than raise):

          * Leaf file does not exist.
          * Leaf file is malformed JSON.
          * Leaf file's ``$schema_version`` differs from the constant.
          * Leaf file declares ``incomplete=True``. Partial entries
            are dispatch-invisible until promoted via the operator
            CLI (PR D).
          * Leaf file's recomputed cache_key digest doesn't match its
            filename (corruption / writer bug).
          * Leaf file's ``hardware_fingerprint`` differs from the
            store's hardware. (Two devices that hash to the same path
            prefix can happen with the truncation; the inner check
            ensures we don't return another device's winner.)
        """
        path = self._leaf_path(cache_key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(
                "runtime_bench cache: malformed leaf at %s (%s); treating as cache miss",
                path,
                exc,
            )
            return None
        return self._validate_and_parse(data, expected_filename=path.name, cache_key=cache_key)

    def _validate_and_parse(
        self,
        data: Any,
        *,
        expected_filename: str,
        cache_key: CacheKey,
    ) -> Optional[Result]:
        if not isinstance(data, dict):
            return None
        # Schema-version mismatch is a cache miss, never a crash.
        if data.get("$schema_version") != CACHE_SCHEMA_VERSION:
            logger.debug(
                "runtime_bench cache: schema_version mismatch at %s (expected %d, got %r)",
                expected_filename,
                CACHE_SCHEMA_VERSION,
                data.get("$schema_version"),
            )
            return None
        # Incomplete entries are not consulted.
        if data.get("incomplete", False):
            return None
        # Leaf-name integrity check: the cache_key in the body must
        # match the requested cache_key. A corrupt write that landed
        # under the wrong filename surfaces as a cache miss here.
        body_cache_key = data.get("cache_key")
        if not isinstance(body_cache_key, dict):
            return None
        try:
            recomputed_filename = hashlib.sha256(_canonical_json_bytes(body_cache_key)).hexdigest() + ".json"
        except Exception:
            return None
        if recomputed_filename != expected_filename:
            logger.warning(
                "runtime_bench cache: filename %s does not match its body's cache_key hash (%s); treating as miss",
                expected_filename,
                recomputed_filename,
            )
            return None
        # Hardware-scope check: defense-in-depth against truncated
        # path-prefix collisions across distinct devices.
        if data.get("hardware_fingerprint") != self._hardware.full_digest():
            logger.debug(
                "runtime_bench cache: hardware_fingerprint mismatch at %s; treating as miss",
                expected_filename,
            )
            return None
        # Reconstruct the typed Result. Failures here also surface as
        # cache miss — any unrecognized shape from a future writer is
        # safe to ignore.
        try:
            return _result_from_dict(data, cache_key=cache_key)
        except Exception as exc:
            logger.debug(
                "runtime_bench cache: leaf at %s did not parse into Result (%s); treating as miss",
                expected_filename,
                exc,
            )
            return None

    # --------- Write ---------

    def put(self, result: Result) -> Path:
        """Atomically write ``result`` to the cache and update the index.

        Returns the leaf path on success. Raises on:

          * Filesystem write failures (parent directory creation, etc.).
          * ``Result`` validation failures (already enforced in
            ``Result.__post_init__``).

        Concurrency: the writer takes an advisory lock on the leaf's
        ``.lock`` sidecar before publishing. A peer writer that holds
        the lock causes this call to **wait** for the peer's result
        (not run its own write); after the wait, the peer's result is
        the published winner. Bounded by
        :data:`_PEER_WRITE_WAIT_TIMEOUT_S` so a stuck peer doesn't
        block this caller indefinitely — on timeout we publish our
        own result on top.

        ``incomplete=True`` results ARE written (so a partial
        observation can survive a process restart for resume / for
        the operator's ``bench accept-partial`` CLI), but lookup
        ignores them.
        """
        if result.cache_key.leaf_filename() not in {
            result.cache_key.leaf_filename(),
        }:  # pragma: no cover — defensive; canonicalization is deterministic
            raise RuntimeError("cache_key produced an unstable leaf filename")
        leaf = self._leaf_path(result.cache_key)
        lock = self._lock_path(result.cache_key)
        leaf.parent.mkdir(parents=True, exist_ok=True)
        with _try_writer_lock(lock) as acquired:
            if not acquired:
                # A peer is benching the same key. Wait for their
                # result rather than racing it.
                if self._wait_for_peer_result(leaf):
                    return leaf
                # Peer timed out. Fall through and write our own.
                logger.warning(
                    "runtime_bench cache: peer write at %s timed out; publishing own result",
                    leaf,
                )
            # We hold the lock (or fell through after waiting). Publish
            # the result file atomically and update the index.
            created_at = result.created_at or _utc_now_iso()
            payload = result.to_dict(created_at=created_at)
            _atomic_write_text(
                leaf,
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            )
            self._update_index(result.cache_key, leaf, payload)
        return leaf

    def _wait_for_peer_result(self, leaf: Path) -> bool:
        """Poll for a peer-written leaf until it appears or the timeout
        elapses. Returns True on success."""
        deadline = time.monotonic() + _PEER_WRITE_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            if leaf.is_file():
                return True
            time.sleep(_PEER_WRITE_POLL_INTERVAL_S)
        return False

    # --------- Index ---------

    def _update_index(self, cache_key: CacheKey, leaf: Path, payload: dict[str, Any]) -> None:
        """Append / replace this cache_key's entry in the per-model index.

        Concurrency: the index is shared across cache_keys of the same
        model, so per-leaf write locks are insufficient. Take a
        per-model index lock around the read-modify-write transaction
        so two concurrent puts (different cache_keys, same model) don't
        race and lose one entry. Reviewer DiD from the engineering-
        debate session.

        Index loss is non-fatal — callers that need the index can call
        :meth:`rebuild_index` to scan leaves. We update on every put
        anyway so that ``octomil bench list`` (PR D) can answer
        questions without parsing every leaf JSON.
        """
        index_path = self._index_path(cache_key)
        index_lock = self._index_lock_path(cache_key)
        # Per-model index lock. Block (not non-blocking) so concurrent
        # writers of different cache_keys serialize on the index
        # update; the actual bench / leaf-write happened outside this
        # lock and isn't blocked by it.
        with _try_writer_lock(index_lock, blocking=True):
            try:
                index = self._read_index(cache_key)
            except Exception:
                index = {
                    "$schema_version": CACHE_SCHEMA_VERSION,
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "model_id": cache_key.model_id,
                    "entries": [],
                }
            # Replace any existing entry for the same leaf filename.
            entries = [e for e in index.get("entries", []) if e.get("leaf_filename") != leaf.name]
            entries.append(
                {
                    "cache_key": payload["cache_key"],
                    "created_at": payload["created_at"],
                    "incomplete": bool(payload.get("incomplete", False)),
                    "leaf_filename": leaf.name,
                    "winner_summary": _winner_summary(payload),
                }
            )
            # Sort entries deterministically so a checked-in fixture
            # comparison (or two writes from different SDKs) produces
            # byte-identical index files.
            entries.sort(key=lambda e: e["leaf_filename"])
            index["entries"] = entries
            try:
                _atomic_write_text(
                    index_path,
                    json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False),
                )
            except OSError as exc:
                logger.warning(
                    "runtime_bench cache: failed to update index at %s (%s); leaf still wrote successfully",
                    index_path,
                    exc,
                )

    def _read_index(self, cache_key: CacheKey) -> dict[str, Any]:
        """Read the per-model index. Returns an empty index on miss /
        malformed file (loss is non-fatal)."""
        path = self._index_path(cache_key)
        if not path.is_file():
            return {
                "$schema_version": CACHE_SCHEMA_VERSION,
                "schema_version": CACHE_SCHEMA_VERSION,
                "model_id": cache_key.model_id,
                "entries": [],
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "$schema_version": CACHE_SCHEMA_VERSION,
                "schema_version": CACHE_SCHEMA_VERSION,
                "model_id": cache_key.model_id,
                "entries": [],
            }
        if not isinstance(data, dict) or data.get("$schema_version") != CACHE_SCHEMA_VERSION:
            return {
                "$schema_version": CACHE_SCHEMA_VERSION,
                "schema_version": CACHE_SCHEMA_VERSION,
                "model_id": cache_key.model_id,
                "entries": [],
            }
        return data

    def rebuild_index(self, *, model_id: str) -> int:
        """Scan every leaf for ``model_id`` and rewrite its index from
        scratch. Returns the number of entries materialized.

        Use case: a corrupt index after a crash, or after a manual
        ``rm`` of leaf files that left the index pointing at ghosts.
        """
        # Need a representative cache_key for the model dir resolution.
        # Construct a synthetic key just for path math; not used for hashing.
        synthetic = _path_only_cache_key(model_id)
        model_dir = self._model_dir(synthetic)
        if not model_dir.is_dir():
            return 0
        entries: list[dict[str, Any]] = []
        for leaf in sorted(model_dir.glob("*.json")):
            if leaf.name == "index.json":
                continue
            try:
                data = json.loads(leaf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("$schema_version") != CACHE_SCHEMA_VERSION:
                continue
            # DiD: the per-fp directory layout means foreign-fp leaves
            # SHOULD never appear here. But operator manual `mv`s,
            # cross-machine cache copies, and runtime-build-tag
            # mismatches can put a wrong-fp leaf in this dir.
            # Refuse to materialize entries from a foreign fingerprint.
            leaf_fp = data.get("hardware_fingerprint")
            if leaf_fp and leaf_fp != self._hardware.full_digest():
                logger.debug(
                    "runtime_bench cache: skipping leaf %s with foreign hardware_fingerprint %s",
                    leaf.name,
                    leaf_fp,
                )
                continue
            body_cache_key = data.get("cache_key")
            if not isinstance(body_cache_key, dict):
                continue
            entries.append(
                {
                    "cache_key": body_cache_key,
                    "created_at": data.get("created_at", ""),
                    "incomplete": bool(data.get("incomplete", False)),
                    "leaf_filename": leaf.name,
                    "winner_summary": _winner_summary(data),
                }
            )
        entries.sort(key=lambda e: e["leaf_filename"])
        index = {
            "$schema_version": CACHE_SCHEMA_VERSION,
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_id": model_id,
            "entries": entries,
        }
        index_path = model_dir / "index.json"
        _atomic_write_text(
            index_path,
            json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False),
        )
        return len(entries)

    # --------- List / clear ---------

    def list_models(self) -> list[str]:
        """Return the original (un-sanitized) ``model_id`` for every
        cache entry under the current hardware fingerprint.

        Reviewer R1 (Codex): an earlier draft returned the
        URL-encoded directory name, then ``clear_model(model_id)``
        re-sanitized that already-encoded string, doubly-encoding
        the percent sign and writing/looking up under a totally
        different path. Round-trip broken.

        Correct shape:
          * The index sidecar at ``<model_dir>/index.json`` carries
            the original ``model_id`` in its body (and again in each
            entry's nested ``cache_key``). Read that as the source
            of truth.
          * Fallback when the index is missing or unreadable:
            URL-decode the directory name. The defensive
            ``%2E``-for-``.`` re-encoding makes the decoded form
            indistinguishable from the original — the only
            difference is empty-string and ``.``/`..`` cases that
            cannot occur in real ``model_id``s anyway.
        """
        hw_dir = self._hardware_dir()
        if not hw_dir.is_dir():
            return []
        out: list[str] = []
        for p in sorted(hw_dir.iterdir()):
            if not p.is_dir():
                continue
            original = self._read_original_model_id(p)
            if original is None:
                # Index missing/unreadable; fall back to decode.
                original = urllib.parse.unquote(p.name)
            out.append(original)
        return sorted(out)

    def _read_original_model_id(self, model_dir: Path) -> Optional[str]:
        """Pull the un-sanitized ``model_id`` out of the model's index
        sidecar. Returns ``None`` if the index is absent or
        unreadable; callers fall back to URL-decoding the directory
        name."""
        index_path = model_dir / "index.json"
        if not index_path.is_file():
            return None
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        model_id = data.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
        return None

    def list_cache_keys(self, *, model_id: str) -> list[dict[str, Any]]:
        """Return the index entries for ``model_id``. Empty list when
        no entries exist."""
        synthetic = _path_only_cache_key(model_id)
        return list(self._read_index(synthetic).get("entries", []))

    def clear_model(self, *, model_id: str) -> int:
        """Delete every leaf + index file for ``model_id`` under the
        current hardware fingerprint. Returns the number of leaf
        files removed.

        Used by the ``octomil bench reset <model>`` CLI in PR D.
        """
        synthetic = _path_only_cache_key(model_id)
        model_dir = self._model_dir(synthetic)
        if not model_dir.is_dir():
            return 0
        removed = 0
        for child in list(model_dir.iterdir()):
            if child.suffix == ".json" or child.suffix == ".lock":
                with contextlib.suppress(OSError):
                    child.unlink()
                if child.suffix == ".json" and child.name != "index.json":
                    removed += 1
        # Best-effort prune of empty directory.
        with contextlib.suppress(OSError):
            model_dir.rmdir()
        return removed

    def clear_all(self) -> None:
        """Delete every cache entry under the current hardware
        fingerprint. ``octomil bench reset --all`` calls this in PR D."""
        for model_id in self.list_models():
            self.clear_model(model_id=model_id)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _path_only_cache_key(model_id: str) -> CacheKey:
    """Construct a CacheKey suitable for path resolution only.

    Used by ``rebuild_index`` / ``list_cache_keys`` / ``clear_model``
    where only the model_dir path matters; the synthetic key's hash
    is never used. Keeps the path-resolution methods uniform without
    forcing callers to fabricate a real key.
    """
    return CacheKey(
        capability="tts",
        model_id=model_id,
        model_digest="sha256:" + "0" * 64,
        quantization_preference="auto",
        candidate_set_version="0.0",
        reference_workload_version="0.0",
        dispatch_shape=DispatchShape(fields={}),
    )


def _winner_summary(payload: dict[str, Any]) -> str:
    """Human-readable winner string for the index. Empty for incomplete
    results."""
    if payload.get("incomplete", False):
        return ""
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        return ""
    parts = [str(winner.get("engine", "?"))]
    provider = winner.get("provider")
    if provider:
        parts.append(f"+ {provider}")
    config = winner.get("config")
    if isinstance(config, dict) and config:
        # First two config items, terse.
        snippet = ", ".join(f"{k}={config[k]}" for k in list(config)[:2])
        parts.append(f"({snippet})")
    return " ".join(parts)


def _result_from_dict(data: dict[str, Any], *, cache_key: CacheKey) -> Result:
    """Reconstruct a Result from a dict. Strict — any shape mismatch
    raises so the caller (``get``) can treat it as a cache miss."""
    writer = data.get("writer", {})
    winner_dict = data.get("winner")
    winner = _winner_from_dict(winner_dict) if isinstance(winner_dict, dict) else None
    runners_up = tuple(_winner_from_dict(w) for w in data.get("runners_up", []) if isinstance(w, dict))
    partials = tuple(_winner_from_dict(w) for w in data.get("partial_observations", []) if isinstance(w, dict))
    disqualified_raw = data.get("disqualified", []) or []
    disqualified = tuple(d for d in disqualified_raw if isinstance(d, dict))
    return Result(
        cache_key=cache_key,
        hardware_fingerprint=data["hardware_fingerprint"],
        hardware_descriptor=dict(data.get("hardware_descriptor", {})),
        writer_runtime_build_tag=writer.get("runtime_build_tag", ""),
        writer_process_kind=writer.get("process_kind", "python_sdk"),
        writer_pid=writer.get("pid"),
        incomplete=bool(data.get("incomplete", False)),
        winner=winner,
        runners_up=runners_up,
        partial_observations=partials,
        disqualified=disqualified,
        confidence=data.get("confidence", "high"),
        created_at=data.get("created_at"),
    )


def _winner_from_dict(d: dict[str, Any]) -> Winner:
    return Winner(
        engine=d["engine"],
        config=dict(d.get("config", {})),
        score=float(d["score"]),
        provider=d.get("provider"),
        first_chunk_ms=d.get("first_chunk_ms"),
        total_latency_ms=d.get("total_latency_ms"),
        quality_metrics=dict(d.get("quality_metrics", {})),
    )


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with seconds precision. The contract uses
    ``date-time`` format which jsonschema accepts in this shape."""
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
