#!/usr/bin/env python3
"""Validator for ``cache_bench_proof`` documents (Python-side mirror).

Mirrors ``ci/validate_cache_bench_proof.py`` in ``octomil-contracts`` so the
benchmark-gate workflow can reject schema-invalid proofs without cloning
the contracts repo (which requires a privileged token).

Codex sweep B4 (URGENT): the prior gate only checked ``cold_p50_ms`` and
``hit_ratio`` for ``skipped=false`` artifacts and never ran the canonical
schema/validator, so a measured proof with extra plaintext fields, an
unknown ``capability``, NaN/Inf metrics, or bad digests would still pass
the release gate. This module closes that hole.

The validator is self-contained: it captures the structural invariants
that block invalid latency claims from supporting cache speedup
narratives. The schema source of truth still lives at
``octomil-contracts/schemas/core/cache_bench_proof.json``; if that schema
gains new constraints the gate must be updated here too. Drift is caught
by ``tests/test_cache_bench_proof_validator.py``.

Usage::

    from scripts.validate_cache_bench_proof import (
        validate_proof, ProofValidationError,
    )

    try:
        validate_proof(doc)
    except ProofValidationError as exc:
        sys.exit(f"cache bench proof rejected: {exc}")
"""

from __future__ import annotations

import math
import re
from typing import Any

# Closed enums — must match the contracts schema. Keep in sync with
# ``octomil-contracts/schemas/core/cache_bench_proof.json``.
_ALLOWED_CAPABILITIES = frozenset(
    {
        "audio.realtime.session",
        "audio.stt.batch",
        "audio.stt.stream",
        "audio.transcription",
        "audio.tts.batch",
        "audio.tts.stream",
        "chat.completion",
        "chat.stream",
        "embeddings.image",
        "embeddings.text",
    }
)

_ALLOWED_SKIP_REASONS = frozenset(
    {
        "staged_artifact_absent",
        "policy_disabled",
        "capability_unsupported",
    }
)

_ALLOWED_PARITY_STATUS = frozenset({"parity_ok", "parity_drift", "n/a"})

# Required top-level keys.
_REQUIRED_KEYS = (
    "$schema_version",
    "schema_version",
    "cache_id",
    "capability",
    "measured_at",
    "skipped",
    "runtime_digest",
    "model_digest",
    "adapter_version",
    "staged_artifact_ref",
)

# Latency / ratio metric fields.
_METRIC_FIELDS = (
    "cold_p50_ms",
    "cold_p95_ms",
    "warm_p50_ms",
    "warm_p95_ms",
    "hit_ratio",
)

# Allowed top-level keys (additionalProperties=false in schema). Extra
# plaintext keys are rejected to prevent input-content leakage.
_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    set(_REQUIRED_KEYS)
    | {
        "skip_reason",
        "input_digest",
        "cold_p50_ms",
        "cold_p95_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "hit_ratio",
        "entries",
        "bytes_overhead",
        "parity_status",
        "parity_drift_threshold_pct",
        "writer",
    }
)

_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
# RFC 3339 / ISO 8601 UTC timestamp shape (Z or ±HH:MM offset).
_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})$")


class ProofValidationError(ValueError):
    """Raised when a cache_bench_proof document fails validation."""


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise ProofValidationError(msg)


def validate_proof(doc: Any) -> None:
    """Validate a cache_bench_proof document.

    Raises ``ProofValidationError`` on any structural or semantic
    violation. Returns ``None`` on success.

    Codex B4 fix: every artifact under ``release/cache_proofs/**`` MUST
    pass this validator before its latency claim is accepted by CI.
    """

    _require(isinstance(doc, dict), "proof root must be an object")

    # additionalProperties=false in the schema — reject unknown fields so
    # plaintext input contents cannot leak past the gate.
    extra = set(doc.keys()) - _ALLOWED_TOP_LEVEL_KEYS
    _require(
        not extra,
        f"unexpected top-level keys: {sorted(extra)} (additionalProperties=false)",
    )

    for key in _REQUIRED_KEYS:
        _require(key in doc, f"missing required key: {key}")

    _require(doc["$schema_version"] == 1, "$schema_version must be 1")
    _require(doc["schema_version"] == 1, "schema_version must be 1")

    _require(
        isinstance(doc["cache_id"], str) and bool(doc["cache_id"]),
        "cache_id must be a non-empty string",
    )
    _require(
        doc["capability"] in _ALLOWED_CAPABILITIES,
        f"capability must be one of {sorted(_ALLOWED_CAPABILITIES)}",
    )
    _require(
        isinstance(doc["measured_at"], str) and bool(_ISO8601_RE.match(doc["measured_at"])),
        "measured_at must be an RFC 3339 / ISO 8601 UTC timestamp",
    )
    _require(isinstance(doc["skipped"], bool), "skipped must be a boolean")
    _require(
        isinstance(doc["runtime_digest"], str) and bool(_SHA256_RE.match(doc["runtime_digest"])),
        "runtime_digest must match ^sha256:[a-f0-9]{64}$",
    )
    _require(
        isinstance(doc["model_digest"], str) and bool(_SHA256_RE.match(doc["model_digest"])),
        "model_digest must match ^sha256:[a-f0-9]{64}$",
    )
    _require(
        isinstance(doc["adapter_version"], str) and bool(doc["adapter_version"]),
        "adapter_version must be a non-empty string",
    )
    _require(
        isinstance(doc["staged_artifact_ref"], str) and bool(doc["staged_artifact_ref"]),
        "staged_artifact_ref must be a non-empty string",
    )

    if "input_digest" in doc:
        _require(
            isinstance(doc["input_digest"], str) and bool(_SHA256_RE.match(doc["input_digest"])),
            "input_digest must match ^sha256:[a-f0-9]{64}$ when present",
        )

    if "parity_status" in doc:
        _require(
            doc["parity_status"] in _ALLOWED_PARITY_STATUS,
            f"parity_status must be one of {sorted(_ALLOWED_PARITY_STATUS)}",
        )

    if doc["skipped"]:
        _require(
            "skip_reason" in doc and doc["skip_reason"] in _ALLOWED_SKIP_REASONS,
            "skipped=true requires skip_reason in " f"{sorted(_ALLOWED_SKIP_REASONS)}",
        )
        for field in _METRIC_FIELDS + ("entries", "bytes_overhead"):
            if field in doc:
                _require(
                    doc[field] is None,
                    f"{field} must be null when skipped=true (got {doc[field]!r})",
                )
    else:
        # skip_reason MUST NOT be set when not skipped (avoid ambiguous state).
        _require(
            "skip_reason" not in doc or doc["skip_reason"] is None,
            "skip_reason must be absent when skipped=false",
        )
        for field in _METRIC_FIELDS:
            _require(field in doc, f"missing required metric field: {field}")
            val = doc[field]
            _require(
                isinstance(val, (int, float)) and not isinstance(val, bool),
                f"{field} must be a finite number (got {val!r})",
            )
            _require(
                math.isfinite(float(val)),
                f"{field}={val!r} is NaN or Inf",
            )
            # Latency fields must be non-negative.
            if field != "hit_ratio":
                _require(
                    float(val) >= 0.0,
                    f"{field}={val!r} must be >= 0",
                )
            else:
                _require(
                    0.0 <= float(val) <= 1.0,
                    f"hit_ratio={val!r} must be in [0, 1]",
                )

        for field in ("entries", "bytes_overhead"):
            if field in doc and doc[field] is not None:
                _require(
                    isinstance(doc[field], int) and not isinstance(doc[field], bool) and doc[field] >= 0,
                    f"{field} must be a non-negative integer when present",
                )


def main() -> int:
    """CLI entry: ``validate_cache_bench_proof.py <path>...``.

    Exit code 0 on success, 1 on any validation failure.
    """
    import json
    import pathlib
    import sys

    if len(sys.argv) < 2:
        print(
            "usage: validate_cache_bench_proof.py <proof.json>...",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: cannot read/parse: {exc}")
            continue
        try:
            validate_proof(doc)
        except ProofValidationError as exc:
            failures.append(f"{path}: {exc}")
            continue
        print(f"OK: {path}")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
