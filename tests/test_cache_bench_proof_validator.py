"""Regression tests for ``scripts/validate_cache_bench_proof.py``.

Codex sweep B4 (URGENT): the prior CI gate accepted schema-invalid proof
artifacts. These tests pin the canonical structural invariants so the
gate cannot silently weaken.

Codex M4: feeds bad capability / digest / extra plaintext fields through
the validator and asserts each is rejected with a typed error.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest

# Load the validator from scripts/ (it is not packaged on PYTHONPATH).
_HERE = pathlib.Path(__file__).resolve().parent
_VALIDATOR_PATH = _HERE.parent / "scripts" / "validate_cache_bench_proof.py"
_spec = importlib.util.spec_from_file_location("validate_cache_bench_proof", _VALIDATOR_PATH)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
sys.modules["validate_cache_bench_proof"] = _module
_spec.loader.exec_module(_module)
validate_proof = _module.validate_proof
ProofValidationError = _module.ProofValidationError


def _zero_sha() -> str:
    return "sha256:" + ("0" * 64)


def _measured_proof() -> dict:
    """A schema-valid measured (skipped=false) proof."""
    return {
        "$schema_version": 1,
        "schema_version": 1,
        "cache_id": "chat.completion.kv",
        "capability": "chat.completion",
        "measured_at": "2026-05-10T00:00:00Z",
        "skipped": False,
        "runtime_digest": _zero_sha(),
        "model_digest": _zero_sha(),
        "adapter_version": "0.1.11-test",
        "staged_artifact_ref": "/tmp/fixtures",
        "input_digest": _zero_sha(),
        "cold_p50_ms": 1.0,
        "cold_p95_ms": 2.0,
        "warm_p50_ms": 0.5,
        "warm_p95_ms": 1.5,
        "hit_ratio": 0.9,
        "entries": 10,
        "bytes_overhead": 1024,
        "parity_status": "parity_ok",
    }


def _skipped_proof() -> dict:
    """A schema-valid skipped proof."""
    return {
        "$schema_version": 1,
        "schema_version": 1,
        "cache_id": "chat.completion.kv",
        "capability": "chat.completion",
        "measured_at": "2026-05-10T00:00:00Z",
        "skipped": True,
        "skip_reason": "staged_artifact_absent",
        "runtime_digest": _zero_sha(),
        "model_digest": _zero_sha(),
        "adapter_version": "0.1.11-test",
        "staged_artifact_ref": "/nonexistent",
    }


def test_valid_measured_proof_passes() -> None:
    validate_proof(_measured_proof())


def test_valid_skipped_proof_passes() -> None:
    validate_proof(_skipped_proof())


def test_unknown_capability_rejected() -> None:
    proof = _measured_proof()
    proof["capability"] = "definitely.not.a.capability"
    with pytest.raises(ProofValidationError, match="capability must be one of"):
        validate_proof(proof)


def test_extra_plaintext_field_rejected() -> None:
    """B4 / M4: additional plaintext top-level keys must fail closed."""
    proof = _measured_proof()
    proof["raw_input_text"] = "the quick brown fox"
    with pytest.raises(ProofValidationError, match="unexpected top-level keys"):
        validate_proof(proof)


def test_bad_runtime_digest_rejected() -> None:
    proof = _measured_proof()
    proof["runtime_digest"] = "sha256:not_hex"
    with pytest.raises(ProofValidationError, match="runtime_digest"):
        validate_proof(proof)


def test_bad_model_digest_rejected() -> None:
    proof = _measured_proof()
    proof["model_digest"] = "md5:" + ("0" * 32)
    with pytest.raises(ProofValidationError, match="model_digest"):
        validate_proof(proof)


def test_bad_input_digest_rejected() -> None:
    proof = _measured_proof()
    proof["input_digest"] = "raw:abc"
    with pytest.raises(ProofValidationError, match="input_digest"):
        validate_proof(proof)


def test_bad_measured_at_rejected() -> None:
    proof = _measured_proof()
    proof["measured_at"] = "yesterday"
    with pytest.raises(ProofValidationError, match="measured_at"):
        validate_proof(proof)


def test_negative_cold_p50_rejected() -> None:
    proof = _measured_proof()
    proof["cold_p50_ms"] = -1.0
    with pytest.raises(ProofValidationError, match="cold_p50_ms"):
        validate_proof(proof)


def test_nan_metric_rejected() -> None:
    proof = _measured_proof()
    proof["warm_p95_ms"] = float("nan")
    with pytest.raises(ProofValidationError, match="warm_p95_ms"):
        validate_proof(proof)


def test_inf_metric_rejected() -> None:
    proof = _measured_proof()
    proof["cold_p95_ms"] = math.inf
    with pytest.raises(ProofValidationError, match="cold_p95_ms"):
        validate_proof(proof)


def test_hit_ratio_out_of_range_rejected() -> None:
    proof = _measured_proof()
    proof["hit_ratio"] = 1.5
    with pytest.raises(ProofValidationError, match="hit_ratio"):
        validate_proof(proof)


def test_skipped_with_metric_rejected() -> None:
    proof = _skipped_proof()
    proof["cold_p50_ms"] = 1.0
    with pytest.raises(ProofValidationError, match="cold_p50_ms"):
        validate_proof(proof)


def test_skipped_without_skip_reason_rejected() -> None:
    proof = _skipped_proof()
    del proof["skip_reason"]
    with pytest.raises(ProofValidationError, match="skip_reason"):
        validate_proof(proof)


def test_skipped_with_unknown_reason_rejected() -> None:
    proof = _skipped_proof()
    proof["skip_reason"] = "i_felt_like_it"
    with pytest.raises(ProofValidationError, match="skip_reason"):
        validate_proof(proof)


def test_measured_with_skip_reason_rejected() -> None:
    """Ambiguous state: skipped=false + skip_reason present → reject."""
    proof = _measured_proof()
    proof["skip_reason"] = "policy_disabled"
    with pytest.raises(ProofValidationError, match="skip_reason"):
        validate_proof(proof)


def test_missing_required_metric_rejected() -> None:
    proof = _measured_proof()
    del proof["warm_p50_ms"]
    with pytest.raises(ProofValidationError, match="warm_p50_ms"):
        validate_proof(proof)


def test_wrong_schema_version_rejected() -> None:
    proof = _measured_proof()
    proof["$schema_version"] = 2
    with pytest.raises(ProofValidationError, match="schema_version"):
        validate_proof(proof)
