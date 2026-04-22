"""Data-driven conformance tests for parse_model_ref using the canonical
fixture from octomil-contracts (fixtures/model_refs/canonical.json).

The fixture is the single source of truth for model ref classification.
If this test fails, fix the parser -- not the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from octomil.runtime.routing.model_ref import parse_model_ref

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_refs"

CANONICAL = json.loads((FIXTURE_DIR / "canonical.json").read_text())
DEPRECATED = json.loads((FIXTURE_DIR / "deprecated_aliases.json").read_text())

CASES = CANONICAL["cases"]

# =========================================================================
# Kind classification
# =========================================================================


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[c.get("description", c["input"]) or "<empty>" for c in CASES],
)
def test_kind_classification(case: dict) -> None:
    result = parse_model_ref(case["input"])
    assert (
        result.kind == case["expected_kind"]
    ), f"input={case['input']!r}: expected kind={case['expected_kind']!r}, got={result.kind!r}"


# =========================================================================
# App ref field extraction
# =========================================================================


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["expected_kind"] == "app"],
    ids=[c["input"] for c in CASES if c["expected_kind"] == "app"],
)
def test_app_ref_fields(case: dict) -> None:
    result = parse_model_ref(case["input"])
    assert result.app_slug == case["expected_app_slug"], f"slug mismatch for {case['input']!r}"
    assert result.capability == case["expected_capability"], f"capability mismatch for {case['input']!r}"


# =========================================================================
# Deployment ref field extraction
# =========================================================================


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["expected_kind"] == "deployment"],
    ids=[c["input"] for c in CASES if c["expected_kind"] == "deployment"],
)
def test_deployment_ref_fields(case: dict) -> None:
    result = parse_model_ref(case["input"])
    assert result.deployment_id == case["expected_deployment_id"], f"deployment_id mismatch for {case['input']!r}"


# =========================================================================
# Experiment ref field extraction
# =========================================================================


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["expected_kind"] == "experiment"],
    ids=[c["input"] for c in CASES if c["expected_kind"] == "experiment"],
)
def test_experiment_ref_fields(case: dict) -> None:
    result = parse_model_ref(case["input"])
    assert result.experiment_id == case["expected_experiment_id"], f"experiment_id mismatch for {case['input']!r}"
    assert result.variant_id == case["expected_variant_id"], f"variant_id mismatch for {case['input']!r}"


# =========================================================================
# Deprecated aliases
# =========================================================================


def test_parser_never_produces_deprecated_kinds() -> None:
    deprecated_kinds = set(DEPRECATED["deprecated_to_canonical"].keys())
    for case in CASES:
        result = parse_model_ref(case["input"])
        assert (
            result.kind not in deprecated_kinds
        ), f"Parser produced deprecated kind {result.kind!r} for input {case['input']!r}"


# =========================================================================
# All 8 canonical kinds covered
# =========================================================================


def test_fixture_covers_all_8_canonical_kinds() -> None:
    expected = {"model", "app", "capability", "deployment", "experiment", "alias", "default", "unknown"}
    covered = {c["expected_kind"] for c in CASES}
    assert expected == covered, f"Missing kinds: {expected - covered}"
