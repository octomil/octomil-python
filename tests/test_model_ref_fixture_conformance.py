"""Fixture-driven conformance tests for model-ref classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from octomil.runtime.routing.model_ref import parse_model_ref

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model_ref_parse_cases.json"
CASES = json.loads(FIXTURE_PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_model_ref_parse_cases(case: dict) -> None:
    result = parse_model_ref(case["input"])
    expected = case["expected"]

    assert result.kind == expected["kind"]
    assert result.raw == expected["raw"]
    assert result.model_slug == expected.get("model_slug")
    assert result.app_slug == expected.get("app_slug")
    assert result.capability == expected.get("capability")
    assert result.deployment_id == expected.get("deployment_id")
    assert result.experiment_id == expected.get("experiment_id")
    assert result.variant_id == expected.get("variant_id")


def test_fixture_covers_all_canonical_kinds() -> None:
    expected = {
        "model",
        "app",
        "capability",
        "deployment",
        "experiment",
        "alias",
        "default",
        "unknown",
    }
    assert {case["expected"]["kind"] for case in CASES} == expected
