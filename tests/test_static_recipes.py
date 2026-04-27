"""PR C: static offline recipes for canonical local models.

The reviewer's bar:

    pip install "octomil[tts]"
    octomil prepare kokoro-82m --capability tts

must work without a server planner round-trip. These tests pin:

  - the recipe table contains Kokoro at the expected ids;
  - the recipe materializes the same files / layout
    ``_SherpaTtsBackend`` reads (``model.onnx``, ``voices.bin``,
    ``tokens.txt``, espeak-ng-data);
  - the synthesized ``RuntimeCandidatePlan`` passes
    ``PrepareManager.can_prepare``'s structural validation
    (non-empty digest, non-empty download_urls);
  - ``client.prepare`` falls back to the static recipe when the
    planner returned no candidate;
  - the fallback fires only for known canonical models — unknown
    ids surface an actionable error so we never silently substitute
    a public mirror for a private artifact.
"""

from __future__ import annotations

import pytest


def test_kokoro_recipes_registered_under_canonical_ids():
    from octomil.runtime.lifecycle.static_recipes import _RECIPES, get_static_recipe

    assert ("kokoro-82m", "tts") in _RECIPES
    assert ("kokoro-en-v0_19", "tts") in _RECIPES
    # Both ids resolve to the same recipe (alias).
    assert get_static_recipe("kokoro-82m", "tts") is get_static_recipe("kokoro-en-v0_19", "tts")


def test_kokoro_recipe_uses_single_file_tarball_until_manifest_support_lands():
    """PrepareManager today rejects multi-file artifacts (the planner
    schema only carries one artifact-level digest). We model Kokoro
    as a single-file tarball — the upstream sherpa-onnx release
    bundles ``model.onnx`` + ``voices.bin`` + ``tokens.txt`` +
    ``espeak-ng-data/`` in one archive. The prepare pipeline
    extracts it downstream so ``_SherpaTtsBackend`` finds the
    expected layout. Once the multi-file ``manifest_uri`` follow-up
    lands, this recipe switches to per-file downloads without
    changing callers."""
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    assert len(recipe.files) == 1, (
        "Kokoro recipe must be single-file until multi-file manifest "
        f"support lands; got {[f.relative_path for f in recipe.files]}"
    )
    f = recipe.files[0]
    assert f.relative_path.endswith(".tar.bz2"), f.relative_path
    assert f.extract is True, "tarball must be flagged for post-download extraction"


def test_recipe_synthesizes_planner_shaped_candidate():
    """The recipe must produce a ``RuntimeCandidatePlan`` that the
    rest of the prepare pipeline accepts unchanged. Key invariants:
    locality=local, delivery_mode=sdk_runtime, prepare_required=True,
    prepare_policy=explicit_only (so lazy-prepare during inference
    doesn't auto-download), non-empty digest + download_urls so
    PrepareManager.can_prepare's structural validator passes."""
    from octomil.runtime.lifecycle.static_recipes import static_recipe_candidate

    candidate = static_recipe_candidate("kokoro-82m", "tts")
    assert candidate is not None
    assert candidate.locality == "local"
    assert candidate.delivery_mode == "sdk_runtime"
    assert candidate.prepare_required is True
    assert candidate.prepare_policy == "explicit_only"
    assert candidate.engine == "sherpa-onnx"
    assert candidate.artifact is not None
    assert candidate.artifact.digest, "digest must be non-empty for can_prepare"
    assert candidate.artifact.digest.startswith("sha256:")
    assert candidate.artifact.download_urls, "download_urls must be non-empty"
    assert candidate.artifact.required_files


def test_recipe_candidate_passes_can_prepare():
    """End-to-end: PrepareManager's structural validator must accept
    the synthesized candidate. If this regresses, prepare would
    raise before any download starts."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.lifecycle.static_recipes import static_recipe_candidate

    candidate = static_recipe_candidate("kokoro-82m", "tts")
    assert candidate is not None
    pm = PrepareManager()
    assert pm.can_prepare(candidate) is True


def test_unknown_model_returns_no_recipe():
    """Recipes must be narrowly scoped — unknown model ids do NOT
    fall through to a generic mirror, otherwise we'd risk shipping
    public bytes for what was meant to be a private artifact."""
    from octomil.runtime.lifecycle.static_recipes import (
        get_static_recipe,
        static_recipe_candidate,
    )

    assert get_static_recipe("nonexistent-private-app-tts", "tts") is None
    assert static_recipe_candidate("nonexistent-private-app-tts", "tts") is None


def test_recipe_only_fires_for_matching_capability():
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    # kokoro-82m exists for tts but not chat / embeddings.
    assert get_static_recipe("kokoro-82m", "tts") is not None
    assert get_static_recipe("kokoro-82m", "chat") is None
    assert get_static_recipe("kokoro-82m", "embedding") is None


# ---------------------------------------------------------------------------
# Kernel.prepare() falls back to the static recipe when planner returns None
# ---------------------------------------------------------------------------


def test_kernel_prepare_falls_back_to_static_recipe(tmp_path, monkeypatch):
    """Reviewer P1 happy path: ``client.prepare(model='kokoro-82m',
    capability='tts')`` must work when ``OCTOMIL_SERVER_KEY`` is
    unset and the planner returns no candidate. The static recipe
    fires in that case."""
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome

    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "kokoro-82m",
            "policy_preset": "local_only",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    # Stub the PrepareManager so the test doesn't actually hit the
    # network. The point of this test is the fallback wiring, not
    # the download itself.
    captured: dict = {}

    class _StubPM:
        def can_prepare(self, candidate):
            return True

        def prepare(self, candidate, *, mode=None):
            captured["candidate"] = candidate
            captured["mode"] = mode
            return PrepareOutcome(
                artifact_id=candidate.artifact.artifact_id,
                artifact_dir=tmp_path / "kokoro",
                files={f: tmp_path / "kokoro" / f for f in candidate.artifact.required_files},
                engine=candidate.engine,
                delivery_mode="sdk_runtime",
                prepare_policy="explicit_only",
                cached=False,
            )

    kernel._prepare_manager = _StubPM()

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        outcome = kernel.prepare(model="kokoro-82m", capability="tts")

    assert outcome.artifact_id == "kokoro-82m"
    # The candidate that flowed into PrepareManager came from the
    # static recipe — assert the recipe's signature shape.
    cand = captured["candidate"]
    assert cand.engine == "sherpa-onnx"
    assert cand.delivery_mode == "sdk_runtime"
    assert cand.prepare_policy == "explicit_only"


def test_kernel_prepare_unknown_model_without_planner_surfaces_actionable_error(monkeypatch):
    """Reviewer requirement: when the planner is offline AND the
    model has no static recipe, the SDK must NOT silently substitute
    a public mirror. The error must name the failure mode and tell
    the operator to set OCTOMIL_SERVER_KEY or use a known model."""
    from unittest.mock import patch

    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "private-org-only-tts-model",
            "policy_preset": "local_only",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        with pytest.raises(OctomilError) as excinfo:
            kernel.prepare(model="private-org-only-tts-model", capability="tts")

    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "private-org-only-tts-model" in msg
    assert "OCTOMIL_SERVER_KEY" in msg
    assert "kokoro-82m" in msg  # canonical example named in the error
