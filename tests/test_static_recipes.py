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


def test_recipe_artifact_digest_is_the_actual_file_digest_not_a_manifest_hash():
    """Reviewer P1: ``PrepareManager._build_descriptor`` hands the
    artifact-level digest directly to the durable downloader as the
    expected SHA-256 of the downloaded file. Setting it to a
    "manifest hash of joined per-file digests" would make
    verification fail on every download even when the bytes are
    correct. For single-file recipes the artifact digest must equal
    the file's own digest verbatim.

    Manifest-style hashing is reserved for the multi-file
    follow-up that lands ``manifest_uri`` support."""
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    candidate = recipe.to_runtime_candidate()
    assert candidate.artifact is not None
    only = recipe.files[0]
    assert candidate.artifact.digest == only.digest, (
        f"single-file recipe artifact digest must equal the file's own SHA-256 "
        f"(durable downloader checks the downloaded bytes against this); "
        f"got artifact={candidate.artifact.digest!r} vs file={only.digest!r}"
    )


def test_recipe_to_runtime_candidate_rejects_multi_file_recipes():
    """Until ``manifest_uri`` support lands, recipes must be
    single-file. Constructing the candidate from a multi-file
    recipe must raise so we never silently downgrade to
    "verify the first file only"."""
    from octomil.runtime.lifecycle.static_recipes import (
        StaticRecipe,
        _StaticArtifactFile,
    )

    bad = StaticRecipe(
        model_id="multi",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(
                relative_path="a.bin",
                url="https://x/a.bin",
                digest="sha256:" + "a" * 64,
            ),
            _StaticArtifactFile(
                relative_path="b.bin",
                url="https://x/b.bin",
                digest="sha256:" + "b" * 64,
            ),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        bad.to_runtime_candidate()
    assert "single-file" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Reviewer P1: tarball recipes must be extracted into the layout the
# backend expects, otherwise the prepared dir is not runnable.
# ---------------------------------------------------------------------------


def test_materialize_recipe_layout_extracts_tarball_into_artifact_dir(tmp_path):
    """``materialize_recipe_layout`` unpacks the downloaded tarball
    so ``_SherpaTtsBackend`` finds ``model.onnx`` / ``voices.bin`` /
    ``tokens.txt`` / ``espeak-ng-data/`` directly under
    ``artifact_dir``. Without this, prepare succeeds with valid
    bytes on disk but the backend can't load the model."""
    import tarfile

    from octomil.runtime.lifecycle.static_recipes import (
        StaticRecipe,
        _StaticArtifactFile,
        materialize_recipe_layout,
    )

    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    # Build a fake tarball that mimics the upstream Kokoro layout.
    tarball_path = artifact_dir / "kokoro-en-v0_19.tar.bz2"
    with tarfile.open(tarball_path, "w:bz2") as tar:
        for name, content in [
            ("model.onnx", b"fake onnx"),
            ("voices.bin", b"fake voices"),
            ("tokens.txt", b"fake tokens"),
            ("espeak-ng-data/dict", b"fake dict"),
        ]:
            data = content
            payload = artifact_dir / name
            payload.parent.mkdir(parents=True, exist_ok=True)
            payload.write_bytes(data)
            tar.add(payload, arcname=name)
            payload.unlink()
    # Drop the staged staging dir so we can verify materialization.
    (artifact_dir / "espeak-ng-data").rmdir()

    recipe = StaticRecipe(
        model_id="kokoro-82m",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(
                relative_path="kokoro-en-v0_19.tar.bz2",
                url="https://example.com/kokoro.tar.bz2",
                digest="sha256:" + "0" * 64,
                extract=True,
            ),
        ],
    )
    materialize_recipe_layout(recipe, artifact_dir)

    assert (artifact_dir / "model.onnx").is_file()
    assert (artifact_dir / "voices.bin").is_file()
    assert (artifact_dir / "tokens.txt").is_file()
    assert (artifact_dir / "espeak-ng-data" / "dict").is_file()


def test_materialize_recipe_layout_refuses_path_traversal(tmp_path):
    """Defense in depth: a hostile tarball with ``../etc/passwd``
    or absolute-path members must be filtered before extraction.
    Even though the recipe URLs are pinned to public CDNs, the
    extractor never trusts archive contents blindly."""
    import tarfile

    from octomil.runtime.lifecycle.static_recipes import (
        StaticRecipe,
        _StaticArtifactFile,
        materialize_recipe_layout,
    )

    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    tarball = artifact_dir / "evil.tar.bz2"
    bad_payload = artifact_dir / "_payload"
    bad_payload.write_bytes(b"safe content")
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(bad_payload, arcname="../escape.bin")  # would land outside artifact_dir
    bad_payload.unlink()

    recipe = StaticRecipe(
        model_id="evil",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(
                relative_path="evil.tar.bz2",
                url="https://example.com/evil.tar.bz2",
                digest="sha256:" + "0" * 64,
                extract=True,
            ),
        ],
    )
    materialize_recipe_layout(recipe, artifact_dir)

    # ``../escape.bin`` must NOT exist outside artifact_dir.
    assert not (artifact_dir.parent / "escape.bin").exists()
    assert not (artifact_dir / ".." / "escape.bin").exists()


def test_materialize_recipe_layout_is_idempotent(tmp_path):
    """Re-running ``materialize_recipe_layout`` against an
    already-extracted artifact dir is a no-op."""
    import tarfile

    from octomil.runtime.lifecycle.static_recipes import (
        StaticRecipe,
        _StaticArtifactFile,
        materialize_recipe_layout,
    )

    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    # Pre-stage the unpacked layout — simulating a prior prepare.
    (artifact_dir / "model.onnx").write_bytes(b"existing-model")
    (artifact_dir / "voices.bin").write_bytes(b"existing-voices")
    tarball = artifact_dir / "kokoro-en-v0_19.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        # Tarball contains DIFFERENT bytes — re-extracting would
        # overwrite the prior model. We assert the helper skips.
        tmp = artifact_dir / "_payload"
        tmp.write_bytes(b"new-model-bytes")
        tar.add(tmp, arcname="model.onnx")
        tmp.unlink()

    recipe = StaticRecipe(
        model_id="kokoro-82m",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(
                relative_path="kokoro-en-v0_19.tar.bz2",
                url="https://example.com/kokoro.tar.bz2",
                digest="sha256:" + "0" * 64,
                extract=True,
            ),
        ],
    )
    materialize_recipe_layout(recipe, artifact_dir)

    assert (artifact_dir / "model.onnx").read_bytes() == b"existing-model"


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
