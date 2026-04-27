"""PR C: static offline recipes + generic materialization layer.

Reviewer's bar:

    pip install "octomil[tts]"
    octomil prepare kokoro-82m --capability tts

must work without a server planner round-trip — and the second
recipe to land must be a *data row*, not a copy of the Kokoro path.

The recipe layer is two halves:

  - ``StaticRecipe`` carries ``files`` (download description) +
    ``materialization`` (``MaterializationPlan``);
  - generic ``Materializer`` consumes ``MaterializationPlan`` and
    leaves ``artifact_dir`` in a backend-ready state.

These tests pin every reviewer-flagged invariant:

  - the Kokoro digest matches the GitHub release asset
    (``test_kokoro_recipe_digest_is_not_the_all_zero_placeholder``);
  - ``RuntimeCandidatePlan.artifact.digest`` IS the file digest the
    durable downloader will check (no synthetic manifest hash);
  - extraction is safe against traversal / symlink / hardlink /
    bomb attacks;
  - partial extractions are retried, never skipped (marker is
    written LAST, only after every required output is on disk);
  - after prepare + materialize, ``artifact_dir`` contains the
    exact files Sherpa needs;
  - ``MaterializationPlan(kind='none')`` is the no-op shape used
    by single-file backends.
"""

from __future__ import annotations

import tarfile

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.lifecycle.materialization import (
    EXTRACTION_MARKER,
    MaterializationPlan,
    MaterializationSafetyPolicy,
    Materializer,
)
from octomil.runtime.lifecycle.static_recipes import (
    StaticRecipe,
    _assert_no_placeholder_digest,
    _StaticArtifactFile,
    get_static_recipe,
    static_recipe_candidate,
)

# ---------------------------------------------------------------------------
# Recipe registration + alias coverage
# ---------------------------------------------------------------------------


def test_kokoro_recipes_registered_under_canonical_ids():
    from octomil.runtime.lifecycle.static_recipes import _RECIPES

    assert ("kokoro-82m", "tts") in _RECIPES
    assert ("kokoro-en-v0_19", "tts") in _RECIPES
    assert get_static_recipe("kokoro-82m", "tts") is get_static_recipe("kokoro-en-v0_19", "tts")


def test_unknown_model_returns_no_recipe():
    """Recipes must be narrowly scoped — unknown ids do NOT fall
    through to a generic mirror, otherwise we'd risk shipping
    public bytes for what was meant to be a private artifact."""
    assert get_static_recipe("nonexistent-private-app-tts", "tts") is None
    assert static_recipe_candidate("nonexistent-private-app-tts", "tts") is None


def test_recipe_only_fires_for_matching_capability():
    assert get_static_recipe("kokoro-82m", "tts") is not None
    assert get_static_recipe("kokoro-82m", "chat") is None
    assert get_static_recipe("kokoro-82m", "embedding") is None


# ---------------------------------------------------------------------------
# Digest + candidate shape
# ---------------------------------------------------------------------------


def test_kokoro_recipe_digest_is_not_the_all_zero_placeholder():
    """Reviewer P1: a recipe with ``sha256:0000…0000`` is structurally
    valid but every download against it fails verification. Catch
    the regression at the recipe-table level."""
    from octomil.runtime.lifecycle.static_recipes import _RECIPES

    placeholder = "sha256:" + "0" * 64
    for (model_id, capability), recipe in _RECIPES.items():
        for f in recipe.files:
            assert f.digest != placeholder, (
                f"recipe {model_id!r}/{capability!r} ships placeholder digest for "
                f"{f.relative_path!r}; replace with the real SHA-256."
            )
            assert f.digest.startswith("sha256:")
            assert len(f.digest) == len("sha256:") + 64


def test_kokoro_recipe_digest_matches_upstream_release_asset():
    """The Kokoro digest pinned in the recipe MUST equal the SHA-256
    GitHub publishes for the release asset. Reviewer flagged a prior
    drift; this test catches it.

    Verified value comes from
    ``GET /repos/k2-fsa/sherpa-onnx/releases/tags/tts-models``
    → ``assets[?name=='kokoro-en-v0_19.tar.bz2'].digest``."""
    from octomil.runtime.lifecycle.static_recipes import _KOKORO_82M_TARBALL_SHA256

    assert _KOKORO_82M_TARBALL_SHA256 == "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"


def test_recipe_module_import_rejects_placeholder_digest_at_construction():
    bad = StaticRecipe(
        model_id="bad",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(
                relative_path="x.tar.bz2",
                url="https://x/x.tar.bz2",
                digest="sha256:" + "0" * 64,
            ),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        _assert_no_placeholder_digest(bad)
    assert "placeholder" in str(excinfo.value).lower()


def test_recipe_artifact_digest_is_the_actual_file_digest_not_a_manifest_hash():
    """``PrepareManager._build_descriptor`` hands the artifact-level
    digest directly to the durable downloader as the expected
    SHA-256 of the downloaded file. For single-file recipes the
    artifact digest must equal the file's own digest verbatim."""
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    candidate = recipe.to_runtime_candidate()
    assert candidate.artifact is not None
    only = recipe.files[0]
    assert candidate.artifact.digest == only.digest


def test_recipe_to_runtime_candidate_rejects_multi_file_recipes():
    """Multi-file recipes need ``manifest_uri`` support; until then,
    ``to_runtime_candidate`` raises so we never silently downgrade."""
    bad = StaticRecipe(
        model_id="multi",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            _StaticArtifactFile(relative_path="a.bin", url="https://x/a", digest="sha256:" + "a" * 64),
            _StaticArtifactFile(relative_path="b.bin", url="https://x/b", digest="sha256:" + "b" * 64),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        bad.to_runtime_candidate()
    assert "single-file" in str(excinfo.value).lower()


def test_recipe_synthesizes_planner_shaped_candidate():
    """Direction-of-travel test on the recipe → candidate
    conversion: locality=local, sdk_runtime, prepare_required=True,
    explicit_only policy, non-empty digest + download_urls."""
    candidate = static_recipe_candidate("kokoro-82m", "tts")
    assert candidate is not None
    assert candidate.locality == "local"
    assert candidate.delivery_mode == "sdk_runtime"
    assert candidate.prepare_required is True
    assert candidate.prepare_policy == "explicit_only"
    assert candidate.engine == "sherpa-onnx"
    assert candidate.artifact is not None
    assert candidate.artifact.digest.startswith("sha256:")
    assert candidate.artifact.download_urls
    assert candidate.artifact.required_files


# ---------------------------------------------------------------------------
# Generic Materializer: archive extraction
# ---------------------------------------------------------------------------


def _make_kokoro_layout_tarball(tmp_path):
    """Produce a tarball that mimics the upstream Kokoro layout:
    members live under ``kokoro-en-v0_19/...`` so the recipe's
    ``strip_prefix`` is exercised."""
    archive_dir = tmp_path / "_archive_src"
    archive_dir.mkdir()
    layout = {
        "kokoro-en-v0_19/model.onnx": b"fake-onnx",
        "kokoro-en-v0_19/voices.bin": b"fake-voices",
        "kokoro-en-v0_19/tokens.txt": b"fake-tokens",
        "kokoro-en-v0_19/espeak-ng-data/phontab": b"fake-phontab",
    }
    paths_to_pack = []
    for relpath, data in layout.items():
        p = archive_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        paths_to_pack.append((p, relpath))
    tarball = tmp_path / "kokoro-en-v0_19.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        for p, name in paths_to_pack:
            tar.add(p, arcname=name)
    return tarball


def test_materializer_extracts_kokoro_layout_with_strip_prefix(tmp_path):
    """End-to-end test of the generic Materializer against the
    Kokoro recipe's MaterializationPlan: download the tarball,
    invoke ``materialize``, assert the strip_prefix unwrapped the
    top-level directory and every required output is on disk."""
    tarball = _make_kokoro_layout_tarball(tmp_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    # Place the downloaded tarball where PrepareManager would have
    # put it (under artifact_dir / source).
    target = artifact_dir / "kokoro-en-v0_19.tar.bz2"
    target.write_bytes(tarball.read_bytes())

    plan = MaterializationPlan(
        kind="archive",
        source="kokoro-en-v0_19.tar.bz2",
        archive_format="tar.bz2",
        strip_prefix="kokoro-en-v0_19/",
        required_outputs=(
            "model.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/phontab",
        ),
    )
    Materializer().materialize(artifact_dir, plan)

    assert (artifact_dir / "model.onnx").is_file()
    assert (artifact_dir / "voices.bin").is_file()
    assert (artifact_dir / "tokens.txt").is_file()
    assert (artifact_dir / "espeak-ng-data" / "phontab").is_file()
    # Marker is written LAST.
    assert (artifact_dir / EXTRACTION_MARKER).is_file()


def test_materializer_kind_none_validates_layout_without_extracting(tmp_path):
    """``kind='none'`` plans don't extract anything, but they MUST
    still validate ``required_outputs`` so a malformed download
    surfaces in the materializer rather than later inside the
    backend."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "ggml-tiny.bin").write_bytes(b"weights")

    plan = MaterializationPlan(kind="none", required_outputs=("ggml-tiny.bin",))
    Materializer().materialize(artifact_dir, plan)  # no exception

    plan_missing = MaterializationPlan(kind="none", required_outputs=("missing.bin",))
    with pytest.raises(OctomilError) as excinfo:
        Materializer().materialize(artifact_dir, plan_missing)
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "missing" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_materializer_refuses_path_traversal(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = artifact_dir / "_payload"
    payload.write_bytes(b"safe content")
    tarball = artifact_dir / "evil.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(payload, arcname="../escape.bin")
    payload.unlink()

    plan = MaterializationPlan(
        kind="archive",
        source="evil.tar.bz2",
        archive_format="tar.bz2",
        # No required_outputs declared → raises after extraction.
    )
    # Materializer either filters the bad member silently OR raises
    # the layout-incomplete error; either way ``../escape.bin``
    # must NOT exist outside artifact_dir.
    try:
        Materializer().materialize(artifact_dir, plan)
    except OctomilError:
        pass  # acceptable — plan declares no outputs
    assert not (artifact_dir.parent / "escape.bin").exists()
    assert not (artifact_dir / ".." / "escape.bin").exists()


def test_materializer_keeps_extraction_inside_artifact_dir_for_absolute_paths(tmp_path):
    """Tarfile strips leading slashes so an ``arcname='/etc/passwd'``
    member ends up as ``etc/passwd`` (no longer absolute) at extract
    time. The critical safety property is that the member NEVER
    escapes ``artifact_dir`` — the absolute path becomes a path
    *under* the artifact dir, never the real ``/etc/passwd``."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = artifact_dir / "_payload"
    payload.write_bytes(b"safe")
    tarball = artifact_dir / "abs.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(payload, arcname="/etc/passwd")
    payload.unlink()

    plan = MaterializationPlan(kind="archive", source="abs.tar.bz2", archive_format="tar.bz2")
    try:
        Materializer().materialize(artifact_dir, plan)
    except OctomilError:
        pass
    # The real ``/etc/passwd`` was never touched (we'd never have
    # write perms anyway, but the materializer also doesn't try).
    # Anything written sits under artifact_dir.
    for child in artifact_dir.rglob("*"):
        assert str(child).startswith(str(artifact_dir)), child


def test_materializer_refuses_symlinks_by_default(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    target = artifact_dir / "_payload"
    target.write_bytes(b"safe")
    tarball = artifact_dir / "sym.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        info = tarfile.TarInfo(name="evil_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
        tar.add(target, arcname="legit.bin")
    target.unlink()

    plan = MaterializationPlan(
        kind="archive",
        source="sym.tar.bz2",
        archive_format="tar.bz2",
        required_outputs=("legit.bin",),
    )
    Materializer().materialize(artifact_dir, plan)
    # Symlink filtered out; legit file present.
    assert not (artifact_dir / "evil_link").exists()
    assert (artifact_dir / "legit.bin").is_file()


def test_materializer_refuses_hardlinks_by_default(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    target = artifact_dir / "_payload"
    target.write_bytes(b"safe")
    tarball = artifact_dir / "hard.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        info = tarfile.TarInfo(name="hardlink_to_outside")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
        tar.add(target, arcname="legit.bin")
    target.unlink()

    plan = MaterializationPlan(
        kind="archive",
        source="hard.tar.bz2",
        archive_format="tar.bz2",
        required_outputs=("legit.bin",),
    )
    Materializer().materialize(artifact_dir, plan)
    assert not (artifact_dir / "hardlink_to_outside").exists()


def test_materializer_enforces_max_uncompressed_size(tmp_path):
    """Tar/zip-bomb defense. Cap is per plan via safety_policy."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    big = artifact_dir / "_big"
    big.write_bytes(b"x" * (1024 * 1024))  # 1 MiB
    tarball = artifact_dir / "big.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(big, arcname="big.bin")
    big.unlink()

    plan = MaterializationPlan(
        kind="archive",
        source="big.tar.bz2",
        archive_format="tar.bz2",
        safety_policy=MaterializationSafetyPolicy(max_total_uncompressed_bytes=1024),
    )
    with pytest.raises(OctomilError):
        Materializer().materialize(artifact_dir, plan)


# ---------------------------------------------------------------------------
# Idempotency: marker logic + partial extraction handling
# ---------------------------------------------------------------------------


def test_materializer_skips_when_full_layout_and_marker_are_present(tmp_path):
    """Re-running materialize against a previously-completed
    artifact dir is a no-op."""
    tarball = _make_kokoro_layout_tarball(tmp_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "kokoro-en-v0_19.tar.bz2").write_bytes(tarball.read_bytes())

    plan = MaterializationPlan(
        kind="archive",
        source="kokoro-en-v0_19.tar.bz2",
        archive_format="tar.bz2",
        strip_prefix="kokoro-en-v0_19/",
        required_outputs=("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/phontab"),
    )
    mat = Materializer()
    mat.materialize(artifact_dir, plan)

    # Mutate one of the extracted files to detect a re-extraction:
    # if the materializer skips correctly, our edit survives.
    (artifact_dir / "model.onnx").write_bytes(b"USER-EDITED")
    mat.materialize(artifact_dir, plan)
    assert (artifact_dir / "model.onnx").read_bytes() == b"USER-EDITED"


def test_materializer_retries_partial_extraction_without_marker(tmp_path):
    """Reviewer P2: if a previous extraction was interrupted after
    writing some files but before the marker, the next materialize
    call MUST re-extract — not silently treat the partial layout
    as complete."""
    tarball = _make_kokoro_layout_tarball(tmp_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "kokoro-en-v0_19.tar.bz2").write_bytes(tarball.read_bytes())
    # Simulate a partial extraction: model.onnx + voices.bin only.
    # No marker, no tokens.txt, no espeak-ng-data/.
    (artifact_dir / "model.onnx").write_bytes(b"PARTIAL-stale")
    (artifact_dir / "voices.bin").write_bytes(b"PARTIAL-stale")

    plan = MaterializationPlan(
        kind="archive",
        source="kokoro-en-v0_19.tar.bz2",
        archive_format="tar.bz2",
        strip_prefix="kokoro-en-v0_19/",
        required_outputs=("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/phontab"),
    )
    Materializer().materialize(artifact_dir, plan)

    # Re-extracted: model.onnx now matches the tarball, not the
    # stale partial content.
    assert (artifact_dir / "model.onnx").read_bytes() == b"fake-onnx"
    assert (artifact_dir / "tokens.txt").is_file()
    assert (artifact_dir / "espeak-ng-data" / "phontab").is_file()
    # Marker now present.
    assert (artifact_dir / EXTRACTION_MARKER).is_file()


def test_materializer_marker_only_present_when_layout_complete(tmp_path):
    """Defense in depth: if the tarball is missing one of the
    declared required outputs, materialize raises BEFORE writing
    the marker. A subsequent run must re-extract (not skip)."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = artifact_dir / "_payload"
    payload.write_bytes(b"only one file")
    tarball = artifact_dir / "incomplete.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tar:
        tar.add(payload, arcname="model.onnx")
    payload.unlink()

    plan = MaterializationPlan(
        kind="archive",
        source="incomplete.tar.bz2",
        archive_format="tar.bz2",
        required_outputs=("model.onnx", "voices.bin"),  # voices.bin missing from archive
    )
    with pytest.raises(OctomilError) as excinfo:
        Materializer().materialize(artifact_dir, plan)
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "voices.bin" in str(excinfo.value)
    # Marker NOT written.
    assert not (artifact_dir / EXTRACTION_MARKER).exists()


# ---------------------------------------------------------------------------
# Kokoro recipe + kernel.prepare integration
# ---------------------------------------------------------------------------


def test_kernel_prepare_falls_back_to_static_recipe(tmp_path, monkeypatch):
    """``client.prepare(model='kokoro-82m', capability='tts')`` works
    when the planner returned no candidate. The kernel hands the
    recipe's MaterializationPlan to the generic Materializer; the
    kernel itself knows nothing about Kokoro / tarballs."""
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome

    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

    # Stub PrepareManager so the test doesn't hit the network. Drop
    # a tarball where the materializer expects to find it.
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    tarball = _make_kokoro_layout_tarball(tmp_path)
    (artifact_dir / "kokoro-en-v0_19.tar.bz2").write_bytes(tarball.read_bytes())

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

    captured = {}

    class _StubPM:
        def can_prepare(self, candidate):
            return True

        def prepare(self, candidate, *, mode=None):
            captured["candidate"] = candidate
            return PrepareOutcome(
                artifact_id=candidate.artifact.artifact_id,
                artifact_dir=artifact_dir,
                files={f: artifact_dir / f for f in candidate.artifact.required_files},
                engine=candidate.engine,
                delivery_mode="sdk_runtime",
                prepare_policy="explicit_only",
                cached=False,
            )

    kernel._prepare_manager = _StubPM()

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        outcome = kernel.prepare(model="kokoro-82m", capability="tts")

    assert outcome.artifact_id == "kokoro-82m"
    # Materializer ran: required outputs on disk + marker present.
    assert (artifact_dir / "model.onnx").is_file()
    assert (artifact_dir / "voices.bin").is_file()
    assert (artifact_dir / "tokens.txt").is_file()
    assert (artifact_dir / "espeak-ng-data" / "phontab").is_file()
    assert (artifact_dir / EXTRACTION_MARKER).is_file()


def test_kernel_prepare_unknown_model_without_planner_surfaces_actionable_error(monkeypatch):
    """Reviewer requirement: when planner is offline AND no static
    recipe matches, the SDK must NOT silently substitute a public
    mirror. Error names ``OCTOMIL_SERVER_KEY`` and ``kokoro-82m``."""
    from unittest.mock import patch

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
    assert "kokoro-82m" in msg
