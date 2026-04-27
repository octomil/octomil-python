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


def test_materializer_zip_refuses_pre_existing_symlink_escape(tmp_path):
    """Reviewer P1 on PR #455 (post-636ff06): the zip path used to
    resolve member destinations as ``artifact_dir / target_name``
    after only string-level validation. If ``artifact_dir`` already
    contains a symlink pointing outside the dir (e.g.
    ``linkdir → /tmp/outside``), an archive member ``linkdir/
    escaped.txt`` lands at the symlink target.

    Fix: every destination goes through ``_safe_join_under`` which
    resolves symlinks and verifies the resolved path is under the
    resolved artifact_dir. Hostile member is skipped; legitimate
    members extract normally."""
    import zipfile

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_dir / "linkdir").symlink_to(outside, target_is_directory=True)

    archive_src = tmp_path / "_zip_src"
    archive_src.mkdir()
    safe_payload = archive_src / "safe.txt"
    safe_payload.write_bytes(b"safe content")
    escape_payload = archive_src / "escaped.txt"
    escape_payload.write_bytes(b"i should never land outside")

    archive = artifact_dir / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(safe_payload, arcname="safe.txt")
        z.write(escape_payload, arcname="linkdir/escaped.txt")

    plan = MaterializationPlan(
        kind="archive",
        source="evil.zip",
        archive_format="zip",
        required_outputs=("safe.txt",),
    )
    Materializer().materialize(artifact_dir, plan)

    assert (artifact_dir / "safe.txt").read_bytes() == b"safe content"
    assert not (outside / "escaped.txt").exists()


def test_materializer_tar_refuses_pre_existing_symlink_escape(tmp_path):
    """Same regression as the zip case, against the tar fallback
    path used on Python 3.9 (no ``filter='data'``). The
    ``_safe_join_under`` check runs both at member-collection time
    AND inside the per-member extract fallback."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_dir / "linkdir").symlink_to(outside, target_is_directory=True)

    src = tmp_path / "_tar_src"
    src.mkdir()
    safe = src / "safe.bin"
    safe.write_bytes(b"safe")
    escape = src / "escape.bin"
    escape.write_bytes(b"would escape")

    archive = artifact_dir / "evil.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(safe, arcname="safe.bin")
        tar.add(escape, arcname="linkdir/escaped.bin")

    plan = MaterializationPlan(
        kind="archive",
        source="evil.tar.bz2",
        archive_format="tar.bz2",
        required_outputs=("safe.bin",),
    )
    Materializer().materialize(artifact_dir, plan)

    assert (artifact_dir / "safe.bin").read_bytes() == b"safe"
    assert not (outside / "escaped.bin").exists()


def test_strip_prefix_is_an_allowlist_boundary(tmp_path):
    """Reviewer P2 on PR #455: when ``strip_prefix`` is set,
    archive members OUTSIDE the prefix used to be returned
    unchanged and accepted. A malformed archive with root-level
    ``model.onnx`` could then satisfy
    ``required_outputs=('model.onnx',)`` for a recipe that
    explicitly told us to expect everything under
    ``kokoro-en-v0_19/``. That defeats the prefix declaration.

    Fix: ``_apply_strip_prefix`` returns ``None`` for members
    outside the prefix, so they're skipped. ``required_outputs``
    can NOT be satisfied by content sitting at the wrong subtree."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    src = tmp_path / "_src"
    src.mkdir()
    # Misshapen archive: root-level ``model.onnx`` instead of the
    # expected ``kokoro-en-v0_19/model.onnx``.
    bad = src / "model.onnx"
    bad.write_bytes(b"misshapen-root-level")
    archive = artifact_dir / "rootfile.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(bad, arcname="model.onnx")

    plan = MaterializationPlan(
        kind="archive",
        source="rootfile.tar.bz2",
        archive_format="tar.bz2",
        strip_prefix="kokoro-en-v0_19/",
        required_outputs=("model.onnx",),
    )
    # Materializer must NOT accept the root-level member as
    # satisfying ``required_outputs`` — it should refuse with
    # missing-output because nothing under the prefix existed.
    with pytest.raises(OctomilError) as excinfo:
        Materializer().materialize(artifact_dir, plan)
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "model.onnx" in str(excinfo.value)
    assert not (artifact_dir / EXTRACTION_MARKER).exists()


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


# ---------------------------------------------------------------------------
# PR D: TTS dispatch consults the static-recipe prepared artifact cache
# ---------------------------------------------------------------------------


def _stage_kokoro_prepared_cache(tmp_path):
    """Materialize a complete Kokoro layout under PrepareManager's
    deterministic cache dir for kokoro-82m. Returns the artifact dir.

    Mirrors what ``client.prepare(model='kokoro-82m', capability='tts')``
    leaves on disk after a successful prepare + materialize: bytes
    extracted, marker present, every required output in place.

    Uses ``PrepareManager()`` with no ``cache_dir`` override so the
    resolution path matches what the kernel does at dispatch time
    (``OCTOMIL_CACHE_DIR`` → ``<root>/artifacts`` → ArtifactCache).
    Tests must set ``OCTOMIL_CACHE_DIR`` via ``monkeypatch`` before
    calling this helper so the staged dir and the kernel's lookup
    land at identical paths.
    """
    from octomil.runtime.lifecycle.materialization import Materializer
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    manager = PrepareManager()
    artifact_dir = manager.artifact_dir_for(recipe.model_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tarball = _make_kokoro_layout_tarball(tmp_path)
    (artifact_dir / "kokoro-en-v0_19.tar.bz2").write_bytes(tarball.read_bytes())
    Materializer().materialize(artifact_dir, recipe.materialization)
    return artifact_dir


def test_has_local_tts_backend_counts_prepared_static_recipe_cache(tmp_path, monkeypatch):
    """PR D contract: a complete prepared layout under PrepareManager's
    artifact cache is the *only* on-disk source counted as local
    availability after the legacy staging cutover. With the runtime
    importable AND the cache present, ``_has_local_tts_backend`` must
    return True even with no planner candidate and no legacy
    ``~/.octomil/models/sherpa`` staging on disk."""
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    _stage_kokoro_prepared_cache(tmp_path)

    kernel = ExecutionKernel()
    # Patch the runtime-loadable check to True so this test isolates
    # the prepared-cache contract from sherpa-onnx availability
    # (covered by the symmetric test below).
    with patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True):
        assert kernel._has_local_tts_backend("kokoro-82m") is True
        # Aliases registered against the same recipe also count.
        assert kernel._has_local_tts_backend("kokoro-en-v0_19") is True


def test_has_local_tts_backend_false_without_prepared_cache(tmp_path, monkeypatch):
    """Cutover symmetry: without a prepared artifact dir there is no
    on-disk source. ``_has_local_tts_backend`` must return False so
    the route-selection chain falls back to ``_can_prepare_local_tts``
    (planner candidate route) or fails closed."""
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty"))

    kernel = ExecutionKernel()
    with patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True):
        assert kernel._has_local_tts_backend("kokoro-82m") is False


def test_has_local_tts_backend_false_when_runtime_not_loadable(tmp_path, monkeypatch):
    """P2 fix: cache-without-runtime is NOT local availability. With
    a complete prepared layout on disk but sherpa-onnx unimportable
    (or the model id unrecognized), ``_has_local_tts_backend`` must
    return False so ``local_first`` falls back to cloud rather than
    committing to local and raising "could not load sherpa backend"."""
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    _stage_kokoro_prepared_cache(tmp_path)

    kernel = ExecutionKernel()
    with patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=False):
        assert kernel._has_local_tts_backend("kokoro-82m") is False


def test_prepared_local_artifact_dir_returns_none_for_unknown_capability(tmp_path, monkeypatch):
    """The helper is generic across capabilities; capabilities with
    no static recipe registered (today: chat, embedding) must return
    None unconditionally — never silently substitute a TTS recipe."""
    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    _stage_kokoro_prepared_cache(tmp_path)

    kernel = ExecutionKernel()
    assert kernel._prepared_local_artifact_dir("chat", "kokoro-82m") is None
    assert kernel._prepared_local_artifact_dir("embedding", "kokoro-82m") is None
    # And TTS still resolves so the negative cases above prove the
    # cache + recipe were correctly staged.
    assert kernel._prepared_local_artifact_dir("tts", "kokoro-82m") is not None


def test_prepared_local_artifact_dir_idempotent_on_complete_layout(tmp_path, monkeypatch):
    """Re-running the helper against an already-materialized cache is
    cheap (marker check; no extraction / no I/O on file contents).
    User-edited bytes survive the second call — proves the helper
    didn't re-run extraction over a complete layout."""
    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    artifact_dir = _stage_kokoro_prepared_cache(tmp_path)

    kernel = ExecutionKernel()
    first = kernel._prepared_local_artifact_dir("tts", "kokoro-82m")
    assert first == str(artifact_dir)
    # Mutate one of the required outputs; if the helper re-extracts
    # we'd see this byte string overwritten by the tarball content.
    (artifact_dir / "model.onnx").write_bytes(b"USER-EDITED-AFTER-PREPARE")
    second = kernel._prepared_local_artifact_dir("tts", "kokoro-82m")
    assert second == str(artifact_dir)
    assert (artifact_dir / "model.onnx").read_bytes() == b"USER-EDITED-AFTER-PREPARE"


def test_synthesize_speech_uses_prepared_static_recipe_when_planner_offline(tmp_path, monkeypatch):
    """End-to-end PR D regression. The user's release-blocking story:

        client.prepare(model='kokoro-82m', capability='tts')   # PR C
        client.audio.speech.create(model='kokoro-82m', ...)    # PR D

    With the planner offline (returning no candidate), step 2 must
    load from the prepared artifact dir step 1 wrote — the kernel
    threads the cache dir into ``SherpaTtsEngine.create_backend(
    model_dir=...)`` instead of falling through to the (removed)
    legacy staging path or routing to cloud."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

    artifact_dir = _stage_kokoro_prepared_cache(tmp_path)

    captured: dict = {}

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            captured["create_backend_kwargs"] = kwargs
            captured["create_backend_model"] = model_name

        def load_model(self, model_name):
            captured["load_model"] = model_name

        def synthesize(self, text, voice, speed):
            return {
                "audio_bytes": b"RIFF\x00\x00\x00\x00WAVEfake",
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 50,
            }

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=None))
        response = asyncio.get_event_loop().run_until_complete(
            kernel.synthesize_speech(model="kokoro-82m", input="hello")
        )

    assert response.route.locality == "on_device"
    assert response.route.engine == "sherpa-onnx"
    assert response.audio_bytes.startswith(b"RIFF")
    # The prepared cache dir was threaded into the backend — not
    # the legacy ``~/.octomil/models/sherpa`` path.
    assert captured["create_backend_kwargs"] == {"model_dir": str(artifact_dir)}
    assert captured["load_model"] == "kokoro-82m"


def test_synthesize_speech_local_first_wins_when_prepared_cache_present(tmp_path, monkeypatch):
    """Reviewer regression: under ``local_first`` with the planner
    offline AND a prepared static-recipe cache on disk, local
    routing must win. Earlier the kernel would treat
    ``local_available=False`` (because ``is_sherpa_tts_model_staged``
    only consulted the legacy dirs) and silently route to cloud
    when cloud creds were present. PR D fixes this by counting the
    prepared cache as local availability."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import AsyncMock, patch

    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    artifact_dir = _stage_kokoro_prepared_cache(tmp_path)

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            self.kwargs = kwargs

        def load_model(self, model_name):
            return None

        def synthesize(self, text, voice, speed):
            return {
                "audio_bytes": b"RIFF\x00\x00\x00\x00WAVE",
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 50,
            }

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

    kernel = ExecutionKernel()
    # local_first defaults; no cloud_profile → cloud_available=False
    # naturally, but spy on _cloud_synthesize_speech to make any
    # accidental cloud dispatch a hard test failure.
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "kokoro-82m",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()
    cloud_spy = AsyncMock()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=None))
        stack.enter_context(patch.object(kernel, "_cloud_synthesize_speech", new=cloud_spy))
        response = asyncio.get_event_loop().run_until_complete(
            kernel.synthesize_speech(model="kokoro-82m", input="hello")
        )

    assert response.route.locality == "on_device"
    cloud_spy.assert_not_called()
    # Sanity: the prepared artifact path landed at the deterministic
    # cache_dir/artifacts/<key> location (no env override leak).
    assert str(artifact_dir).startswith(str(tmp_path / "cache" / "artifacts"))


def test_synthesize_speech_prepared_cache_short_circuits_echo_only_synthetic_candidate(tmp_path, monkeypatch):
    """The cache short-circuit fires when the planner echoes the
    runtime model name back without committing to a specific
    artifact (no digest, no urls, ``artifact_id`` either missing or
    equal to the runtime model). For a *direct* request — the user
    typed ``model='kokoro-82m'`` — the public static recipe is
    exactly what they asked for; using it as the silent fallback is
    safe.

    Contrast: a candidate with a *different* artifact_id (e.g.
    ``'private-kokoro-v2'``) names a specific other artifact and
    must NOT be substituted by the cache — pinned in
    ``test_synthesize_speech_app_or_mismatched_identity_does_not_short_circuit_cache``."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.planner.schemas import RuntimeArtifactPlan, RuntimeCandidatePlan

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    artifact_dir = _stage_kokoro_prepared_cache(tmp_path)

    class _Selection:
        candidates: list
        locality = None
        engine = None
        artifact = None
        source = None
        fallback_allowed = True
        reason = ""
        app_resolution = None
        resolution = None

        def __init__(self, candidates):
            self.candidates = candidates

    # Echo-only synthetic candidate: artifact_id == runtime_model,
    # no digest, no download_urls. The planner gave us the model
    # name back without naming a specific artifact version.
    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="echo-only-synthetic",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(model_id="kokoro-82m", artifact_id="kokoro-82m"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[synthetic])

    captured: dict = {}

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            captured["create_backend_kwargs"] = kwargs

        def load_model(self, model_name):
            return None

        def synthesize(self, text, voice, speed):
            return {
                "audio_bytes": b"RIFF\x00\x00\x00\x00WAVE",
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 50,
            }

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

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
    # Spy on _prepare_local_tts_artifact: it must NOT be called when
    # the prepared cache short-circuits routing.
    prepare_spy = patch.object(
        kernel,
        "_prepare_local_tts_artifact",
        side_effect=AssertionError(
            "_prepare_local_tts_artifact must not run when the prepared cache short-circuits routing"
        ),
    )

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(prepare_spy)
        response = asyncio.get_event_loop().run_until_complete(
            kernel.synthesize_speech(model="kokoro-82m", input="hello")
        )

    assert response.route.locality == "on_device"
    assert response.route.engine == "sherpa-onnx"
    # The cache dir — NOT a synthetic candidate's prepare outcome —
    # was threaded into the backend.
    assert captured["create_backend_kwargs"] == {"model_dir": str(artifact_dir)}


def test_synthesize_speech_prepared_cache_does_not_shadow_planner_selected_artifact(tmp_path, monkeypatch):
    """Reviewer P1 (round 2): the prepared static cache must NOT
    shadow a *legitimate* planner-selected artifact whose identity
    differs from the static recipe's. Repro: complete ``kokoro-82m``
    static cache on disk + planner local candidate for the same
    runtime model with ``artifact_id='private-kokoro-v2'``, a real
    digest, and a real download_url. Earlier the cache short-circuit
    was keyed only by ``runtime_model``, so dispatch loaded the
    static cache dir instead of running the planner candidate's
    ``prepare()`` and serving the planner-selected artifact.

    Fix: short-circuit only when (no candidate) OR (candidate
    unpreparable) OR (candidate identity matches the static recipe).
    A real, preparable, identity-mismatched candidate must run
    through ``_prepare_local_tts_artifact`` — the static cache stays
    untouched."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome
    from octomil.runtime.planner.schemas import (
        ArtifactDownloadEndpoint,
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    static_cache_dir = _stage_kokoro_prepared_cache(tmp_path)

    # A *real* planner candidate for the same runtime model but a
    # different artifact identity — same shape PrepareManager
    # accepts (digest + at least one url), so ``can_prepare`` returns
    # True and the unpreparable veto does NOT fire.
    private_candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="planner-selected-private-artifact",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-82m",
            artifact_id="private-kokoro-v2",
            digest="sha256:" + "a" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://private.example.com/")],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )

    class _Selection:
        candidates: list
        locality = None
        engine = None
        artifact = None
        source = None
        fallback_allowed = True
        reason = ""
        app_resolution = None
        resolution = None

        def __init__(self, candidates):
            self.candidates = candidates

    selection = _Selection(candidates=[private_candidate])

    private_dir = tmp_path / "private-kokoro-v2-dir"
    private_dir.mkdir()

    captured: dict = {}
    prepare_calls: list = []

    class _StubPM:
        def can_prepare(self, candidate):
            return True

        def prepare(self, candidate, *, mode=None):
            prepare_calls.append(candidate.artifact.artifact_id)
            return PrepareOutcome(
                artifact_id=candidate.artifact.artifact_id,
                artifact_dir=private_dir,
                files={"": private_dir / "artifact"},
                engine=candidate.engine,
                delivery_mode=candidate.delivery_mode or "sdk_runtime",
                prepare_policy=candidate.prepare_policy,
                cached=False,
            )

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            captured["create_backend_kwargs"] = kwargs

        def load_model(self, model_name):
            return None

        def synthesize(self, text, voice, speed):
            return {
                "audio_bytes": b"RIFF\x00\x00\x00\x00WAVE",
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 50,
            }

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

    kernel = ExecutionKernel(prepare_manager=_StubPM())
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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        response = asyncio.get_event_loop().run_until_complete(
            kernel.synthesize_speech(model="kokoro-82m", input="hello")
        )

    # The planner candidate's prepare() ran exactly once for the
    # private artifact id — the static cache did NOT shadow it.
    assert prepare_calls == ["private-kokoro-v2"]
    # The backend loaded the planner-selected artifact dir, NOT the
    # static-recipe cache dir.
    assert captured["create_backend_kwargs"] == {"model_dir": str(private_dir)}
    assert captured["create_backend_kwargs"]["model_dir"] != str(static_cache_dir)
    assert response.route.locality == "on_device"


def _selection_with(candidates):
    class _Selection:
        locality = None
        engine = None
        artifact = None
        source = None
        fallback_allowed = True
        reason = ""
        app_resolution = None
        resolution = None

        def __init__(self, c):
            self.candidates = c

    return _Selection(candidates)


def test_synthesize_speech_app_scoped_synthetic_candidate_surfaces_planner_error(tmp_path, monkeypatch):
    """Reviewer P1 (round 4): app-scoped requests must NOT silently
    fall back to the public static-recipe cache when the planner
    returns a synthetic candidate. The user asked for the app's
    artifact (``@app/tts-tester/tts``); substituting the public
    Kokoro hides Task #51 (server bug) and serves the wrong bytes.

    Repro: ``@app/tts-tester/tts`` resolves to runtime model
    ``kokoro-82m``; planner returns local candidate with
    ``artifact_id='private-kokoro-v2'`` and no ``download_urls``;
    static cache present. Expected: ``OctomilError`` from
    PrepareManager (no download_urls), NOT a successful WAV from
    the public cache."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.errors import OctomilError
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.planner.schemas import RuntimeArtifactPlan, RuntimeCandidatePlan

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    _stage_kokoro_prepared_cache(tmp_path)

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="server-bug-synthetic",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(model_id="kokoro-82m", artifact_id="private-kokoro-v2"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _selection_with([synthetic])

    backend_calls: list = []

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            backend_calls.append(kwargs)

        def load_model(self, model_name):
            return None

        def synthesize(self, *a, **kw):
            return {"audio_bytes": b"unreachable", "content_type": "audio/wav", "format": "wav"}

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        with pytest.raises(OctomilError) as excinfo:
            asyncio.get_event_loop().run_until_complete(
                kernel.synthesize_speech(model="@app/tts-tester/tts", input="hello")
            )

    # The error came from the routing/prepare path, not the cache.
    # Backend never loaded — the public Kokoro bytes were not
    # silently substituted for the app's artifact.
    assert backend_calls == []
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE


def test_synthesize_speech_direct_mismatched_identity_does_not_short_circuit_cache(tmp_path, monkeypatch):
    """Reviewer P1 (round 4) symmetric: a *direct* request with a
    candidate whose artifact_id names a different artifact must
    also surface the planner error, not silently use the public
    cache. The candidate has a meaningful identity (the planner
    named ``private-kokoro-v2``); substituting the public Kokoro
    would serve different bytes than what was selected."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.errors import OctomilError
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.planner.schemas import RuntimeArtifactPlan, RuntimeCandidatePlan

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    _stage_kokoro_prepared_cache(tmp_path)

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="mismatched-id-no-urls",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(model_id="kokoro-82m", artifact_id="private-kokoro-v2"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _selection_with([synthetic])

    backend_calls: list = []

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            backend_calls.append(kwargs)

        def load_model(self, model_name):
            return None

        def synthesize(self, *a, **kw):
            return {"audio_bytes": b"unreachable", "content_type": "audio/wav", "format": "wav"}

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        with pytest.raises(OctomilError):
            asyncio.get_event_loop().run_until_complete(kernel.synthesize_speech(model="kokoro-82m", input="hello"))

    assert backend_calls == [], "backend must not load when the planner candidate's identity is mismatched"


def test_synthesize_speech_app_scoped_no_candidate_does_not_short_circuit_cache(tmp_path, monkeypatch):
    """Reviewer P1 (round 4): an app-scoped request with NO local
    candidate (planner offline / could not resolve the app) must
    also refuse the cache substitution. The user asked for the
    app's artifact; substituting the public static recipe would
    swallow the planner outage. Identity match (a) does not apply
    because there is no candidate to match against."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.errors import OctomilError
    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    _stage_kokoro_prepared_cache(tmp_path)

    selection = _selection_with([])

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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        with pytest.raises(OctomilError):
            asyncio.get_event_loop().run_until_complete(
                kernel.synthesize_speech(model="@app/tts-tester/tts", input="hello")
            )


def test_synthesize_speech_prepared_cache_used_when_candidate_matches_static_recipe(tmp_path, monkeypatch):
    """Symmetric pin: when the planner *does* select the static-recipe
    artifact (same artifact_id, same digest), the cache short-circuit
    fires and the planner's ``prepare()`` is skipped — the cached
    bytes are bit-identical to what prepare would have produced, so
    re-downloading is wasted work."""
    import asyncio
    from contextlib import ExitStack
    from unittest.mock import patch

    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe
    from octomil.runtime.planner.schemas import (
        ArtifactDownloadEndpoint,
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
    monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
    static_cache_dir = _stage_kokoro_prepared_cache(tmp_path)

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    matching_candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=1.0,
        reason="planner-selected-static-recipe-shape",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-82m",
            artifact_id=recipe.model_id,  # matches static recipe
            digest=recipe.files[0].digest,  # matches static recipe
            download_urls=[ArtifactDownloadEndpoint(url=recipe.files[0].url)],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )

    class _Selection:
        candidates: list
        locality = None
        engine = None
        artifact = None
        source = None
        fallback_allowed = True
        reason = ""
        app_resolution = None
        resolution = None

        def __init__(self, candidates):
            self.candidates = candidates

    selection = _Selection(candidates=[matching_candidate])

    captured: dict = {}
    prepare_calls: list = []

    # Wrap a real PrepareManager so ``artifact_dir_for(...)`` returns
    # the same cache path the static-recipe staging used. The stub
    # only intercepts ``prepare`` to assert it isn't called.
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager as _RealPM

    real_pm = _RealPM()

    class _StubPM:
        def can_prepare(self, candidate):
            return True

        def artifact_dir_for(self, artifact_id):
            return real_pm.artifact_dir_for(artifact_id)

        def prepare(self, candidate, *, mode=None):
            prepare_calls.append(candidate.artifact.artifact_id)
            raise AssertionError(
                "prepare() must not run when the planner candidate matches the static recipe and the cache is on disk"
            )

    class _FakeBackend:
        def __init__(self, model_name, **kwargs):
            captured["create_backend_kwargs"] = kwargs

        def load_model(self, model_name):
            return None

        def synthesize(self, text, voice, speed):
            return {
                "audio_bytes": b"RIFF\x00\x00\x00\x00WAVE",
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 50,
            }

    class _FakeEngine:
        def create_backend(self, model_name, **kwargs):
            return _FakeBackend(model_name, **kwargs)

    kernel = ExecutionKernel(prepare_manager=_StubPM())
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

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available", return_value=True))
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        response = asyncio.get_event_loop().run_until_complete(
            kernel.synthesize_speech(model="kokoro-82m", input="hello")
        )

    assert prepare_calls == [], "matching candidate must not invoke PrepareManager.prepare"
    assert captured["create_backend_kwargs"] == {"model_dir": str(static_cache_dir)}
    assert response.route.locality == "on_device"
