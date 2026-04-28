"""PR C-followup: multi-file ``manifest_uri`` + first-class
``source="static_recipe"`` planner contract.

Two halves:

  1. ``octomil.runtime.lifecycle.manifest`` parses + verifies a
     ``manifest.v1.json`` payload, fetches it under the artifact-
     level ``digest`` pin, and emits a per-file ``RequiredFile`` list
     ``DurableDownloader`` consumes.
  2. ``RuntimeArtifactPlan.source="static_recipe"`` + ``recipe_id``
     instructs the SDK to expand the candidate from its built-in
     recipe table — the dashboard says "use kokoro-82m" without
     re-publishing canonical URL/digest/required_files in every plan.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.lifecycle.manifest import (
    ParsedManifest,
    parse_manifest_payload,
)
from octomil.runtime.lifecycle.prepare_manager import PrepareManager
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)

# ---------------------------------------------------------------------------
# parse_manifest_payload
# ---------------------------------------------------------------------------


def test_parses_well_formed_manifest():
    payload = {
        "version": 1,
        "files": [
            {"relative_path": "model.onnx", "digest": "sha256:" + "a" * 64, "size_bytes": 1024},
            {"relative_path": "voices.bin", "digest": "sha256:" + "b" * 64, "size_bytes": 2048},
        ],
    }
    parsed = parse_manifest_payload(payload)
    assert isinstance(parsed, ParsedManifest)
    assert parsed.version == 1
    assert [f.relative_path for f in parsed.files] == ["model.onnx", "voices.bin"]
    assert parsed.files[0].digest == "sha256:" + "a" * 64
    assert parsed.files[1].size_bytes == 2048


def test_rejects_unknown_version():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload({"version": 2, "files": [{"relative_path": "x", "digest": "sha256:" + "0" * 64}]})
    assert exc.value.code == OctomilErrorCode.INVALID_INPUT
    assert "version" in str(exc.value).lower()


def test_rejects_non_object_payload():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload(["not", "an", "object"])
    assert "object" in str(exc.value).lower()


def test_rejects_missing_files():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload({"version": 1, "files": []})
    assert "non-empty list" in str(exc.value)


def test_rejects_empty_relative_path_in_multi_file_manifest():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload(
            {
                "version": 1,
                "files": [
                    {"relative_path": "", "digest": "sha256:" + "a" * 64},
                ],
            }
        )
    assert "must not be empty" in str(exc.value)


def test_rejects_traversal_relative_path():
    with pytest.raises(OctomilError):
        parse_manifest_payload(
            {
                "version": 1,
                "files": [
                    {"relative_path": "../etc/passwd", "digest": "sha256:" + "a" * 64},
                ],
            }
        )


def test_rejects_duplicate_relative_paths():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload(
            {
                "version": 1,
                "files": [
                    {"relative_path": "model.onnx", "digest": "sha256:" + "a" * 64},
                    {"relative_path": "model.onnx", "digest": "sha256:" + "b" * 64},
                ],
            }
        )
    assert "duplicate" in str(exc.value)


def test_rejects_malformed_digest():
    with pytest.raises(OctomilError) as exc:
        parse_manifest_payload(
            {
                "version": 1,
                "files": [
                    {"relative_path": "model.onnx", "digest": "not-hex"},
                ],
            }
        )
    assert "digest" in str(exc.value).lower()


def test_accepts_bare_hex_digest_without_prefix():
    payload = {
        "version": 1,
        "files": [
            {"relative_path": "model.onnx", "digest": "a" * 64},
        ],
    }
    parsed = parse_manifest_payload(payload)
    assert parsed.files[0].digest == "a" * 64


def test_rejects_negative_size():
    with pytest.raises(OctomilError):
        parse_manifest_payload(
            {
                "version": 1,
                "files": [
                    {"relative_path": "x", "digest": "sha256:" + "a" * 64, "size_bytes": -1},
                ],
            }
        )


# ---------------------------------------------------------------------------
# PrepareManager._build_descriptor with manifest_uri
# ---------------------------------------------------------------------------


def test_build_descriptor_rejects_multifile_without_manifest_uri():
    pm = PrepareManager()
    artifact = RuntimeArtifactPlan(
        model_id="multi",
        digest="sha256:" + "0" * 64,
        required_files=["a.bin", "b.bin"],
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
    )
    with pytest.raises(OctomilError) as exc:
        pm._build_descriptor(artifact)
    assert "manifest_uri" in str(exc.value)


def test_build_descriptor_with_manifest_uri_routes_through_manifest(tmp_path, monkeypatch):
    """Multi-file artifact + manifest_uri -> per-file digests from the
    manifest, planner-required files cross-checked."""
    manifest_payload = {
        "version": 1,
        "files": [
            {"relative_path": "model.onnx", "digest": "sha256:" + "a" * 64, "size_bytes": 1024},
            {"relative_path": "voices.bin", "digest": "sha256:" + "b" * 64, "size_bytes": 2048},
        ],
    }
    body = json.dumps(manifest_payload).encode()
    body_digest = "sha256:" + hashlib.sha256(body).hexdigest()

    # Stub fetch_and_parse_manifest so the test stays offline.
    from octomil.runtime.lifecycle import manifest as manifest_mod

    def fake_fetch(uri, *, artifact_digest, **_kw):
        assert uri == "https://x/manifest.json"
        assert artifact_digest == body_digest
        return manifest_mod.parse_manifest_payload(manifest_payload)

    monkeypatch.setattr(manifest_mod, "fetch_and_parse_manifest", fake_fetch)
    # Also patch the import used inside prepare_manager.
    from octomil.runtime.lifecycle import prepare_manager as pm_mod

    monkeypatch.setattr(pm_mod, "_validate_relative_path", pm_mod._validate_relative_path)

    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-multi",
        digest=body_digest,
        required_files=["model.onnx", "voices.bin"],
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
        manifest_uri="https://x/manifest.json",
    )
    descriptor = pm._build_descriptor(artifact)
    assert [f.relative_path for f in descriptor.required_files] == ["model.onnx", "voices.bin"]
    assert descriptor.required_files[0].digest == "sha256:" + "a" * 64
    assert descriptor.required_files[1].digest == "sha256:" + "b" * 64


def test_build_descriptor_rejects_required_files_not_in_manifest(tmp_path, monkeypatch):
    """Planner says we need ``z.bin`` but manifest doesn't include it."""
    manifest_payload = {
        "version": 1,
        "files": [{"relative_path": "model.onnx", "digest": "sha256:" + "a" * 64}],
    }
    from octomil.runtime.lifecycle import manifest as manifest_mod

    def fake_fetch(uri, *, artifact_digest, **_kw):
        return manifest_mod.parse_manifest_payload(manifest_payload)

    monkeypatch.setattr(manifest_mod, "fetch_and_parse_manifest", fake_fetch)

    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="multi",
        digest="sha256:" + "0" * 64,
        required_files=["model.onnx", "z.bin"],
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
        manifest_uri="https://x/manifest.json",
    )
    with pytest.raises(OctomilError) as exc:
        pm._build_descriptor(artifact)
    assert "z.bin" in str(exc.value)
    assert "manifest" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# source="static_recipe" expansion
# ---------------------------------------------------------------------------


def test_static_recipe_source_expands_from_table(tmp_path):
    """Planner emits ``source='static_recipe', recipe_id='kokoro-82m'``;
    SDK expands canonical URL/digest from its built-in recipe table."""
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-82m",
        source="static_recipe",
        recipe_id="kokoro-82m",
    )
    expanded, recipe = pm._expand_static_recipe_source(artifact)
    # Canonical Kokoro digest from static_recipes.py
    assert expanded.digest == "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"
    assert expanded.required_files == ["kokoro-en-v0_19.tar.bz2"]
    assert len(expanded.download_urls) == 1
    assert expanded.download_urls[0].url == "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"


def test_static_recipe_source_passes_through_when_absent(tmp_path):
    """When ``source`` is not set, behavior is unchanged."""
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-82m",
        artifact_id="kokoro-82m",
        digest="sha256:" + "1" * 64,
        download_urls=[ArtifactDownloadEndpoint(url="https://planner.example/x")],
    )
    expanded, recipe = pm._expand_static_recipe_source(artifact)
    assert expanded is artifact
    assert recipe is None


def test_static_recipe_source_rejects_unknown_source(tmp_path):
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(model_id="x", source="hf_snapshot")
    with pytest.raises(OctomilError) as exc:
        pm._expand_static_recipe_source(artifact)
    assert "hf_snapshot" in str(exc.value)


def test_static_recipe_source_rejects_missing_recipe_id(tmp_path):
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(model_id="x", source="static_recipe")
    with pytest.raises(OctomilError) as exc:
        pm._expand_static_recipe_source(artifact)
    assert "recipe_id" in str(exc.value)


def test_static_recipe_source_rejects_unknown_recipe_id(tmp_path):
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(model_id="x", source="static_recipe", recipe_id="nonexistent-private-app")
    with pytest.raises(OctomilError) as exc:
        pm._expand_static_recipe_source(artifact)
    assert "nonexistent-private-app" in str(exc.value)


def test_static_recipe_source_refuses_planner_digest_mismatch(tmp_path):
    """Reviewer's safety: a compromised planner cannot point users at
    forged bytes under a known recipe id by overriding the digest."""
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-82m",
        source="static_recipe",
        recipe_id="kokoro-82m",
        digest="sha256:" + "f" * 64,  # not the canonical Kokoro digest
    )
    with pytest.raises(OctomilError) as exc:
        pm._expand_static_recipe_source(artifact)
    assert exc.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
    assert "kokoro-82m" in str(exc.value)


def test_static_recipe_source_refuses_required_files_mismatch(tmp_path):
    pm = PrepareManager(cache_dir=tmp_path)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-82m",
        source="static_recipe",
        recipe_id="kokoro-82m",
        required_files=["wrong-file.bin"],
    )
    with pytest.raises(OctomilError) as exc:
        pm._expand_static_recipe_source(artifact)
    assert "wrong-file.bin" in str(exc.value)


def test_static_recipe_source_full_prepare_pipeline(tmp_path, monkeypatch):
    """End-to-end: static-recipe candidate -> expanded artifact ->
    descriptor matches the canonical Kokoro shape."""
    pm = PrepareManager(cache_dir=tmp_path)
    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=1.0,
        reason="planner-static-recipe",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-82m",
            source="static_recipe",
            recipe_id="kokoro-82m",
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    expanded, recipe = pm._expand_static_recipe_source(candidate.artifact)
    assert recipe is not None
    descriptor = pm._build_descriptor(expanded)
    assert descriptor.artifact_id == "kokoro-82m"
    assert len(descriptor.required_files) == 1
    assert descriptor.required_files[0].relative_path == "kokoro-en-v0_19.tar.bz2"
    assert (
        descriptor.required_files[0].digest == "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"
    )


# ---------------------------------------------------------------------------
# End-to-end PrepareManager.prepare contracts (P1 reviewer regressions)
# ---------------------------------------------------------------------------


def _make_kokoro_layout_tarball(tmp_path):
    """Mimic the upstream Kokoro tarball layout for offline testing."""
    import tarfile

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


def test_prepare_with_source_static_recipe_runs_validation_and_materialization(tmp_path, monkeypatch):
    """Reviewer P1 (#4 + #6): full ``PrepareManager.prepare(...)``
    pipeline with ``source='static_recipe'``.

    1. Validator must NOT reject the candidate for missing
       download_urls / digest — those come from the registry.
    2. Download runs; bytes land at ``<cache>/<key>/`` as the
       canonical Kokoro tarball.
    3. Post-download Materializer extracts the tarball so the
       artifact_dir contains ``model.onnx`` + ``voices.bin`` +
       ``tokens.txt`` + ``espeak-ng-data/phontab`` — NOT just
       ``kokoro-en-v0_19.tar.bz2``.
    """
    import hashlib

    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    # Build a Kokoro-shape tarball, compute its real digest, register
    # an in-process recipe pointing at our local fixture, and stub
    # the downloader to copy bytes from disk instead of hitting the
    # network. We can't override _RECIPES at import (frozen
    # dataclasses), so we override the durable downloader and
    # inject the digest via a stubbed _RECIPES.
    tarball = _make_kokoro_layout_tarball(tmp_path)
    real_digest = "sha256:" + hashlib.sha256(tarball.read_bytes()).hexdigest()

    # Build a fresh recipe under a new id so we don't tread on the
    # global Kokoro entry, and register it.
    from octomil.runtime.lifecycle import static_recipes as recipes_mod
    from octomil.runtime.lifecycle.materialization import (
        MaterializationPlan,
        MaterializationSafetyPolicy,
    )

    test_recipe = recipes_mod.StaticRecipe(
        model_id="kokoro-test",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            recipes_mod._StaticArtifactFile(
                relative_path="kokoro-en-v0_19.tar.bz2",
                url="https://test.example.com/",
                digest=real_digest,
            )
        ],
        materialization=MaterializationPlan(
            kind="archive",
            source="kokoro-en-v0_19.tar.bz2",
            archive_format="tar.bz2",
            strip_prefix="kokoro-en-v0_19/",
            required_outputs=("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/phontab"),
            safety_policy=MaterializationSafetyPolicy(),
        ),
    )
    monkeypatch.setitem(recipes_mod._RECIPES, ("kokoro-test", "tts"), test_recipe)

    # Stub the downloader: copy our local tarball to the .part path.
    cache_dir = tmp_path / "cache"
    pm = PrepareManager(cache_dir=cache_dir)

    from octomil.runtime.lifecycle.durable_download import DownloadResult

    real_download = pm._downloader.download

    def stub_download(descriptor, dest_dir):  # noqa: ARG001
        for required in descriptor.required_files:
            target = dest_dir / required.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tarball.read_bytes())
        return DownloadResult(
            artifact_id=descriptor.artifact_id,
            files={r.relative_path: dest_dir / r.relative_path for r in descriptor.required_files},
        )

    pm._downloader.download = stub_download  # type: ignore[method-assign]

    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=1.0,
        reason="planner-static-recipe",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-test",
            source="static_recipe",
            recipe_id="kokoro-test",
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )

    outcome = pm.prepare(candidate)

    # Materializer ran: required outputs exist on disk.
    assert (outcome.artifact_dir / "model.onnx").is_file()
    assert (outcome.artifact_dir / "voices.bin").is_file()
    assert (outcome.artifact_dir / "tokens.txt").is_file()
    assert (outcome.artifact_dir / "espeak-ng-data" / "phontab").is_file()
    # Restore for the rest of the suite
    pm._downloader.download = real_download  # type: ignore[method-assign]


def test_prepare_with_explicit_only_admitted_under_explicit_mode(tmp_path, monkeypatch):
    """Reviewer P1 (Node parallel): explicit_only candidates must
    succeed under PrepareMode.EXPLICIT through the public path."""
    import hashlib

    from octomil.runtime.lifecycle.durable_download import DownloadResult
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

    payload = b"explicit prepare payload"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    pm = PrepareManager(cache_dir=tmp_path)

    def stub_download(descriptor, dest_dir):  # noqa: ARG001
        for required in descriptor.required_files:
            target = (dest_dir / "artifact") if not required.relative_path else dest_dir / required.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return DownloadResult(
            artifact_id=descriptor.artifact_id,
            files={
                r.relative_path: (dest_dir / "artifact") if not r.relative_path else dest_dir / r.relative_path
                for r in descriptor.required_files
            },
        )

    pm._downloader.download = stub_download  # type: ignore[method-assign]

    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=1.0,
        reason="explicit-only-test",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="m",
            artifact_id="m",
            digest=digest,
            download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="explicit_only",
    )

    # Lazy mode rejects.
    with pytest.raises(OctomilError) as exc:
        pm.prepare(candidate, mode=PrepareMode.LAZY)
    assert "explicit_only" in str(exc.value)

    # Explicit mode admits.
    outcome = pm.prepare(candidate, mode=PrepareMode.EXPLICIT)
    assert outcome.cached is False
    assert (outcome.artifact_dir / "artifact").read_bytes() == payload


def test_prepare_admits_multifile_with_manifest_uri(tmp_path, monkeypatch):
    """Reviewer P1 (#5): ``PrepareManager.prepare`` (not just
    _build_descriptor) must accept multi-file artifacts when paired
    with a manifest_uri. Pre-fix, the validator rejected with
    'planned follow-up' before the descriptor builder could resolve
    the manifest."""
    import hashlib

    from octomil.runtime.lifecycle import manifest as manifest_mod
    from octomil.runtime.lifecycle.durable_download import DownloadResult
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    file_a = b"file-a-bytes"
    file_b = b"file-b-bytes"
    digest_a = "sha256:" + hashlib.sha256(file_a).hexdigest()
    digest_b = "sha256:" + hashlib.sha256(file_b).hexdigest()
    manifest_payload = {
        "version": 1,
        "files": [
            {"relative_path": "a.bin", "digest": digest_a, "size_bytes": len(file_a)},
            {"relative_path": "b.bin", "digest": digest_b, "size_bytes": len(file_b)},
        ],
    }
    import json as _json

    body = _json.dumps(manifest_payload).encode()
    body_digest = "sha256:" + hashlib.sha256(body).hexdigest()

    def fake_fetch(uri, *, artifact_digest, **_kw):  # noqa: ARG001
        return manifest_mod.parse_manifest_payload(manifest_payload)

    monkeypatch.setattr(manifest_mod, "fetch_and_parse_manifest", fake_fetch)

    pm = PrepareManager(cache_dir=tmp_path)

    def stub_download(descriptor, dest_dir):  # noqa: ARG001
        files = {}
        for r in descriptor.required_files:
            target = dest_dir / r.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file_a if r.relative_path == "a.bin" else file_b)
            files[r.relative_path] = target
        return DownloadResult(artifact_id=descriptor.artifact_id, files=files)

    pm._downloader.download = stub_download  # type: ignore[method-assign]

    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=1.0,
        reason="multifile-test",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="multi",
            artifact_id="multi",
            digest=body_digest,
            required_files=["a.bin", "b.bin"],
            download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            manifest_uri="https://x/manifest.json",
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )

    outcome = pm.prepare(candidate)
    assert (outcome.artifact_dir / "a.bin").read_bytes() == file_a
    assert (outcome.artifact_dir / "b.bin").read_bytes() == file_b
