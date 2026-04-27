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
    expanded = pm._expand_static_recipe_source(artifact)
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
    expanded = pm._expand_static_recipe_source(artifact)
    assert expanded is artifact


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
    expanded = pm._expand_static_recipe_source(candidate.artifact)
    descriptor = pm._build_descriptor(expanded)
    assert descriptor.artifact_id == "kokoro-82m"
    assert len(descriptor.required_files) == 1
    assert descriptor.required_files[0].relative_path == "kokoro-en-v0_19.tar.bz2"
    assert (
        descriptor.required_files[0].digest == "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"
    )
