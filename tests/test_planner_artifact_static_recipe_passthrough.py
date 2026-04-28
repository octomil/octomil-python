"""Reviewer P1: planner artifact dict → SDK static-recipe round-trip.

The runtime planner's app resolver now emits canonical TTS
artifacts with ``source='static_recipe'`` + ``recipe_id`` so the
SDK can run the recipe's MaterializationPlan after download. The
SDK has two parsing paths into :class:`RuntimeArtifactPlan` —
the live HTTP planner client and the cached-plan rehydration —
and they previously dropped the prepare-lifecycle fields silently
(``required_files`` / ``download_urls`` / ``manifest_uri`` /
``source`` / ``recipe_id``). This module pins both projection
sites against the server's wire shape and end-to-ends through
``PrepareManager.prepare()`` so app-scoped Kokoro materializes.
"""

from __future__ import annotations

import hashlib
import tarfile
from io import BytesIO
from pathlib import Path


def _server_shaped_kokoro_artifact_dict() -> dict:
    """Mirror the JSON the server's ``app_resolver`` emits."""
    return {
        "model_id": "kokoro-82m",
        "artifact_id": "kokoro-82m",
        "model_version": None,
        "format": "onnx",
        "quantization": "fp16",
        "uri": None,
        "digest": "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
        "size_bytes": None,
        "min_ram_bytes": None,
        "required_files": ["kokoro-en-v0_19.tar.bz2"],
        "download_urls": [
            {
                "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models",
                "expires_at": None,
                "headers": {"X-Octomil-Recipe-Path": "kokoro-en-v0_19.tar.bz2"},
            }
        ],
        "manifest_uri": None,
        "source": "static_recipe",
        "recipe_id": "kokoro-82m",
    }


# ---------------------------------------------------------------------------
# Live HTTP plan parsing
# ---------------------------------------------------------------------------


def test_client_parses_static_recipe_discriminators():
    """``RuntimePlannerClient._parse_artifact`` must keep ``source``
    and ``recipe_id`` so the SDK's ``PrepareManager`` runs the
    recipe's MaterializationPlan after download. Pre-fix this path
    silently dropped both fields, leaving ``@app/.../tts`` stuck
    on "downloaded the tarball but didn't extract it"."""
    from octomil.runtime.planner.client import _parse_artifact

    artifact_dict = _server_shaped_kokoro_artifact_dict()
    parsed = _parse_artifact(artifact_dict)

    assert parsed.source == "static_recipe"
    assert parsed.recipe_id == "kokoro-82m"
    assert parsed.required_files == ["kokoro-en-v0_19.tar.bz2"]
    assert len(parsed.download_urls) == 1
    assert parsed.download_urls[0].headers == {"X-Octomil-Recipe-Path": "kokoro-en-v0_19.tar.bz2"}


# ---------------------------------------------------------------------------
# Cached plan rehydration
# ---------------------------------------------------------------------------


def test_cache_rehydration_preserves_static_recipe_discriminators():
    """Cached plans (stored as JSON dicts in the planner store)
    rehydrate through ``plan_dict_to_*`` helpers. Those helpers
    previously dropped ``required_files`` / ``download_urls`` /
    ``manifest_uri`` / ``source`` / ``recipe_id`` even when the
    cached JSON contained them — silently breaking app-scoped
    Kokoro on cache hits."""
    from octomil.runtime.planner.planner import (
        plan_dict_to_app_resolution,
        plan_dict_to_candidates,
    )

    artifact_dict = _server_shaped_kokoro_artifact_dict()

    # AppResolution path — server emits artifact_candidates here.
    app_dict = {
        "app_id": "tts-tester",
        "capability": "tts",
        "routing_policy": "private",
        "selected_model": "kokoro-82m",
        "artifact_candidates": [artifact_dict],
    }
    app_res = plan_dict_to_app_resolution(app_dict)
    assert app_res is not None
    [art] = app_res.artifact_candidates
    assert art.source == "static_recipe"
    assert art.recipe_id == "kokoro-82m"
    assert art.required_files == ["kokoro-en-v0_19.tar.bz2"]
    assert art.download_urls and art.download_urls[0].headers == {"X-Octomil-Recipe-Path": "kokoro-en-v0_19.tar.bz2"}

    # Candidate-list path — used by the broader plan cache.
    cand_dict = {
        "locality": "local",
        "priority": 0,
        "confidence": 1.0,
        "reason": "test",
        "engine": "sherpa-onnx",
        "delivery_mode": "sdk_runtime",
        "prepare_required": True,
        "prepare_policy": "lazy",
        "artifact": artifact_dict,
    }
    [cand] = plan_dict_to_candidates([cand_dict])
    assert cand.artifact is not None
    assert cand.artifact.source == "static_recipe"
    assert cand.artifact.recipe_id == "kokoro-82m"


# ---------------------------------------------------------------------------
# End-to-end through PrepareManager
# ---------------------------------------------------------------------------


def _make_kokoro_layout_tarball(dst: Path) -> Path:
    """Build a fixture ``kokoro-en-v0_19.tar.bz2`` whose SHA-256
    we'll register as a test recipe so PrepareManager's digest
    check passes against the local fixture."""
    archive = dst / "kokoro-en-v0_19.tar.bz2"
    layout = {
        "kokoro-en-v0_19/model.onnx": b"fake-onnx",
        "kokoro-en-v0_19/voices.bin": b"fake-voices",
        "kokoro-en-v0_19/tokens.txt": b"fake-tokens",
        "kokoro-en-v0_19/espeak-ng-data/phontab": b"fake-phontab",
    }
    with tarfile.open(archive, "w:bz2") as tar:
        for name, data in layout.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return archive


def test_server_shaped_artifact_round_trips_through_prepare_manager(tmp_path, monkeypatch):
    """The release-blocker pin: a server-shaped Kokoro app
    artifact, parsed through the SDK's planner client, must drive
    ``PrepareManager.prepare()`` to materialize ``model.onnx`` /
    ``voices.bin`` / ``tokens.txt`` / ``espeak-ng-data/phontab``
    on disk — not leave the raw tarball.
    """
    from octomil.runtime.lifecycle import static_recipes as recipes_mod
    from octomil.runtime.lifecycle.durable_download import DownloadResult
    from octomil.runtime.lifecycle.materialization import (
        MaterializationPlan,
        MaterializationSafetyPolicy,
    )
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.planner.client import _parse_artifact
    from octomil.runtime.planner.schemas import RuntimeCandidatePlan

    tarball = _make_kokoro_layout_tarball(tmp_path)
    real_digest = "sha256:" + hashlib.sha256(tarball.read_bytes()).hexdigest()

    # Register a recipe under a fresh id pinned to our fixture's
    # digest so the durable downloader's verification passes
    # against the local copy.
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

    # Build the wire-shape dict the server emits, then run it
    # through the SDK's planner client so we exercise the real
    # _parse_artifact path the @app/.../tts flow takes.
    wire_dict = _server_shaped_kokoro_artifact_dict()
    wire_dict["model_id"] = "kokoro-test"
    wire_dict["artifact_id"] = "kokoro-test"
    wire_dict["recipe_id"] = "kokoro-test"
    wire_dict["digest"] = real_digest
    artifact = _parse_artifact(wire_dict)
    # Sanity: the discriminators that drive the materializer
    # actually survived the parse.
    assert artifact.source == "static_recipe"
    assert artifact.recipe_id == "kokoro-test"

    pm = PrepareManager(cache_dir=tmp_path / "cache")
    real_download = pm._downloader.download

    def stub_download(descriptor, dest_dir):
        for required in descriptor.required_files:
            target = dest_dir / required.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tarball.read_bytes())
        return DownloadResult(
            artifact_id=descriptor.artifact_id,
            files={r.relative_path: dest_dir / r.relative_path for r in descriptor.required_files},
        )

    pm._downloader.download = stub_download  # type: ignore[method-assign]
    try:
        candidate = RuntimeCandidatePlan(
            locality="local",
            priority=0,
            confidence=1.0,
            reason="planner-static-recipe",
            engine="sherpa-onnx",
            artifact=artifact,
            delivery_mode="sdk_runtime",
            prepare_required=True,
            prepare_policy="lazy",
        )
        outcome = pm.prepare(candidate)
        # Materialization actually ran — Sherpa-ready layout on
        # disk, NOT just the tarball.
        assert (outcome.artifact_dir / "model.onnx").is_file()
        assert (outcome.artifact_dir / "voices.bin").is_file()
        assert (outcome.artifact_dir / "tokens.txt").is_file()
        assert (outcome.artifact_dir / "espeak-ng-data" / "phontab").is_file()
    finally:
        pm._downloader.download = real_download  # type: ignore[method-assign]
