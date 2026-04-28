"""Issue D end-to-end: ``prepare(...)`` then ``synthesize_speech(...)``
must reuse the prepared artifact dir.

Reviewer P1 frame:

    Add an e2e regression: same ExecutionKernel,
    prepare(model="kokoro-82m", capability="tts"),
    then synthesize_speech(model="kokoro-82m", policy="private");
    assert the TTS backend receives model_dir=outcome.artifact_dir.

The bug shape that Issue D names: a developer runs
``client.prepare(model='kokoro-82m', capability='tts')``, the
durable downloader + materializer succeed, ``model.onnx`` /
``voices.bin`` / ``tokens.txt`` / ``espeak-ng-data/`` are on
disk — but the very next ``client.audio.speech.create(...)``
call still raises ``local_tts_runtime_unavailable`` because
the dispatch path either re-resolves to a different artifact
dir or surfaces a vague error that hides the real cause.

These tests pin the contract by inspecting how the kernel
calls the Sherpa engine: the ``model_dir`` it passes MUST be
the same path ``PrepareManager.prepare()`` returned.
"""

from __future__ import annotations

import hashlib
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_kokoro_layout_tarball(dst: Path) -> tuple[Path, str]:
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
    digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def _register_test_recipe(monkeypatch, digest: str) -> str:
    """Register a fresh ``kokoro-test`` recipe pinned to ``digest``."""
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
                digest=digest,
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
    return "kokoro-test"


@pytest.mark.asyncio
async def test_prepare_then_synthesize_threads_artifact_dir_to_backend(tmp_path, monkeypatch):
    """The release-blocker pin: when the caller runs
    ``prepare(kokoro-test)`` and then ``synthesize_speech(kokoro-test,
    policy='private')`` on the same kernel, the Sherpa backend MUST
    be created with ``model_dir=outcome.artifact_dir`` — the same
    path PrepareManager wrote to. No silent re-resolution to a
    different cache dir, no scanning of arbitrary ``<model>-*``
    siblings, no fallthrough to the legacy staging path."""
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.durable_download import DownloadResult
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    tarball, digest = _make_kokoro_layout_tarball(tmp_path)
    model = _register_test_recipe(monkeypatch, digest)

    # Single PrepareManager threaded into the kernel — same instance
    # for prepare() and the dispatch-time cache lookup. Without this
    # the cache dir would be different across the two calls because
    # ``defaultCacheDir`` keys off env vars.
    cache_dir = tmp_path / "cache"
    pm = PrepareManager(cache_dir=cache_dir)

    # Stub the durable downloader so the recipe's URL is irrelevant
    # — copy our fixture tarball into the artifact dir.
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
        # Build a kernel that returns our PrepareManager instance via
        # the standard injection path.
        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()
        kernel._prepare_manager = pm
        kernel._warmed_backends = {}

        # Step 1: pretend client.prepare(...) ran. Build a candidate
        # using the standard registry-shaped artifact (so it lands
        # under the same artifact_dir_for(recipe.model_id) key the
        # dispatch path computes).
        from octomil.runtime.planner.schemas import (
            ArtifactDownloadEndpoint,
            RuntimeArtifactPlan,
            RuntimeCandidatePlan,
        )

        candidate = RuntimeCandidatePlan(
            locality="local",
            engine="sherpa-onnx",
            artifact=RuntimeArtifactPlan(
                model_id=model,
                artifact_id=model,
                digest=digest,
                required_files=["kokoro-en-v0_19.tar.bz2"],
                download_urls=[ArtifactDownloadEndpoint(url="https://test.example.com/")],
                source="static_recipe",
                recipe_id=model,
            ),
            priority=0,
            confidence=1.0,
            reason="test",
            delivery_mode="sdk_runtime",
            prepare_required=True,
            prepare_policy="lazy",
        )
        prepare_outcome = pm.prepare(candidate)
        assert (prepare_outcome.artifact_dir / "model.onnx").is_file()
        prepared_dir = str(prepare_outcome.artifact_dir)

        # Step 2: synthesize_speech with policy='private' (local-only
        # by config) and a direct non-app model ref. Stub _resolve so
        # we don't need a real OctomilConfig + _resolve_planner_selection
        # so the planner is "offline". Mock the SherpaTtsEngine so we
        # can capture the model_dir kwarg the kernel passes.
        defaults = ResolvedExecutionDefaults(
            model=model,
            policy_preset="local_only",
            inline_policy=None,
            cloud_profile=None,
        )
        fake_backend = MagicMock()
        fake_backend.synthesize.return_value = {
            "audio_bytes": b"\x52\x49\x46\x46\x00",
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": 24000,
            "duration_ms": 1,
            "voice": "af_bella",
            "model": model,
        }
        fake_engine = MagicMock()
        fake_engine.create_backend.return_value = fake_backend

        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
            patch("octomil.execution.kernel._resolve_routing_policy"),
            patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
            patch.object(
                kernel,
                "_candidate_matches_static_recipe",
                return_value=False,
            ),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
            patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", return_value=fake_engine),
        ):
            response = await kernel.synthesize_speech(
                model=model,
                input="hello",
                policy="private",
            )

        # Backend received model_dir=prepared_dir — the kernel did
        # NOT scan, NOT re-resolve, NOT silently fall through to a
        # different cache. This is the tight roundtrip Issue D
        # demanded.
        fake_engine.create_backend.assert_called_once()
        kwargs = fake_engine.create_backend.call_args.kwargs
        assert kwargs.get("model_dir") == prepared_dir, (
            f"Sherpa backend should be created with model_dir={prepared_dir!r}, "
            f"got {kwargs.get('model_dir')!r} — kernel is not threading the prepared "
            f"artifact dir into dispatch."
        )
        assert response.route.locality == "on_device"
    finally:
        pm._downloader.download = real_download  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_local_tts_unavailable_message_distinguishes_three_failure_modes(tmp_path, monkeypatch):
    """The vague single-line ``local_tts_runtime_unavailable`` was
    hiding three distinct root causes. After Issue D the dispatch
    path picks one of three messages so the user can tell which
    layer failed:

      (1) ``sherpa_onnx`` not importable
      (2) prepared artifact dir missing
      (3) prepared dir exists but backend load failed
    """
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    kernel._warmed_backends = {}

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    # ---- (1) sherpa import failure ----------------------------
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=True),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value="/tmp/missing"),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
        patch.object(
            kernel,
            "_resolve_local_tts_backend",
            side_effect=lambda *a, load_error=None, **kw: (
                load_error.append('sherpa_import: ImportError("_sherpa_onnx")') if load_error is not None else None,
                None,
            )[1],
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech(model="kokoro-82m", input="hi")
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "sherpa_onnx is not importable" in str(ei.value)

    # ---- (2) prepared dir missing -----------------------------
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=False),
        # cache short-circuit refuses; _prepared_cache_may_short_circuit returns False
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_prepare_local_tts_artifact", return_value=None),
        patch.object(kernel, "_can_prepare_local_tts", return_value=True),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
        patch.object(kernel, "_resolve_local_tts_backend", return_value=None),
    ):
        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech(model="kokoro-82m", input="hi")
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "no prepared artifact dir on disk" in str(ei.value)

    # ---- (3) backend load failed ------------------------------
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=True),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=str(tmp_path)),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
        patch.object(
            kernel,
            "_resolve_local_tts_backend",
            side_effect=lambda *a, load_error=None, **kw: (
                load_error.append("backend_load: RuntimeError('missing model.onnx')")
                if load_error is not None
                else None,
                None,
            )[1],
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech(model="kokoro-82m", input="hi")
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        msg = str(ei.value)
        assert "prepared artifact dir" in msg
        assert "exists but the sherpa backend failed to load" in msg
        assert "backend_load:" in msg
