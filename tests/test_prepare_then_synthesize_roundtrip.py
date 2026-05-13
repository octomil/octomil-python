"""Issue D end-to-end: ``prepare(...)`` then ``synthesize_speech(...)``
must respect the native TTS batch gate.

Reviewer P1 frame:

    Add an e2e regression: same ExecutionKernel,
    prepare(model="piper-en-amy", capability="tts"),
    then synthesize_speech(model="piper-en-amy", policy="private");
    assert the speech path uses the native batch backend gate.

The bug shape that Issue D names: a developer runs
``client.prepare(model='piper-en-amy', capability='tts')``, the
durable downloader + materializer succeed, ``model.onnx`` /
``voices.bin`` / ``tokens.txt`` / ``espeak-ng-data/`` are on
disk — but the very next ``client.audio.speech.create(...)``
call still raises ``local_tts_runtime_unavailable`` because
the dispatch path either re-resolves to a different artifact
dir or surfaces a vague error that hides the real cause.

These tests pin the native cutover contract by inspecting how the
kernel calls ``NativeTtsBatchBackend``: prepare may materialize an
artifact dir, but the speech path must route through the native
batch backend and capability gate instead of threading a legacy
Sherpa ``model_dir``.
"""

from __future__ import annotations

import asyncio
import hashlib
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_piper_layout_tarball(dst: Path) -> tuple[Path, str]:
    archive = dst / "piper-en-amy.tar.bz2"
    layout = {
        "piper-en-amy/model.onnx": b"fake-onnx",
        "piper-en-amy/voices.bin": b"fake-voices",
        "piper-en-amy/tokens.txt": b"fake-tokens",
        "piper-en-amy/espeak-ng-data/phontab": b"fake-phontab",
    }
    with tarfile.open(archive, "w:bz2") as tar:
        for name, data in layout.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def _register_test_recipe(monkeypatch, digest: str) -> str:
    """Register a fresh ``piper-en-amy`` recipe pinned to ``digest``."""
    from octomil.runtime.lifecycle import static_recipes as recipes_mod
    from octomil.runtime.lifecycle.materialization import (
        MaterializationPlan,
        MaterializationSafetyPolicy,
    )

    test_recipe = recipes_mod.StaticRecipe(
        model_id="piper-en-amy",
        capability="tts",
        engine="sherpa-onnx",
        files=[
            recipes_mod._StaticArtifactFile(
                relative_path="piper-en-amy.tar.bz2",
                url="https://test.example.com/",
                digest=digest,
            )
        ],
        materialization=MaterializationPlan(
            kind="archive",
            source="piper-en-amy.tar.bz2",
            archive_format="tar.bz2",
            strip_prefix="piper-en-amy/",
            required_outputs=("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/phontab"),
            safety_policy=MaterializationSafetyPolicy(),
        ),
    )
    monkeypatch.setitem(recipes_mod._RECIPES, ("piper-en-amy", "tts"), test_recipe)
    return "piper-en-amy"


def test_prepare_then_synthesize_uses_native_batch_backend(tmp_path, monkeypatch):
    """The release-blocker pin: when the caller runs
    ``prepare(piper-en-amy)`` and then ``synthesize_speech(piper-en-amy,
    policy='private')`` on the same kernel, the speech path must use
    ``NativeTtsBatchBackend`` and its native capability gate. Prepare
    may still write bytes to disk, but the dispatch path should not
    thread that artifact dir into a legacy Sherpa backend."""
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.durable_download import DownloadResult
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    tarball, digest = _make_piper_layout_tarball(tmp_path)
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
                required_files=["piper-en-amy.tar.bz2"],
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
        # Step 2: synthesize_speech with policy='private' (local-only
        # by config) and a direct non-app model ref. Stub _resolve so
        # we don't need a real OctomilConfig + _resolve_planner_selection
        # so the planner is "offline". Mock the native batch backend so
        # we can capture the model passed to the runtime load path.
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
            patch(
                "octomil.runtime.native.tts_batch_backend.NativeTtsBatchBackend",
                return_value=fake_backend,
            ),
        ):
            response = asyncio.run(
                kernel.synthesize_speech(
                    model=model,
                    input="hello",
                    policy="private",
                )
            )

        # Native backend was used, and the prepared artifact dir was
        # not threaded into a legacy Sherpa backend path.
        assert fake_backend.load_model.called
        fake_backend.load_model.assert_called_once_with(model)
        fake_backend.synthesize.assert_called_once()
        assert "model_dir" not in fake_backend.load_model.call_args.kwargs
        assert "model_dir" not in fake_backend.synthesize.call_args.kwargs
        assert response.route.locality == "on_device"
    finally:
        pm._downloader.download = real_download  # type: ignore[method-assign]


def test_local_tts_unavailable_message_distinguishes_three_failure_modes(tmp_path, monkeypatch):
    """The vague single-line ``local_tts_runtime_unavailable`` was
    hiding three distinct root causes. After the native cutover the
    dispatch path picks one of three messages so the user can tell
    which layer failed:

      (1) native batch backend import failure
      (2) native capability gate not advertised
      (3) prepared dir exists but native backend load failed
    """
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    kernel._warmed_backends = {}

    defaults = ResolvedExecutionDefaults(
        model="piper-en-amy",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    # ---- (1) native batch import failure ----------------------
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
                load_error.append('native_tts_import: ImportError("_native_tts")') if load_error is not None else None,
                None,
            )[1],
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            asyncio.run(kernel.synthesize_speech(model="piper-en-amy", input="hi"))
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "native audio.tts.batch backend could not import or bind the runtime loader" in str(ei.value)

    # ---- (2) native capability gate unavailable ---------------
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=False),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            asyncio.run(kernel.synthesize_speech(model="piper-en-amy", input="hi"))
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "native audio.tts.batch is not advertised" in str(ei.value)

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
                load_error.append("backend_load: RuntimeError('missing native model')")
                if load_error is not None
                else None,
                None,
            )[1],
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            asyncio.run(kernel.synthesize_speech(model="piper-en-amy", input="hi"))
        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        msg = str(ei.value)
        assert "native audio.tts.batch" in msg
        assert "failed to load or synthesize" in msg
        assert "backend_load:" in msg
