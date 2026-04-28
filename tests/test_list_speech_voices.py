"""Tests for ``client.audio.voices.list`` and the shared catalog
resolver that powers it.

Closure-of-loop guarantee: the same resolver
(:func:`octomil.runtime.engines.sherpa.resolve_voice_catalog`)
feeds the listing API, the kernel preflight, and the engine's
sid resolution. A voice that listing advertises must resolve
under synthesis, and vice versa — these tests pin that.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octomil.audio.speech import FacadeVoices, VoiceCatalog, VoiceInfo
from octomil.errors import OctomilError
from octomil.runtime.engines.sherpa.engine import (
    _SherpaTtsBackend,
    resolve_voice_catalog,
)
from octomil.runtime.lifecycle.static_recipes import (
    KOKORO_EN_V0_19_VOICES,
    KOKORO_MULTI_LANG_V1_0_VOICES,
    get_static_recipe,
)

# ---------------------------------------------------------------------------
# Resolver primitives
# ---------------------------------------------------------------------------


def test_resolver_reads_voices_txt_sidecar_first(tmp_path):
    """A prepared dir with a voices.txt sidecar uses that catalog
    verbatim. Source == 'voices_txt'."""
    custom = ("custom_a", "custom_b", "custom_c")
    (tmp_path / "voices.txt").write_text("\n".join(custom) + "\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("custom-bundle-vX\n", encoding="utf-8")

    resolved = resolve_voice_catalog("kokoro-82m", prepared_model_dir=str(tmp_path))
    assert resolved.voices == custom
    assert resolved.source == "voices_txt"
    assert resolved.artifact_version == "custom-bundle-vX"


def test_resolver_falls_back_to_layout_when_sidecar_missing(tmp_path):
    """Prepared dir without voices.txt but with VERSION + a
    recognized layout → use the per-version catalog. Source still
    'voices_txt' (artifact-on-disk derived)."""
    (tmp_path / "VERSION").write_text("kokoro-multi-lang-v1_0\n", encoding="utf-8")
    (tmp_path / "lexicon-us-en.txt").write_bytes(b"x")
    (tmp_path / "dict").mkdir()
    (tmp_path / "dict" / "jieba.dict.utf8").write_bytes(b"x")

    resolved = resolve_voice_catalog("kokoro-82m", prepared_model_dir=str(tmp_path))
    assert resolved.voices == KOKORO_MULTI_LANG_V1_0_VOICES
    assert resolved.source == "voices_txt"
    assert resolved.artifact_version == "kokoro-multi-lang-v1_0"


def test_resolver_refuses_ambiguous_prepared_dir(tmp_path):
    """An old v0.19 layout under the kokoro-82m id (post-cutover)
    is ambiguous: layout says v0.19 but model id catalog is v1.0.
    Resolver returns empty rather than guessing — this is the
    closure-of-loop guarantee for sid resolution."""
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")
    # No VERSION, no voices.txt, no dict/, no lexicon.

    resolved = resolve_voice_catalog("kokoro-82m", prepared_model_dir=str(tmp_path))
    assert resolved.voices == ()
    assert resolved.source == ""


def test_resolver_uses_static_recipe_manifest_when_no_prepared_dir():
    """Listing path with no prepared cache yet → use the recipe's
    voice_manifest as a *preview* so the UI can render before
    forcing a download. Source == 'static_recipe'."""
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None

    resolved = resolve_voice_catalog(
        "kokoro-82m",
        prepared_model_dir=None,
        static_recipe_manifest=recipe.materialization.voice_manifest,
        static_recipe_artifact_version=recipe.materialization.artifact_version or "",
    )
    assert resolved.voices == KOKORO_MULTI_LANG_V1_0_VOICES
    assert resolved.source == "static_recipe"
    assert resolved.artifact_version == "kokoro-multi-lang-v1_0"


def test_resolver_returns_empty_for_unknown_model_with_no_signals():
    """An unknown model with no prepared dir and no recipe manifest
    yields an empty catalog — callers translate that to a strict
    refusal for the explicit-voice path."""
    resolved = resolve_voice_catalog("piper-en-amy")
    assert resolved.voices == ()
    assert resolved.source == ""


# ---------------------------------------------------------------------------
# Closure-of-loop: resolver feeds both _voice_to_sid AND list_speech_voices
# ---------------------------------------------------------------------------


def test_engine_voice_to_sid_walks_through_resolver(tmp_path):
    """Pin the closure-of-loop: ``_SherpaTtsBackend._voice_to_sid``
    resolves names through the same code path the listing API
    uses. If the resolver advertises ``am_echo`` at index 12, the
    engine resolves it to sid=12; if the resolver excludes it,
    the engine raises."""
    (tmp_path / "voices.txt").write_text("\n".join(KOKORO_MULTI_LANG_V1_0_VOICES) + "\n", encoding="utf-8")
    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))

    resolved = resolve_voice_catalog("kokoro-82m", prepared_model_dir=str(tmp_path))
    for expected_sid, name in enumerate(resolved.voices):
        assert backend._voice_to_sid(name) == expected_sid


def test_engine_voice_to_sid_rejects_when_resolver_returns_empty(tmp_path):
    """When the resolver returns an empty catalog (ambiguous
    prepared dir), the engine raises voice_not_supported_for_model
    rather than aliasing to sid=0."""
    # v0.19 layout under kokoro-82m id → resolver returns ().
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")
    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))

    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("bm_george")
    assert "voice_not_supported_for_model" in str(ei.value)


# ---------------------------------------------------------------------------
# kernel.list_speech_voices — local path
# ---------------------------------------------------------------------------


def _kernel_with_static_route():
    """Return an ExecutionKernel preloaded with the helpers the
    listing path expects, stubbed so we don't go through real
    routing/planner code in unit tests."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    return kernel


@pytest.mark.asyncio
async def test_list_speech_voices_returns_v1_0_catalog_from_prepared_cache(tmp_path):
    """Happy path: prepared cache on disk → catalog comes from
    voices.txt. 53 entries, source='voices_txt', sample_rate=24000,
    artifact identity populated from the recipe."""
    from octomil.config.local import ResolvedExecutionDefaults

    kernel = _kernel_with_static_route()
    (tmp_path / "voices.txt").write_text("\n".join(KOKORO_MULTI_LANG_V1_0_VOICES) + "\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("kokoro-multi-lang-v1_0\n", encoding="utf-8")

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
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
    ):
        catalog = await kernel.list_speech_voices(model="kokoro-82m")

    assert isinstance(catalog, VoiceCatalog)
    assert catalog.locality == "on_device"
    assert catalog.source == "voices_txt"
    assert catalog.model == "kokoro-82m"
    assert catalog.artifact_id == "kokoro-82m"
    assert catalog.artifact_version == "kokoro-multi-lang-v1_0"
    assert catalog.digest and catalog.digest.startswith("sha256:c133d263")
    assert catalog.sample_rate == 24000
    assert len(catalog.voices) == 53
    assert catalog.voice_ids[:3] == ("af_alloy", "af_aoede", "af_bella")
    assert catalog.voice_ids[12] == "am_echo"
    # default voice flagged.
    assert catalog.default_voice == "af_bella"
    bella = catalog.get("af_bella")
    assert bella is not None
    assert bella.default is True
    assert bella.sid == 2
    # Unknown voices return None from .get().
    assert catalog.get("not_a_voice") is None


@pytest.mark.asyncio
async def test_list_speech_voices_previews_recipe_when_no_prepared_cache():
    """When no prepared cache exists, the listing API still works:
    it previews the static recipe's voice_manifest so the UI can
    render the catalog without forcing a 349 MB download."""
    from octomil.config.local import ResolvedExecutionDefaults

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=None),
        patch.object(kernel, "_local_candidate_is_unpreparable", return_value=False),
        patch.object(kernel, "_can_prepare_local_tts", return_value=True),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ):
        catalog = await kernel.list_speech_voices(model="kokoro-82m")

    assert catalog.source == "static_recipe"
    assert len(catalog.voices) == 53
    assert catalog.artifact_version == "kokoro-multi-lang-v1_0"


@pytest.mark.asyncio
async def test_list_speech_voices_returns_v0_19_for_legacy_id():
    """``kokoro-en-v0_19`` resolves to its own 11-speaker catalog,
    NOT the v1.0 catalog. The two recipes are independent."""
    from octomil.config.local import ResolvedExecutionDefaults

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-en-v0_19",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=None),
        patch.object(kernel, "_local_candidate_is_unpreparable", return_value=False),
        patch.object(kernel, "_can_prepare_local_tts", return_value=True),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ):
        catalog = await kernel.list_speech_voices(model="kokoro-en-v0_19")

    assert len(catalog.voices) == 11
    assert catalog.voice_ids == KOKORO_EN_V0_19_VOICES
    assert catalog.artifact_version == "kokoro-en-v0_19"


@pytest.mark.asyncio
async def test_list_speech_voices_returns_cloud_locality_for_hosted_routing():
    """When routing dispatches to cloud, the listing API returns
    locality='cloud' with source='hosted'. The voices tuple is
    empty because the SDK doesn't ship a curated hosted catalog —
    UI consumers should hit the provider directly or wait for the
    server-side ``/v1/audio/voices`` route."""
    from octomil.config.local import ResolvedExecutionDefaults

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="tts-1",
        policy_preset="cloud_only",
        inline_policy=None,
        cloud_profile=MagicMock(),
    )
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch("octomil.execution.kernel._cloud_available", return_value=True),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=False),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=None),
        patch.object(kernel, "_local_candidate_is_unpreparable", return_value=True),
        patch.object(kernel, "_can_prepare_local_tts", return_value=False),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("cloud", False),
        ),
    ):
        catalog = await kernel.list_speech_voices(model="tts-1")

    assert catalog.locality == "cloud"
    assert catalog.source == "hosted"
    assert catalog.voices == ()
    assert catalog.model == "tts-1"


# ---------------------------------------------------------------------------
# FacadeVoices: client.audio.voices.list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_voices_delegates_to_kernel():
    """``client.audio.voices.list(...)`` is a thin shim — pass
    model/policy/app through to ``kernel.list_speech_voices`` and
    return the kernel's VoiceCatalog verbatim."""
    fake_kernel = MagicMock()

    async def fake_list_speech_voices(*, model, policy=None, app=None):
        fake_kernel.captured = {"model": model, "policy": policy, "app": app}
        return VoiceCatalog(
            model=model,
            locality="on_device",
            source="voices_txt",
            voices=(VoiceInfo(id="af_bella", sid=2, default=True),),
        )

    fake_kernel.list_speech_voices = fake_list_speech_voices

    facade = FacadeVoices(fake_kernel)
    catalog = await facade.list(model="kokoro-82m", policy="local_only", app="myapp")

    assert fake_kernel.captured == {"model": "kokoro-82m", "policy": "local_only", "app": "myapp"}
    assert catalog.voice_ids == ("af_bella",)
    assert catalog.locality == "on_device"


@pytest.mark.asyncio
async def test_facade_audio_exposes_voices_namespace():
    """``client.audio.voices`` exists on FacadeAudio."""
    from octomil.audio import FacadeAudio

    fake_kernel = MagicMock()
    facade = FacadeAudio(fake_kernel)
    assert isinstance(facade.voices, FacadeVoices)


# ---------------------------------------------------------------------------
# P1 — planner-selected non-static artifacts must NOT inherit the public recipe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_speech_voices_planner_artifact_without_prepared_cache_returns_planner_pending():
    """P1: planner picked a non-static artifact (e.g. a private
    Kokoro bundle) for runtime_model=kokoro-82m, but it isn't on
    disk yet. The SDK has no authoritative catalog source — return
    locality='on_device', source='planner_pending', empty voices,
    with the planner artifact's identity (NOT the public static
    recipe's). Listing must NOT advertise the public v1.0 catalog
    for what synthesis will run as a private artifact."""
    from octomil.config.local import ResolvedExecutionDefaults

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    selection = MagicMock()
    candidate = MagicMock()
    artifact = MagicMock()
    artifact.digest = "sha256:" + "a" * 64
    artifact.artifact_id = "private-kokoro-v2"
    candidate.artifact = artifact

    import octomil.execution.kernel as kernel_mod

    original = kernel_mod._local_sdk_runtime_candidate
    kernel_mod._local_sdk_runtime_candidate = lambda _sel: candidate
    try:
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch("octomil.execution.kernel._resolve_routing_policy"),
            patch("octomil.execution.kernel._enforce_app_ref_routing_policy"),
            patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
            patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
            patch.object(kernel, "_prepared_local_artifact_dir", return_value=None),
            patch.object(kernel, "_local_candidate_is_unpreparable", return_value=False),
            patch.object(kernel, "_can_prepare_local_tts", return_value=True),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
        ):
            catalog = await kernel.list_speech_voices(model="kokoro-82m")
    finally:
        kernel_mod._local_sdk_runtime_candidate = original

    assert catalog.locality == "on_device"
    assert catalog.source == "planner_pending"
    assert catalog.voices == ()
    # Identity is the planner artifact's, not the public recipe's.
    assert catalog.artifact_id == "private-kokoro-v2"
    assert catalog.digest == "sha256:" + "a" * 64
    # Default voice unset until prepare materializes the artifact.
    assert catalog.default_voice is None
    assert catalog.sample_rate is None


@pytest.mark.asyncio
async def test_list_speech_voices_planner_artifact_with_prepared_cache_uses_artifact_voices_txt(tmp_path):
    """P1: when the planner artifact IS on disk, voices.txt is
    authoritative — listing returns those voices, plus the planner
    artifact's identity, never falling through to the public static
    recipe's manifest/digest."""
    from octomil.config.local import ResolvedExecutionDefaults

    custom_catalog = ("private_voice_a", "private_voice_b", "private_voice_c")
    (tmp_path / "voices.txt").write_text("\n".join(custom_catalog) + "\n", encoding="utf-8")

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    selection = MagicMock()
    candidate = MagicMock()
    artifact = MagicMock()
    artifact.digest = "sha256:" + "b" * 64
    artifact.artifact_id = "private-kokoro-v2"
    candidate.artifact = artifact

    import octomil.execution.kernel as kernel_mod

    original = kernel_mod._local_sdk_runtime_candidate
    kernel_mod._local_sdk_runtime_candidate = lambda _sel: candidate
    try:
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch("octomil.execution.kernel._resolve_routing_policy"),
            patch("octomil.execution.kernel._enforce_app_ref_routing_policy"),
            patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
            patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=True),
            patch.object(kernel, "_prepared_local_artifact_dir", return_value=str(tmp_path)),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
        ):
            catalog = await kernel.list_speech_voices(model="kokoro-82m")
    finally:
        kernel_mod._local_sdk_runtime_candidate = original

    # Voices come from the artifact's own voices.txt, not the public recipe.
    assert catalog.voice_ids == custom_catalog
    assert catalog.source == "voices_txt"
    # Identity is the planner artifact's.
    assert catalog.artifact_id == "private-kokoro-v2"
    assert catalog.digest == "sha256:" + "b" * 64
    # P2 invariant: default_voice falls back to the catalog's first
    # entry because the model-table default (af_bella) isn't in the
    # private artifact's catalog. The flagged VoiceInfo agrees.
    assert catalog.default_voice == "private_voice_a"
    flagged = [v for v in catalog.voices if v.default]
    assert len(flagged) == 1
    assert flagged[0].id == "private_voice_a"


# ---------------------------------------------------------------------------
# P2 — default_voice and the flagged VoiceInfo always agree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_speech_voices_default_voice_aligned_with_flagged_entry(tmp_path):
    """P2 invariant — happy path: when the model-table default IS
    present in the catalog, default_voice points at it AND the
    flagged VoiceInfo is exactly that entry."""
    from octomil.config.local import ResolvedExecutionDefaults

    (tmp_path / "voices.txt").write_text("\n".join(KOKORO_MULTI_LANG_V1_0_VOICES) + "\n", encoding="utf-8")

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch("octomil.execution.kernel._enforce_app_ref_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=True),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=str(tmp_path)),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ):
        catalog = await kernel.list_speech_voices(model="kokoro-82m")

    assert catalog.default_voice == "af_bella"
    flagged = [v for v in catalog.voices if v.default]
    assert len(flagged) == 1
    assert flagged[0].id == "af_bella"


# ---------------------------------------------------------------------------
# P2 — routing failures surface as OctomilError, not silent cloud routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_speech_voices_raises_on_local_route_failure():
    """P2: a routing RuntimeError that synthesis would surface as
    local_tts_runtime_unavailable must surface here too. The
    listing API mirrors synthesis's failure modes — silently
    routing to cloud would let UIs preview a public hosted catalog
    for a request synthesis is going to refuse."""
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilErrorCode

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    def raise_no_route(*args, **kwargs):
        raise RuntimeError("local TTS is unavailable: no usable route")

    expected = OctomilError(
        code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        message="local_tts_runtime_unavailable: no usable local route",
    )

    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch("octomil.execution.kernel._enforce_app_ref_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=False),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value=None),
        patch.object(kernel, "_local_candidate_is_unpreparable", return_value=True),
        patch.object(kernel, "_can_prepare_local_tts", return_value=False),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            side_effect=raise_no_route,
        ),
        patch.object(kernel, "_tts_local_unavailable_error", return_value=expected),
    ):
        with pytest.raises(OctomilError) as ei:
            await kernel.list_speech_voices(model="kokoro-82m")
    assert "local_tts_runtime_unavailable" in str(ei.value)
    assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE


# ---------------------------------------------------------------------------
# Closure-of-loop: listing ↔ synthesis agree on the catalog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listing_and_synthesis_share_catalog_for_prepared_artifact(tmp_path):
    """End-to-end closure: every voice the listing API advertises
    resolves through the engine's _voice_to_sid to the same sid,
    and any voice the listing API DOESN'T advertise raises."""
    from octomil.config.local import ResolvedExecutionDefaults

    # Prepared cache with the canonical v1.0 sidecar.
    (tmp_path / "voices.txt").write_text("\n".join(KOKORO_MULTI_LANG_V1_0_VOICES) + "\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("kokoro-multi-lang-v1_0\n", encoding="utf-8")

    kernel = _kernel_with_static_route()
    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
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
    ):
        catalog = await kernel.list_speech_voices(model="kokoro-82m")

    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))
    # Every voice listing advertises resolves to its declared sid.
    for v in catalog.voices:
        assert backend._voice_to_sid(v.id) == v.sid

    # A voice NOT in the listing raises.
    not_in_catalog = "completely_made_up_voice"
    assert catalog.get(not_in_catalog) is None
    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid(not_in_catalog)
    assert "voice_not_supported_for_model" in str(ei.value)
