"""Regression tests for the Kokoro voice-manifest fix.

Background: octomil[tts] 4.12.x with sherpa-onnx 1.13.0 used a
hardcoded 28-name catalog (``_KOKORO_VOICES``) to map voice strings
to sherpa-onnx speaker ids. The static recipe for ``kokoro-82m``
downloads the upstream ``kokoro-en-v0_19.tar.bz2`` bundle, which
ships an 11-speaker ``voices.bin``. Because the global catalog had
a different order AND included names absent from the bundle, voices
like ``am_echo`` / ``af_heart`` resolved to ``sid >= 11``;
sherpa-onnx then logged ``sid should be in range [0, 10]`` and
silently rendered every bad voice with ``sid=0``. Even
"in-range" voices were misnamed because the table order didn't
match the bundle.

The fix is two-part and these tests pin both halves:

  1. The static recipe declares ``voice_manifest`` for
     ``kokoro-82m`` as the v0_19 11-speaker catalog
     (``af, af_bella, af_nicole, af_sarah, af_sky, am_adam,
     am_michael, bf_emma, bf_isabella, bm_george, bm_lewis``);
     the materializer writes it as ``voices.txt`` next to
     ``model.onnx`` / ``voices.bin``.
  2. ``_SherpaTtsBackend._voice_to_sid`` reads ``voices.txt`` from
     the prepared artifact directory as the authoritative ordered
     catalog. Unknown voices raise ``OctomilError`` with
     ``voice_not_supported_for_model`` instead of silently
     returning ``sid=0``. Empty voice still maps to ``sid=0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.engines.sherpa.engine import (
    _KOKORO_EN_V0_19_VOICES,
    _SherpaTtsBackend,
    catalog_for_model,
)
from octomil.runtime.lifecycle.materialization import (
    EXTRACTION_MARKER,
    MaterializationPlan,
    Materializer,
)
from octomil.runtime.lifecycle.static_recipes import (
    KOKORO_EN_V0_19_VOICES,
    get_static_recipe,
)

# ---------------------------------------------------------------------------
# Recipe-side: kokoro-82m declares the 11-speaker v0_19 catalog
# ---------------------------------------------------------------------------


_EXPECTED_V0_19 = (
    "af",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)


def test_static_recipe_declares_v0_19_voice_manifest():
    """The kokoro-82m recipe carries the bundle's 11-speaker catalog
    in its MaterializationPlan, in the exact order voices.bin uses.
    """
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    assert recipe.materialization.voice_manifest == _EXPECTED_V0_19
    # Aliased recipe id resolves to the same data.
    assert get_static_recipe("kokoro-en-v0_19", "tts") is recipe
    # Module-level export matches.
    assert KOKORO_EN_V0_19_VOICES == _EXPECTED_V0_19


def test_static_recipe_declares_artifact_version():
    """``artifact_version`` is materialized as a VERSION sidecar so
    operators can disambiguate v0_19 from a future bundle."""
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    assert recipe.materialization.artifact_version == "kokoro-en-v0_19"


# ---------------------------------------------------------------------------
# Materializer writes voices.txt + VERSION
# ---------------------------------------------------------------------------


def test_materializer_writes_voices_txt_for_kokoro_recipe(tmp_path):
    """After materialize(), voices.txt contains the manifest in
    order and VERSION carries the artifact version."""
    plan = MaterializationPlan(
        kind="none",
        required_outputs=(),
        voice_manifest=_EXPECTED_V0_19,
        artifact_version="kokoro-en-v0_19",
    )
    Materializer().materialize(tmp_path, plan)

    voices_txt = (tmp_path / "voices.txt").read_text(encoding="utf-8")
    assert voices_txt.splitlines() == list(_EXPECTED_V0_19)

    version = (tmp_path / "VERSION").read_text(encoding="utf-8").strip()
    assert version == "kokoro-en-v0_19"


def test_materializer_overwrites_stale_voices_txt(tmp_path):
    """A stale voices.txt left by a previous recipe revision is
    replaced — the manifest is the single source of truth, so a
    silent merge would risk shifting speaker ids."""
    (tmp_path / "voices.txt").write_text("old_voice_a\nold_voice_b\n", encoding="utf-8")
    plan = MaterializationPlan(
        kind="none",
        required_outputs=(),
        voice_manifest=_EXPECTED_V0_19,
    )
    Materializer().materialize(tmp_path, plan)

    voices_txt = (tmp_path / "voices.txt").read_text(encoding="utf-8")
    assert voices_txt.splitlines() == list(_EXPECTED_V0_19)


def test_materializer_no_op_when_manifest_empty(tmp_path):
    """Recipes that don't declare a voice_manifest (single-speaker
    Piper bundles, the whisper recipe, …) get no voices.txt."""
    plan = MaterializationPlan(kind="none", required_outputs=())
    Materializer().materialize(tmp_path, plan)
    assert not (tmp_path / "voices.txt").exists()
    assert not (tmp_path / "VERSION").exists()


def test_materializer_writes_voice_manifest_after_archive_extraction(tmp_path):
    """Archive recipes get voices.txt written *after* the extraction
    completes, so a half-extracted artifact never carries a manifest
    that would mislead the engine."""
    import tarfile
    from io import BytesIO

    # Build a minimal tarball mimicking the kokoro-en-v0_19 layout.
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for rel, content in [
            ("kokoro-en-v0_19/model.onnx", b"\x00onnx"),
            ("kokoro-en-v0_19/voices.bin", b"\x00voicebin"),
            ("kokoro-en-v0_19/tokens.txt", b"a 1\n"),
            ("kokoro-en-v0_19/espeak-ng-data/phontab", b"\x00phontab"),
        ]:
            data = content
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    archive = tmp_path / "kokoro-en-v0_19.tar.bz2"
    archive.write_bytes(buf.getvalue())

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
        voice_manifest=_EXPECTED_V0_19,
        artifact_version="kokoro-en-v0_19",
    )
    Materializer().materialize(tmp_path, plan)

    assert (tmp_path / EXTRACTION_MARKER).is_file()
    assert (tmp_path / "voices.txt").read_text(encoding="utf-8").splitlines() == list(_EXPECTED_V0_19)
    assert (tmp_path / "VERSION").read_text(encoding="utf-8").strip() == "kokoro-en-v0_19"


# ---------------------------------------------------------------------------
# Engine voice resolution: voices.txt is authoritative
# ---------------------------------------------------------------------------


def _make_backend(model_dir: Path, model: str = "kokoro-82m") -> _SherpaTtsBackend:
    return _SherpaTtsBackend(model, model_dir=str(model_dir))


def _write_v0_19_manifest(model_dir: Path) -> None:
    (model_dir / "voices.txt").write_text("\n".join(_EXPECTED_V0_19) + "\n", encoding="utf-8")


def test_voice_to_sid_resolves_each_v0_19_voice_to_correct_index(tmp_path):
    """Every voice in the v0_19 catalog resolves to its position
    in voices.txt — the regression that motivated this whole patch.
    Pre-fix, ``bm_george`` mapped to ``sid=0`` because the global
    28-item catalog put it at index 26.
    """
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)

    for expected_sid, voice in enumerate(_EXPECTED_V0_19):
        assert backend._voice_to_sid(voice) == expected_sid, f"voice={voice!r} should resolve to sid={expected_sid}"


def test_voice_to_sid_specific_kokoro_v0_19_assignments(tmp_path):
    """Explicit assertions per the task spec, in case the loop above
    masks a per-name regression behind tuple-equality."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)

    assert backend._voice_to_sid("af") == 0
    assert backend._voice_to_sid("af_bella") == 1
    assert backend._voice_to_sid("af_nicole") == 2
    assert backend._voice_to_sid("af_sarah") == 3
    assert backend._voice_to_sid("af_sky") == 4
    assert backend._voice_to_sid("am_adam") == 5
    assert backend._voice_to_sid("am_michael") == 6
    assert backend._voice_to_sid("bf_emma") == 7
    assert backend._voice_to_sid("bf_isabella") == 8
    assert backend._voice_to_sid("bm_george") == 9
    assert backend._voice_to_sid("bm_lewis") == 10


def test_voice_to_sid_bm_george_no_longer_aliases_to_sid_zero(tmp_path):
    """Direct regression: pre-fix, ``bm_george`` resolved to ``sid=0``
    via the legacy 28-item catalog (index 26 → out of range → clamp).
    Post-fix it must resolve to its real position in v0_19's bundle."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("bm_george") != 0
    assert backend._voice_to_sid("bm_george") == 9


def test_voice_to_sid_empty_voice_returns_default_speaker(tmp_path):
    """Empty / missing voice intentionally maps to ``sid=0``: the
    backend's first/default speaker is the contracted default."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("") == 0


def test_voice_to_sid_rejects_am_echo_for_v0_19_artifact(tmp_path):
    """``am_echo`` lives in the modern Kokoro catalog but NOT in
    the v0_19 bundle. Pre-fix it silently fell through to sid=0;
    post-fix it must raise voice_not_supported_for_model so callers
    can correct their request."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("am_echo")
    assert ei.value.code == OctomilErrorCode.INVALID_INPUT
    msg = str(ei.value)
    assert "voice_not_supported_for_model" in msg
    assert "kokoro-82m" in msg
    assert "am_adam" in msg  # supported voices listed in message


def test_voice_to_sid_rejects_old_global_catalog_voices(tmp_path):
    """Every name that lived in the old 28-item global catalog but
    NOT in v0_19 must now raise. These are the exact voices that
    sherpa-onnx logged "sid should be in range [0, 10]" for."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    out_of_bundle = [
        "af_alloy",
        "af_aoede",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nova",
        "af_river",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
    ]
    for voice in out_of_bundle:
        with pytest.raises(OctomilError) as ei:
            backend._voice_to_sid(voice)
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT
        assert "voice_not_supported_for_model" in str(ei.value), voice


def test_voice_to_sid_rejects_completely_unknown_voice(tmp_path):
    """A voice that isn't in any Kokoro catalog (e.g. an OpenAI
    cloud voice) must raise rather than fall back to sid=0."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("alloy")
    assert "voice_not_supported_for_model" in str(ei.value)


def test_voice_to_sid_voices_txt_overrides_legacy_fallback(tmp_path):
    """A voices.txt sidecar takes precedence over the per-model
    legacy fallback. Future Kokoro recipes can ship a different
    speaker order without code changes."""
    custom = ("custom_a", "custom_b", "custom_c")
    (tmp_path / "voices.txt").write_text("\n".join(custom) + "\n", encoding="utf-8")
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("custom_a") == 0
    assert backend._voice_to_sid("custom_b") == 1
    assert backend._voice_to_sid("custom_c") == 2
    # And v0_19 names are NOT in the custom catalog → reject.
    with pytest.raises(OctomilError):
        backend._voice_to_sid("af_bella")


def test_voice_to_sid_falls_back_to_per_model_legacy_when_sidecar_missing(tmp_path):
    """An artifact directory hand-staged before voices.txt
    materialization shipped (no sidecar) still resolves voices via
    the per-model legacy fallback. The fallback is keyed by model
    id, so it can't accidentally inherit some other artifact's
    catalog."""
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("af_bella") == 1
    assert backend._voice_to_sid("bm_george") == 9
    # Voice outside the legacy fallback still raises.
    with pytest.raises(OctomilError):
        backend._voice_to_sid("am_echo")


def test_voice_to_sid_voice_lookup_is_case_insensitive(tmp_path):
    """Caller-supplied voice ids are compared case-insensitively
    so accidentally up-cased ``AF_BELLA`` doesn't fail loudly."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("AF_BELLA") == 1
    assert backend._voice_to_sid("Bm_George") == 9


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def test_catalog_for_model_returns_v0_19_for_kokoro_aliases():
    assert catalog_for_model("kokoro-82m") == _KOKORO_EN_V0_19_VOICES
    assert catalog_for_model("kokoro-en-v0_19") == _KOKORO_EN_V0_19_VOICES
    assert catalog_for_model("KOKORO-82M") == _KOKORO_EN_V0_19_VOICES


def test_catalog_for_model_returns_empty_tuple_for_unknown_model():
    assert catalog_for_model("piper-en-amy") == ()
    assert catalog_for_model("not-a-real-model") == ()


# ---------------------------------------------------------------------------
# Kernel pre-flight uses the recipe manifest, not the global catalog
# ---------------------------------------------------------------------------


def test_kernel_validate_local_voice_uses_recipe_manifest():
    """The kernel's pre-flight must validate against the static
    recipe's voice_manifest so callers get the supported voices for
    THEIR model id, not a global "kokoro = these names" list."""
    from unittest.mock import MagicMock

    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()

    # Voice that's in v0_19 → no raise.
    kernel._validate_local_voice("kokoro-82m", "bm_george")

    # Voice that's not in v0_19 → raises with the model id and
    # the supported voices in the message.
    with pytest.raises(OctomilError) as ei:
        kernel._validate_local_voice("kokoro-82m", "am_echo")
    msg = str(ei.value)
    assert "voice_not_supported_for_model" in msg
    assert "kokoro-82m" in msg
    assert "bm_george" in msg
    # Legacy tag preserved for callers that still grep for it.
    assert "voice_not_supported_for_locality" in msg

    # Empty voice → no raise (kernel skips early).
    kernel._validate_local_voice("kokoro-82m", None)
    kernel._validate_local_voice("kokoro-82m", "")


def test_kernel_validate_local_voice_skips_when_no_recipe():
    """For models the SDK has no static recipe for (e.g. Piper
    bundles), the kernel skips voice validation and lets the
    backend surface mismatches."""
    from unittest.mock import MagicMock

    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()

    # piper-en-amy is in _SHERPA_TTS_MODELS but has no static recipe
    # / voice manifest → skip rather than raise.
    kernel._validate_local_voice("piper-en-amy", "any-voice")
