"""Regression tests for the Kokoro voice-manifest fix and v1.0 cutover.

Background — two related bugs the fix had to address:

  - octomil[tts] 4.12.x with sherpa-onnx 1.13.0 used a hardcoded
    28-name catalog (``_KOKORO_VOICES``) to map voice strings to
    sherpa-onnx speaker ids. The static recipe for ``kokoro-82m``
    downloaded ``kokoro-en-v0_19.tar.bz2``, an 11-speaker bundle.
    Because the global catalog had a different order AND included
    names absent from the bundle, voices like ``am_echo`` /
    ``af_heart`` resolved to ``sid >= 11``; sherpa-onnx then logged
    ``sid should be in range [0, 10]`` and silently rendered every
    bad voice with ``sid=0``. Even "in-range" voices were misnamed
    because the table order didn't match the bundle.

  - The first patch fixed the catalog drift but broke catalog-less
    Piper models (``synthesize()`` resolved the model default voice
    string and ran it through the strict catalog check, raising for
    ``amy`` / ``ryan``); and the kernel's preflight rejected valid
    voices for planner-selected non-static artifacts because it only
    knew the static recipe's manifest.

The post-fix invariants pinned here:

  1. Static recipe for ``kokoro-82m`` declares the 53-speaker
     kokoro-multi-lang-v1_0 catalog; ``kokoro-en-v0_19`` keeps the
     legacy 11-speaker catalog as a separate recipe.
  2. ``MaterializationPlan`` writes ``voices.txt`` + ``VERSION``
     sidecars deterministically.
  3. Backend ``_voice_to_sid`` reads ``voices.txt`` first, falls
     back to a per-model legacy catalog, raises
     ``voice_not_supported_for_model`` for explicit unknown voices,
     and returns ``sid=0`` for the model-default fallback when no
     catalog is available (Piper) — never silently aliases.
  4. Kernel preflight only enforces the static-manifest check when
     the static recipe is the chosen artifact.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.engines.sherpa.engine import (
    _KOKORO_EN_V0_19_VOICES,
    _KOKORO_MULTI_LANG_V1_0_VOICES,
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
    KOKORO_MULTI_LANG_V1_0_VOICES,
    get_static_recipe,
)

# ---------------------------------------------------------------------------
# v1.0 catalog — authoritative ordering from upstream
# scripts/kokoro/v1.0/generate_voices_bin.py.
# ---------------------------------------------------------------------------


_EXPECTED_V1_0 = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)


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


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def test_kokoro_82m_recipe_pins_v1_0_bundle():
    """``kokoro-82m`` resolves to the v1.0 multi-lang bundle, with
    its 53-speaker catalog declared in the MaterializationPlan."""
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    assert recipe.files[0].relative_path == "kokoro-multi-lang-v1_0.tar.bz2"
    assert recipe.materialization.source == "kokoro-multi-lang-v1_0.tar.bz2"
    assert recipe.materialization.strip_prefix == "kokoro-multi-lang-v1_0/"
    assert recipe.materialization.artifact_version == "kokoro-multi-lang-v1_0"
    assert recipe.materialization.voice_manifest == _EXPECTED_V1_0
    # 28-name English subset matches the front of the v1.0 catalog.
    assert recipe.materialization.voice_manifest[:28] == (
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
        "bm_fable", "bm_george", "bm_lewis",
    )  # fmt: skip


def test_kokoro_en_v0_19_recipe_kept_as_legacy():
    """The explicit ``kokoro-en-v0_19`` id keeps resolving to the
    legacy 11-speaker bundle. It is NOT aliased to the v1.0 recipe;
    digest, layout, and voice manifest are all v0.19."""
    recipe = get_static_recipe("kokoro-en-v0_19", "tts")
    assert recipe is not None
    assert recipe.files[0].relative_path == "kokoro-en-v0_19.tar.bz2"
    assert recipe.materialization.strip_prefix == "kokoro-en-v0_19/"
    assert recipe.materialization.artifact_version == "kokoro-en-v0_19"
    assert recipe.materialization.voice_manifest == _EXPECTED_V0_19
    assert "espeak-ng-data/phontab" in recipe.materialization.required_outputs

    # Distinct from the v1.0 recipe.
    v1_recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not v1_recipe
    assert recipe.files[0].digest != v1_recipe.files[0].digest


def test_module_exports_match_recipe_manifests():
    """``KOKORO_*`` exported tuples match recipe manifests so
    external callers can import the catalog without going through
    ``get_static_recipe``."""
    assert KOKORO_MULTI_LANG_V1_0_VOICES == _EXPECTED_V1_0
    assert KOKORO_EN_V0_19_VOICES == _EXPECTED_V0_19
    assert _KOKORO_MULTI_LANG_V1_0_VOICES == _EXPECTED_V1_0
    assert _KOKORO_EN_V0_19_VOICES == _EXPECTED_V0_19


# ---------------------------------------------------------------------------
# Materializer writes voices.txt + VERSION
# ---------------------------------------------------------------------------


def test_materializer_writes_voices_txt_for_kokoro_recipe(tmp_path):
    """After materialize(), voices.txt contains the manifest in
    order and VERSION carries the artifact version."""
    plan = MaterializationPlan(
        kind="none",
        required_outputs=(),
        voice_manifest=_EXPECTED_V1_0,
        artifact_version="kokoro-multi-lang-v1_0",
    )
    Materializer().materialize(tmp_path, plan)

    voices_txt = (tmp_path / "voices.txt").read_text(encoding="utf-8")
    assert voices_txt.splitlines() == list(_EXPECTED_V1_0)

    version = (tmp_path / "VERSION").read_text(encoding="utf-8").strip()
    assert version == "kokoro-multi-lang-v1_0"


def test_materializer_overwrites_stale_voices_txt(tmp_path):
    """A stale voices.txt left by a previous recipe revision is
    replaced — the manifest is the single source of truth, so a
    silent merge would risk shifting speaker ids."""
    (tmp_path / "voices.txt").write_text("old_voice_a\nold_voice_b\n", encoding="utf-8")
    plan = MaterializationPlan(
        kind="none",
        required_outputs=(),
        voice_manifest=_EXPECTED_V1_0,
    )
    Materializer().materialize(tmp_path, plan)

    voices_txt = (tmp_path / "voices.txt").read_text(encoding="utf-8")
    assert voices_txt.splitlines() == list(_EXPECTED_V1_0)


def test_materializer_no_op_when_manifest_empty(tmp_path):
    """Recipes that don't declare a voice_manifest (single-speaker
    Piper bundles, the whisper recipe, …) get no voices.txt."""
    plan = MaterializationPlan(kind="none", required_outputs=())
    Materializer().materialize(tmp_path, plan)
    assert not (tmp_path / "voices.txt").exists()
    assert not (tmp_path / "VERSION").exists()


def test_materializer_writes_voice_manifest_after_archive_extraction(tmp_path):
    """Archive recipes get voices.txt written *after* extraction so
    a half-extracted artifact never carries a manifest that would
    mislead the engine."""
    import tarfile
    from io import BytesIO

    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for rel, content in [
            ("kokoro-multi-lang-v1_0/model.onnx", b"\x00onnx"),
            ("kokoro-multi-lang-v1_0/voices.bin", b"\x00voicebin"),
            ("kokoro-multi-lang-v1_0/tokens.txt", b"a 1\n"),
            ("kokoro-multi-lang-v1_0/lexicon-us-en.txt", b"a a\n"),
            ("kokoro-multi-lang-v1_0/lexicon-gb-en.txt", b"a a\n"),
            ("kokoro-multi-lang-v1_0/lexicon-zh.txt", b"a a\n"),
            ("kokoro-multi-lang-v1_0/dict/jieba.dict.utf8", b"\x00\x00"),
        ]:
            data = content
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    archive = tmp_path / "kokoro-multi-lang-v1_0.tar.bz2"
    archive.write_bytes(buf.getvalue())

    plan = MaterializationPlan(
        kind="archive",
        source="kokoro-multi-lang-v1_0.tar.bz2",
        archive_format="tar.bz2",
        strip_prefix="kokoro-multi-lang-v1_0/",
        required_outputs=(
            "model.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "lexicon-gb-en.txt",
            "lexicon-zh.txt",
            "dict/jieba.dict.utf8",
        ),
        voice_manifest=_EXPECTED_V1_0,
        artifact_version="kokoro-multi-lang-v1_0",
    )
    Materializer().materialize(tmp_path, plan)

    assert (tmp_path / EXTRACTION_MARKER).is_file()
    assert (tmp_path / "voices.txt").read_text(encoding="utf-8").splitlines() == list(_EXPECTED_V1_0)
    assert (tmp_path / "VERSION").read_text(encoding="utf-8").strip() == "kokoro-multi-lang-v1_0"


# ---------------------------------------------------------------------------
# Engine voice resolution: voices.txt is authoritative
# ---------------------------------------------------------------------------


def _make_backend(model_dir: Path, model: str = "kokoro-82m") -> _SherpaTtsBackend:
    return _SherpaTtsBackend(model, model_dir=str(model_dir))


def _write_v1_0_manifest(model_dir: Path) -> None:
    (model_dir / "voices.txt").write_text("\n".join(_EXPECTED_V1_0) + "\n", encoding="utf-8")


def _write_v0_19_manifest(model_dir: Path) -> None:
    (model_dir / "voices.txt").write_text("\n".join(_EXPECTED_V0_19) + "\n", encoding="utf-8")


def test_voice_to_sid_resolves_each_v1_0_voice_to_correct_index(tmp_path):
    """Every voice in the v1.0 catalog resolves to its position in
    voices.txt — pre-fix, ``bm_george`` resolved to ``sid=0`` because
    the (different) v0_19 bundle didn't have it at index 26."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    for expected_sid, voice in enumerate(_EXPECTED_V1_0):
        assert backend._voice_to_sid(voice) == expected_sid, f"voice={voice!r} should resolve to sid={expected_sid}"


def test_voice_to_sid_specific_v1_0_assignments(tmp_path):
    """Explicit per-name assertions for the previously-missing
    voices that motivated the v1.0 cutover."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("af_alloy") == 0
    assert backend._voice_to_sid("af_bella") == 2
    assert backend._voice_to_sid("am_adam") == 11
    assert backend._voice_to_sid("am_echo") == 12  # the canonical regression case
    assert backend._voice_to_sid("am_michael") == 16
    assert backend._voice_to_sid("bf_emma") == 21
    assert backend._voice_to_sid("bm_george") == 26
    assert backend._voice_to_sid("bm_lewis") == 27
    assert backend._voice_to_sid("zm_yunyang") == 52


def test_voice_to_sid_v0_19_legacy_assignments(tmp_path):
    """The v0_19 11-speaker catalog still resolves correctly when
    its sidecar is on disk (kokoro-en-v0_19 model id retained)."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path, model="kokoro-en-v0_19")
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
    """Direct regression: pre-fix ``bm_george`` resolved to ``sid=0``
    because the global catalog had it at index 26 (out of range for
    the 11-speaker bundle). Post-fix it must resolve to its real
    speaker id under whichever bundle is active."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("bm_george") == 26
    assert backend._voice_to_sid("bm_george") != 0


def test_voice_to_sid_empty_voice_returns_default_speaker(tmp_path):
    """Empty / missing voice intentionally maps to ``sid=0``."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("") == 0


def test_voice_to_sid_rejects_unknown_voice_against_v1_0(tmp_path):
    """A voice that isn't in the v1.0 catalog raises rather than
    falling back to sid=0."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("not_a_real_voice")
    assert ei.value.code == OctomilErrorCode.INVALID_INPUT
    msg = str(ei.value)
    assert "voice_not_supported_for_model" in msg
    assert "kokoro-82m" in msg
    assert "af_bella" in msg  # supported voices are listed


def test_voice_to_sid_rejects_v1_0_voice_against_v0_19_artifact(tmp_path):
    """A voice from the v1.0 catalog must NOT resolve under a
    v0_19 sidecar — the bundles disagree on speaker ids and silent
    aliasing was the original bug. ``af_alloy`` belongs to v1.0."""
    _write_v0_19_manifest(tmp_path)
    backend = _make_backend(tmp_path, model="kokoro-en-v0_19")
    with pytest.raises(OctomilError):
        backend._voice_to_sid("af_alloy")


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
    with pytest.raises(OctomilError):
        backend._voice_to_sid("af_bella")  # not in the custom list


def test_voice_to_sid_fallback_keys_off_VERSION_sidecar(tmp_path):
    """Sidecar-less artifact whose VERSION advertises v1.0 → use
    the v1.0 catalog. This is the post-cutover happy path for
    artifacts staged before the voices.txt patch shipped."""
    (tmp_path / "VERSION").write_text("kokoro-multi-lang-v1_0\n", encoding="utf-8")
    # Layout signals consistent with v1.0.
    (tmp_path / "lexicon-us-en.txt").write_bytes(b"x")
    (tmp_path / "dict").mkdir()
    (tmp_path / "dict" / "jieba.dict.utf8").write_bytes(b"x")

    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("af_alloy") == 0
    assert backend._voice_to_sid("am_echo") == 12
    assert backend._voice_to_sid("bm_george") == 26


def test_voice_to_sid_fallback_refuses_pre_cutover_v0_19_dir_under_kokoro_82m(tmp_path):
    """The P2 reviewer reproducer: an OLD ``kokoro-82m`` prepared
    dir from before the cutover still has the v0.19 layout
    (espeak-ng-data/, no dict/, no lexicon-*.txt) and no VERSION
    sidecar. The fallback must NOT inherit the v1.0 catalog by
    model id alone — that would map ``bm_george`` to sid 26 against
    a bundle whose voices.bin only carries 11 speakers, and
    sherpa-onnx would clamp it back to sid 0 silently. Refuse the
    explicit voice path instead."""
    # v0.19 layout under the kokoro-82m model id (the pre-cutover
    # state for any user upgrading without re-prepare).
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")
    backend = _make_backend(tmp_path)  # model="kokoro-82m"

    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("bm_george")
    assert "voice_not_supported_for_model" in str(ei.value)


def test_voice_to_sid_fallback_uses_v0_19_for_unambiguous_legacy_dir(tmp_path):
    """A sidecar-less dir under the EXPLICIT ``kokoro-en-v0_19``
    model id, with v0.19 layout, resolves correctly via the
    v0_19 catalog — no ambiguity with the v1.0 bundle."""
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")
    backend = _make_backend(tmp_path, model="kokoro-en-v0_19")
    assert backend._voice_to_sid("af_bella") == 1
    assert backend._voice_to_sid("bm_george") == 9


def test_voice_to_sid_fallback_uses_v1_0_when_layout_unambiguous(tmp_path):
    """Sidecar-less, VERSION-less dir whose layout uniquely matches
    v1.0 (dict/ + lexicon-*.txt) under the kokoro-82m id: layout
    + model id agree, so fall back to the v1.0 catalog."""
    (tmp_path / "lexicon-us-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-gb-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-zh.txt").write_bytes(b"x")
    (tmp_path / "dict").mkdir()
    (tmp_path / "dict" / "jieba.dict.utf8").write_bytes(b"x")
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")

    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("am_echo") == 12
    assert backend._voice_to_sid("bm_george") == 26


def test_voice_to_sid_voice_lookup_is_case_insensitive(tmp_path):
    """Voice ids are compared case-insensitively."""
    _write_v1_0_manifest(tmp_path)
    backend = _make_backend(tmp_path)
    assert backend._voice_to_sid("AM_ECHO") == 12
    assert backend._voice_to_sid("Bm_George") == 26


# ---------------------------------------------------------------------------
# Default-voice path: catalog-less Piper models must not raise
# ---------------------------------------------------------------------------


def test_default_voice_for_catalog_less_model_returns_sid_zero(tmp_path):
    """``synthesize(voice=None)`` resolves to the model default
    string (``amy`` for piper-en-amy). Piper has no catalog and no
    sidecar; the default-voice path must NOT raise — single-speaker
    bundles have nothing to validate against."""
    backend = _SherpaTtsBackend("piper-en-amy", model_dir=str(tmp_path))
    # explicit=False is what synthesize() passes for a defaulted voice.
    assert backend._voice_to_sid("amy", explicit=False) == 0


def test_explicit_voice_for_catalog_less_model_still_raises(tmp_path):
    """When the caller explicitly supplies a voice for a catalog-
    less model, refuse loudly — we have no way to validate the
    request, and silent ``sid=0`` aliasing is exactly the bug class
    the manifest fix exists to prevent."""
    backend = _SherpaTtsBackend("piper-en-amy", model_dir=str(tmp_path))
    with pytest.raises(OctomilError) as ei:
        backend._voice_to_sid("amy", explicit=True)
    assert "voice_not_supported_for_model" in str(ei.value)


def test_default_voice_outside_catalog_falls_back_to_sid_zero(tmp_path):
    """If the model's documented default voice happens not to be in
    the artifact's catalog (rare; e.g. recipe drift), the
    default-voice path lands on ``sid=0`` instead of raising. The
    explicit path stays strict."""
    custom = ("custom_a", "custom_b")
    (tmp_path / "voices.txt").write_text("\n".join(custom) + "\n", encoding="utf-8")
    backend = _make_backend(tmp_path)
    # af_bella is the documented kokoro-82m default; not in the
    # custom catalog. explicit=False → sid=0.
    assert backend._voice_to_sid("af_bella", explicit=False) == 0
    # explicit=True → raise.
    with pytest.raises(OctomilError):
        backend._voice_to_sid("af_bella", explicit=True)


def test_synthesize_call_site_marks_default_voice_inexplicit(monkeypatch, tmp_path):
    """Belt-and-suspenders: the synthesize() call path resolves a
    None voice through the model default and calls _voice_to_sid
    with explicit=False. Verify by stubbing _tts and capturing the
    call shape."""
    backend = _SherpaTtsBackend("piper-en-amy", model_dir=str(tmp_path))

    fake_tts = MagicMock()
    fake_audio = MagicMock()
    fake_audio.samples = []
    fake_audio.sample_rate = 24000
    fake_tts.generate.return_value = fake_audio
    backend._tts = fake_tts

    captured: dict[str, object] = {}

    def spy(voice, *, explicit=True):
        captured["voice"] = voice
        captured["explicit"] = explicit
        return 0  # short-circuit; we're testing the call shape, not resolution

    monkeypatch.setattr(backend, "_voice_to_sid", spy)
    backend.synthesize("hello", voice=None)
    assert captured["voice"] == "amy"
    assert captured["explicit"] is False

    # And explicit voice passes explicit=True.
    captured.clear()
    backend.synthesize("hello", voice="custom")
    assert captured["voice"] == "custom"
    assert captured["explicit"] is True


# ---------------------------------------------------------------------------
# sherpa-onnx config builder: data_dir is required even for v1.0
# ---------------------------------------------------------------------------


def test_build_kokoro_model_config_v0_19_layout_sets_data_dir(tmp_path):
    """v0.19 layout (espeak-ng-data/) → data_dir wires the espeak
    directory. model/voices/tokens point at the artifact dir."""
    from octomil.runtime.engines.sherpa.engine import _build_kokoro_model_config

    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "voices.bin").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_bytes(b"x")
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")

    captured: dict[str, str] = {}

    class _FakeSherpa:
        @staticmethod
        def OfflineTtsKokoroModelConfig(**kwargs):
            captured.update(kwargs)
            return object()

    _build_kokoro_model_config(_FakeSherpa, str(tmp_path))
    assert captured["data_dir"] == str(tmp_path / "espeak-ng-data")
    assert captured["model"].endswith("/model.onnx")
    assert captured["voices"].endswith("/voices.bin")
    assert captured["tokens"].endswith("/tokens.txt")
    # v0.19 must NOT set the v1.0 lexicon/dict knobs.
    assert "lexicon" not in captured
    assert "dict_dir" not in captured


def test_build_kokoro_model_config_v1_0_layout_wires_espeak_alongside_lexicon(tmp_path):
    """sherpa-onnx 1.13.0's OfflineTtsKokoroModelConfig requires
    ``data_dir`` as a keyword argument (omitting it raises
    TypeError). The upstream v1.0 bundle ships ``espeak-ng-data/``
    AND lexicon files; upstream's own invocation passes both, so we
    must do the same — an empty ``data_dir`` when espeak data is on
    disk risks broken / OOV phonemization for languages not covered
    by the lexicon."""
    from octomil.runtime.engines.sherpa.engine import _build_kokoro_model_config

    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "voices.bin").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_bytes(b"x")
    (tmp_path / "lexicon-us-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-gb-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-zh.txt").write_bytes(b"x")
    (tmp_path / "dict").mkdir()
    (tmp_path / "dict" / "jieba.dict.utf8").write_bytes(b"x")
    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "espeak-ng-data" / "phontab").write_bytes(b"x")

    captured: dict[str, str] = {}

    class _FakeSherpa:
        @staticmethod
        def OfflineTtsKokoroModelConfig(**kwargs):
            captured.update(kwargs)
            return object()

    _build_kokoro_model_config(_FakeSherpa, str(tmp_path))
    # data_dir wired to the espeak-ng-data path, not left empty —
    # the v1.0 bundle ships espeak data and upstream uses it.
    assert captured["data_dir"] == str(tmp_path / "espeak-ng-data")
    # v1.0 lexicon + dict_dir wired alongside espeak.
    assert captured["dict_dir"] == str(tmp_path / "dict")
    lex = captured["lexicon"]
    assert "lexicon-us-en.txt" in lex
    assert "lexicon-gb-en.txt" in lex
    assert "lexicon-zh.txt" in lex


def test_build_kokoro_model_config_v1_0_without_espeak_uses_empty_data_dir(tmp_path):
    """Defensive: if a hand-staged or stripped v1.0 dir lacks
    espeak data, ``data_dir`` falls back to '' so the
    sherpa-onnx 1.13.0 constructor still accepts the kwargs.
    Lexicon + dict_dir still wire correctly so English/Chinese
    paths keep working."""
    from octomil.runtime.engines.sherpa.engine import _build_kokoro_model_config

    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "voices.bin").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_bytes(b"x")
    (tmp_path / "lexicon-us-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-gb-en.txt").write_bytes(b"x")
    (tmp_path / "lexicon-zh.txt").write_bytes(b"x")
    (tmp_path / "dict").mkdir()
    (tmp_path / "dict" / "jieba.dict.utf8").write_bytes(b"x")

    captured: dict[str, str] = {}

    class _FakeSherpa:
        @staticmethod
        def OfflineTtsKokoroModelConfig(**kwargs):
            captured.update(kwargs)
            return object()

    _build_kokoro_model_config(_FakeSherpa, str(tmp_path))
    assert "data_dir" in captured
    assert captured["data_dir"] == ""
    assert "dict_dir" in captured
    assert "lexicon" in captured


def test_build_kokoro_model_config_rejects_unknown_layout(tmp_path):
    """Neither espeak-ng-data/ nor dict/+lexicon-*.txt → fail loudly
    rather than build a half-configured backend that errors at
    inference time."""
    from octomil.runtime.engines.sherpa.engine import _build_kokoro_model_config

    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "voices.bin").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_bytes(b"x")

    class _FakeSherpa:
        @staticmethod
        def OfflineTtsKokoroModelConfig(**kwargs):
            return object()

    with pytest.raises(OctomilError) as ei:
        _build_kokoro_model_config(_FakeSherpa, str(tmp_path))
    assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "espeak-ng-data" in str(ei.value)
    assert "lexicon" in str(ei.value)


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def test_catalog_for_model_returns_v1_0_for_kokoro_82m():
    """kokoro-82m's legacy fallback IS the v1.0 catalog post-cutover."""
    assert catalog_for_model("kokoro-82m") == _KOKORO_MULTI_LANG_V1_0_VOICES
    assert catalog_for_model("KOKORO-82M") == _KOKORO_MULTI_LANG_V1_0_VOICES


def test_catalog_for_model_returns_v0_19_for_legacy_id():
    """kokoro-en-v0_19 keeps its own 11-speaker fallback so
    artifacts pinned to that id resolve correctly even without a
    sidecar."""
    assert catalog_for_model("kokoro-en-v0_19") == _KOKORO_EN_V0_19_VOICES


def test_catalog_for_model_returns_empty_tuple_for_unknown_model():
    assert catalog_for_model("piper-en-amy") == ()
    assert catalog_for_model("not-a-real-model") == ()


# ---------------------------------------------------------------------------
# Kernel pre-flight: scoped to the static-recipe path
# ---------------------------------------------------------------------------


def _kernel():
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()
    return kernel


def test_kernel_validate_local_voice_uses_recipe_manifest_for_static_path():
    """When no planner candidate is in play (selection=None), the
    kernel uses the static recipe's manifest to fast-reject."""
    kernel = _kernel()

    # Voice in v1.0 catalog → no raise.
    kernel._validate_local_voice("kokoro-82m", "am_echo", selection=None)

    # Voice not in v1.0 catalog → raises with model id and supported voices.
    with pytest.raises(OctomilError) as ei:
        kernel._validate_local_voice("kokoro-82m", "not_a_voice", selection=None)
    msg = str(ei.value)
    assert "voice_not_supported_for_model" in msg
    assert "kokoro-82m" in msg
    assert "am_echo" in msg
    # Legacy substring preserved for callers grepping for it.
    assert "voice_not_supported_for_locality" in msg


def test_kernel_validate_local_voice_skips_for_planner_selected_artifact():
    """When the planner returns a non-static candidate (different
    artifact_id and digest), kernel preflight defers to the
    backend's voices.txt-based check — preventing the static
    manifest from rejecting voices the planner artifact actually
    supports (e.g. a private 28-speaker bundle pre-v1.0)."""
    kernel = _kernel()

    # Mock a planner selection whose local sdk_runtime candidate
    # points at a different artifact (different artifact_id, no
    # matching digest).
    selection = MagicMock()
    candidate = MagicMock()
    artifact = MagicMock()
    artifact.digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001"
    artifact.artifact_id = "private-kokoro-v9000"
    candidate.artifact = artifact

    import octomil.execution.kernel as kernel_mod

    original = kernel_mod._local_sdk_runtime_candidate
    kernel_mod._local_sdk_runtime_candidate = lambda _sel: candidate
    try:
        # Voice not in v1.0 catalog — but the planner artifact is
        # different, so we must NOT raise; defer to backend.
        kernel._validate_local_voice(
            "kokoro-82m",
            "am_echo_xyz",
            selection=selection,
            prepared_cache_dir=None,
        )
    finally:
        kernel_mod._local_sdk_runtime_candidate = original


def test_kernel_validate_local_voice_runs_when_prepared_cache_dir_set():
    """The static-cache short-circuit path is exactly the path
    where preflight matters — the static recipe IS what's serving
    the request. Confirm preflight runs."""
    kernel = _kernel()
    with pytest.raises(OctomilError) as ei:
        kernel._validate_local_voice(
            "kokoro-82m",
            "not_a_voice",
            selection=MagicMock(),
            prepared_cache_dir="/tmp/whatever",
        )
    assert "voice_not_supported_for_model" in str(ei.value)


def test_kernel_validate_local_voice_runs_when_planner_picked_static_recipe():
    """Planner can legitimately return the same artifact identity
    as the static recipe (digest match). Preflight should run."""
    kernel = _kernel()

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    static_digest = recipe.files[0].digest

    selection = MagicMock()
    candidate = MagicMock()
    artifact = MagicMock()
    artifact.digest = static_digest  # same as static recipe
    artifact.artifact_id = "kokoro-82m"
    candidate.artifact = artifact

    import octomil.execution.kernel as kernel_mod

    original = kernel_mod._local_sdk_runtime_candidate
    kernel_mod._local_sdk_runtime_candidate = lambda _sel: candidate
    try:
        with pytest.raises(OctomilError) as ei:
            kernel._validate_local_voice("kokoro-82m", "not_a_voice", selection=selection)
        assert "voice_not_supported_for_model" in str(ei.value)
    finally:
        kernel_mod._local_sdk_runtime_candidate = original


def test_kernel_validate_local_voice_skips_when_no_recipe():
    """For models the SDK has no static recipe for (e.g. Piper
    bundles), the kernel skips voice validation and lets the
    backend surface mismatches."""
    kernel = _kernel()
    # piper-en-amy is in _SHERPA_TTS_MODELS but has no static recipe
    # → skip rather than raise.
    kernel._validate_local_voice("piper-en-amy", "any-voice", selection=None)


def test_kernel_validate_local_voice_no_voice_no_raise():
    """Empty / None voice never raises (default speaker path)."""
    kernel = _kernel()
    kernel._validate_local_voice("kokoro-82m", None, selection=None)
    kernel._validate_local_voice("kokoro-82m", "", selection=None)
