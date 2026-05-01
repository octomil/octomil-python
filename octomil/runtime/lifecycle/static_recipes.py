"""Static offline recipes for canonical local models.

A "recipe" is a *pair*:

  - a download description (``RuntimeCandidatePlan`` in the planner
    schema), describing where the bytes live on the public CDN and
    how to verify them;
  - a :class:`MaterializationPlan` describing how to arrange those
    bytes on disk so the downstream backend can read them.

The kernel knows nothing about Kokoro, tarballs, or sherpa-onnx. It
runs:

    candidate = static_recipe_candidate(model, capability)
    outcome = PrepareManager.prepare(candidate)
    Materializer().materialize(
        outcome.artifact_dir,
        get_static_recipe(model, capability).materialization,
    )

Adding a new model is a *data row* in ``_RECIPES``, not a code path.
The principal-engineer review of PR #455 was specifically that the
Kokoro path had been hard-coded into the kernel and the materializer;
moving everything Kokoro-specific into a single declarative recipe
ensures the second model is data, not duplicated logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from octomil.runtime.lifecycle.materialization import (
    MaterializationPlan,
    MaterializationSafetyPolicy,
)
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)


@dataclass(frozen=True)
class _StaticArtifactFile:
    """One file inside a static recipe's download manifest.

    Fields mirror what the planner-side schema carries — relative
    path, public URL, SHA-256 digest, optional size hint.
    """

    relative_path: str
    url: str
    digest: str
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class StaticRecipe:
    """Static metadata for a known canonical local model.

    Two halves:

      - ``files`` describes the download (passed through to
        ``RuntimeCandidatePlan`` / ``ArtifactDownloadEndpoint``);
      - ``materialization`` describes the post-download arrangement
        the backend reads at inference time. ``MaterializationPlan
        (kind='none')`` covers single-file backends like
        ``whisper.cpp``; ``kind='archive'`` covers Kokoro and any
        future tarball-shipped bundle.

    Recipes are intentionally narrow: lookup keys to canonical
    public bundles only. Models that need auth or private CDNs go
    through the planner.
    """

    model_id: str
    capability: str
    engine: str
    files: list[_StaticArtifactFile] = field(default_factory=list)
    materialization: MaterializationPlan = field(default_factory=MaterializationPlan)
    notes: str = ""

    def to_runtime_candidate(self) -> RuntimeCandidatePlan:
        """Build a planner-shaped candidate from this recipe.

        Single-file recipes (today's only shape) wire the artifact's
        ``digest`` directly to the file's SHA-256 — that's the value
        ``PrepareManager._build_descriptor`` hands to the durable
        downloader for verification. Multi-file recipes need
        ``manifest_uri`` support in PrepareManager and are rejected
        until that lands.
        """
        if len(self.files) != 1:
            raise ValueError(
                f"static recipe {self.model_id!r} has {len(self.files)} files; "
                "PrepareManager today supports single-file recipes only — "
                "the planner schema carries one artifact-level digest. "
                "Multi-file recipes need manifest_uri support (tracked as a follow-up)."
            )
        only = self.files[0]
        endpoints = [
            ArtifactDownloadEndpoint(
                url=only.url,
                # Sentinel header: PR-tracked metadata for the
                # post-prepare materialization step. Keeps the
                # download path unaware of recipe semantics.
                headers={"X-Octomil-Recipe-Path": only.relative_path},
            )
        ]
        return RuntimeCandidatePlan(
            locality="local",
            priority=0,
            confidence=1.0,
            reason=f"static offline recipe for {self.model_id!r}",
            engine=self.engine,
            artifact=RuntimeArtifactPlan(
                model_id=self.model_id,
                artifact_id=self.model_id,
                format="onnx",
                digest=only.digest,
                size_bytes=only.size_bytes,
                required_files=[only.relative_path],
                download_urls=endpoints,
            ),
            delivery_mode="sdk_runtime",
            prepare_required=True,
            prepare_policy="explicit_only",
        )


# ---------------------------------------------------------------------------
# Catalog (data only)
# ---------------------------------------------------------------------------


# Kokoro v1.0 multi-language single-file tarball.
#
# Digest + URL come from the public sherpa-onnx ``tts-models``
# GitHub release. Cross-check with::
#
#     gh api repos/k2-fsa/sherpa-onnx/releases/tags/tts-models \
#         --jq '.assets[] | select(.name == "kokoro-multi-lang-v1_0.tar.bz2")'
#
# v1.0 supersedes v0.19 (which we previously shipped under this
# same model id). Why cut over rather than introduce a new id:
#
#   - v0.19 carried only 11 English speakers, while the SDK's prior
#     hardcoded catalog advertised 28. Voices like ``am_echo`` /
#     ``bm_george`` always silently aliased to ``sid=0``. The
#     long-term fix needs the artifact to actually carry the
#     advertised voices.
#   - v1.0 ships the full 53-speaker multilingual catalog
#     (English af/am/bf/bm + Spanish ef/em + French ff + Hindi
#     hf/hm + Italian if/im + Japanese jf/jm + Portuguese pf/pm +
#     Chinese zf/zm) — a strict superset of the names anyone could
#     have legitimately tried against ``kokoro-82m``.
#   - The on-disk layout is different (lexicon-*.txt + dict/ in
#     place of espeak-ng-data/), so reusing the same id with the
#     new digest forces a clean re-prepare; old prepared dirs
#     cannot be confused with the new bundle.
#
# Authoritative speaker ordering for ``KOKORO_MULTI_LANG_V1_0_VOICES``
# below comes directly from upstream's
# ``scripts/kokoro/v1.0/generate_voices_bin.py`` — that script
# enumerates the speakers in the exact order their float32
# embeddings are concatenated into ``voices.bin``, which IS the
# sherpa-onnx speaker-id table for this bundle.
_KOKORO_82M_TARBALL_SHA256 = "sha256:c133d26353d776da730870dac7da07dbfc9a5e3bc80cc5e8e83ab6e823be7046"
_KOKORO_82M_TARBALL_SIZE = 349_418_188


# Frozen, for tests and diagnostic tooling that still need the
# v0.19 catalog (e.g. snapshot tests of historical artifact dirs).
# Not used by the active recipe.
KOKORO_EN_V0_19_VOICES: tuple[str, ...] = (
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


# kokoro-multi-lang-v1_0 — 53 speakers. Position == sherpa-onnx
# speaker id, transcribed verbatim from upstream's
# ``generate_voices_bin.py``. Any drift here would re-introduce the
# silent ``sid=0`` aliasing bug, so the regression tests in
# ``tests/test_kokoro_voice_manifest.py`` pin every entry.
KOKORO_MULTI_LANG_V1_0_VOICES: tuple[str, ...] = (
    # American English — female (af_*) and male (am_*).
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
    # British English — female (bf_*) and male (bm_*).
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    # Spanish.
    "ef_dora",
    "em_alex",
    # French.
    "ff_siwis",
    # Hindi.
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    # Italian.
    "if_sara",
    "im_nicola",
    # Japanese.
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    # Portuguese.
    "pf_dora",
    "pm_alex",
    "pm_santa",
    # Mandarin Chinese.
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)


_KOKORO_82M_RECIPE = StaticRecipe(
    model_id="kokoro-82m",
    capability="tts",
    engine="sherpa-onnx",
    files=[
        _StaticArtifactFile(
            relative_path="kokoro-multi-lang-v1_0.tar.bz2",
            # ``DurableDownloader._resolve_url`` joins
            # ``endpoint.url`` with ``relative_path`` as
            # ``f"{base.rstrip('/')}/{rel}"``. The ``url`` here is
            # the *parent directory* — the file name comes from
            # ``relative_path``. Setting ``url`` to the full file
            # URL would produce ``…/kokoro-multi-lang-v1_0.tar.bz2/
            # kokoro-multi-lang-v1_0.tar.bz2`` which 404s.
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models",
            digest=_KOKORO_82M_TARBALL_SHA256,
            size_bytes=_KOKORO_82M_TARBALL_SIZE,
        ),
    ],
    materialization=MaterializationPlan(
        kind="archive",
        source="kokoro-multi-lang-v1_0.tar.bz2",
        archive_format="tar.bz2",
        # Upstream tar wraps everything in a top-level directory
        # named ``kokoro-multi-lang-v1_0/``. Strip it so the
        # backend reads ``model.onnx`` directly under
        # ``artifact_dir`` rather than
        # ``artifact_dir/kokoro-multi-lang-v1_0/model.onnx``.
        strip_prefix="kokoro-multi-lang-v1_0/",
        # v1.0 keeps espeak-ng-data/ AND adds lexicon-*.txt + dict/
        # for Chinese segmentation. Upstream's own invocation passes
        # all three directories (--kokoro-data-dir + --kokoro-lexicon
        # + --kokoro-dict-dir), so we require them all here — a
        # partial extraction that drops espeak would silently degrade
        # phonemization at runtime instead of failing at prepare.
        required_outputs=(
            "model.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/phontab",
            "lexicon-us-en.txt",
            "lexicon-gb-en.txt",
            "lexicon-zh.txt",
            "dict/jieba.dict.utf8",
        ),
        # Kokoro's tarball contains no symlinks/hardlinks; default
        # safety policy refuses both, which is what we want.
        safety_policy=MaterializationSafetyPolicy(),
        # Materialize the bundle's authoritative 53-speaker table
        # as a voices.txt sidecar so the sherpa engine resolves
        # voices against THIS artifact's catalog rather than a
        # global hardcoded list. See
        # ``KOKORO_MULTI_LANG_V1_0_VOICES`` for rationale.
        voice_manifest=KOKORO_MULTI_LANG_V1_0_VOICES,
        artifact_version="kokoro-multi-lang-v1_0",
    ),
    notes=(
        "Kokoro v1.0 multi-language TTS published by sherpa-onnx. "
        "53 speakers across English, Spanish, French, Hindi, "
        "Italian, Japanese, Portuguese, and Mandarin. The "
        "MaterializationPlan covers tarball extraction and the "
        "voices.txt sidecar so the kernel doesn't have to know "
        "about archive shapes or speaker ordering."
    ),
)


def _assert_no_placeholder_digest(recipe: StaticRecipe) -> None:
    """Refuse to register a recipe whose file digest is the
    all-zero placeholder.

    A recipe with ``sha256:0000…0000`` round-trips structurally but
    every download against it fails verification because no real
    file's SHA-256 is 64 zeroes. Catch the regression at import
    time of this module so the broken recipe can never silently
    ship to users.
    """
    placeholder = "sha256:" + "0" * 64
    for f in recipe.files:
        if f.digest == placeholder:
            raise ValueError(
                f"static recipe {recipe.model_id!r} has placeholder "
                f"digest {f.digest!r} for file {f.relative_path!r}; "
                "replace with the real SHA-256 before registering."
            )


_assert_no_placeholder_digest(_KOKORO_82M_RECIPE)


# Legacy v0.19 single-language English bundle. Kept around so callers
# who explicitly pin ``kokoro-en-v0_19`` keep the 11-speaker bundle
# (and its catalog) they were getting before the kokoro-82m cutover
# to v1.0. The recipe carries its OWN digest, layout, and voice
# manifest — no aliasing to ``_KOKORO_82M_RECIPE``, because v0.19
# and v1.0 disagree on every one of those fields.
_KOKORO_EN_V0_19_TARBALL_SHA256 = "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"
_KOKORO_EN_V0_19_TARBALL_SIZE = 319_625_534


_KOKORO_EN_V0_19_RECIPE = StaticRecipe(
    model_id="kokoro-en-v0_19",
    capability="tts",
    engine="sherpa-onnx",
    files=[
        _StaticArtifactFile(
            relative_path="kokoro-en-v0_19.tar.bz2",
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models",
            digest=_KOKORO_EN_V0_19_TARBALL_SHA256,
            size_bytes=_KOKORO_EN_V0_19_TARBALL_SIZE,
        ),
    ],
    materialization=MaterializationPlan(
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
        safety_policy=MaterializationSafetyPolicy(),
        voice_manifest=KOKORO_EN_V0_19_VOICES,
        artifact_version="kokoro-en-v0_19",
    ),
    notes=(
        "Legacy Kokoro v0.19 English-only bundle (11 speakers). "
        "Retained for callers that explicitly pin this id; the "
        "default ``kokoro-82m`` resolves to v1.0 multi-lang."
    ),
)
_assert_no_placeholder_digest(_KOKORO_EN_V0_19_RECIPE)


# PocketTTS — int8-quantized few-shot voice-cloning bundle published
# by sherpa-onnx (2026-01-26 release). NON-COMMERCIAL: the bundle's
# README states the underlying model weights are licensed for
# non-commercial use only. The recipe is registered under
# ``_NON_DEFAULT_RECIPES`` and the planner / app-config layer is
# responsible for eligibility gating — the SDK never registers
# Pocket as a default candidate. Apps that ship Pocket with their
# own commercial license carry their license check in app config.
_POCKET_TTS_INT8_TARBALL_SHA256 = "sha256:2f3b88823cbbb9bf0b2477ec8ae7b3fec417b3a87b6bb5f256dba66f2ad967cb"
_POCKET_TTS_INT8_TARBALL_SIZE = 98_336_520

_POCKET_TTS_INT8_RECIPE = StaticRecipe(
    model_id="pocket-tts-int8",
    capability="tts",
    engine="sherpa-onnx",
    files=[
        _StaticArtifactFile(
            relative_path="sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models",
            digest=_POCKET_TTS_INT8_TARBALL_SHA256,
            size_bytes=_POCKET_TTS_INT8_TARBALL_SIZE,
        ),
    ],
    materialization=MaterializationPlan(
        kind="archive",
        source="sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
        archive_format="tar.bz2",
        # Upstream tar wraps everything in a top-level directory
        # named ``sherpa-onnx-pocket-tts-int8-2026-01-26/``. Strip
        # so the engine reads ``encoder.onnx`` directly under
        # ``artifact_dir/``.
        strip_prefix="sherpa-onnx-pocket-tts-int8-2026-01-26/",
        required_outputs=(
            "text_conditioner.onnx",
            "encoder.onnx",
            "lm_flow.int8.onnx",
            "decoder.int8.onnx",
            "lm_main.int8.onnx",
            "vocab.json",
            "token_scores.json",
        ),
        safety_policy=MaterializationSafetyPolicy(),
        # Pocket has no native voice catalog — its "voices" are
        # reference profiles owned by the planner. Empty manifest
        # signals to the engine "look up speakers from the planner
        # tts_speakers map".
        voice_manifest=(),
        artifact_version="sherpa-onnx-pocket-tts-int8-2026-01-26",
    ),
    notes=(
        "PocketTTS int8 (sherpa-onnx 2026-01-26 release). "
        "Few-shot voice cloning: the engine takes reference audio "
        "+ optional reference text instead of a sid. "
        "NON-COMMERCIAL: the upstream bundle README restricts use "
        "to non-commercial purposes; the SDK does NOT register "
        "this recipe as a default candidate. Apps must opt in "
        "via planner/app config and own their license check."
    ),
)
_assert_no_placeholder_digest(_POCKET_TTS_INT8_RECIPE)


_RECIPES: dict[tuple[str, str], StaticRecipe] = {
    # (model_id, capability) → recipe.
    ("kokoro-82m", "tts"): _KOKORO_82M_RECIPE,
    # Explicit pin to the legacy v0.19 bundle. Carries its own
    # digest + voice catalog so it can't drift with kokoro-82m.
    ("kokoro-en-v0_19", "tts"): _KOKORO_EN_V0_19_RECIPE,
}


# Recipes that are *available* via explicit lookup but NOT advertised
# as a default candidate. The planner / app-config layer must opt in
# explicitly (e.g. ``client.prepare(model='pocket-tts-int8',
# capability='tts')`` for a dev box that wants to test Pocket
# locally; an app's planner-side config to expose Pocket for that
# app). Live in a separate map so a future
# ``list_default_recipes()`` helper can't accidentally include them.
_NON_DEFAULT_RECIPES: dict[tuple[str, str], StaticRecipe] = {
    ("pocket-tts-int8", "tts"): _POCKET_TTS_INT8_RECIPE,
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_static_recipe(model: str, capability: str) -> Optional[StaticRecipe]:
    """Return the recipe for ``(model, capability)``, or ``None``.

    Walks the default recipe table first, then the non-default table
    (Pocket and similar opt-in bundles). Empty result means the SDK
    has no offline knowledge of this model; the caller must go
    through the planner. We do NOT fall through to a generic "guess
    from model_id" path because that would risk shipping public
    bytes for what was meant to be a private artifact.

    The non-default table is consulted last so a same-key default
    recipe always wins; today there's no overlap, but the lookup
    order is explicit so future regressions surface in code review.
    """
    recipe = _RECIPES.get((model, capability))
    if recipe is not None:
        return recipe
    return _NON_DEFAULT_RECIPES.get((model, capability))


def static_recipe_candidate(model: str, capability: str) -> Optional[RuntimeCandidatePlan]:
    """Convenience: recipe → ``RuntimeCandidatePlan`` shape, or None.

    Used by the kernel's ``prepare`` / ``warmup`` fallback path when
    ``_resolve_planner_selection`` returned ``None`` and there's no
    local sdk_runtime candidate to act on.
    """
    recipe = get_static_recipe(model, capability)
    if recipe is None:
        return None
    return recipe.to_runtime_candidate()
