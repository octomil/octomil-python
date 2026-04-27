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


# Kokoro v0.19 single-file tarball.
#
# URL pinned to the upstream sherpa-onnx tts-models GitHub release
# (public, no Octomil auth). Digest is the SHA-256 of the released
# ``kokoro-en-v0_19.tar.bz2`` asset reported by the GitHub release
# API for the ``tts-models`` release tag at PR #455 review time.
# When upstream re-cuts the bundle (rare; the v0_19 tag is frozen),
# regenerate by downloading the file and running ``shasum -a 256``
# AND cross-check against
# ``GET /repos/k2-fsa/sherpa-onnx/releases/tags/tts-models``.
_KOKORO_82M_TARBALL_SHA256 = "sha256:912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"


_KOKORO_82M_RECIPE = StaticRecipe(
    model_id="kokoro-82m",
    capability="tts",
    engine="sherpa-onnx",
    files=[
        _StaticArtifactFile(
            relative_path="kokoro-en-v0_19.tar.bz2",
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
            digest=_KOKORO_82M_TARBALL_SHA256,
        ),
    ],
    materialization=MaterializationPlan(
        kind="archive",
        source="kokoro-en-v0_19.tar.bz2",
        archive_format="tar.bz2",
        # Upstream tar wraps everything in a top-level directory
        # named ``kokoro-en-v0_19/``. Strip it so the backend reads
        # ``model.onnx`` directly under ``artifact_dir`` rather
        # than ``artifact_dir/kokoro-en-v0_19/model.onnx``.
        strip_prefix="kokoro-en-v0_19/",
        required_outputs=(
            "model.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/phontab",
        ),
        # Kokoro's tarball contains no symlinks/hardlinks; default
        # safety policy refuses both, which is what we want.
        safety_policy=MaterializationSafetyPolicy(),
    ),
    notes=(
        "Kokoro v0.19 multi-speaker English TTS published by "
        "sherpa-onnx. The MaterializationPlan covers tarball "
        "extraction so the kernel doesn't have to know about "
        "archive shapes."
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


_RECIPES: dict[tuple[str, str], StaticRecipe] = {
    # (model_id, capability) → recipe.
    ("kokoro-82m", "tts"): _KOKORO_82M_RECIPE,
    # Aliases the SDK's existing catalog already understands. Kept
    # in lockstep so users who type either id get the same recipe.
    ("kokoro-en-v0_19", "tts"): _KOKORO_82M_RECIPE,
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_static_recipe(model: str, capability: str) -> Optional[StaticRecipe]:
    """Return the recipe for ``(model, capability)``, or ``None``.

    Empty result means the SDK has no offline knowledge of this
    model; the caller must go through the planner. We do NOT fall
    through to a generic "guess from model_id" path because that
    would risk shipping public bytes for what was meant to be a
    private artifact.
    """
    return _RECIPES.get((model, capability))


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
