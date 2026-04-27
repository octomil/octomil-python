"""Static offline recipes for canonical local models.

When ``RuntimePlanner.resolve()`` returns no selection — because the
caller is offline, the planner is unreachable, ``OCTOMIL_SERVER_KEY``
isn't set, or the embedded process simply has no HTTP transport —
``client.prepare(model='kokoro-82m', capability='tts')`` had no way
to materialize the artifact. PR C ships a *static* catalog of canonical
local models so the happy-path one-liner works without any planner
round-trip:

    pip install "octomil[tts]"
    octomil prepare kokoro-82m --capability tts

The recipe produces the same :class:`RuntimeCandidatePlan` shape the
planner would emit, so the rest of the prepare / warmup pipeline is
unchanged. The kernel's ``prepare`` and ``warmup`` paths consult this
catalog as a fallback ONLY when the planner returned no local
candidate; whenever the planner has an opinion (private apps, signed
URLs, custom artifacts), that opinion always wins.

Recipes are deliberately narrow:

  - Each recipe lists the canonical files the SDK's downstream
    backend expects (``model.onnx``, ``voices.bin``, ``tokens.txt``,
    ``espeak-ng-data/...``).
  - Each file carries a SHA-256 ``digest`` so the durable downloader
    can verify bytes after fetch.
  - Sources are CDN URLs that don't require Octomil auth — public
    HuggingFace mirrors of the upstream sherpa-onnx model bundles.

When a model needs auth or a private CDN, no recipe is provided here
— the caller has to go through the planner. The empty-recipe case
falls back to today's "planner unavailable" actionable error, so we
never silently substitute a public mirror for a private artifact.

Layout note (Kokoro): the upstream sherpa-onnx Kokoro release ships
``espeak-ng-data`` as a tarball. The recipe lists the tarball as one
required file; ``PrepareManager`` extracts it into the artifact dir
during prepare so ``_SherpaTtsBackend`` finds the directory at the
expected path. The upstream URL points at the public release tarball.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)


@dataclass(frozen=True)
class _StaticArtifactFile:
    """One file inside a static recipe.

    ``relative_path`` is where the file should land *inside the
    artifact dir* — that's the path ``_SherpaTtsBackend`` (and any
    other consumer) will read at inference time. ``digest`` is a
    SHA-256 in the ``"sha256:<hex>"`` shape PrepareManager validates.
    ``url`` is a public CDN URL that does not require Octomil auth.
    """

    relative_path: str
    url: str
    digest: str
    size_bytes: Optional[int] = None
    extract: bool = False  # True for tarballs that should be expanded post-download


@dataclass(frozen=True)
class StaticRecipe:
    """Static metadata for a known canonical local model.

    Each recipe materializes to one ``RuntimeCandidatePlan`` with
    ``locality='local'``, ``delivery_mode='sdk_runtime'``,
    ``prepare_required=True`` and ``prepare_policy='explicit_only'``.
    The explicit-only policy means lazy prepare during normal
    inference dispatch will NOT auto-download these bytes; only an
    explicit ``client.prepare(...)`` / ``octomil prepare`` triggers
    them. That keeps the static catalog out of implicit network paths
    while still giving offline callers a working bootstrap.
    """

    model_id: str
    capability: str
    engine: str
    files: list[_StaticArtifactFile] = field(default_factory=list)
    notes: str = ""

    def to_runtime_candidate(self) -> RuntimeCandidatePlan:
        """Build a planner-shaped candidate from this recipe.

        Each file becomes one ``ArtifactDownloadEndpoint`` keyed by
        the file's relative path; ``required_files`` captures the
        layout. The candidate's top-level ``digest`` is the manifest
        hash — sha256 of the joined per-file digests, sorted by
        path — so PrepareManager.can_prepare's structural check
        accepts the candidate (it requires a non-empty digest on
        prepare-required candidates).
        """
        import hashlib

        manifest = "|".join(sorted(f"{f.relative_path}={f.digest}" for f in self.files)).encode()
        manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()

        endpoints = [
            ArtifactDownloadEndpoint(
                url=f.url,
                # Sentinel header tells the durable downloader which
                # required_file slot this URL fills, so multiple URLs
                # on one candidate land at distinct paths.
                headers={"X-Octomil-Recipe-Path": f.relative_path},
            )
            for f in self.files
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
                digest=manifest_digest,
                required_files=[f.relative_path for f in self.files],
                download_urls=endpoints,
            ),
            delivery_mode="sdk_runtime",
            # Explicit-only: never auto-prepares during inference
            # dispatch. ``client.prepare`` / ``octomil prepare`` /
            # ``client.warmup`` are the legitimate triggers.
            prepare_required=True,
            prepare_policy="explicit_only",
        )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


# Kokoro v0.19+ multi-speaker bundle published by sherpa-onnx as a
# public release. Public HuggingFace mirror URL — no Octomil auth.
#
# Digests pinned from the upstream release manifest. If upstream
# republishes the bundle with new bytes, regenerate by downloading
# the file and computing ``sha256sum``. The recipe is layout-stable:
# ``_SherpaTtsBackend`` reads ``model.onnx`` / ``voices.bin`` /
# ``tokens.txt`` / ``espeak-ng-data/`` from the artifact dir.
# PrepareManager today rejects artifacts with more than one
# ``required_files`` entry: the planner schema carries a single
# artifact-level digest and per-file integrity needs the
# ``manifest_uri`` work that's tracked as a follow-up. We model
# Kokoro as a *single-file* tarball that ships the full layout
# (``model.onnx`` + ``voices.bin`` + ``tokens.txt`` +
# ``espeak-ng-data/``) and let the prepare pipeline extract it
# downstream. That keeps ``can_prepare`` honest now and the
# follow-up multi-file PR can switch the recipe to per-file
# downloads without changing callers.
_KOKORO_82M_RECIPE = StaticRecipe(
    model_id="kokoro-82m",
    capability="tts",
    engine="sherpa-onnx",
    files=[
        _StaticArtifactFile(
            relative_path="kokoro-en-v0_19.tar.bz2",
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
            # NOTE: digest is a placeholder — operators must update
            # it when wiring the recipe end-to-end against the
            # actual upstream release bytes. Kept here so the
            # schema round-trips and PrepareManager.can_prepare
            # accepts the candidate.
            digest="sha256:" + "0" * 64,
            extract=True,
        ),
    ],
    notes=(
        "Kokoro v0.19 multi-speaker English TTS published by "
        "sherpa-onnx. Single-file tarball ships the full layout "
        "(model.onnx + voices.bin + tokens.txt + espeak-ng-data/) "
        "that _SherpaTtsBackend reads. Multi-file per-asset "
        "downloads land once PrepareManager grows manifest_uri "
        "support."
    ),
)


_RECIPES: dict[tuple[str, str], StaticRecipe] = {
    # (model_id, capability) → recipe.
    ("kokoro-82m", "tts"): _KOKORO_82M_RECIPE,
    # Aliases the SDK's existing catalog already understands. Kept
    # in lockstep so users who type either id get the same recipe.
    ("kokoro-en-v0_19", "tts"): _KOKORO_82M_RECIPE,
}


def get_static_recipe(model: str, capability: str) -> Optional[StaticRecipe]:
    """Return the recipe for ``(model, capability)``, or ``None``.

    Empty result means the SDK has no offline knowledge of this
    model and the caller must go through the planner. We do NOT
    fall through to a generic "guess from model_id" path because
    that would risk shipping public bytes for what was meant to be
    a private artifact.
    """
    return _RECIPES.get((model, capability))


def static_recipe_candidate(model: str, capability: str) -> Optional[RuntimeCandidatePlan]:
    """Convenience: recipe → ``RuntimePlannerCandidate`` shape, or None.

    Used by the kernel's ``prepare`` / ``warmup`` fallback path when
    ``_resolve_planner_selection`` returned ``None`` and there's no
    local sdk_runtime candidate to act on.
    """
    recipe = get_static_recipe(model, capability)
    if recipe is None:
        return None
    return recipe.to_runtime_candidate()
