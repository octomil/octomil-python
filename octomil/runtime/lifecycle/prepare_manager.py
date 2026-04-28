"""PrepareManager — bridge planner output to on-disk artifact readiness.

Given a planner ``RuntimeCandidatePlan`` whose ``delivery_mode`` is
``"sdk_runtime"`` and whose ``prepare_required`` is true, the manager
materializes the artifact's files into a local directory, honoring the
candidate's ``prepare_policy``:

- ``lazy``           — prepare on demand (e.g. from the runtime path).
- ``explicit_only``  — prepare only when the caller explicitly opted in.
- ``disabled``       — never prepare; raise.

The manager is the sole consumer of :class:`DurableDownloader` and the
sole producer of "this candidate is ready" outcomes. PR 4+ wires it into
runtime adapters; this PR only adds the manager and its tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.runtime.lifecycle._fs_key import safe_filesystem_key
from octomil.runtime.lifecycle.artifact_cache import ArtifactCache
from octomil.runtime.lifecycle.durable_download import (
    ArtifactDescriptor,
    DownloadEndpoint,
    DurableDownloader,
    RequiredFile,
    _digest_matches,
    _safe_join,
    _validate_relative_path,
)
from octomil.runtime.planner.schemas import RuntimeCandidatePlan

logger = logging.getLogger(__name__)


class PrepareMode(str, Enum):
    """Why ``prepare`` was called.

    ``LAZY`` is the default and represents a runtime-driven prepare (e.g.
    just-in-time during inference dispatch). ``EXPLICIT`` represents a
    caller-driven prepare (e.g. a pre-warm CLI command); explicit calls are
    permitted even when the candidate's ``prepare_policy`` is
    ``explicit_only``.
    """

    LAZY = "lazy"
    EXPLICIT = "explicit"


@dataclass
class PrepareOutcome:
    """Result of a successful prepare.

    ``artifact_dir`` is the directory the candidate's files live under.
    ``files`` maps each ``required_files`` entry to its on-disk path. For
    single-file artifacts (empty ``relative_path``), ``files[""]`` is the
    same as ``artifact_dir / "artifact"``. ``cached`` is true when the
    files were already present and verified, so the manager did no I/O.

    Plain ``@dataclass`` (no ``slots=True``). The slots= kwarg was added
    in Python 3.10, but ``octomil`` still declares
    ``requires-python=">=3.9"`` — using it would silently break 3.9
    users on a minor release.
    """

    artifact_id: str
    artifact_dir: Path
    files: dict[str, Path]
    engine: str | None
    delivery_mode: str
    prepare_policy: str
    cached: bool


class PrepareManager:
    """Single owner of artifact readiness for ``sdk_runtime`` candidates."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        downloader: DurableDownloader | None = None,
        artifact_cache: ArtifactCache | None = None,
    ) -> None:
        self._cache = artifact_cache or ArtifactCache()
        self._cache_dir = cache_dir or self._cache.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._downloader = downloader or DurableDownloader(cache_dir=self._cache_dir)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def can_prepare(self, candidate: RuntimeCandidatePlan) -> bool:
        """Dry-run preparability check.

        Returns ``True`` only when :meth:`prepare` is structurally
        guaranteed to succeed on ``candidate``'s metadata. The check
        mirrors *every* validation ``prepare()`` performs before any
        disk/network work, including the path-safety checks for
        ``required_files[0]`` and ``artifact_id``. Synthetic or malformed
        planner metadata (no digest, no urls, traversal in
        ``required_files``, NUL-byte ``artifact_id``, etc.) returns
        ``False`` so the routing layer can treat the candidate as
        unavailable instead of committing to local and failing in
        ``prepare()``.

        Pure inspection — never touches disk or network.
        """
        try:
            _validate_for_prepare(candidate)
            return True
        except OctomilError:
            return False

    def prepare(
        self,
        candidate: RuntimeCandidatePlan,
        *,
        mode: PrepareMode = PrepareMode.LAZY,
    ) -> PrepareOutcome:
        """Bring ``candidate``'s artifact to a ready local state.

        Raises :class:`OctomilError` (with an actionable message) if the
        candidate is not preparable, the policy forbids the call site, or
        the underlying download fails after exhausting all endpoints.
        """
        # Single source of truth for structural validation. Anything that
        # raises here would also make ``can_prepare`` return False.
        _validate_for_prepare(candidate)
        # explicit_only-vs-mode is the one check ``can_prepare`` cannot
        # perform (it has no mode). Run it here for the canonical message.
        self._check_explicit_only_vs_mode(candidate, mode)

        if not candidate.prepare_required:
            # Server says no preparation is needed — usually because the
            # engine manages its own artifacts (e.g. ollama). Surface a
            # cached outcome with no files so callers have a uniform shape.
            return PrepareOutcome(
                artifact_id=_artifact_id(candidate),
                artifact_dir=self._cache_dir,
                files={},
                engine=candidate.engine,
                delivery_mode=candidate.delivery_mode or "sdk_runtime",
                prepare_policy=candidate.prepare_policy,
                cached=True,
            )

        artifact = candidate.artifact
        if artifact is None:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    "Candidate marks prepare_required=True but carries no artifact plan. "
                    "This is a server contract violation; refusing to prepare."
                ),
            )

        # PR C-followup option 2: when the planner emits
        # ``source="static_recipe"`` + ``recipe_id``, expand the
        # candidate from the SDK's built-in recipe table. The
        # registered ``StaticRecipe`` carries both download metadata
        # AND a ``MaterializationPlan`` (e.g. Kokoro's tarball
        # extraction). Capture the recipe so we can run the plan
        # after the download lands — without this, a server-emitted
        # ``source='static_recipe'`` candidate would leave Kokoro
        # as ``kokoro-en-v0_19.tar.bz2`` on disk instead of the
        # extracted ``model.onnx`` / ``voices.bin`` / ``tokens.txt`` /
        # ``espeak-ng-data/``.
        artifact, used_static_recipe = self._expand_static_recipe_source(artifact)

        descriptor = self._build_descriptor(artifact)
        artifact_dir = self.artifact_dir_for(descriptor.artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cached_files = self._already_verified(descriptor, artifact_dir)
        if cached_files is not None:
            # Even on a cache hit, run materialization idempotently
            # — the marker check inside Materializer makes a
            # complete layout a no-op, but a partial extraction
            # (interrupted before ``required_outputs`` all landed)
            # gets re-extracted from the on-disk archive. Without
            # this, the cache hit could return ``files`` pointing
            # only at the tarball.
            if used_static_recipe is not None:
                from octomil.runtime.lifecycle.materialization import Materializer

                Materializer().materialize(artifact_dir, used_static_recipe.materialization)
            return PrepareOutcome(
                artifact_id=descriptor.artifact_id,
                artifact_dir=artifact_dir,
                files=cached_files,
                engine=candidate.engine,
                delivery_mode=candidate.delivery_mode or "sdk_runtime",
                prepare_policy=candidate.prepare_policy,
                cached=True,
            )

        result = self._downloader.download(descriptor, artifact_dir)
        # Post-download materialization. The kernel does NOT know
        # archive shapes / extension matchers / Kokoro layouts — it
        # just hands the recipe's ``MaterializationPlan`` to the
        # generic Materializer (archive extraction, safety filtering,
        # idempotency, required-output verification). ``kind='none'``
        # plans (single-file backends like whisper.cpp) are a no-op
        # aside from layout validation.
        if used_static_recipe is not None:
            from octomil.runtime.lifecycle.materialization import Materializer

            Materializer().materialize(artifact_dir, used_static_recipe.materialization)
        return PrepareOutcome(
            artifact_id=descriptor.artifact_id,
            artifact_dir=artifact_dir,
            files=result.files,
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )

    def _check_explicit_only_vs_mode(self, candidate: RuntimeCandidatePlan, mode: PrepareMode) -> None:
        """Explicit-only candidates may only be prepared via mode=EXPLICIT.

        This is the one check that depends on the *call site* (mode) rather
        than the candidate alone, so it lives outside ``_validate_for_prepare``
        and is not consulted by ``can_prepare``.
        """
        if candidate.prepare_policy == "explicit_only" and mode is not PrepareMode.EXPLICIT:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    "Candidate's prepare_policy is 'explicit_only'. Call client.prepare(...) before "
                    "invoking inference, or rerun with mode=PrepareMode.EXPLICIT."
                ),
            )

    def _expand_static_recipe_source(self, artifact):
        """Resolve ``source="static_recipe"`` artifacts via the recipe
        table.

        When the planner picks a canonical built-in artifact and
        emits ``source="static_recipe"`` + ``recipe_id``, the SDK
        treats the recipe as the source of truth for URL / digest /
        required_files / **MaterializationPlan**. Any of these
        fields the planner *also* emitted is treated as a hint and
        cross-checked: a mismatch between the planner's declared
        digest and the recipe's digest is rejected loudly (would
        otherwise let a compromised planner direct users to forged
        bytes under a known recipe id).

        Returns ``(artifact, recipe_or_None)``. ``recipe`` is the
        registered :class:`StaticRecipe` (or ``None`` when the
        planner didn't request expansion); the caller threads it
        through to :class:`Materializer` so the post-download
        archive layout (Kokoro tarball, etc.) actually gets
        unpacked. Without this round-trip a planner-emitted
        ``source='static_recipe'`` would download
        ``kokoro-en-v0_19.tar.bz2`` and return the tarball path —
        not ``model.onnx`` / ``voices.bin`` / etc.
        """
        source = getattr(artifact, "source", None)
        if source is None:
            return artifact, None
        if source != "static_recipe":
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact source {source!r} is not recognized by this SDK release. "
                    f"Known sources: 'static_recipe'. Upgrade the SDK or have the "
                    f"planner omit ``source`` to use planner-supplied metadata directly."
                ),
            )
        recipe_id = getattr(artifact, "recipe_id", None)
        if not recipe_id:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    "Artifact has source='static_recipe' but no recipe_id. " "The planner must name a specific recipe."
                ),
            )

        from octomil.runtime.lifecycle.static_recipes import _RECIPES

        recipe = next(
            (r for (mid, _cap), r in _RECIPES.items() if r.model_id == recipe_id or mid == recipe_id),
            None,
        )
        if recipe is None:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact source='static_recipe' but recipe_id {recipe_id!r} is "
                    f"not in this SDK's built-in recipe table. Either upgrade the SDK "
                    f"or have the planner switch to ``source=None`` and emit the "
                    f"artifact metadata directly."
                ),
            )
        if len(recipe.files) != 1:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Static recipe {recipe_id!r} has {len(recipe.files)} files; "
                    f"multi-file recipes need manifest_uri support to expand under "
                    f"source='static_recipe' (planned)."
                ),
            )
        only = recipe.files[0]

        # Cross-check planner-supplied metadata against the recipe.
        # Mismatches mean the server is asking us to use a different
        # artifact under a known recipe id — refuse rather than
        # silently substitute.
        if getattr(artifact, "digest", None) and artifact.digest != only.digest:
            raise OctomilError(
                code=ErrorCode.CHECKSUM_MISMATCH,
                message=(
                    f"Static recipe {recipe_id!r} digest {only.digest!r} does not match "
                    f"planner-declared digest {artifact.digest!r}. Refusing to substitute "
                    f"a different artifact under a known recipe id."
                ),
            )
        if artifact.required_files and artifact.required_files != [only.relative_path]:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Static recipe {recipe_id!r} ships file {only.relative_path!r}; "
                    f"planner-declared required_files {artifact.required_files!r} "
                    f"does not match. Refusing to substitute."
                ),
            )

        # Recipe wins. Build a fresh artifact with canonical metadata,
        # preserving any planner-supplied overrides we don't override
        # (engine, format, model_id).
        from dataclasses import replace

        from octomil.runtime.planner.schemas import ArtifactDownloadEndpoint

        expanded = replace(
            artifact,
            artifact_id=artifact.artifact_id or recipe.model_id,
            digest=only.digest,
            required_files=[only.relative_path],
            download_urls=[
                ArtifactDownloadEndpoint(
                    url=only.url,
                    headers={"X-Octomil-Recipe-Path": only.relative_path},
                )
            ],
            size_bytes=artifact.size_bytes or only.size_bytes,
            source=None,
            recipe_id=None,
        )
        return expanded, recipe

    def _build_descriptor(self, artifact) -> ArtifactDescriptor:
        if not artifact.download_urls:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact '{artifact.artifact_id or artifact.model_id}' has no download_urls. "
                    f"Cannot prepare; the planner must emit at least one endpoint."
                ),
            )
        if not artifact.digest:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact '{artifact.artifact_id or artifact.model_id}' has no digest. "
                    f"Refusing to prepare without integrity verification."
                ),
            )

        endpoints = [
            DownloadEndpoint(url=ep.url, expires_at=ep.expires_at, headers=dict(ep.headers or {}))
            for ep in artifact.download_urls
        ]

        # PR C-followup: multi-file artifacts go through ``manifest_uri``.
        # The single artifact-level digest cannot verify multiple files
        # (every file would be checked against the same hash and at least
        # one would fail). The manifest body itself is digest-pinned by
        # ``RuntimeArtifactPlan.digest``, so a tampered manifest is caught
        # before any per-file URL is fetched. See
        # ``octomil.runtime.lifecycle.manifest`` for the schema and
        # validation rules.
        if len(artifact.required_files) > 1:
            if not artifact.manifest_uri:
                raise OctomilError(
                    code=ErrorCode.INVALID_INPUT,
                    message=(
                        f"Artifact '{artifact.artifact_id or artifact.model_id}' lists "
                        f"{len(artifact.required_files)} required_files but the planner "
                        f"emitted no manifest_uri. The single artifact-level digest cannot "
                        f"verify multiple files; refusing to prepare without per-file "
                        f"integrity. Have the planner emit ``manifest_uri`` (manifest.v1)."
                    ),
                )
            from octomil.runtime.lifecycle.manifest import fetch_and_parse_manifest

            manifest = fetch_and_parse_manifest(artifact.manifest_uri, artifact_digest=artifact.digest)
            # Cross-check: every required_files entry the planner advertises
            # MUST appear in the manifest. Manifest may be a strict superset
            # (unused files), but the planner's claimed file set must be
            # fully covered by per-file digests.
            manifest_paths = {f.relative_path: f for f in manifest.files}
            for advertised in artifact.required_files:
                safe_rel = _validate_relative_path(advertised)
                if safe_rel not in manifest_paths:
                    raise OctomilError(
                        code=ErrorCode.INVALID_INPUT,
                        message=(
                            f"Artifact '{artifact.artifact_id or artifact.model_id}' "
                            f"lists required_file {advertised!r} but the manifest at "
                            f"{artifact.manifest_uri!r} does not include it. The planner "
                            f"and the manifest must agree on the file set."
                        ),
                    )
            # Order-preserving: the descriptor walks files in the planner's
            # ``required_files`` order so dispatch / progress is stable
            # regardless of manifest ordering.
            required = [manifest_paths[_validate_relative_path(p)] for p in artifact.required_files]
            return ArtifactDescriptor(
                artifact_id=artifact.artifact_id or artifact.model_id,
                required_files=required,
                endpoints=endpoints,
            )

        if artifact.required_files:
            # Validate the planner-supplied path before it is ever used as a
            # filesystem name or URL component. _validate_relative_path is the
            # same helper DurableDownloader uses, so PrepareManager and the
            # downloader share one trust boundary.
            safe_path = _validate_relative_path(artifact.required_files[0])
            required = [RequiredFile(relative_path=safe_path, digest=artifact.digest, size_bytes=artifact.size_bytes)]
        else:
            required = [RequiredFile(relative_path="", digest=artifact.digest, size_bytes=artifact.size_bytes)]

        return ArtifactDescriptor(
            artifact_id=artifact.artifact_id or artifact.model_id,
            required_files=required,
            endpoints=endpoints,
        )

    def artifact_dir_for(self, artifact_id: str) -> Path:
        """Return the on-disk directory for ``artifact_id``, guaranteed to
        live under ``<cache_dir>/artifacts``.

        Delegates the key shape to :func:`safe_filesystem_key`, which is
        also used by :class:`FileLock`. Re-checks containment as defense
        in depth.
        """
        if not artifact_id:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message="Refusing to prepare artifact with empty artifact_id.",
            )
        try:
            key = safe_filesystem_key(artifact_id)
        except ValueError as exc:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=f"artifact_id is not a valid filesystem key: {exc}",
            ) from exc

        artifacts_root = (self._cache_dir / "artifacts").resolve()
        artifacts_root.mkdir(parents=True, exist_ok=True)
        candidate = (artifacts_root / key).resolve()
        try:
            candidate.relative_to(artifacts_root)
        except ValueError as exc:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"artifact_id resolves outside the cache root: {artifact_id!r} -> {candidate}"),
            ) from exc
        return candidate

    def _already_verified(self, descriptor: ArtifactDescriptor, artifact_dir: Path) -> dict[str, Path] | None:
        """Return the file map iff every required file exists, sits under
        ``artifact_dir`` after symlink/path resolution, and matches its digest.

        Returns ``None`` (cache miss) on any mismatch — the caller then runs
        the downloader, which performs the same containment + digest checks
        as the authoritative path.
        """
        verified: dict[str, Path] = {}
        for required in descriptor.required_files:
            try:
                if required.relative_path:
                    target = _safe_join(artifact_dir, required.relative_path)
                else:
                    target = artifact_dir.resolve() / "artifact"
            except OctomilError:
                # Path validation already failed at descriptor build, so this
                # is a defense-in-depth path. Treat as cache miss; the
                # downloader call that follows will surface the real error.
                return None
            if not target.is_file():
                return None
            if not _digest_matches(target, required.digest):
                logger.info(
                    "Artifact %s file %r exists but digest mismatch; will re-download",
                    descriptor.artifact_id,
                    required.relative_path,
                )
                return None
            verified[required.relative_path] = target
        return verified


def _validate_for_prepare(candidate: RuntimeCandidatePlan) -> None:
    """Apply every structural check :meth:`PrepareManager.prepare` performs
    before any disk/network work.

    Raises :class:`OctomilError(INVALID_INPUT)` with an actionable message
    on the first failure. ``prepare()`` calls this for the canonical error
    text; ``can_prepare()`` calls this and treats any exception as
    "not preparable". Putting the rules in one place is what keeps
    ``can_prepare`` honest: anything ``prepare`` would reject is also
    rejected by the dry-run.
    """
    # Locality + delivery_mode: PrepareManager only handles local sdk_runtime.
    if getattr(candidate, "locality", None) != "local":
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"PrepareManager only handles local candidates; got locality={getattr(candidate, 'locality', None)!r}."
            ),
        )
    delivery_mode = getattr(candidate, "delivery_mode", None) or "sdk_runtime"
    if delivery_mode != "sdk_runtime":
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(f"PrepareManager only handles sdk_runtime delivery; got delivery_mode={delivery_mode!r}."),
        )

    # Policy: 'disabled' is a hard stop. 'explicit_only' is policy-vs-mode
    # and can only be checked at prepare() call time, so can_prepare lets
    # it pass — the kernel's lazy path will surface the canonical message.
    policy = getattr(candidate, "prepare_policy", "lazy")
    if policy == "disabled":
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                "Candidate's prepare_policy is 'disabled'. The server has marked this artifact "
                "as ineligible for SDK-side preparation; resolve via a different routing policy."
            ),
        )

    # prepare_required=False short-circuits to a no-files cached outcome
    # — every shape past the locality/policy gates is fine.
    if not getattr(candidate, "prepare_required", False):
        return

    artifact = getattr(candidate, "artifact", None)
    if artifact is None:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                "Candidate marks prepare_required=True but carries no artifact plan. "
                "This is a server contract violation; refusing to prepare."
            ),
        )
    # PR C-followup option 2: ``source="static_recipe"`` lets the
    # planner select a canonical built-in artifact by ``recipe_id``
    # alone — the SDK fills download_urls / digest / required_files
    # from its registry. Validate by checking the recipe is known
    # rather than insisting on planner-supplied url/digest. Unknown
    # ``source`` values are rejected so we don't silently pass
    # through future shapes the SDK doesn't understand.
    source = getattr(artifact, "source", None)
    if source is not None:
        if source != "static_recipe":
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact source {source!r} is not recognized by this SDK release. "
                    f"Known sources: 'static_recipe'."
                ),
            )
        recipe_id = getattr(artifact, "recipe_id", None)
        if not recipe_id:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    "Artifact has source='static_recipe' but no recipe_id. " "The planner must name a specific recipe."
                ),
            )
        from octomil.runtime.lifecycle.static_recipes import _RECIPES

        if not any(r.model_id == recipe_id or mid == recipe_id for (mid, _cap), r in _RECIPES.items()):
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact source='static_recipe' but recipe_id {recipe_id!r} is "
                    f"not in this SDK's built-in recipe table."
                ),
            )
        # Static-recipe expansion happens at prepare time (after
        # this validator runs); the rest of the structural checks
        # apply to the expanded artifact.
        return

    if not getattr(artifact, "download_urls", None):
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Artifact '{artifact.artifact_id or artifact.model_id}' has no download_urls. "
                f"Cannot prepare; the planner must emit at least one endpoint."
            ),
        )
    if not getattr(artifact, "digest", None):
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Artifact '{artifact.artifact_id or artifact.model_id}' has no digest. "
                f"Refusing to prepare without integrity verification."
            ),
        )

    required_files = getattr(artifact, "required_files", None) or []
    # Multi-file artifacts are now allowed when the planner pairs
    # them with a ``manifest_uri`` (PR C-followup). The descriptor
    # builder fetches + verifies the manifest under the artifact-
    # level digest pin and uses per-file digests from there.
    if len(required_files) > 1 and not getattr(artifact, "manifest_uri", None):
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Artifact '{artifact.artifact_id or artifact.model_id}' lists "
                f"{len(required_files)} required_files but the planner emitted no "
                f"manifest_uri. The single artifact-level digest cannot verify multiple "
                f"files; refusing to prepare without per-file integrity. Have the planner "
                f"emit ``manifest_uri`` (manifest.v1)."
            ),
        )
    if required_files:
        # Untrusted planner path. _validate_relative_path raises
        # OctomilError(INVALID_INPUT) on traversal, dot segments,
        # backslashes, NUL bytes, absolute paths, etc.
        _validate_relative_path(required_files[0])

    # artifact_id is used as a filesystem key. Match the checks
    # ``artifact_dir_for`` performs (empty/NUL) so can_prepare cannot lie.
    artifact_id = artifact.artifact_id or artifact.model_id
    if not artifact_id:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message="Refusing to prepare artifact with empty artifact_id.",
        )
    if "\x00" in artifact_id:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=f"artifact_id contains a NUL byte: {artifact_id!r}",
        )


def _artifact_id(candidate: RuntimeCandidatePlan) -> str:
    artifact = candidate.artifact
    if artifact is None:
        return f"<no-artifact:{candidate.engine or 'unknown'}>"
    return artifact.artifact_id or artifact.model_id
