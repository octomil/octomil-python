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


@dataclass(slots=True)
class PrepareOutcome:
    """Result of a successful prepare.

    ``artifact_dir`` is the directory the candidate's files live under.
    ``files`` maps each ``required_files`` entry to its on-disk path. For
    single-file artifacts (empty ``relative_path``), ``files[""]`` is the
    same as ``artifact_dir / "artifact"``. ``cached`` is true when the
    files were already present and verified, so the manager did no I/O.
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

        descriptor = self._build_descriptor(artifact)
        artifact_dir = self.artifact_dir_for(descriptor.artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cached_files = self._already_verified(descriptor, artifact_dir)
        if cached_files is not None:
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

        # The current planner schema gives us a single artifact-level digest
        # and a flat list of required_files. There is no per-file manifest
        # yet, so a multi-file artifact cannot be verified — every file would
        # be checked against the same digest and at least one would fail. A
        # later PR adds manifest_uri parsing; for now, refuse multi-file
        # artifacts loudly rather than silently broadcast one digest across
        # files.
        if len(artifact.required_files) > 1:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Artifact '{artifact.artifact_id or artifact.model_id}' lists "
                    f"{len(artifact.required_files)} required_files but the planner "
                    f"schema in this release only carries a single artifact-level digest. "
                    f"Multi-file artifacts require a per-file manifest_uri (planned in a "
                    f"follow-up PR); refusing to prepare without per-file integrity."
                ),
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
    if len(required_files) > 1:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Artifact '{artifact.artifact_id or artifact.model_id}' lists "
                f"{len(required_files)} required_files but the planner schema only carries a "
                f"single artifact-level digest. Multi-file artifacts require a per-file "
                f"manifest_uri (planned in a follow-up PR); refusing to prepare without "
                f"per-file integrity."
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
