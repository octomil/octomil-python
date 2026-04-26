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
from octomil.runtime.lifecycle.artifact_cache import ArtifactCache
from octomil.runtime.lifecycle.durable_download import (
    ArtifactDescriptor,
    DownloadEndpoint,
    DurableDownloader,
    RequiredFile,
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
        self._reject_non_local(candidate)
        self._check_policy(candidate, mode)

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
        artifact_dir = self._cache_dir / "artifacts" / descriptor.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        all_present = self._already_present(descriptor, artifact_dir)
        if all_present:
            files = {f.relative_path: _resolve_path(artifact_dir, f.relative_path) for f in descriptor.required_files}
            return PrepareOutcome(
                artifact_id=descriptor.artifact_id,
                artifact_dir=artifact_dir,
                files=files,
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

    def _reject_non_local(self, candidate: RuntimeCandidatePlan) -> None:
        if candidate.locality != "local":
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"PrepareManager only handles local candidates; got locality={candidate.locality!r}. "
                    f"Cloud candidates are dispatched through the hosted gateway."
                ),
            )
        delivery_mode = candidate.delivery_mode or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"PrepareManager only handles sdk_runtime delivery; got delivery_mode={delivery_mode!r}."),
            )

    def _check_policy(self, candidate: RuntimeCandidatePlan, mode: PrepareMode) -> None:
        policy = candidate.prepare_policy
        if policy == "disabled":
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    "Candidate's prepare_policy is 'disabled'. The server has marked this artifact "
                    "as ineligible for SDK-side preparation; resolve via a different routing policy."
                ),
            )
        if policy == "explicit_only" and mode is not PrepareMode.EXPLICIT:
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

        if artifact.required_files:
            required = [
                RequiredFile(relative_path=p, digest=artifact.digest, size_bytes=artifact.size_bytes)
                for p in artifact.required_files
            ]
        else:
            required = [RequiredFile(relative_path="", digest=artifact.digest, size_bytes=artifact.size_bytes)]

        return ArtifactDescriptor(
            artifact_id=artifact.artifact_id or artifact.model_id,
            required_files=required,
            endpoints=endpoints,
        )

    def _already_present(self, descriptor: ArtifactDescriptor, artifact_dir: Path) -> bool:
        for required in descriptor.required_files:
            target = _resolve_path(artifact_dir, required.relative_path)
            if not target.exists():
                return False
        return True


def _artifact_id(candidate: RuntimeCandidatePlan) -> str:
    artifact = candidate.artifact
    if artifact is None:
        return f"<no-artifact:{candidate.engine or 'unknown'}>"
    return artifact.artifact_id or artifact.model_id


def _resolve_path(artifact_dir: Path, relative_path: str) -> Path:
    if not relative_path:
        return artifact_dir / "artifact"
    return artifact_dir / relative_path
