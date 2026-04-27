"""Generic materialization layer for prepared artifacts.

Once :class:`PrepareManager` has verified bytes on disk, callers
often need a *backend-ready layout* — not just the raw downloaded
files. The Sherpa TTS backend wants ``model.onnx`` + ``voices.bin``
+ ``tokens.txt`` + ``espeak-ng-data/`` under one directory; a Whisper
recipe might want ``ggml-tiny.bin``; a future MLX recipe might want a
HuggingFace snapshot tree. Every one of those shapes is "downloaded
file → arrange-on-disk", and they all share the same safety / idempotency
concerns.

This module ships a single generic abstraction:

  - :class:`MaterializationPlan` — a *data* description of what the
    backend needs after prepare:

      * ``kind``: ``"none"`` (downloaded files are already in the
        layout the backend reads) or ``"archive"`` (one of the
        downloaded files is an archive that must be unpacked).
      * ``source``: relative path of the downloaded file the plan
        operates on (only used by ``kind != "none"``).
      * ``archive_format``: ``"tar.bz2"`` / ``"tar.gz"`` / ``"tar"``
        / ``"zip"``; defaults to inference from ``source``.
      * ``strip_prefix``: optional path prefix removed from each
        archive member's destination (matches ``tar
        --strip-components``).
      * ``required_outputs``: paths the backend will read at
        inference time; the materializer's idempotency check and
        completeness assertion both key off this list.
      * ``safety_policy``: refusal rules — absolute paths, traversal,
        symlinks, hardlinks are blocked by default. ``allow_symlinks``
        / ``allow_hardlinks`` can opt into them when a downstream
        backend genuinely needs them.

  - :class:`Materializer` — a single class that consumes a
    :class:`MaterializationPlan` against an ``artifact_dir`` and
    leaves the directory in a backend-ready state. Idempotent and
    safe: a partial extraction (interrupted before the marker is
    written) is detected on the next run and re-extracted, never
    silently treated as complete.

The kernel does NOT know Kokoro / tarballs / sherpa-onnx. It does:

    outcome = PrepareManager.prepare(candidate)
    Materializer().materialize(outcome.artifact_dir, recipe.materialization)

A "recipe" in this codebase is therefore the pair of
``(RuntimeCandidatePlan-compatible candidate, MaterializationPlan)``
— the candidate covers download + verification, the plan covers
post-download arrangement. Adding a new model is a data row, not a
new code path.
"""

from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from octomil.errors import OctomilError, OctomilErrorCode

logger = logging.getLogger(__name__)


# Sentinel marker file written into the artifact dir after a
# successful materialization. Its presence + every required output
# being on disk together prove the directory is backend-ready.
EXTRACTION_MARKER = ".octomil-materialized"


@dataclass(frozen=True)
class MaterializationSafetyPolicy:
    """Per-plan safety rules applied to archive members.

    Defaults are conservative: refuse traversal, absolute paths,
    symlinks, and hardlinks. Backends that genuinely need symlinks
    (e.g. an HF snapshot pointer to a blob) can opt in via
    ``allow_symlinks=True``; the rule still applies destination-
    path validation.
    """

    allow_symlinks: bool = False
    allow_hardlinks: bool = False
    # Maximum total uncompressed size we'll extract. Defends against
    # zip/tarbomb attacks. Default is a generous 8 GiB — plenty for
    # the largest TTS bundle, far below disk-fill territory.
    max_total_uncompressed_bytes: int = 8 * 1024**3


@dataclass(frozen=True)
class MaterializationPlan:
    """How to turn ``artifact_dir`` into a backend-ready layout.

    See module docstring for field semantics. The plan is *data*:
    it carries no behaviour. ``Materializer`` consumes it.
    """

    kind: Literal["none", "archive"] = "none"
    source: Optional[str] = None
    archive_format: Optional[Literal["tar", "tar.gz", "tar.bz2", "tar.xz", "zip"]] = None
    strip_prefix: Optional[str] = None
    required_outputs: tuple[str, ...] = ()
    safety_policy: MaterializationSafetyPolicy = field(default_factory=MaterializationSafetyPolicy)

    def __post_init__(self) -> None:
        if self.kind == "archive" and not self.source:
            raise ValueError("MaterializationPlan(kind='archive') requires source=")
        if self.kind not in ("none", "archive"):
            raise ValueError(f"MaterializationPlan: unknown kind={self.kind!r}")

    def infer_archive_format(self) -> str:
        """Resolve ``archive_format`` from explicit field or extension."""
        if self.archive_format:
            return self.archive_format
        if not self.source:
            raise ValueError("cannot infer archive_format without source")
        s = self.source.lower()
        if s.endswith(".tar.bz2") or s.endswith(".tbz2") or s.endswith(".tbz"):
            return "tar.bz2"
        if s.endswith(".tar.gz") or s.endswith(".tgz"):
            return "tar.gz"
        if s.endswith(".tar.xz") or s.endswith(".txz"):
            return "tar.xz"
        if s.endswith(".tar"):
            return "tar"
        if s.endswith(".zip"):
            return "zip"
        raise ValueError(f"cannot infer archive_format from source={self.source!r}; set archive_format explicitly")


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


class Materializer:
    """Apply a :class:`MaterializationPlan` to a prepared artifact dir.

    Single entry point: :meth:`materialize`. Idempotent across runs;
    safe against tar/zip-bomb / traversal / symlink-escape attacks.
    Plans with ``kind='none'`` are a no-op (the downloaded layout
    is already what the backend reads). Plans with ``kind='archive'``
    extract ``source`` into ``artifact_dir`` honoring
    ``strip_prefix`` and the safety policy, then write the
    extraction-complete marker only after every entry in
    ``required_outputs`` is present on disk.
    """

    def materialize(self, artifact_dir: "str | Path", plan: MaterializationPlan) -> None:
        artifact_dir = Path(artifact_dir)
        if not artifact_dir.is_dir():
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"Materializer: artifact_dir {artifact_dir!r} does not exist or is not "
                    "a directory. Run PrepareManager.prepare(candidate) first."
                ),
            )

        if plan.kind == "none":
            # Downloaded files ARE the layout. Nothing to unpack;
            # we still validate the required outputs so a
            # malformed download surfaces here instead of inside
            # the backend.
            self._assert_layout_complete(artifact_dir, plan)
            return

        if plan.kind == "archive":
            self._materialize_archive(artifact_dir, plan)
            return

        # __post_init__ already rejected unknown kinds; defensive.
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"Materializer: unknown plan kind {plan.kind!r}",
        )

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def _materialize_archive(self, artifact_dir: Path, plan: MaterializationPlan) -> None:
        assert plan.source is not None  # __post_init__ enforced
        archive_path = artifact_dir / plan.source
        if not archive_path.is_file():
            # PrepareManager didn't materialize the archive (failed
            # download, manual cache poke, etc.). Surface a clear
            # error rather than silently no-op.
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"Materializer: archive {plan.source!r} not found under "
                    f"{artifact_dir!r}. Was PrepareManager.prepare(candidate) "
                    f"actually run?"
                ),
            )

        # Idempotency: skip extraction only when the FULL required
        # output layout is on disk AND the marker is present.
        # Reviewer P2 on PR #455: a partial extraction (interrupted
        # mid-write) used to satisfy a "first two files" shortcut,
        # so the next ``prepare`` would skip extraction and leave
        # the artifact dir un-runnable. We additionally write the
        # marker LAST so its presence is the authoritative
        # "ready" signal.
        if self._extraction_marker_valid(artifact_dir, plan):
            return

        # Reset any half-written marker from a previous interrupted
        # run; otherwise a stale marker could confuse the post-
        # extraction completeness check.
        marker = artifact_dir / EXTRACTION_MARKER
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass

        archive_format = plan.infer_archive_format()
        if archive_format == "zip":
            self._extract_zip(archive_path, artifact_dir, plan)
        else:
            self._extract_tar(archive_path, artifact_dir, plan, archive_format)

        # Post-extraction completeness check — refuse to write the
        # marker if anything required is missing.
        self._assert_layout_complete(artifact_dir, plan)

        # Atomic marker write: tmp + rename so a crash mid-write
        # doesn't leave a half-written marker.
        tmp_marker = artifact_dir / f"{EXTRACTION_MARKER}.tmp"
        tmp_marker.write_text(f"plan_kind={plan.kind}\nsource={plan.source}\n")
        os.replace(tmp_marker, marker)

    # ------------------------------------------------------------------
    # Tar / Zip safety
    # ------------------------------------------------------------------

    def _extract_tar(
        self,
        archive_path: Path,
        artifact_dir: Path,
        plan: MaterializationPlan,
        archive_format: str,
    ) -> None:
        mode = {
            "tar": "r:",
            "tar.gz": "r:gz",
            "tar.bz2": "r:bz2",
            "tar.xz": "r:xz",
        }[archive_format]
        policy = plan.safety_policy
        total_unpacked = 0
        # mode is one of the safe "r:..." literals built from the
        # validated archive_format; the union signature on
        # ``tarfile.open`` doesn't see that narrowing.
        with tarfile.open(archive_path, mode) as tar:  # type: ignore[call-overload]
            safe_members = []
            for m in tar.getmembers():
                if m.issym() and not policy.allow_symlinks:
                    continue
                if m.islnk() and not policy.allow_hardlinks:
                    continue
                # Apply strip_prefix in-place so safe_join still
                # validates the FINAL destination path.
                target_name = self._apply_strip_prefix(m.name, plan.strip_prefix)
                if target_name is None:
                    continue  # member entirely covered by strip_prefix
                if not _safe_relative(target_name):
                    continue
                m.name = target_name
                # Symlink targets and hardlink targets need the
                # same validation as the member name itself.
                if (m.issym() or m.islnk()) and not _safe_relative(m.linkname):
                    continue
                # Bomb defense.
                total_unpacked += int(getattr(m, "size", 0) or 0)
                if total_unpacked > policy.max_total_uncompressed_bytes:
                    raise OctomilError(
                        code=OctomilErrorCode.INVALID_INPUT,
                        message=(
                            f"Materializer: archive {archive_path.name!r} exceeds "
                            f"safety policy max_total_uncompressed_bytes "
                            f"({policy.max_total_uncompressed_bytes} bytes); refusing."
                        ),
                    )
                safe_members.append(m)
            try:
                tar.extractall(path=artifact_dir, members=safe_members, filter="data")  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                tar.extractall(path=artifact_dir, members=safe_members)

    def _extract_zip(self, archive_path: Path, artifact_dir: Path, plan: MaterializationPlan) -> None:
        policy = plan.safety_policy
        total_unpacked = 0
        with zipfile.ZipFile(archive_path) as z:
            for info in z.infolist():
                # zipfile has no symlink/hardlink concept by default.
                target_name = self._apply_strip_prefix(info.filename, plan.strip_prefix)
                if target_name is None:
                    continue
                if not _safe_relative(target_name):
                    continue
                total_unpacked += info.file_size
                if total_unpacked > policy.max_total_uncompressed_bytes:
                    raise OctomilError(
                        code=OctomilErrorCode.INVALID_INPUT,
                        message=(
                            f"Materializer: zip {archive_path.name!r} exceeds safety policy "
                            f"max_total_uncompressed_bytes; refusing."
                        ),
                    )
                # Manual extract (zipfile.extract honours zip slip
                # only on 3.12+; we filter ourselves to be safe).
                dest = artifact_dir / target_name
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_strip_prefix(member_name: str, strip_prefix: Optional[str]) -> Optional[str]:
        if not strip_prefix:
            return member_name
        # Normalize trailing slash so ``"foo/"`` and ``"foo"`` both
        # strip the leading directory.
        prefix = strip_prefix.rstrip("/") + "/"
        if member_name == strip_prefix.rstrip("/"):
            return None  # the directory itself; nothing to extract
        if member_name.startswith(prefix):
            stripped = member_name[len(prefix) :]
            return stripped or None
        # Member outside the strip prefix — let it through as-is so
        # the safety filter rejects unexpected layouts.
        return member_name

    def _extraction_marker_valid(self, artifact_dir: Path, plan: MaterializationPlan) -> bool:
        marker = artifact_dir / EXTRACTION_MARKER
        if not marker.is_file():
            return False
        if not plan.required_outputs:
            # No layout declared → can't reason about completeness;
            # be conservative and re-run the extraction.
            return False
        return all((artifact_dir / rel).exists() for rel in plan.required_outputs)

    @staticmethod
    def _assert_layout_complete(artifact_dir: Path, plan: MaterializationPlan) -> None:
        if not plan.required_outputs:
            return
        missing = [rel for rel in plan.required_outputs if not (artifact_dir / rel).exists()]
        if missing:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"Materializer: required outputs missing under {artifact_dir!r}: {missing!r}. "
                    f"The downloaded artifact may be incomplete or the plan's required_outputs / "
                    f"strip_prefix may be wrong."
                ),
            )


def _safe_relative(name: str) -> bool:
    """Return False for member paths that would escape extraction root.

    Rejects absolute paths, drive letters, and any path component
    equal to ``..``. Empty / ``.`` paths also rejected.
    """
    if not name or name in (".", ".."):
        return False
    if os.path.isabs(name):
        return False
    parts = Path(name).parts
    if not parts:
        return False
    for p in parts:
        if p == "..":
            return False
    return True
