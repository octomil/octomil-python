"""Runtime preparation — verify engine + artifact readiness before inference.

Coordinates detection, cache lookup, and download into a single
prepare step that either succeeds with a ready-to-use path or
returns an actionable error. NEVER synthesizes fake output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.lifecycle.artifact_cache import ArtifactCache
from octomil.runtime.lifecycle.detection import InstalledRuntime, detect_installed_runtimes
from octomil.runtime.lifecycle.download import DownloadManager

logger = logging.getLogger(__name__)


# Engine install instructions for actionable errors
_INSTALL_INSTRUCTIONS: dict[str, str] = {
    "mlx-lm": "pip install mlx-lm  # Apple Silicon only",
    "llama.cpp": "pip install llama-cpp-python  # Cross-platform CPU/GPU",
    "onnxruntime": "pip install onnxruntime  # Cross-platform",
}


@dataclass
class RuntimeCandidate:
    """Input to prepare_runtime: what engine and artifact to prepare."""

    engine_id: str
    artifact_id: str | None = None
    artifact_url: str | None = None
    expected_digest: str | None = None
    filename: str | None = None


@dataclass
class PrepareResult:
    """Result of prepare_runtime.

    On success: ok=True, artifact_path is set.
    On failure: ok=False, error is set with actionable message.
    """

    ok: bool
    engine_id: str
    artifact_path: Path | None = None
    installed_runtime: InstalledRuntime | None = None
    error: str | None = None
    error_code: OctomilErrorCode | None = None
    extras: dict[str, str] = field(default_factory=dict)


def prepare_runtime(
    candidate: RuntimeCandidate,
    cache: ArtifactCache | None = None,
    download_manager: DownloadManager | None = None,
    skip_download: bool = False,
) -> PrepareResult:
    """Prepare a local runtime for inference.

    Steps:
    1. Detect if the requested engine is installed.
    2. If artifact_id is provided, check cache for existing artifact.
    3. If not cached and URL is available, download with lock + verification.
    4. Return PrepareResult with either success or actionable error.

    NEVER returns fake/synthetic inference output. On failure, the error
    message tells the user exactly what to do.
    """
    engine_id = candidate.engine_id

    # Step 1: Check engine is installed
    installed = detect_installed_runtimes()
    matching = [r for r in installed if r.engine_id == engine_id]

    if not matching:
        install_hint = _INSTALL_INSTRUCTIONS.get(engine_id, f"Install the '{engine_id}' runtime")
        return PrepareResult(
            ok=False,
            engine_id=engine_id,
            error=(f"Engine '{engine_id}' is not installed. " f"Install with: {install_hint}"),
            error_code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        )

    runtime = matching[0]

    # Step 2: If no artifact needed (engine manages own download), we're done
    if candidate.artifact_id is None:
        return PrepareResult(
            ok=True,
            engine_id=engine_id,
            installed_runtime=runtime,
        )

    # Step 3: Check artifact cache
    _cache = cache or ArtifactCache()
    artifact_id = candidate.artifact_id
    expected_digest = candidate.expected_digest or ""

    if expected_digest:
        cached_path = _cache.get(artifact_id, expected_digest)
        if cached_path is not None:
            return PrepareResult(
                ok=True,
                engine_id=engine_id,
                artifact_path=cached_path,
                installed_runtime=runtime,
            )

    # Step 4: Download if URL provided and not skipping
    if skip_download:
        return PrepareResult(
            ok=False,
            engine_id=engine_id,
            installed_runtime=runtime,
            error=(
                f"Artifact '{artifact_id}' not in cache and download is disabled. "
                f"Run with --yes to allow automatic downloads."
            ),
            error_code=OctomilErrorCode.DOWNLOAD_FAILED,
        )

    if candidate.artifact_url is None:
        return PrepareResult(
            ok=False,
            engine_id=engine_id,
            installed_runtime=runtime,
            error=(
                f"Artifact '{artifact_id}' not in cache and no download URL available. "
                f"Provide a download URL or manually place the artifact in the cache."
            ),
            error_code=OctomilErrorCode.DOWNLOAD_FAILED,
        )

    if not expected_digest:
        return PrepareResult(
            ok=False,
            engine_id=engine_id,
            installed_runtime=runtime,
            error=(
                f"Cannot download artifact '{artifact_id}' without an expected digest. "
                f"Integrity verification requires a SHA-256 digest."
            ),
            error_code=OctomilErrorCode.CHECKSUM_MISMATCH,
        )

    # Download with verification
    mgr = download_manager or DownloadManager(cache=_cache)
    try:
        artifact_path = mgr.download(
            artifact_id=artifact_id,
            url=candidate.artifact_url,
            expected_digest=expected_digest,
            filename=candidate.filename,
        )
    except OctomilError as exc:
        return PrepareResult(
            ok=False,
            engine_id=engine_id,
            installed_runtime=runtime,
            error=str(exc),
            error_code=exc.code,
        )

    return PrepareResult(
        ok=True,
        engine_id=engine_id,
        artifact_path=artifact_path,
        installed_runtime=runtime,
    )
