"""Per-file manifest parser for multi-file artifacts.

PR C-followup: ``RuntimeArtifactPlan.manifest_uri`` lets the planner
emit per-file integrity metadata for artifacts whose
``required_files`` carries more than one entry. The single
artifact-level ``digest`` cannot verify multiple files; this module
fetches and validates the ``manifest_uri`` JSON and produces a
per-file :class:`RequiredFile` list that
:class:`DurableDownloader` can consume.

Manifest schema (``manifest.v1.json``):

.. code-block:: json

    {
      "version": 1,
      "files": [
        {
          "relative_path": "model.onnx",
          "digest": "sha256:912804855a04...",
          "size_bytes": 311040000
        },
        {
          "relative_path": "voices.bin",
          "digest": "sha256:abc123...",
          "size_bytes": 21504
        }
      ]
    }

Future versions add fields under the same ``version`` discriminator.
The SDK rejects unknown ``version`` values rather than silently
ignoring fields it doesn't understand — same shape as the rest of
the planner schemas.

Trust boundary: ``manifest_uri`` is server-supplied. Every relative
path is validated by ``_validate_relative_path`` before any
filesystem use, every digest is checked for the ``sha256:`` prefix
plus 64 hex chars, and the manifest itself is digest-pinned by the
artifact-level ``digest`` so a tampered manifest body is rejected
before any per-file URL is fetched.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.runtime.lifecycle.durable_download import (
    RequiredFile,
    _validate_relative_path,
)

logger = logging.getLogger(__name__)

_MANIFEST_FETCH_TIMEOUT = 30.0
_MANIFEST_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB cap — manifests are tiny by definition.
_DIGEST_PREFIX = "sha256:"
_DIGEST_HEX_LEN = 64


@dataclass(frozen=True)
class ParsedManifest:
    """Validated per-file manifest, ready for descriptor construction."""

    version: int
    files: list[RequiredFile]


def fetch_and_parse_manifest(
    manifest_uri: str,
    *,
    artifact_digest: str,
    client: httpx.Client | None = None,
    timeout: float = _MANIFEST_FETCH_TIMEOUT,
) -> ParsedManifest:
    """Fetch ``manifest_uri``, verify its body against ``artifact_digest``,
    and return the parsed :class:`ParsedManifest`.

    The artifact-level ``digest`` is reused as the manifest body's
    expected SHA-256: the server commits to a specific manifest by
    publishing its hash as ``RuntimeArtifactPlan.digest``. This means
    a tampered ``manifest_uri`` (different host, modified files) is
    caught before any per-file URL is fetched.

    Raises :class:`OctomilError` (``INVALID_INPUT`` / ``CHECKSUM_MISMATCH``
    / ``DOWNLOAD_FAILED``) on any contract violation.
    """
    if not manifest_uri:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message="Manifest URI is empty.",
        )
    expected_hex = _normalize_digest(artifact_digest)

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = http.get(manifest_uri)
        if response.status_code != 200:
            raise OctomilError(
                code=ErrorCode.DOWNLOAD_FAILED,
                message=(f"Manifest fetch failed: HTTP {response.status_code} " f"for {manifest_uri!r}."),
            )
        body = response.content
    except httpx.RequestError as exc:
        raise OctomilError(
            code=ErrorCode.DOWNLOAD_FAILED,
            message=f"Manifest fetch failed: {exc}",
        ) from exc
    finally:
        if owns_client:
            http.close()

    if len(body) > _MANIFEST_MAX_BYTES:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Manifest at {manifest_uri!r} exceeds the safety cap "
                f"({len(body)} > {_MANIFEST_MAX_BYTES} bytes); refusing to "
                f"parse."
            ),
        )

    actual_hex = hashlib.sha256(body).hexdigest()
    if actual_hex != expected_hex:
        raise OctomilError(
            code=ErrorCode.CHECKSUM_MISMATCH,
            message=(
                f"Manifest at {manifest_uri!r} digest mismatch: "
                f"expected {expected_hex}, got {actual_hex}. The manifest "
                f"body the server published does not match the digest in "
                f"RuntimeArtifactPlan; refusing to honor."
            ),
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Manifest at {manifest_uri!r} is not valid JSON: {exc}",
        ) from exc

    return parse_manifest_payload(payload)


def parse_manifest_payload(payload: Any) -> ParsedManifest:
    """Validate a parsed manifest object and produce a
    :class:`ParsedManifest`.

    Split out from :func:`fetch_and_parse_manifest` so the parsing
    half can be tested without a network round-trip.
    """
    if not isinstance(payload, dict):
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Manifest payload must be a JSON object, got {type(payload).__name__}",
        )
    version = payload.get("version")
    if not isinstance(version, int):
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(f"Manifest 'version' must be an integer, got {type(version).__name__}"),
        )
    if version != 1:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Manifest version {version} is not supported by this SDK release. "
                f"Only manifest v1 is recognized; upgrade the SDK or have the "
                f"planner emit a v1 manifest."
            ),
        )
    files_raw = payload.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=("Manifest 'files' must be a non-empty list of " "{relative_path, digest, size_bytes?} entries."),
        )
    files: list[RequiredFile] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(files_raw):
        if not isinstance(entry, dict):
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"Manifest entry {index} must be an object, got " f"{type(entry).__name__}."),
            )
        rel = entry.get("relative_path")
        digest = entry.get("digest")
        size = entry.get("size_bytes")
        if not isinstance(rel, str):
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"Manifest entry {index}: 'relative_path' must be a string."),
            )
        if not isinstance(digest, str):
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"Manifest entry {index}: 'digest' must be a string."),
            )
        if size is not None and (not isinstance(size, int) or size < 0):
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(f"Manifest entry {index}: 'size_bytes' must be a non-negative integer " f"or null."),
            )
        # Empty relative_path is the single-file shape; not allowed in
        # a multi-file manifest. The single-file path goes through the
        # artifact-level digest, not the manifest.
        if not rel:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Manifest entry {index}: 'relative_path' must not be empty in "
                    f"a multi-file manifest. Single-file artifacts use "
                    f"RuntimeArtifactPlan.digest directly."
                ),
            )
        safe_rel = _validate_relative_path(rel)
        if safe_rel in seen_paths:
            raise OctomilError(
                code=ErrorCode.INVALID_INPUT,
                message=(
                    f"Manifest entry {index}: duplicate relative_path " f"{rel!r}. Each path must appear at most once."
                ),
            )
        seen_paths.add(safe_rel)
        _validate_digest_format(digest, index=index)
        files.append(RequiredFile(relative_path=safe_rel, digest=digest, size_bytes=size))
    return ParsedManifest(version=version, files=files)


def _normalize_digest(value: str) -> str:
    """Strip the optional ``sha256:`` prefix and lowercase the hex."""
    if value.startswith(_DIGEST_PREFIX):
        value = value[len(_DIGEST_PREFIX) :]
    return value.lower()


def _validate_digest_format(digest: str, *, index: int) -> None:
    """Reject digests that aren't ``sha256:<64 hex>`` or bare 64 hex."""
    body = digest[len(_DIGEST_PREFIX) :] if digest.startswith(_DIGEST_PREFIX) else digest
    if len(body) != _DIGEST_HEX_LEN:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"Manifest entry {index}: digest must be 64 hex chars (with "
                f"or without 'sha256:' prefix), got {len(body)} chars."
            ),
        )
    try:
        int(body, 16)
    except ValueError:
        raise OctomilError(
            code=ErrorCode.INVALID_INPUT,
            message=(f"Manifest entry {index}: digest is not valid hex: {digest!r}."),
        ) from None
