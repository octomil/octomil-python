"""Tests for PrepareManager — the bridge from planner output to on-disk artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import httpx
import pytest

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.runtime.lifecycle.durable_download import DurableDownloader
from octomil.runtime.lifecycle.prepare_manager import (
    PrepareManager,
    PrepareMode,
)
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)


def _candidate(
    *,
    artifact: RuntimeArtifactPlan | None,
    delivery_mode: Literal["hosted_gateway", "sdk_runtime", "external_endpoint"] = "sdk_runtime",
    prepare_required: bool = True,
    prepare_policy: Literal["lazy", "explicit_only", "disabled"] = "lazy",
    locality: Literal["local", "cloud"] = "local",
    engine: str = "sherpa-onnx",
) -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality=locality,
        priority=0,
        confidence=1.0,
        reason="test",
        engine=engine,
        artifact=artifact,
        delivery_mode=delivery_mode,
        prepare_required=prepare_required,
        prepare_policy=prepare_policy,
    )


def _artifact(
    *,
    payload: bytes = b"weights" * 100,
    artifact_id: str = "art-1",
    required_files: list[str] | None = None,
    endpoints: list[ArtifactDownloadEndpoint] | None = None,
) -> RuntimeArtifactPlan:
    return RuntimeArtifactPlan(
        model_id="kokoro-en-v0_19",
        artifact_id=artifact_id,
        digest=_digest(payload),
        size_bytes=len(payload),
        required_files=required_files if required_files is not None else [],
        download_urls=endpoints
        if endpoints is not None
        else [ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "octomil-cache"
    d.mkdir()
    return d


def _manager_with_handler(cache_dir: Path, handler) -> PrepareManager:
    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    return PrepareManager(cache_dir=cache_dir, downloader=downloader)


def test_prepare_downloads_and_returns_outcome(cache_dir):
    payload = b"voice tokens"
    artifact = _artifact(payload=payload)

    served: list[str] = []

    def handler(request):
        served.append(str(request.url))
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(_candidate(artifact=artifact))

    assert not outcome.cached
    assert outcome.delivery_mode == "sdk_runtime"
    assert outcome.prepare_policy == "lazy"
    assert outcome.engine == "sherpa-onnx"
    assert outcome.artifact_dir == cache_dir / "artifacts" / artifact.artifact_id
    assert outcome.files[""].read_bytes() == payload
    assert served, "downloader should have hit the network"


def test_prepare_is_idempotent_on_second_call(cache_dir):
    payload = b"twice"
    artifact = _artifact(payload=payload)
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    first = mgr.prepare(_candidate(artifact=artifact))
    second = mgr.prepare(_candidate(artifact=artifact))

    assert not first.cached
    assert second.cached
    assert second.files[""] == first.files[""]
    assert len(calls) == 1, "second prepare must not re-download"


def test_prepare_rejects_cloud_candidate(cache_dir):
    artifact = _artifact()
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError) as excinfo:
        mgr.prepare(_candidate(artifact=artifact, locality="cloud"))
    assert excinfo.value.code == ErrorCode.INVALID_INPUT


def test_prepare_rejects_non_sdk_runtime_delivery_mode(cache_dir):
    artifact = _artifact()
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError) as excinfo:
        mgr.prepare(_candidate(artifact=artifact, delivery_mode="hosted_gateway"))
    assert excinfo.value.code == ErrorCode.INVALID_INPUT


def test_prepare_rejects_disabled_policy(cache_dir):
    artifact = _artifact()
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError, match="disabled"):
        mgr.prepare(_candidate(artifact=artifact, prepare_policy="disabled"))


def test_prepare_explicit_only_blocks_lazy_call(cache_dir):
    artifact = _artifact()
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError, match="explicit_only"):
        mgr.prepare(_candidate(artifact=artifact, prepare_policy="explicit_only"))


def test_prepare_explicit_only_allows_explicit_mode(cache_dir):
    payload = b"explicit"
    artifact = _artifact(payload=payload)

    def handler(request):
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(
        _candidate(artifact=artifact, prepare_policy="explicit_only"),
        mode=PrepareMode.EXPLICIT,
    )
    assert outcome.files[""].read_bytes() == payload


def test_prepare_required_false_returns_cached_no_files(cache_dir):
    artifact = _artifact()
    mgr = PrepareManager(cache_dir=cache_dir)  # no downloader, must not call network
    outcome = mgr.prepare(_candidate(artifact=artifact, prepare_required=False))
    assert outcome.cached
    assert outcome.files == {}


def test_prepare_rejects_when_artifact_missing_but_required(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    candidate = _candidate(artifact=None)
    with pytest.raises(OctomilError, match="prepare_required"):
        mgr.prepare(candidate)


def test_prepare_rejects_when_no_download_urls(cache_dir):
    artifact = _artifact(endpoints=[])
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError, match="download_urls"):
        mgr.prepare(_candidate(artifact=artifact))


def test_prepare_rejects_when_no_digest(cache_dir):
    artifact = RuntimeArtifactPlan(
        model_id="kokoro",
        artifact_id="art-x",
        digest=None,
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError, match="digest"):
        mgr.prepare(_candidate(artifact=artifact))


def test_prepare_single_required_file_uses_relative_path_under_artifact_dir(cache_dir):
    payload = b"\x00\x01" * 50
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="multi",
        artifact_id="art-multi",
        digest=digest,
        required_files=["model.onnx"],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/art-multi/")],
    )

    def handler(request):
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(_candidate(artifact=artifact))
    assert outcome.files["model.onnx"].read_bytes() == payload
    assert outcome.artifact_dir == cache_dir / "artifacts" / "art-multi"
    assert outcome.files["model.onnx"] == (cache_dir / "artifacts" / "art-multi" / "model.onnx").resolve()


def test_prepare_rejects_multi_file_artifact_until_per_file_manifest_exists(cache_dir):
    payload = b"a"
    artifact = RuntimeArtifactPlan(
        model_id="kokoro",
        artifact_id="art-multi-real",
        digest=_digest(payload),
        required_files=["model.onnx", "voices.bin"],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError, match="manifest_uri"):
        mgr.prepare(_candidate(artifact=artifact))


def test_prepare_rejects_traversal_in_required_files(cache_dir):
    payload = b"pwn"
    artifact = RuntimeArtifactPlan(
        model_id="evil",
        artifact_id="art-evil",
        digest=_digest(payload),
        required_files=["../escaped.bin"],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    # Plant the file the planner is trying to claim, so a permissive cached
    # shortcut would happily return it. The fix must reject before checking.
    outside = cache_dir / "artifacts" / "escaped.bin"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(payload)

    mgr = PrepareManager(cache_dir=cache_dir)
    with pytest.raises(OctomilError) as excinfo:
        mgr.prepare(_candidate(artifact=artifact))
    assert excinfo.value.code == ErrorCode.INVALID_INPUT


def test_prepare_cached_shortcut_rejects_corrupt_existing_file(cache_dir):
    payload = b"good bytes" * 50
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="corrupt-test",
        artifact_id="art-corrupt",
        digest=digest,
        required_files=[],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    # Pre-place a corrupt file at the destination so a permissive cached
    # shortcut would treat it as ready. Manager must re-download.
    artifact_dir = cache_dir / "artifacts" / "art-corrupt"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact").write_bytes(b"WRONG BYTES")

    served: list[str] = []

    def handler(request):
        served.append(str(request.url))
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(_candidate(artifact=artifact))
    assert not outcome.cached, "corrupt file must not be returned as cached"
    assert outcome.files[""].read_bytes() == payload
    assert served, "downloader must have been called to recover from corrupt cache"


def test_prepare_cached_shortcut_rejects_directory_at_target(cache_dir):
    payload = b"contents"
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="dir-test",
        artifact_id="art-dir",
        digest=digest,
        required_files=[],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    # Plant a directory where the artifact file is expected.
    artifact_dir = cache_dir / "artifacts" / "art-dir"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact").mkdir()

    def handler(request):
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    # The downloader will try to write a file where a directory exists and
    # raise; but the cached shortcut must NOT return cached=True.
    with pytest.raises(Exception):
        mgr.prepare(_candidate(artifact=artifact))


def test_prepare_cached_shortcut_returns_cached_only_after_digest_verifies(cache_dir):
    payload = b"verified bytes" * 30
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="cache-good",
        artifact_id="art-cache-good",
        digest=digest,
        required_files=[],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    # Pre-place the correct file. Manager must serve from cache without
    # touching the network.
    artifact_dir = cache_dir / "artifacts" / "art-cache-good"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact").write_bytes(payload)

    def handler(request):
        raise AssertionError("network must not be touched on a digest-verified cache hit")

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(_candidate(artifact=artifact))
    assert outcome.cached
    assert outcome.files[""].read_bytes() == payload
