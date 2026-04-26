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
    artifacts_root = (cache_dir / "artifacts").resolve()
    assert outcome.artifact_dir.parent == artifacts_root
    assert outcome.artifact_dir.name.startswith(artifact.artifact_id + "-")
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
    artifacts_root = (cache_dir / "artifacts").resolve()
    assert outcome.artifact_dir.parent == artifacts_root
    assert outcome.artifact_dir.name.startswith("art-multi-")
    assert outcome.files["model.onnx"] == (outcome.artifact_dir / "model.onnx").resolve()


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
    served: list[str] = []

    def handler(request):
        served.append(str(request.url))
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    # Pre-place a corrupt file at the destination so a permissive cached
    # shortcut would treat it as ready. Manager must re-download.
    artifact_dir = mgr.artifact_dir_for("art-corrupt")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "artifact").write_bytes(b"WRONG BYTES")
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

    def handler(request):
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    # Plant a directory where the artifact file is expected.
    artifact_dir = mgr.artifact_dir_for("art-dir")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "artifact").mkdir()
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

    def handler(request):
        raise AssertionError("network must not be touched on a digest-verified cache hit")

    mgr = _manager_with_handler(cache_dir, handler)
    # Pre-place the correct file at the manager-derived location so the
    # cached path can find it. Manager must serve from cache without
    # touching the network.
    artifact_dir = mgr.artifact_dir_for("art-cache-good")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "artifact").write_bytes(payload)
    outcome = mgr.prepare(_candidate(artifact=artifact))
    assert outcome.cached
    assert outcome.files[""].read_bytes() == payload


@pytest.mark.parametrize(
    "evil_id",
    [
        "../../escaped-artifact",
        "/tmp/octomil-absolute-artifact",
        "..",
        ".",
        "a/b/c",
        "subdir/../escape",
        "with\x00null",
    ],
)
def test_prepare_sanitizes_or_rejects_unsafe_artifact_id(tmp_path, cache_dir, evil_id):
    payload = b"data"
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="m",
        artifact_id=evil_id,
        digest=digest,
        required_files=[],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )

    served: list[str] = []

    def handler(request):
        served.append(str(request.url))
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    artifacts_root = (cache_dir / "artifacts").resolve()
    # Track every path that exists under tmp_path before and after to be
    # sure no write escaped the per-test sandbox.
    pre_existing = {p for p in tmp_path.rglob("*") if p.is_file()}

    try:
        outcome = mgr.prepare(_candidate(artifact=artifact))
    except OctomilError as exc:
        assert exc.code == ErrorCode.INVALID_INPUT
        return

    # If accepted, the resolved artifact_dir must live under <cache>/artifacts.
    assert outcome.artifact_dir.parent == artifacts_root
    assert outcome.artifact_dir.is_relative_to(artifacts_root)
    # And no file landed outside the per-test sandbox.
    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    new_files = written - pre_existing
    for new_file in new_files:
        assert new_file.resolve().is_relative_to(tmp_path.resolve())


def test_prepare_distinct_planner_ids_get_distinct_directories(cache_dir):
    # Two artifact ids that sanitize to the same visible name must still
    # land in distinct directories — the digest suffix prevents collision.
    mgr = PrepareManager(cache_dir=cache_dir)
    # Both ids sanitize to "a_b" but should not share an artifact_dir.
    dir_a = mgr.artifact_dir_for("a/b")
    dir_b = mgr.artifact_dir_for("a b")
    assert dir_a != dir_b
    artifacts_root = (cache_dir / "artifacts").resolve()
    assert dir_a.parent == artifacts_root
    assert dir_b.parent == artifacts_root
    # Same input twice -> same directory (deterministic).
    assert mgr.artifact_dir_for("a/b") == dir_a


def test_prepare_caps_long_but_safe_artifact_id_to_filesystem_safe_length(cache_dir):
    # 5000 safe characters would exceed NAME_MAX on every common filesystem.
    # Manager must keep the on-disk key under 256 bytes and surface a usable
    # path, not a raw OSError.
    long_id = "a" * 5000
    payload = b"x"
    digest = _digest(payload)
    artifact = RuntimeArtifactPlan(
        model_id="m",
        artifact_id=long_id,
        digest=digest,
        required_files=[],
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )

    def handler(request):
        return httpx.Response(200, content=payload)

    mgr = _manager_with_handler(cache_dir, handler)
    outcome = mgr.prepare(_candidate(artifact=artifact))

    artifacts_root = (cache_dir / "artifacts").resolve()
    assert outcome.artifact_dir.parent == artifacts_root
    assert (
        len(outcome.artifact_dir.name.encode("utf-8")) <= 255
    ), f"artifact_dir component must fit NAME_MAX, got {len(outcome.artifact_dir.name)} chars"
    # Two distinct long ids that share their first chars must still get
    # distinct keys (the digest covers the full original id).
    other_id = "a" * 4999 + "b"
    other_dir = mgr.artifact_dir_for(other_id)
    assert other_dir != outcome.artifact_dir
    assert len(other_dir.name.encode("utf-8")) <= 255


def test_artifact_dir_for_caps_visible_portion_directly(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    long_id = "z" * 5000
    d = mgr.artifact_dir_for(long_id)
    # Visible (everything before the trailing -<hash>) is capped; the dir
    # name overall stays comfortably below NAME_MAX (96 + 1 + 12 = 109).
    assert len(d.name) <= 109
    assert d.name.endswith("-" + hashlib.sha256(long_id.encode()).hexdigest()[:12])


# --- can_prepare dry-run ----------------------------------------------------


def test_can_prepare_returns_true_for_complete_local_sdk_runtime_candidate(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    assert mgr.can_prepare(_candidate(artifact=_artifact()))


def test_can_prepare_returns_false_when_artifact_missing_digest(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-82m",
        artifact_id="art-x",
        digest=None,
        download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
    )
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_false_when_artifact_missing_download_urls(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = _artifact(endpoints=[])
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_false_for_synthetic_artifact_with_only_model_id(cache_dir):
    """Reviewer's reproducer: planner can emit prepare_required=True with no
    digest or download_urls. PrepareManager's prepare() rejects this — so
    can_prepare() must not lie to the routing layer."""
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = RuntimeArtifactPlan(model_id="kokoro-82m")
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_false_when_artifact_is_none(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    assert not mgr.can_prepare(_candidate(artifact=None))


def test_can_prepare_returns_false_for_disabled_policy(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    assert not mgr.can_prepare(_candidate(artifact=_artifact(), prepare_policy="disabled"))


def test_can_prepare_returns_false_for_cloud_candidate(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    assert not mgr.can_prepare(_candidate(artifact=_artifact(), locality="cloud"))


def test_can_prepare_returns_false_for_hosted_gateway_delivery(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    assert not mgr.can_prepare(_candidate(artifact=_artifact(), delivery_mode="hosted_gateway"))


def test_can_prepare_returns_false_for_multi_file_artifact(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = _artifact(required_files=["model.onnx", "voices.bin"])
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_true_when_prepare_required_false(cache_dir):
    """prepare() short-circuits to a cached no-files outcome — that is a
    valid, non-failing call, so can_prepare must return True."""
    mgr = PrepareManager(cache_dir=cache_dir)
    assert mgr.can_prepare(_candidate(artifact=_artifact(), prepare_required=False))


def test_can_prepare_does_not_touch_disk_or_network(cache_dir):
    # Empty cache dir, no downloader injected — can_prepare must work
    # entirely from the candidate inspection.
    mgr = PrepareManager(cache_dir=cache_dir)
    assert mgr.can_prepare(_candidate(artifact=_artifact()))
    # Cache dir has only what __init__ created (.locks would be lazy).
    assert not (cache_dir / "artifacts").exists() or not any((cache_dir / "artifacts").iterdir())


# --- can_prepare mirrors prepare structural validation --------------------


@pytest.mark.parametrize(
    "bad_relative_path",
    [
        "../escaped.bin",
        ".",
        "./",
        "/abs/path",
        "subdir/../escape",
        "with\x00null",
        "back\\slash",
    ],
)
def test_can_prepare_returns_false_for_unsafe_required_files_path(cache_dir, bad_relative_path):
    """Reviewer's reproducer: can_prepare must reject any required_files
    entry that prepare() would reject via _validate_relative_path. Without
    this, local_first commits to local on a malformed planner plan and
    fails in prepare instead of falling back."""
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = _artifact(required_files=[bad_relative_path])
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_false_for_empty_artifact_id_and_model_id(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = RuntimeArtifactPlan(
        model_id="",
        artifact_id="",
        digest=_digest(b"x"),
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
    )
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_false_for_artifact_id_with_nul_byte(cache_dir):
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = RuntimeArtifactPlan(
        model_id="m",
        artifact_id="evil\x00\x00",
        digest=_digest(b"x"),
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
    )
    assert not mgr.can_prepare(_candidate(artifact=artifact))


def test_can_prepare_returns_true_when_artifact_id_empty_but_model_id_present(cache_dir):
    """artifact_id falls back to model_id at the build site, so an empty
    artifact_id with a non-empty model_id is preparable."""
    mgr = PrepareManager(cache_dir=cache_dir)
    artifact = RuntimeArtifactPlan(
        model_id="kokoro-en-v0_19",
        artifact_id="",
        digest=_digest(b"x"),
        download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
    )
    assert mgr.can_prepare(_candidate(artifact=artifact))


@pytest.mark.parametrize(
    "build_artifact,description",
    [
        (lambda: RuntimeArtifactPlan(model_id="m"), "no digest, no urls"),
        (
            lambda: RuntimeArtifactPlan(
                model_id="m",
                artifact_id="ok",
                digest="sha256:" + "0" * 64,
                required_files=["../escape.bin"],
                download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            ),
            "traversal in required_files",
        ),
        (
            lambda: RuntimeArtifactPlan(
                model_id="m",
                artifact_id="ok",
                digest="sha256:" + "0" * 64,
                required_files=["."],
                download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            ),
            "dot path",
        ),
        (
            lambda: RuntimeArtifactPlan(
                model_id="",
                artifact_id="",
                digest="sha256:" + "0" * 64,
                download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            ),
            "empty artifact_id and model_id",
        ),
        (
            lambda: RuntimeArtifactPlan(
                model_id="m",
                artifact_id="evil\x00",
                digest="sha256:" + "0" * 64,
                download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            ),
            "NUL in artifact_id",
        ),
        (
            lambda: RuntimeArtifactPlan(
                model_id="m",
                artifact_id="ok",
                digest="sha256:" + "0" * 64,
                required_files=["a.bin", "b.bin"],
                download_urls=[ArtifactDownloadEndpoint(url="https://x/")],
            ),
            "multi-file",
        ),
    ],
)
def test_can_prepare_and_prepare_agree_on_structurally_invalid_inputs(cache_dir, build_artifact, description):
    """Property: anything can_prepare rejects, prepare also rejects. This is
    the contract the routing layer relies on to avoid committing to local
    and failing in prepare."""
    mgr = PrepareManager(cache_dir=cache_dir)
    candidate = _candidate(artifact=build_artifact())
    assert not mgr.can_prepare(candidate), f"{description}: can_prepare lied"
    with pytest.raises(OctomilError):
        mgr.prepare(candidate)
