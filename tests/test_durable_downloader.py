"""Tests for the durable, multi-URL artifact downloader."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.runtime.lifecycle.durable_download import (
    ArtifactDescriptor,
    DownloadEndpoint,
    DurableDownloader,
    RequiredFile,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def dest_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dest"
    d.mkdir()
    return d


def test_downloads_single_file_artifact(cache_dir, dest_dir):
    payload = b"hello world" * 1000
    digest = _digest(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-1",
        required_files=[RequiredFile(relative_path="", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/file.bin")],
    )

    result = downloader.download(desc, dest_dir)

    assert set(result.files) == {""}
    assert result.files[""].read_bytes() == payload


def test_downloads_multi_file_artifact_resolves_urls_relative_to_endpoint(cache_dir, dest_dir):
    files = {
        "tokens.json": b'{"tokens": []}',
        "weights/model.bin": b"BIN" * 100,
    }
    digests = {name: _digest(data) for name, data in files.items()}
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        for name, data in files.items():
            if request.url.path.endswith("/" + name):
                return httpx.Response(200, content=data)
        return httpx.Response(404)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-multi",
        required_files=[
            RequiredFile(relative_path="tokens.json", digest=digests["tokens.json"]),
            RequiredFile(relative_path="weights/model.bin", digest=digests["weights/model.bin"]),
        ],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/art-multi/")],
    )

    result = downloader.download(desc, dest_dir)

    assert set(result.files) == {"tokens.json", "weights/model.bin"}
    assert (dest_dir / "tokens.json").read_bytes() == files["tokens.json"]
    assert (dest_dir / "weights/model.bin").read_bytes() == files["weights/model.bin"]
    assert any(p.endswith("/weights/model.bin") for p in seen_paths)


def test_idempotent_when_file_already_present_with_correct_digest(cache_dir, dest_dir):
    payload = b"already here"
    digest = _digest(payload)
    target = dest_dir / "out.bin"
    target.write_bytes(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have called network")

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-cached",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/")],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"] == target
    assert target.read_bytes() == payload


def test_resumes_partial_download_via_range_header(cache_dir, dest_dir):
    payload = b"abcdefghijklmnopqrstuvwxyz" * 100
    digest = _digest(payload)
    head, tail = payload[:300], payload[300:]

    parts = dest_dir / ".parts"
    parts.mkdir()
    (parts / "out.bin.part").write_bytes(head)

    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        if request.headers.get("range") == "bytes=300-":
            return httpx.Response(
                206, content=tail, headers={"content-range": f"bytes 300-{len(payload) - 1}/{len(payload)}"}
            )
        return httpx.Response(200, content=payload)

    journal = cache_dir / ".progress.sqlite"
    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    # Seed the journal so the resume offset is taken from there.
    downloader._journal.record("art-resume", "out.bin", 300, 0)

    desc = ArtifactDescriptor(
        artifact_id="art-resume",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/")],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    assert "bytes=300-" in seen_ranges
    assert journal.exists()


def test_rotates_to_next_endpoint_on_403(cache_dir, dest_dir):
    payload = b"fallback win"
    digest = _digest(payload)

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "primary" in str(request.url):
            return httpx.Response(403, content=b"forbidden")
        return httpx.Response(200, content=payload)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-fb",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[
            DownloadEndpoint(url="https://primary.example.com/"),
            DownloadEndpoint(url="https://fallback.example.com/"),
        ],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    assert any("primary" in u for u in calls)
    assert any("fallback" in u for u in calls)


def test_skips_endpoint_with_expires_at_in_the_past(cache_dir, dest_dir):
    payload = b"after-expiry"
    digest = _digest(payload)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=payload)

    fixed_now = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    downloader = DurableDownloader(
        cache_dir=cache_dir,
        client=_client_with(handler),
        clock=lambda: fixed_now,
    )

    desc = ArtifactDescriptor(
        artifact_id="art-exp",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[
            DownloadEndpoint(
                url="https://expired.example.com/",
                expires_at=(fixed_now - timedelta(minutes=5)).isoformat(),
            ),
            DownloadEndpoint(url="https://fresh.example.com/"),
        ],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    assert all("expired.example.com" not in u for u in calls)


def test_corrupt_response_falls_through_to_next_endpoint(cache_dir, dest_dir):
    payload = b"good data"
    digest = _digest(payload)
    bad = b"corrupted"

    def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in str(request.url):
            return httpx.Response(200, content=bad)
        return httpx.Response(200, content=payload)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-corrupt",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[
            DownloadEndpoint(url="https://bad.example.com/"),
            DownloadEndpoint(url="https://good.example.com/"),
        ],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload


def test_raises_when_all_endpoints_fail(cache_dir, dest_dir):
    digest = _digest(b"never seen")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-dead",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[
            DownloadEndpoint(url="https://a.example.com/"),
            DownloadEndpoint(url="https://b.example.com/"),
        ],
    )

    with pytest.raises(OctomilError) as excinfo:
        downloader.download(desc, dest_dir)
    assert excinfo.value.code == ErrorCode.DOWNLOAD_FAILED


def test_progress_journal_survives_new_downloader_instance(cache_dir, dest_dir):
    payload = b"persisted progress" * 50
    digest = _digest(payload)
    head, tail = payload[:200], payload[200:]
    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        if request.headers.get("range") == "bytes=200-":
            return httpx.Response(
                206, content=tail, headers={"content-range": f"bytes 200-{len(payload) - 1}/{len(payload)}"}
            )
        return httpx.Response(200, content=payload)

    parts = dest_dir / ".parts"
    parts.mkdir()
    (parts / "out.bin.part").write_bytes(head)

    # Instance A seeds the journal.
    a = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    a._journal.record("art-pers", "out.bin", 200, 0)

    # Instance B is fresh — must read journal from disk.
    b = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-pers",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/")],
    )
    result = b.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    assert "bytes=200-" in seen_ranges


def test_clamps_offset_when_disk_is_behind_journal(cache_dir, dest_dir):
    payload = b"recovery data" * 40
    digest = _digest(payload)
    head, tail = payload[:50], payload[50:]
    seen_ranges: list[str | None] = []

    parts = dest_dir / ".parts"
    parts.mkdir()
    (parts / "out.bin.part").write_bytes(head)  # only 50 bytes on disk

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        if request.headers.get("range") == "bytes=50-":
            return httpx.Response(
                206, content=tail, headers={"content-range": f"bytes 50-{len(payload) - 1}/{len(payload)}"}
            )
        return httpx.Response(200, content=payload)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    # Journal claims 200 bytes written, but disk only has 50.
    downloader._journal.record("art-clamp", "out.bin", 200, 0)

    desc = ArtifactDescriptor(
        artifact_id="art-clamp",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://cdn.example.com/")],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    assert "bytes=50-" in seen_ranges


def test_endpoint_headers_are_sent(cache_dir, dest_dir):
    payload = b"with auth"
    digest = _digest(payload)
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("x-signed-url-token"))
        return httpx.Response(200, content=payload)

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-hdr",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[
            DownloadEndpoint(
                url="https://signed.example.com/",
                headers={"X-Signed-Url-Token": "abc123"},
            )
        ],
    )

    downloader.download(desc, dest_dir)
    assert "abc123" in captured


def test_empty_endpoints_raises(cache_dir, dest_dir):
    downloader = DurableDownloader(cache_dir=cache_dir)
    desc = ArtifactDescriptor(
        artifact_id="art-empty",
        required_files=[RequiredFile(relative_path="x", digest=_digest(b"x"))],
        endpoints=[],
    )
    with pytest.raises(OctomilError) as excinfo:
        downloader.download(desc, dest_dir)
    assert excinfo.value.code == ErrorCode.DOWNLOAD_FAILED


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escaped.bin",
        "subdir/../../escaped.bin",
        "/etc/passwd",
        "C:\\Windows\\system32",
        "weights\\model.bin",
        "weights/../../../../etc/passwd",
        "//absolute/posix",
        "with\x00null.bin",
    ],
)
def test_rejects_unsafe_relative_paths(cache_dir, dest_dir, bad_path):
    def handler(request):
        raise AssertionError("network must not be touched on path-validation failure")

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-evil",
        required_files=[RequiredFile(relative_path=bad_path, digest=_digest(b"x"))],
        endpoints=[DownloadEndpoint(url="https://x.example.com/")],
    )
    with pytest.raises(OctomilError) as excinfo:
        downloader.download(desc, dest_dir)
    assert excinfo.value.code == ErrorCode.INVALID_INPUT
    # No file or .part may have been written outside dest_dir.
    assert not (dest_dir.parent / "escaped.bin").exists()


def test_rejects_relative_path_through_symlink_escape(tmp_path, cache_dir, dest_dir):
    # Place a symlink inside dest_dir that points outside it. A naive
    # implementation that joined relative_path under dest_dir without
    # resolving symlinks could still write through the link.
    outside = tmp_path / "outside"
    outside.mkdir()
    link = dest_dir / "outlink"
    link.symlink_to(outside, target_is_directory=True)

    def handler(request):
        raise AssertionError("network must not be touched")

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    desc = ArtifactDescriptor(
        artifact_id="art-link",
        required_files=[RequiredFile(relative_path="outlink/escaped.bin", digest=_digest(b"x"))],
        endpoints=[DownloadEndpoint(url="https://x.example.com/")],
    )
    with pytest.raises(OctomilError) as excinfo:
        downloader.download(desc, dest_dir)
    assert excinfo.value.code == ErrorCode.INVALID_INPUT
    assert not (outside / "escaped.bin").exists()


def test_416_retries_same_endpoint_from_zero_when_offset_is_stale(cache_dir, dest_dir):
    payload = b"recovered after 416" * 50
    digest = _digest(payload)
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        seen.append((str(request.url), rng))
        if rng == "bytes=999999-":
            return httpx.Response(416)
        if rng is None:
            return httpx.Response(200, content=payload)
        return httpx.Response(
            206, content=payload, headers={"content-range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}
        )

    downloader = DurableDownloader(cache_dir=cache_dir, client=_client_with(handler))
    # Stale journal — claims we wrote 999999 bytes but the file is much smaller.
    parts = dest_dir / ".parts"
    parts.mkdir()
    (parts / "out.bin.part").write_bytes(b"x" * 999999)
    downloader._journal.record("art-416", "out.bin", 999999, 0)

    desc = ArtifactDescriptor(
        artifact_id="art-416",
        required_files=[RequiredFile(relative_path="out.bin", digest=digest)],
        endpoints=[DownloadEndpoint(url="https://only.example.com/")],
    )

    result = downloader.download(desc, dest_dir)
    assert result.files["out.bin"].read_bytes() == payload
    # The same endpoint URL must have been retried — verifies we didn't just
    # surface the 416 to the next-endpoint loop.
    urls = [u for u, _ in seen]
    assert urls.count(urls[0]) >= 2
    # First request was the stale-range request; a later one had no Range.
    assert seen[0][1] == "bytes=999999-"
    assert any(rng is None for _, rng in seen)


def test_empty_required_files_raises(cache_dir, dest_dir):
    downloader = DurableDownloader(cache_dir=cache_dir)
    desc = ArtifactDescriptor(
        artifact_id="art-noreq",
        required_files=[],
        endpoints=[DownloadEndpoint(url="https://x.example.com/")],
    )
    with pytest.raises(OctomilError):
        downloader.download(desc, dest_dir)
