from __future__ import annotations

import hashlib
from pathlib import Path

from octomil.runtime.lifecycle.artifact_cache import ArtifactCache, ArtifactEntry


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_put_get_roundtrip_with_hostile_artifact_id(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    src = tmp_path / "m.bin"
    src.write_bytes(b"data")
    dest = cache.put("..", _digest(b"data"), src)
    assert (tmp_path / "cache").resolve() in dest.resolve().parents
    assert cache.get("..", _digest(b"data")) == dest


def test_distinct_ids_do_not_collide(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    src = tmp_path / "m.bin"
    src.write_bytes(b"data")
    a = cache.put("a/b", _digest(b"data"), src)
    b = cache.put("a_b", _digest(b"data"), src)
    assert a.parent != b.parent


def test_manifest_path_escape_is_not_deleted(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("x")
    cache = ArtifactCache(tmp_path / "cache")
    cache.manifest.entries["k::d"] = ArtifactEntry("k", "d", "../victim", 1, 0.0, 0.0)
    assert cache.remove("k", "d") is True
    assert (victim / "keep").exists()

    cache.manifest.entries["k::d"] = ArtifactEntry("k", "d", "../victim", 1, 0.0, 0.0)
    assert cache.get("k", "d") is None
    assert (victim / "keep").exists()


def test_manifest_written_atomically(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    src = tmp_path / "m.bin"
    src.write_bytes(b"data")
    cache.put("m", _digest(b"data"), src)
    assert not list((tmp_path / "cache").glob(".manifest-*"))
    assert ArtifactCache(tmp_path / "cache").get("m", _digest(b"data")) is not None


def test_filename_from_url_never_returns_dot_segments() -> None:
    from octomil.runtime.lifecycle.download import _filename_from_url

    assert _filename_from_url("https://h/a/..", "art") not in ("..", ".", "")
    assert _filename_from_url("https://h/a/.", "../x") not in ("..", ".", "")
    assert _filename_from_url("https://h/a/model.gguf", "art") == "model.gguf"
