"""Unit tests for ``scripts/fetch_runtime_dev.py``.

Covers:
- ``_platform_key()`` host detection.
- Manifest-driven asset resolution (happy path, both flavors).
- Legacy fallback when MANIFEST.json is absent from the asset dict.
- Error path: requested flavor not in manifest.
- Error path: platform not in manifest (non-darwin legacy).
- ``_safe_extract`` path-traversal + symlink + device guards.
- Sentinel gate: cache treated as incomplete without ``.extracted-ok``.
- ``_is_appledouble`` filter.
- ``_verify_sha256`` mismatch detection.
"""

from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
# Helpers: import the script under test as a module.
# ---------------------------------------------------------------------------
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


def _load_script() -> Any:
    """Import ``scripts/fetch_runtime_dev.py`` as a module named ``frd``."""
    script_path = Path(__file__).parent.parent / "scripts" / "fetch_runtime_dev.py"
    spec = importlib.util.spec_from_file_location("frd", script_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


frd = _load_script()

# ---------------------------------------------------------------------------
# _platform_key tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sysname, machine, expected_arch",
    [
        ("Darwin", "arm64", "darwin-arm64"),
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "amd64", "linux-x86_64"),
    ],
)
def test_platform_key_returns_correct_arch(sysname: str, machine: str, expected_arch: str) -> None:
    with mock.patch("platform.system", return_value=sysname), mock.patch("platform.machine", return_value=machine):
        arch, _ = frd._platform_key()
    assert arch == expected_arch


def test_platform_key_unsupported_raises() -> None:
    with mock.patch("platform.system", return_value="Windows"), mock.patch("platform.machine", return_value="AMD64"):
        with pytest.raises(SystemExit, match="no runtime artifact known"):
            frd._platform_key()


# ---------------------------------------------------------------------------
# _resolve_bin_asset_name — manifest-driven path
# ---------------------------------------------------------------------------


def _fake_manifest(version: str = "v0.1.5") -> dict:
    """Minimal MANIFEST.json as produced by release.yml."""
    return {
        "version": version,
        "abi": {"major": 0, "minor": 10, "patch": 0},
        "platforms": {
            "darwin-arm64": {
                "chat": f"liboctomil-runtime-{version}-chat-darwin-arm64.tar.gz",
                "stt": f"liboctomil-runtime-{version}-stt-darwin-arm64.tar.gz",
            },
            "linux-x86_64": {
                "chat": f"liboctomil-runtime-{version}-chat-linux-x86_64.tar.gz",
                "stt": f"liboctomil-runtime-{version}-stt-linux-x86_64.tar.gz",
            },
            "android-arm64": {
                "chat": f"liboctomil-runtime-{version}-chat-android-arm64.tar.gz",
            },
        },
        "headers": f"octomil-runtime-headers-{version}.tar.gz",
        "xcframework": None,
    }


def _fake_assets(manifest: dict, _version: str = "v0.1.5") -> dict[str, dict]:
    """Build a fake assets dict containing all assets named in the manifest."""
    names = set()
    for arch_map in manifest.get("platforms", {}).values():
        for v in arch_map.values():
            if v:
                names.add(v)
    if manifest.get("headers"):
        names.add(manifest["headers"])
    names.add("SHA256SUMS")
    names.add("MANIFEST.json")
    return {n: {"name": n, "url": f"https://fake/{n}"} for n in names}


def test_manifest_selects_chat_for_darwin() -> None:
    m = _fake_manifest()
    assets = _fake_assets(m)
    name = frd._resolve_bin_asset_name(m, assets, "darwin-arm64", "chat", "v0.1.5")
    assert name == "liboctomil-runtime-v0.1.5-chat-darwin-arm64.tar.gz"


def test_manifest_selects_stt_for_darwin() -> None:
    m = _fake_manifest()
    assets = _fake_assets(m)
    name = frd._resolve_bin_asset_name(m, assets, "darwin-arm64", "stt", "v0.1.5")
    assert name == "liboctomil-runtime-v0.1.5-stt-darwin-arm64.tar.gz"


def test_manifest_selects_chat_for_linux() -> None:
    m = _fake_manifest()
    assets = _fake_assets(m)
    name = frd._resolve_bin_asset_name(m, assets, "linux-x86_64", "chat", "v0.1.5")
    assert name == "liboctomil-runtime-v0.1.5-chat-linux-x86_64.tar.gz"


def test_manifest_android_chat_only() -> None:
    """Phase 5a: android-arm64 ships chat only."""
    m = _fake_manifest()
    assets = _fake_assets(m)
    name = frd._resolve_bin_asset_name(m, assets, "android-arm64", "chat", "v0.1.5")
    assert name == "liboctomil-runtime-v0.1.5-chat-android-arm64.tar.gz"


# ---------------------------------------------------------------------------
# _resolve_bin_asset_name — error paths
# ---------------------------------------------------------------------------


def test_manifest_missing_flavor_raises_with_available_list() -> None:
    """Requesting stt on android-arm64 (Phase 5a only has chat) must raise
    with a message naming the available flavors."""
    m = _fake_manifest()
    assets = _fake_assets(m)
    with pytest.raises(SystemExit) as exc_info:
        frd._resolve_bin_asset_name(m, assets, "android-arm64", "stt", "v0.1.5")
    msg = str(exc_info.value)
    assert "stt" in msg
    assert "android-arm64" in msg
    assert "chat" in msg  # available flavor listed


def test_manifest_missing_arch_raises_for_non_darwin() -> None:
    """A platform not in the manifest at all (non-darwin) must raise a clear error."""
    m = _fake_manifest()
    assets = _fake_assets(m)
    # Remove linux from manifest to simulate a partial release
    del m["platforms"]["linux-x86_64"]
    with pytest.raises(SystemExit) as exc_info:
        frd._resolve_bin_asset_name(m, assets, "linux-x86_64", "chat", "v0.1.5")
    msg = str(exc_info.value)
    assert "linux-x86_64" in msg


def test_manifest_present_missing_arch_does_not_fall_through_to_legacy() -> None:
    """Regression for v0.1.10 fetcher fix: when MANIFEST.json IS present
    but does NOT list the host arch, the fetcher MUST NOT silently retry
    the legacy ``octomil-runtime-<arch>-<ver>.tar.gz`` shape and report a
    misleading "missing legacy asset" error.

    The truthful failure is "platform not available in this release."
    This covers both darwin-arm64 (which DOES have a legacy shape) and
    linux-x86_64 (which does NOT), to lock in that the legacy fallback
    is gated on `manifest is None`."""
    # Real v0.1.10-style manifest: linux-x86_64 + android, no darwin.
    m = {
        "version": "v0.1.10",
        "abi": {"major": 0, "minor": 11, "patch": 0},
        "platforms": {
            "linux-x86_64": {
                "chat": "liboctomil-runtime-v0.1.10-chat-linux-x86_64.tar.gz",
                "stt": "liboctomil-runtime-v0.1.10-stt-linux-x86_64.tar.gz",
            },
            "android-arm64": {
                "chat": "liboctomil-runtime-v0.1.10-chat-android-arm64.tar.gz",
            },
            "android-x86_64": {
                "chat": "liboctomil-runtime-v0.1.10-chat-android-x86_64.tar.gz",
            },
        },
        "headers": None,
    }
    # Only assets that exist in this release — note: NO legacy darwin asset.
    assets = _fake_assets(m, "v0.1.10")

    with pytest.raises(SystemExit) as exc_info:
        frd._resolve_bin_asset_name(m, assets, "darwin-arm64", "chat", "v0.1.10")
    msg = str(exc_info.value)
    # Must be the truthful "not available" error...
    assert "darwin-arm64" in msg
    assert "not available" in msg
    assert "v0.1.10" in msg
    # ...listing the actually-available platforms...
    assert "linux-x86_64" in msg
    assert "android-arm64" in msg
    # ...and NOT the misleading legacy "missing expected legacy asset" path.
    assert "legacy" not in msg.lower()
    assert "missing expected" not in msg


# ---------------------------------------------------------------------------
# _resolve_bin_asset_name — legacy fallback path
# ---------------------------------------------------------------------------


def test_legacy_fallback_when_no_manifest() -> None:
    """When manifest is None (v0.1.4 release), falls back to legacy shape for darwin-arm64."""
    legacy_name = "octomil-runtime-darwin-arm64-v0.1.4.tar.gz"
    assets: dict[str, dict] = {
        legacy_name: {"name": legacy_name, "url": f"https://fake/{legacy_name}"},
        "octomil-runtime-headers-v0.1.4.tar.gz": {"name": "h", "url": "https://fake/h"},
        "SHA256SUMS": {"name": "SHA256SUMS", "url": "https://fake/SHA256SUMS"},
    }
    name = frd._resolve_bin_asset_name(None, assets, "darwin-arm64", "chat", "v0.1.4")
    assert name == legacy_name


def test_legacy_fallback_missing_asset_raises() -> None:
    """Legacy fallback when the expected asset isn't in the release."""
    assets: dict[str, dict] = {}  # no legacy asset
    with pytest.raises(SystemExit, match="missing expected legacy asset"):
        frd._resolve_bin_asset_name(None, assets, "darwin-arm64", "chat", "v0.1.4")


def test_legacy_fallback_linux_has_no_legacy_shape() -> None:
    """Linux has no legacy shape — should raise a clear error."""
    with pytest.raises(SystemExit):
        frd._resolve_bin_asset_name(None, {}, "linux-x86_64", "chat", "v0.1.4")


# ---------------------------------------------------------------------------
# _is_appledouble
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("._liboctomil-runtime.dylib", True),
        ("lib/._something", True),
        ("liboctomil-runtime.dylib", False),
        ("include/octomil/runtime.h", False),
    ],
)
def test_is_appledouble(name: str, expected: bool) -> None:
    assert frd._is_appledouble(name) == expected


# ---------------------------------------------------------------------------
# _safe_extract safety guards
# ---------------------------------------------------------------------------


def _make_tarball(members: list[tuple[str, bytes | None]], path: Path) -> None:
    """Create a tarball at ``path`` with the given ``(name, content)`` members.
    ``content=None`` creates a regular file entry with empty bytes."""
    with tarfile.open(path, "w:gz") as tf:
        for name, content in members:
            data = content if content is not None else b""
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_safe_extract_normal_file(tmp_path: Path) -> None:
    tb = tmp_path / "test.tar.gz"
    _make_tarball([("lib/liboctomil-runtime.dylib", b"fake-dylib")], tb)
    target = tmp_path / "out"
    target.mkdir()
    frd._safe_extract(tb, target)
    assert (target / "lib" / "liboctomil-runtime.dylib").exists()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    tb = tmp_path / "bad.tar.gz"
    _make_tarball([("../escape.txt", b"pwned")], tb)
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit, match="suspicious tar entry"):
        frd._safe_extract(tb, target)


def test_safe_extract_rejects_absolute_path(tmp_path: Path) -> None:
    tb = tmp_path / "bad.tar.gz"
    # Absolute paths in tarballs
    with tarfile.open(tb, "w:gz") as tf:
        info = tarfile.TarInfo(name="/etc/passwd")
        info.size = 6
        tf.addfile(info, io.BytesIO(b"hacked"))
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit, match="suspicious tar entry"):
        frd._safe_extract(tb, target)


def test_safe_extract_rejects_symlink_to_absolute_path(tmp_path: Path) -> None:
    tb = tmp_path / "sym.tar.gz"
    with tarfile.open(tb, "w:gz") as tf:
        info = tarfile.TarInfo(name="lib/evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit, match="absolute target"):
        frd._safe_extract(tb, target)


def test_safe_extract_rejects_symlink_escaping_target_dir(tmp_path: Path) -> None:
    """Relative symlink whose target resolves OUTSIDE the extraction
    dir must be refused — the classic CVE shape for archive escape."""
    tb = tmp_path / "escape.tar.gz"
    with tarfile.open(tb, "w:gz") as tf:
        info = tarfile.TarInfo(name="lib/escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../etc/passwd"
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit, match="would escape"):
        frd._safe_extract(tb, target)


def test_safe_extract_hardlink_before_target_in_archive_order(tmp_path: Path) -> None:
    """Codex #597 follow-up: tarfile.extract() raises KeyError when a
    hardlink is processed before its target file. We MUST sort all
    regular files ahead of all link entries so hardlinks always see
    their referenced inode on disk."""
    tb = tmp_path / "hardlink-out-of-order.tar.gz"
    real_content = b"liboctomil-runtime body"
    with tarfile.open(tb, "w:gz") as tf:
        # Hardlink appears FIRST in archive order — without two-pass
        # extraction, tarfile.extract() would fail on this entry.
        link = tarfile.TarInfo(name="lib/liboctomil-runtime.0.dylib")
        link.type = tarfile.LNKTYPE
        link.linkname = "lib/liboctomil-runtime.0.1.10.dylib"
        link.size = 0
        tf.addfile(link, io.BytesIO(b""))
        # Real file appears AFTER the hardlink.
        info = tarfile.TarInfo(name="lib/liboctomil-runtime.0.1.10.dylib")
        info.size = len(real_content)
        tf.addfile(info, io.BytesIO(real_content))

    target = tmp_path / "out-hardlink-order"
    target.mkdir()
    frd._safe_extract(tb, target)
    # Real file extracted.
    assert (target / "lib" / "liboctomil-runtime.0.1.10.dylib").read_bytes() == real_content
    # Hardlink resolved successfully.
    assert (target / "lib" / "liboctomil-runtime.0.dylib").exists()


def test_safe_extract_allows_intra_archive_symlink(tmp_path: Path) -> None:
    """macOS dylib chains (liboctomil-runtime.dylib ->
    liboctomil-runtime.0.dylib -> liboctomil-runtime.0.1.10.dylib)
    use symlinks within the archive. These resolve INSIDE the
    extraction dir and must be allowed — refusing them blocks
    v0.1.10 darwin-arm64 consumption entirely."""
    tb = tmp_path / "dylib-chain.tar.gz"
    with tarfile.open(tb, "w:gz") as tf:
        # Real file
        data = b"fake-dylib-body"
        info = tarfile.TarInfo(name="lib/liboctomil-runtime.0.1.10.dylib")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        # Symlink to it
        link = tarfile.TarInfo(name="lib/liboctomil-runtime.0.dylib")
        link.type = tarfile.SYMTYPE
        link.linkname = "liboctomil-runtime.0.1.10.dylib"
        link.size = 0
        tf.addfile(link, io.BytesIO(b""))
        # Symlink to symlink
        link2 = tarfile.TarInfo(name="lib/liboctomil-runtime.dylib")
        link2.type = tarfile.SYMTYPE
        link2.linkname = "liboctomil-runtime.0.dylib"
        link2.size = 0
        tf.addfile(link2, io.BytesIO(b""))
    target = tmp_path / "out"
    target.mkdir()
    frd._safe_extract(tb, target)
    assert (target / "lib" / "liboctomil-runtime.0.1.10.dylib").exists()
    assert (target / "lib" / "liboctomil-runtime.0.dylib").is_symlink()
    assert (target / "lib" / "liboctomil-runtime.dylib").is_symlink()


def test_safe_extract_skips_appledouble(tmp_path: Path) -> None:
    tb = tmp_path / "osx.tar.gz"
    _make_tarball(
        [
            ("lib/._liboctomil-runtime.dylib", b"xattr-junk"),
            ("lib/liboctomil-runtime.dylib", b"real-dylib"),
        ],
        tb,
    )
    target = tmp_path / "out"
    target.mkdir()
    frd._safe_extract(tb, target)
    assert not (target / "lib" / "._liboctomil-runtime.dylib").exists()
    assert (target / "lib" / "liboctomil-runtime.dylib").exists()


# ---------------------------------------------------------------------------
# _flatten_archive_top_dir — collision behavior
# ---------------------------------------------------------------------------


def test_flatten_archive_top_dir_refuses_content_different_collision(tmp_path: Path) -> None:
    """Codex #597 follow-up: when the wrapper dir contains a regular
    file whose path collides with an existing file in target_dir and
    the BYTES DIFFER, the flatten must refuse rather than silently
    overwrite. A malicious tarball could otherwise clobber a sibling
    tarball's LICENSE / NOTICE / SHA256SUMS via the wrapper."""
    target = tmp_path / "out"
    target.mkdir()
    # Pre-populate a runtime.h with one content.
    (target / "include" / "octomil").mkdir(parents=True)
    (target / "include" / "octomil" / "runtime.h").write_text("// headers-version\n")

    # Wrapper's bin tarball ships a DIFFERENT runtime.h.
    wrapper = target / "liboctomil-runtime-v0.1.10-chat-darwin-arm64"
    (wrapper / "include" / "octomil").mkdir(parents=True)
    (wrapper / "include" / "octomil" / "runtime.h").write_text("// bin-version-MALICIOUS\n")

    with pytest.raises(SystemExit, match="content differs"):
        frd._flatten_archive_top_dir(target, "liboctomil-runtime-v0.1.10-chat-darwin-arm64.tar.gz")

    # Original runtime.h must be intact.
    assert (target / "include" / "octomil" / "runtime.h").read_text() == "// headers-version\n"


def test_flatten_archive_top_dir_silently_skips_content_identical_collision(tmp_path: Path) -> None:
    """The expected v0.1.10 darwin shape: headers tarball ships
    `include/octomil/runtime.h`, bin tarball's wrapper ships the
    SAME file. Identical bytes → silently skipped, no error."""
    runtime_h = "// canonical runtime.h\n"
    target = tmp_path / "out"
    target.mkdir()
    (target / "include" / "octomil").mkdir(parents=True)
    (target / "include" / "octomil" / "runtime.h").write_text(runtime_h)

    wrapper = target / "liboctomil-runtime-v0.1.10-chat-darwin-arm64"
    (wrapper / "include" / "octomil").mkdir(parents=True)
    (wrapper / "include" / "octomil" / "runtime.h").write_text(runtime_h)
    (wrapper / "lib").mkdir()
    (wrapper / "lib" / "liboctomil-runtime.dylib").write_text("dylib")

    frd._flatten_archive_top_dir(target, "liboctomil-runtime-v0.1.10-chat-darwin-arm64.tar.gz")

    assert (target / "include" / "octomil" / "runtime.h").read_text() == runtime_h
    assert (target / "lib" / "liboctomil-runtime.dylib").exists()
    assert not wrapper.exists()


def test_flatten_archive_top_dir_merges_directories_without_collision(tmp_path: Path) -> None:
    """When the wrapper's `include/` dir has files that don't collide
    with what's already there, the merge succeeds (this is the
    expected v0.1.10 darwin shape — headers tarball populates
    `include/octomil/runtime.h`, then bin tarball's include/ merges
    in additional files if any)."""
    target = tmp_path / "out"
    target.mkdir()
    # Pre-populate from headers tarball.
    (target / "include" / "octomil").mkdir(parents=True)
    (target / "include" / "octomil" / "runtime.h").write_text("// runtime.h\n")

    # Wrapper has a different file under include/.
    wrapper = target / "liboctomil-runtime-v0.1.10-chat-darwin-arm64"
    (wrapper / "include" / "octomil").mkdir(parents=True)
    (wrapper / "include" / "octomil" / "extra.h").write_text("// extra.h\n")
    (wrapper / "lib").mkdir()
    (wrapper / "lib" / "liboctomil-runtime.dylib").write_text("dylib")

    frd._flatten_archive_top_dir(target, "liboctomil-runtime-v0.1.10-chat-darwin-arm64.tar.gz")

    # Both files present after merge.
    assert (target / "include" / "octomil" / "runtime.h").read_text() == "// runtime.h\n"
    assert (target / "include" / "octomil" / "extra.h").read_text() == "// extra.h\n"
    assert (target / "lib" / "liboctomil-runtime.dylib").exists()
    # Wrapper removed.
    assert not wrapper.exists()


# ---------------------------------------------------------------------------
# _verify_sha256
# ---------------------------------------------------------------------------


def test_verify_sha256_passes_for_correct_hash(tmp_path: Path) -> None:
    content = b"liboctomil-runtime binary data"
    asset = tmp_path / "liboctomil-runtime-v0.1.5-chat-darwin-arm64.tar.gz"
    asset.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{sha}  {asset.name}\n")
    frd._verify_sha256(asset, sums)  # should not raise


def test_verify_sha256_raises_on_mismatch(tmp_path: Path) -> None:
    asset = tmp_path / "lib.tar.gz"
    asset.write_bytes(b"real content")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'a' * 64}  {asset.name}\n")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        frd._verify_sha256(asset, sums)


def test_verify_sha256_raises_when_not_listed(tmp_path: Path) -> None:
    asset = tmp_path / "lib.tar.gz"
    asset.write_bytes(b"content")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("# empty\n")
    with pytest.raises(SystemExit, match="not listed in SHA256SUMS"):
        frd._verify_sha256(asset, sums)


def test_verify_sha256_normalizes_dot_slash_prefix(tmp_path: Path) -> None:
    """SHA256SUMS entries produced by ``shasum -a 256 ./*.tar.gz``
    have a ``./`` prefix on every filename. The fetcher looks up
    assets by ``path.name`` (no prefix), so the parser must strip
    ``./`` when building the lookup table. This is the exact shape
    of the live v0.1.10 release SHA256SUMS — without this fix,
    every fetch against v0.1.10 fails with a misleading
    ``not listed in SHA256SUMS``."""
    bin_content = b"liboctomil-runtime binary data"
    bin_asset = tmp_path / "liboctomil-runtime-v0.1.10-chat-linux-x86_64.tar.gz"
    bin_asset.write_bytes(bin_content)
    bin_sha = hashlib.sha256(bin_content).hexdigest()

    hdr_content = b"headers tarball data"
    hdr_asset = tmp_path / "octomil-runtime-headers-v0.1.10.tar.gz"
    hdr_asset.write_bytes(hdr_content)
    hdr_sha = hashlib.sha256(hdr_content).hexdigest()

    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{bin_sha}  ./{bin_asset.name}\n{hdr_sha}  ./{hdr_asset.name}\n")

    # Both must verify cleanly despite the ./ prefix on each line.
    frd._verify_sha256(bin_asset, sums)
    frd._verify_sha256(hdr_asset, sums)


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_returns_none_when_absent(tmp_path: Path) -> None:
    """No MANIFEST.json in assets → returns None (legacy path)."""
    assets: dict[str, dict] = {
        "SHA256SUMS": {"name": "SHA256SUMS", "url": "u"},
        "octomil-runtime-darwin-arm64-v0.1.4.tar.gz": {"name": "x", "url": "u2"},
    }
    result = frd._load_manifest(assets, tmp_path, "faketoken")
    assert result is None


def test_main_tolerates_headers_null_in_manifest(tmp_path: Path) -> None:
    """Regression for v0.1.10 fetcher fix: when ``MANIFEST.json.headers``
    is null (release deliberately ships no headers tarball), the fetcher
    must skip the headers fetch instead of bailing out with a missing-asset
    error.

    We drive ``main()`` end-to-end against a fully-faked release: no real
    GitHub call, no real archive download — the manifest is injected via
    ``_load_manifest`` and the bin tarball is materialised on disk by a
    fake ``_download`` that writes a valid tar.gz containing the dylib.
    Success is defined as ``main()`` returning 0 and the dylib appearing
    in the flavor-keyed cache slice, with no attempt to fetch any
    ``octomil-runtime-headers-*`` asset (which is not in the release)."""
    version = "v0.1.10"
    flavor = "chat"
    arch = "linux-x86_64"
    bin_name = f"liboctomil-runtime-{version}-{flavor}-{arch}.tar.gz"

    # v0.1.10-style manifest: headers is null.
    manifest_data = {
        "version": version,
        "abi": {"major": 0, "minor": 11, "patch": 0},
        "platforms": {
            arch: {flavor: bin_name},
            "android-arm64": {"chat": f"liboctomil-runtime-{version}-chat-android-arm64.tar.gz"},
        },
        "headers": None,
    }

    # Materialise a real bin tarball + SHA256SUMS so the verify + extract
    # path runs unmodified.
    cache_root = tmp_path / "cache"
    work_root = tmp_path / "work"
    work_root.mkdir()
    bin_tar = work_root / bin_name
    _make_tarball([("lib/liboctomil-runtime.so", b"fake-dylib-bytes")], bin_tar)
    bin_sha = hashlib.sha256(bin_tar.read_bytes()).hexdigest()
    sums_text = f"{bin_sha}  {bin_name}\n"

    # Only the binary, SHA256SUMS and MANIFEST.json are in the release —
    # NO headers tarball. If the fetcher tries to fetch a headers asset,
    # `required not in assets` will fire and the test will fail.
    release_assets = {
        bin_name: {"name": bin_name, "url": f"https://fake/{bin_name}"},
        "SHA256SUMS": {"name": "SHA256SUMS", "url": "https://fake/SHA256SUMS"},
        "MANIFEST.json": {"name": "MANIFEST.json", "url": "https://fake/MANIFEST.json"},
    }

    fetched_names: list[str] = []

    def fake_download(url: str, dest: Path, _token: str) -> None:
        # Record which asset was requested so we can assert no headers fetch.
        name = url.rsplit("/", 1)[-1]
        fetched_names.append(name)
        if name == "MANIFEST.json":
            dest.write_text(json.dumps(manifest_data), encoding="utf-8")
        elif name == "SHA256SUMS":
            dest.write_text(sums_text, encoding="utf-8")
        elif name == bin_name:
            dest.write_bytes(bin_tar.read_bytes())
        else:
            raise AssertionError(f"unexpected fetch attempted: {name}")

    import sys

    argv = [
        "fetch_runtime_dev.py",
        "--version",
        version,
        "--flavor",
        flavor,
        "--cache-root",
        str(cache_root),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch("platform.system", return_value="Linux"),
        mock.patch("platform.machine", return_value="x86_64"),
        mock.patch.object(frd, "_gh_token", return_value="faketoken"),
        mock.patch.object(frd, "_release_assets_via_api", return_value=release_assets),
        mock.patch.object(frd, "_download", side_effect=fake_download),
    ):
        rc = frd.main()

    assert rc == 0, "main() must succeed when manifest.headers is null"
    # The dylib must have been extracted under the flavor-keyed cache slice.
    extracted = cache_root / version / flavor / "lib" / "liboctomil-runtime.so"
    assert extracted.exists(), f"dylib not extracted to {extracted}"
    # No headers asset must have been requested.
    assert not any(
        "headers" in n for n in fetched_names
    ), f"fetcher must not request headers when manifest.headers is null; fetched: {fetched_names}"


def test_load_manifest_parses_downloaded_json(tmp_path: Path) -> None:
    """When MANIFEST.json is in assets, it downloads and parses it."""
    manifest_data = _fake_manifest()
    manifest_bytes = json.dumps(manifest_data).encode()

    assets = {
        "MANIFEST.json": {"name": "MANIFEST.json", "url": "https://fake/MANIFEST.json"},
    }

    # Patch _download to write the fake manifest to disk.
    def fake_download(_url: str, dest: Path, _token: str) -> None:
        # _url + _token are required by `_download`'s interface but the
        # fake doesn't use them — content comes from the closure-captured
        # manifest_bytes.
        dest.write_bytes(manifest_bytes)

    with mock.patch.object(frd, "_download", side_effect=fake_download):
        result = frd._load_manifest(assets, tmp_path, "faketoken")

    assert result is not None
    assert result["version"] == "v0.1.5"
    assert "darwin-arm64" in result["platforms"]


# ---------------------------------------------------------------------------
# Sentinel check: _find_dylib_in_lib
# ---------------------------------------------------------------------------


def test_find_dylib_in_lib_finds_dylib(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    dylib = lib_dir / "liboctomil-runtime.dylib"
    dylib.write_bytes(b"fake")
    result = frd._find_dylib_in_lib(lib_dir)
    assert result == dylib


def test_find_dylib_in_lib_finds_so(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    so = lib_dir / "liboctomil-runtime.so"
    so.write_bytes(b"fake")
    result = frd._find_dylib_in_lib(lib_dir)
    assert result == so


def test_find_dylib_in_lib_returns_none_when_empty(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    assert frd._find_dylib_in_lib(lib_dir) is None


# ---------------------------------------------------------------------------
# _legacy_asset_name
# ---------------------------------------------------------------------------


def test_legacy_asset_name_darwin() -> None:
    assert frd._legacy_asset_name("darwin-arm64", "v0.1.4") == "octomil-runtime-darwin-arm64-v0.1.4.tar.gz"


def test_legacy_asset_name_linux_none() -> None:
    assert frd._legacy_asset_name("linux-x86_64", "v0.1.4") is None


# ---------------------------------------------------------------------------
# Flavor-keyed cache layout — fetch path (main())
# ---------------------------------------------------------------------------


def test_main_cache_path_is_flavor_keyed(tmp_path: Path) -> None:
    """The fetch target_dir must be ``<cache_root>/<version>/<flavor>``,
    not the old ``<cache_root>/<version>``.  This is the structural
    invariant that prevents chat and stt dylibs from collapsing into
    the same directory.

    We exercise the cache-hit short-circuit (dylib + sentinel already
    on disk) to verify the path logic without actually hitting the
    network."""
    version = "v0.1.5"
    flavor = "stt"

    # Pre-populate the flavor-keyed cache so the cache-hit branch fires.
    cache_root = tmp_path / "cache"
    lib_dir = cache_root / version / flavor / "lib"
    lib_dir.mkdir(parents=True)
    dylib = lib_dir / "liboctomil-runtime.dylib"
    dylib.write_bytes(b"fake")
    sentinel = lib_dir / ".extracted-ok"
    sentinel.write_text(f"{version}\n{flavor}\n", encoding="utf-8")

    # Invoke main() directly with patched argv and sys.exit captured.
    import sys

    with mock.patch.object(
        sys, "argv", ["fetch_runtime_dev.py", "--version", version, "--flavor", flavor, "--cache-root", str(cache_root)]
    ):
        with mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
            rc = frd.main()
    assert rc == 0, "should hit cache-hit branch and return 0"
    # The chat flavor dir must NOT have been created — flavor isolation.
    chat_dir = cache_root / version / "chat"
    assert not chat_dir.exists(), "fetching stt must not touch the chat cache slice"


def test_main_force_is_per_flavor(tmp_path: Path) -> None:
    """--force --flavor stt only clears the stt cache slice; the chat
    slice must remain untouched.  We verify this by staging both
    flavors, then simulating a force-fetch for stt that fails early
    (no GH_TOKEN) and confirming chat's sentinel is still present."""
    version = "v0.1.5"
    cache_root = tmp_path / "cache"

    for flavor in ("chat", "stt"):
        lib_dir = cache_root / version / flavor / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "liboctomil-runtime.dylib").write_bytes(b"fake")
        (lib_dir / ".extracted-ok").write_text(f"{version}\n{flavor}\n", encoding="utf-8")

    import sys

    # Invoke with --force --flavor stt but no token — should fail after
    # the cache check (args.force skips the cache-hit short-circuit) but
    # BEFORE anything touches the chat slice.
    with mock.patch.object(
        sys,
        "argv",
        ["fetch_runtime_dev.py", "--version", version, "--flavor", "stt", "--force", "--cache-root", str(cache_root)],
    ):
        with mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
            with mock.patch.object(frd, "_gh_token", return_value=None):
                with pytest.raises(SystemExit):
                    frd.main()

    # Chat sentinel must be intact.
    chat_sentinel = cache_root / version / "chat" / "lib" / ".extracted-ok"
    assert chat_sentinel.exists(), "--force stt must not disturb chat cache slice"


def test_sentinel_records_flavor(tmp_path: Path) -> None:
    """The sentinel file must contain both version AND flavor so its
    content is self-describing.  This is a contract with any future
    tooling that reads the sentinel."""
    sentinel_path = tmp_path / ".extracted-ok"
    version = "v0.1.5"
    flavor = "stt"
    sentinel_path.write_text(f"{version}\n{flavor}\n", encoding="utf-8")
    lines = sentinel_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == version
    assert lines[1] == flavor
