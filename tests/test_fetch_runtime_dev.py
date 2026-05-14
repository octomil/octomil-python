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


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    tb = tmp_path / "sym.tar.gz"
    with tarfile.open(tb, "w:gz") as tf:
        info = tarfile.TarInfo(name="lib/evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit, match="link entry"):
        frd._safe_extract(tb, target)


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
