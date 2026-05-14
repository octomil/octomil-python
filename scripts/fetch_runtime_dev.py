#!/usr/bin/env python3
"""Fetch a dev-only ``liboctomil-runtime`` release from the private
``octomil/octomil-runtime`` repo and unpack it into the local cache.

This is the supported path for local + CI dev environments that
need the dylib without a source build. Production / customer
distribution will eventually use signed-and-notarized binaries;
this script is for the dev range only.

Resolution:
  * Reads ``$GH_TOKEN`` / ``$GITHUB_TOKEN`` / ``$OCTOMIL_RUNTIME_TOKEN``
    for private-repo auth.  Falls back to ``gh auth token`` if available.
  * For v0.1.5+ releases that carry a ``MANIFEST.json`` asset:
    - Downloads ``MANIFEST.json``, parses the ``platforms`` map.
    - Resolves the asset name for ``(arch, --flavor)`` without
      guessing filenames.
    - Falls back to the legacy name shape if the requested
      ``(arch, flavor)`` pair is absent from the manifest.
  * For v0.1.4 and earlier (no ``MANIFEST.json``):
    - Falls back to the legacy name shape:
      ``octomil-runtime-<arch>-<ver>.tar.gz`` (darwin-arm64 only).
  * Downloads the platform tarball, ``octomil-runtime-headers-<ver>.tar.gz``,
    and ``SHA256SUMS``.
  * Verifies sha256 for each asset.
  * Extracts to ``~/.cache/octomil-runtime/<version>/lib`` and
    ``~/.cache/octomil-runtime/<version>/include``.
  * Prints the resolved dylib path.

Supported platforms (v0.1.5+ manifest-driven):
  - darwin-arm64  (chat, stt)
  - linux-x86_64  (chat, stt)
  - android-arm64 (chat only — Phase 5a)

Once extracted, the cffi loader picks the dylib up automatically
via ``octomil.runtime.native.loader._fetched_dylib_candidates()``;
no env var needed for the default-version case. Operators that
want a specific path use ``OCTOMIL_RUNTIME_DYLIB``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "octomil/octomil-runtime"
DEFAULT_VERSION = "v0.1.4"  # v0.1.4: native embeddings.text + chat-template control-token cleanup (OCT_EVENT_EMBEDDING_VECTOR + LlamaCppEmbeddingsSession + per-context pooling-type gate)
CACHE_ROOT = Path.home() / ".cache" / "octomil-runtime"
MANIFEST_ASSET_NAME = "MANIFEST.json"

VALID_FLAVORS = ("chat", "stt")


def _gh_token() -> str | None:
    for env in ("GH_TOKEN", "GITHUB_TOKEN", "OCTOMIL_RUNTIME_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    if shutil.which("gh"):
        try:
            out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _platform_key() -> tuple[str, str]:
    """Return the ``(arch, os_name)`` string for the running host.

    The arch component matches the field names used in ``MANIFEST.json``'s
    ``platforms`` map (e.g. ``"darwin-arm64"``, ``"linux-x86_64"``).

    Android is not a realistic host for Python callers; it's listed here
    for completeness so manifest lookups work in test stubs.

    Raises ``SystemExit`` if the host platform has no known mapping."""
    sysname = platform.system()
    machine = platform.machine()
    if sysname == "Darwin" and machine == "arm64":
        return ("darwin-arm64", sysname)
    if sysname == "Linux" and machine in ("x86_64", "amd64"):
        return ("linux-x86_64", sysname)
    # Android (hypothetical Python-on-Android path)
    if sysname == "Linux" and machine == "aarch64":
        # Distinguish Android from Linux arm64 via /proc/sys/kernel/osrelease
        try:
            osrelease = Path("/proc/sys/kernel/osrelease").read_text().lower()
        except OSError:
            osrelease = ""
        if "android" in osrelease:
            return ("android-arm64", sysname)
    raise SystemExit(
        f"error: no runtime artifact known for {sysname}/{machine}.\n"
        f"Supported platforms: darwin-arm64, linux-x86_64, android-arm64.\n"
        f"See octomil-runtime release notes."
    )


def _legacy_asset_name(arch: str, version: str) -> str | None:
    """Return the legacy (pre-v0.1.5) asset name for ``arch``, or ``None``
    if no legacy shape exists for that arch.

    Legacy shape: ``octomil-runtime-<arch>-<ver>.tar.gz``.
    Only ``darwin-arm64`` was shipped in the v0.1.x legacy line."""
    if arch == "darwin-arm64":
        return f"octomil-runtime-darwin-arm64-{version}.tar.gz"
    return None


def _download(url: str, dest: Path, token: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            "User-Agent": "octomil-python/fetch_runtime_dev.py",
        },
    )
    print(f"  download {url}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(
            f"HTTP {e.code} fetching {url}\nResponse: {body}\n"
            f"If this is a 404 or 401, confirm your token has read access "
            f"to the private octomil/octomil-runtime repo."
        ) from e


def _release_assets_via_api(version: str, token: str) -> dict[str, dict]:
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{version}"
    req = urllib.request.Request(
        api,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "octomil-python/fetch_runtime_dev.py",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return {a["name"]: a for a in data.get("assets", [])}


def _load_manifest(assets: dict[str, dict], work: Path, token: str) -> dict | None:
    """Download and parse ``MANIFEST.json`` from the release assets dict.

    Returns the parsed manifest dict if ``MANIFEST.json`` is present in
    ``assets``, or ``None`` when it is absent (legacy v0.1.4 releases).

    The manifest schema produced by ``octomil-runtime/.github/workflows/
    release.yml`` (lines ~780-828) is:

    .. code-block:: json

        {
          "version": "v0.1.5",
          "abi": {"major": 0, "minor": 10, "patch": 0},
          "platforms": {
            "darwin-arm64": {
              "chat": "liboctomil-runtime-v0.1.5-chat-darwin-arm64.tar.gz",
              "stt":  "liboctomil-runtime-v0.1.5-stt-darwin-arm64.tar.gz"
            },
            ...
          },
          "headers": "octomil-runtime-headers-v0.1.5.tar.gz",
          "xcframework": {"chat": "...", "stt": "..."} | null
        }

    Only platform-flavor entries whose value is non-null are present."""
    if MANIFEST_ASSET_NAME not in assets:
        return None
    dest = work / MANIFEST_ASSET_NAME
    _download(assets[MANIFEST_ASSET_NAME]["url"], dest, token)
    with dest.open("rb") as fh:
        return json.load(fh)


def _resolve_bin_asset_name(
    manifest: dict | None,
    assets: dict[str, dict],
    arch: str,
    flavor: str,
    version: str,
) -> str:
    """Return the binary tarball asset name for ``(arch, flavor)``.

    Resolution order:
    1. Manifest-driven: ``manifest["platforms"][arch][flavor]``.
    2. Legacy fallback: ``_legacy_asset_name(arch, version)`` when the
       manifest is absent or the arch/flavor pair is missing.

    Raises ``SystemExit`` with a clear message when neither path works."""
    if manifest is not None:
        platforms: dict = manifest.get("platforms", {})
        if arch in platforms:
            arch_map: dict = platforms[arch]
            if flavor in arch_map:
                name = arch_map[flavor]
                if name is not None:
                    return name
            # Arch present but requested flavor absent.
            available = sorted(k for k, v in arch_map.items() if v is not None)
            raise SystemExit(
                f"error: flavor {flavor!r} not available for {arch} in this release.\n"
                f"Available flavors for {arch}: {available}\n"
                f"Use --flavor to pick one of the available flavors."
            )
        # Arch not in manifest at all — fall through to legacy for darwin-arm64,
        # hard error for anything else (no legacy shape exists).
        if arch != "darwin-arm64":
            available_arches = sorted(platforms.keys())
            raise SystemExit(
                f"error: platform {arch!r} not available in this release.\n"
                f"Available platforms: {available_arches}\n"
                f"Check octomil-runtime release notes."
            )

    # Legacy fallback (no manifest, or arch not in manifest for darwin-arm64).
    legacy = _legacy_asset_name(arch, version)
    if legacy is not None and legacy in assets:
        return legacy
    if legacy is not None:
        raise SystemExit(
            f"error: release {version} missing expected legacy asset {legacy!r}.\n"
            f"Confirm the tag exists and the token has access."
        )
    raise SystemExit(
        f"error: no asset found for {arch}/{flavor} in release {version}.\n"
        f"This release may not include a runtime artifact for your platform.\n"
        f"See octomil-runtime release notes."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_appledouble(name: str) -> bool:
    """macOS xattr ``._*`` files in tar bundles. Filter out so they
    don't pollute the dev cache. (Bundles are produced on the
    runtime-repo CI runner; the AppleDouble entries appear when the
    tarball is built from a working tree that has xattrs.)"""
    base = name.rsplit("/", 1)[-1]
    return base.startswith("._")


def _safe_extract(tarball: Path, target_dir: Path) -> None:
    """Extract ``tarball`` into ``target_dir`` with the same safety
    properties as Python 3.12's ``filter="data"`` (which we can't
    use because the package floor is 3.9). Refuses:

      * path-traversal / absolute paths,
      * symlinks (any),
      * hardlinks (any),
      * character / block / fifo device entries,
      * any link target that escapes ``target_dir``.

    Filters out macOS AppleDouble (``._*``) metadata. The link-target
    refusal is intentionally strict: SDK dev artifacts don't need
    them and allowing one creates an arbitrary-write primitive. Codex
    R2 blocker fix."""
    target_real = target_dir.resolve()
    with tarfile.open(tarball) as tf:
        for member in tf.getmembers():
            mname = member.name
            if _is_appledouble(mname):
                continue
            if mname.startswith("/") or ".." in Path(mname).parts:
                raise SystemExit(f"error: refusing to extract suspicious tar entry {mname!r} from {tarball.name}")
            if member.issym() or member.islnk():
                raise SystemExit(
                    f"error: refusing to extract link entry {mname!r} "
                    f"(symlinks/hardlinks not allowed in dev artifacts)."
                )
            if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                raise SystemExit(f"error: refusing to extract device entry {mname!r} from {tarball.name}")
            # Final safety: confirm the resolved destination stays
            # inside target_dir even if the member's name passes the
            # textual checks above.
            dest = (target_dir / mname).resolve()
            try:
                dest.relative_to(target_real)
            except ValueError as e:
                raise SystemExit(f"error: tar entry {mname!r} would escape {target_dir} on resolution") from e
            tf.extract(member, target_dir)


def _verify_sha256(path: Path, sums_file: Path) -> None:
    expected: dict[str, str] = {}
    with sums_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]
    name = path.name
    if name not in expected:
        raise SystemExit(f"error: {name} not listed in SHA256SUMS")
    got = _sha256(path)
    if got != expected[name]:
        raise SystemExit(
            f"error: sha256 mismatch for {name}\n"
            f"  expected: {expected[name]}\n"
            f"  got:      {got}\n"
            f"Refuse to extract a corrupt or tampered artifact."
        )


def _dylib_extensions() -> list[str]:
    """Return candidate dylib extensions for the running OS."""
    if platform.system() == "Darwin":
        return [".dylib"]
    return [".so"]


def _find_dylib_in_lib(lib_dir: Path) -> Path | None:
    """Return the first ``liboctomil-runtime.*`` dylib found in ``lib_dir``,
    or ``None`` if none exist. Handles both ``.dylib`` (macOS) and ``.so``
    (Linux/Android)."""
    for ext in _dylib_extensions():
        candidate = lib_dir / f"liboctomil-runtime{ext}"
        if candidate.exists():
            return candidate
    # Wildcard fallback for unexpected extension variants.
    for p in lib_dir.glob("liboctomil-runtime*"):
        if p.is_file() and not p.name.startswith("._"):
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=DEFAULT_VERSION, help=f"release tag (default {DEFAULT_VERSION})")
    p.add_argument("--cache-root", default=str(CACHE_ROOT), help=argparse.SUPPRESS)
    p.add_argument(
        "--flavor",
        choices=VALID_FLAVORS,
        default="chat",
        help=(
            "runtime flavor to fetch: 'chat' (default) ships llama.cpp-based "
            "chat + embeddings; 'stt' ships whisper.cpp-based speech-to-text. "
            "Phase 5a: chat is available on all platforms; stt is darwin-arm64 / "
            "linux-x86_64 only."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-extract even if the cache is populated",
    )
    args = p.parse_args()

    version = args.version
    flavor: str = args.flavor
    cache_root = Path(args.cache_root).expanduser()
    target_dir = cache_root / version
    lib_dir = target_dir / "lib"
    inc_dir = target_dir / "include"

    # Codex R2 missed-case fix: a previous partial run could leave
    # a dylib on disk without the sentinel (e.g. mid-write crash).
    # We check for the sentinel file written ONLY after a complete
    # extraction succeeds. If absent, treat the cache as suspect and
    # re-fetch.
    sentinel = lib_dir / ".extracted-ok"

    # Resolve the expected dylib path for a quick cache-hit check.
    # We do this before hitting the network. Note: on Linux the
    # extension is .so, not .dylib.
    arch, _sysname = _platform_key()
    dylib_ext = ".dylib" if platform.system() == "Darwin" else ".so"
    dylib = lib_dir / f"liboctomil-runtime{dylib_ext}"

    if not args.force and dylib.exists() and sentinel.exists():
        print(f"already cached: {dylib}")
        return 0
    if dylib.exists() and not sentinel.exists():
        print(f"cache at {target_dir} looks incomplete; re-fetching")

    token = _gh_token()
    if not token:
        raise SystemExit(
            "error: no GitHub token available.\n"
            "Set GH_TOKEN/GITHUB_TOKEN, or run `gh auth login` so this script\n"
            "can call `gh auth token` for the private repo's release assets."
        )

    print(f"fetching octomil-runtime {version} ({flavor}/{arch}) into {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    work = target_dir / "_download"
    work.mkdir(exist_ok=True)

    try:
        try:
            assets = _release_assets_via_api(version, token)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise SystemExit(
                f"HTTP {e.code} listing release {version}.\nResponse: {body}\n"
                f"Confirm the tag exists and the token has access."
            ) from e

        # --- Manifest-driven lookup (v0.1.5+) with legacy fallback (v0.1.4) ---
        manifest = _load_manifest(assets, work, token)
        if manifest is not None:
            print(f"  manifest found: v{manifest.get('version', '?')} ABI {manifest.get('abi', {})}")
        else:
            print("  no MANIFEST.json in release; using legacy asset name shape")

        bin_name = _resolve_bin_asset_name(manifest, assets, arch, flavor, version)

        # Headers asset name: canonical shape is the same in both eras.
        # Prefer manifest["headers"] when available; fall back to the
        # well-known pattern.
        if manifest is not None and manifest.get("headers"):
            headers_name: str = manifest["headers"]
        else:
            headers_name = f"octomil-runtime-headers-{version}.tar.gz"

        sums_name = "SHA256SUMS"

        for required in (bin_name, headers_name, sums_name):
            if required not in assets:
                raise SystemExit(f"error: release {version} missing asset {required!r}")
            _download(assets[required]["url"], work / required, token)

        _verify_sha256(work / bin_name, work / sums_name)
        _verify_sha256(work / headers_name, work / sums_name)

        if lib_dir.exists():
            shutil.rmtree(lib_dir)
        if inc_dir.exists():
            shutil.rmtree(inc_dir)

        for tarball in (work / bin_name, work / headers_name):
            _safe_extract(tarball, target_dir)

        # Locate the dylib after extraction (handles .dylib / .so / glob).
        resolved_dylib = _find_dylib_in_lib(lib_dir)
        if resolved_dylib is None:
            raise SystemExit(
                f"error: extracted {bin_name} but no liboctomil-runtime dylib "
                f"found under {lib_dir}.\nBundle layout may have changed."
            )
        # Sentinel: indicates a fully-completed extraction. The next
        # non-force run treats the cache as valid only if this file
        # exists. Codex R2 missed-case fix.
        sentinel.write_text(version + "\n", encoding="utf-8")
    finally:
        # Clean up the download scratch dir whether we succeeded or
        # bailed on a sha256 mismatch / missing asset / extraction
        # error. Codex R1 missed-case fix: previously a partial run
        # could leave _download/ on disk and a subsequent --force run
        # would overwrite into it without noticing.
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    print(f"runtime ready: {resolved_dylib}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
