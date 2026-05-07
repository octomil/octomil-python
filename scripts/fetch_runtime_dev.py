#!/usr/bin/env python3
"""Fetch a dev-only ``liboctomil-runtime`` release from the private
``octomil/octomil-runtime`` repo and unpack it into the local cache.

This is the supported path for local + CI dev environments that
need the dylib without a source build. Production / customer
distribution will eventually use signed-and-notarized binaries;
this script is for the v0.0.x dev range only.

Resolution:
  * Reads ``$GH_TOKEN`` / ``$GITHUB_TOKEN`` for private-repo auth.
    Falls back to ``gh auth token`` if available.
  * Downloads ``octomil-runtime-darwin-arm64-<version>.tar.gz`` and
    ``octomil-runtime-headers-<version>.tar.gz`` plus ``SHA256SUMS``.
  * Verifies sha256.
  * Extracts to ``~/.cache/octomil-runtime/<version>/lib`` and
    ``~/.cache/octomil-runtime/<version>/include``.
  * Prints the resolved dylib path.

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


def _platform_asset_name(version: str) -> str:
    sysname = platform.system()
    machine = platform.machine()
    if sysname == "Darwin" and machine == "arm64":
        return f"octomil-runtime-darwin-arm64-{version}.tar.gz"
    raise SystemExit(
        f"error: no dev artifact for {sysname}/{machine} at {version}.\n"
        f"v0.0.x ships macOS arm64 only. See octomil-runtime release notes."
    )


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=DEFAULT_VERSION, help=f"release tag (default {DEFAULT_VERSION})")
    p.add_argument("--cache-root", default=str(CACHE_ROOT), help=argparse.SUPPRESS)
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-extract even if the cache is populated",
    )
    args = p.parse_args()

    version = args.version
    cache_root = Path(args.cache_root).expanduser()
    target_dir = cache_root / version
    lib_dir = target_dir / "lib"
    inc_dir = target_dir / "include"
    dylib = lib_dir / "liboctomil-runtime.dylib"

    # Codex R2 missed-case fix: a previous partial run could leave
    # `dylib.exists()` true with corrupt or stale content (e.g. mid-
    # write crash); without this gate the next non-force run trusts
    # it. We check for a sentinel file written ONLY after a complete
    # extraction succeeds. If absent, treat the cache as suspect and
    # re-fetch.
    sentinel = lib_dir / ".extracted-ok"
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

    print(f"fetching octomil-runtime {version} into {target_dir}")
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

        bin_name = _platform_asset_name(version)
        headers_name = f"octomil-runtime-headers-{version}.tar.gz"
        sums_name = "SHA256SUMS"

        for required in (bin_name, headers_name, sums_name):
            if required not in assets:
                raise SystemExit(f"error: release {version} missing asset {required}")
            _download(assets[required]["url"], work / required, token)

        _verify_sha256(work / bin_name, work / sums_name)
        _verify_sha256(work / headers_name, work / sums_name)

        if lib_dir.exists():
            shutil.rmtree(lib_dir)
        if inc_dir.exists():
            shutil.rmtree(inc_dir)

        for tarball in (work / bin_name, work / headers_name):
            _safe_extract(tarball, target_dir)

        if not dylib.exists():
            raise SystemExit(
                f"error: extracted {bin_name} but {dylib} is not present.\nBundle layout may have changed."
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

    print(f"runtime ready: {dylib}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
