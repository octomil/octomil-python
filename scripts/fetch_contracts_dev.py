#!/usr/bin/env python3
"""Fetch a dev-only conformance artifact from ``octomil/octomil-contracts``
and unpack it into the local cache.

Mirrors ``scripts/fetch_runtime_dev.py`` for the conformance bundle:
    - Pin-driven version selection via ``octomil-python/CONFORMANCE_PIN``.
    - Two-stage resolution:
        1. local sibling ``octomil-contracts/build/conformance/`` checkout
           (dev-loop happy path; no GitHub round-trip required);
        2. GitHub Release asset on the private contracts repo.
    - SHA-256 verification before extract.
    - Hard-bounded extraction (no path traversal, no symlinks/hardlinks).
    - Cache root: ``~/.cache/octomil-conformance/<version>/``.

The conformance artifact bundles:
    - conformance/CONFORMANCE_VERSION
    - conformance/_schema/capability.schema.json
    - conformance/<capability>.yaml (every YAML matching the schema)
    - conformance/{model_lifecycle,error_mapping,event_sequence}.yaml
    - scripts/generate_conformance.py
    - schemas/core/{runtime_capability,runtime_metric}.json
    - enums/error_code.yaml

Once extracted, the pytest hook (``tests/conformance/conftest.py``) drives
``generate_conformance.py --target python ...`` over each capability YAML
and collects the resulting tests. The generated tests are NEVER committed.

Soft-skip policy: if the artifact is not reachable (no local checkout +
no GitHub token / 404 / 401), this script exits 0 with a clear stderr
message. The pytest hook then reports "no conformance cache" and skips
collection cleanly. Conformance is opt-in for dev environments without
network access (Lane G PR3 spec: "Don't fail pytest when fetch can't
reach the artifact — soft skip").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "octomil/octomil-contracts"
CACHE_ROOT = Path.home() / ".cache" / "octomil-conformance"
PIN_FILENAME = "CONFORMANCE_PIN"


def _read_pin() -> str:
    """Read ``octomil-python/CONFORMANCE_PIN``. Single-line version pin."""
    here = Path(__file__).resolve().parent.parent
    pin_path = here / PIN_FILENAME
    if not pin_path.is_file():
        raise SystemExit(f"error: missing {pin_path} (single-line version pin)")
    pin = pin_path.read_text(encoding="utf-8").strip()
    if not pin:
        raise SystemExit(f"error: {pin_path} is empty")
    if "\n" in pin:
        raise SystemExit(f"error: {pin_path} must be a single line; got {pin!r}")
    return pin


def _gh_token() -> str | None:
    for env in ("GH_TOKEN", "GITHUB_TOKEN", "OCTOMIL_CONTRACTS_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    if shutil.which("gh"):
        try:
            out = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_appledouble(name: str) -> bool:
    """macOS ``._*`` xattr metadata; filtered to keep the cache clean."""
    base = name.rsplit("/", 1)[-1]
    return base.startswith("._")


def _safe_extract(tarball: Path, target_dir: Path) -> None:
    """Hard-bounded tar extraction. Same policy as fetch_runtime_dev.py:
    no path traversal, no symlinks/hardlinks, no device entries, every
    member's resolved destination MUST stay inside ``target_dir``."""
    target_real = target_dir.resolve()
    with tarfile.open(tarball) as tf:
        for member in tf.getmembers():
            mname = member.name
            if _is_appledouble(mname):
                continue
            if mname.startswith("/") or ".." in Path(mname).parts:
                raise SystemExit(f"error: refusing to extract suspicious tar entry " f"{mname!r} from {tarball.name}")
            if member.issym() or member.islnk():
                raise SystemExit(
                    f"error: refusing to extract link entry {mname!r} "
                    f"(symlinks/hardlinks not allowed in dev artifacts)."
                )
            if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                raise SystemExit(f"error: refusing to extract device entry " f"{mname!r} from {tarball.name}")
            dest = (target_dir / mname).resolve()
            try:
                dest.relative_to(target_real)
            except ValueError as e:
                raise SystemExit(f"error: tar entry {mname!r} would escape " f"{target_dir} on resolution") from e
            tf.extract(member, target_dir)


def _verify_sha256_against_sums(path: Path, sums_file: Path) -> None:
    expected: dict[str, str] = {}
    with sums_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                # Strip leading "*" or " " from tarball name (sha256sum format).
                name = parts[1].lstrip(" *")
                expected[name] = parts[0]
    name = path.name
    if name not in expected:
        raise SystemExit(f"error: {name} not listed in {sums_file.name}")
    got = _sha256(path)
    if got != expected[name]:
        raise SystemExit(
            f"error: sha256 mismatch for {name}\n"
            f"  expected: {expected[name]}\n"
            f"  got:      {got}\n"
            f"Refuse to extract a corrupt or tampered artifact."
        )


def _local_contracts_checkout() -> Path | None:
    """Two-up sibling ``octomil-contracts/build/conformance/`` (the same
    dev layout the runtime fetch uses for its bundle). Returns None if
    not present so the caller can fall through to GitHub."""
    here = Path(__file__).resolve().parent.parent
    candidate = here.parent / "octomil-contracts" / "build" / "conformance"
    return candidate if candidate.is_dir() else None


def _artifact_basename(version: str) -> str:
    """Map a release tag (PIN value, e.g. ``v0.1.5-rc1``) to the
    artifact filename produced by ``scripts/build_conformance_artifact.py``
    (which prepends ``v`` to ``CONFORMANCE_VERSION``'s contents,
    e.g. ``0.1.5-rc1`` → ``octomil-contracts-conformance-v0.1.5-rc1.tar.gz``).

    The PIN format MAY use a ``v`` prefix (matching the GitHub release
    tag); we strip it before re-prepending so the filename is stable
    regardless of which form the operator wrote in CONFORMANCE_PIN."""
    bare = version.removeprefix("v")
    return f"octomil-contracts-conformance-v{bare}.tar.gz"


def _try_local_artifact(version: str, target_dir: Path) -> bool:
    """Promote a local contracts ``build/conformance/`` build into the
    cache. Returns True on success."""
    src = _local_contracts_checkout()
    if src is None:
        return False
    tar_name = _artifact_basename(version)
    tar_path = src / tar_name
    sha_path = src / f"{tar_name}.sha256"
    if not tar_path.is_file() or not sha_path.is_file():
        sys.stderr.write(
            f"  local checkout {src} present but missing "
            f"{tar_name} or its .sha256 sidecar; falling through to GitHub\n"
        )
        return False

    sys.stderr.write(f"  using local contracts checkout: {tar_path}\n")
    work = target_dir / "_download"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tar_path, work / tar_name)
    shutil.copy2(sha_path, work / f"{tar_name}.sha256")
    _verify_sha256_against_sums(work / tar_name, work / f"{tar_name}.sha256")
    _safe_extract(work / tar_name, target_dir)
    return True


def _release_assets_via_api(version: str, token: str) -> dict[str, dict]:
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{version}"
    req = urllib.request.Request(
        api,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "octomil-python/fetch_contracts_dev.py",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return {a["name"]: a for a in data.get("assets", [])}


def _download(url: str, dest: Path, token: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            "User-Agent": "octomil-python/fetch_contracts_dev.py",
        },
    )
    sys.stderr.write(f"  download {url}\n")
    with urllib.request.urlopen(req, timeout=120) as resp:
        with dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)


def _try_github_artifact(version: str, target_dir: Path) -> bool:
    """Fetch the conformance tarball + its sha256 sidecar from a Release
    asset. Returns True on success, False on soft-fail (no token / 404 /
    401). Hard SystemExit only on integrity failures (sha mismatch,
    suspicious tar entry, etc.)."""
    token = _gh_token()
    if not token:
        sys.stderr.write(
            "  no GitHub token (GH_TOKEN/GITHUB_TOKEN or `gh auth login`); " "soft-skipping GitHub fetch\n"
        )
        return False

    try:
        assets = _release_assets_via_api(version, token)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        sys.stderr.write(f"  HTTP {e.code} listing release {version}: {body!r}\n" f"  soft-skipping GitHub fetch\n")
        return False
    except (urllib.error.URLError, OSError) as e:
        sys.stderr.write(f"  network error listing release {version}: {e!r}\n" f"  soft-skipping GitHub fetch\n")
        return False

    tar_name = _artifact_basename(version)
    sha_name = f"{tar_name}.sha256"
    if tar_name not in assets or sha_name not in assets:
        sys.stderr.write(f"  release {version} present but missing " f"{tar_name} or {sha_name}; soft-skipping\n")
        return False

    work = target_dir / "_download"
    work.mkdir(parents=True, exist_ok=True)
    try:
        _download(assets[tar_name]["url"], work / tar_name, token)
        _download(assets[sha_name]["url"], work / sha_name, token)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        sys.stderr.write(f"  HTTP {e.code} downloading asset: {body!r}\n" f"  soft-skipping GitHub fetch\n")
        return False
    except (urllib.error.URLError, OSError) as e:
        sys.stderr.write(f"  network error downloading asset: {e!r}\n" f"  soft-skipping GitHub fetch\n")
        return False

    _verify_sha256_against_sums(work / tar_name, work / sha_name)
    _safe_extract(work / tar_name, target_dir)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version",
        default=None,
        help="release tag (default: read CONFORMANCE_PIN)",
    )
    p.add_argument("--cache-root", default=str(CACHE_ROOT), help=argparse.SUPPRESS)
    p.add_argument(
        "--force",
        action="store_true",
        help="re-fetch and re-extract even if the cache is populated",
    )
    args = p.parse_args()

    version = args.version or _read_pin()
    cache_root = Path(args.cache_root).expanduser()
    target_dir = cache_root / version
    sentinel = target_dir / ".extracted-ok"

    if not args.force and sentinel.exists():
        sys.stderr.write(f"already cached: {target_dir}\n")
        sys.stdout.write(str(target_dir) + "\n")
        return 0

    # Codex R1 G-001 fix: with --force OR a partial cache (sentinel
    # missing while target exists), drop the sentinel FIRST and wipe
    # the target before extracting fresh. Without this, a forced
    # re-fetch that crashes mid-extract leaves the old sentinel +
    # partial fresh content visible to the next non-force run, which
    # would then trust the corrupt cache.
    if target_dir.is_dir():
        if sentinel.exists():
            sentinel.unlink()
        sys.stderr.write(f"cache at {target_dir} stale (force={args.force} sentinel-cleared); wiping\n")
        for child in target_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    sys.stderr.write(f"fetching octomil-conformance {version} into {target_dir}\n")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        ok = _try_local_artifact(version, target_dir)
        if not ok:
            ok = _try_github_artifact(version, target_dir)
        if not ok:
            sys.stderr.write(
                "  no source resolved (local checkout absent + GitHub fetch " "soft-skipped); leaving cache untouched\n"
            )
            # Soft-skip: per Lane G PR3 spec, do NOT fail when the
            # artifact is unreachable. The pytest hook detects an empty
            # cache via the absence of the sentinel and skips collection.
            return 0
    finally:
        # Whether we succeeded, soft-skipped, or hard-failed mid-flight,
        # tear down the scratch dir so a subsequent --force run starts
        # clean.
        work = target_dir / "_download"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    # Sentinel: indicates a fully-completed extraction. The next non-force
    # run treats the cache as valid only if this file exists.
    sentinel.write_text(version + "\n", encoding="utf-8")
    sys.stderr.write(f"conformance cache ready: {target_dir}\n")
    sys.stdout.write(str(target_dir) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
