#!/usr/bin/env python3
"""
v0.1.6 PR3 — SDK-side release parity runner.

Drives the runtime-side parity scripts via subprocess and writes
SDK-perspective reports to ``release/parity/`` (this repo) so the
SDK release can carry the same parity attestations as the runtime
release. Useful for SDK release-candidate validation before
publishing to PyPI.

Order of operations:
  1. Resolve the local octomil-runtime checkout (env override
     OCTOMIL_RUNTIME_REPO, else ../octomil-runtime relative to this
     script, else fail with setup instructions).
  2. Resolve the runtime build dir (env override
     OCTOMIL_RUNTIME_BUILD_DIR, else <runtime>/build). If the build
     dir is missing or the parity helper / smoke binaries aren't
     built, attempt ``cmake --build <build_dir>`` (only if
     <build_dir>/CMakeCache.txt exists — we don't auto-configure).
  3. Invoke each runtime parity script via subprocess, with
     --output-dir pointed at this repo's ``release/parity/<cap>/``
     and SDK-prefix filenames so SDK and runtime artifacts can
     coexist if needed.
  4. Aggregate exit codes; non-zero on any gate fail.

Skip behavior matches the runtime-side scripts: missing binary /
env / pywhispercpp = SKIP (gate_pass=true). A real bound miss is
exit 1 from the runtime script and propagates here.

Usage:
  OCTOMIL_RUNTIME_REPO=/abs/path/to/octomil-runtime \\
      python3 scripts/release_run_parity.py

Or with everything defaulted (sibling-checkout layout assumed):
  python3 scripts/release_run_parity.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
SDK_PREFIX = "sdk_"
DEFAULT_OUTPUT_DIR = Path("release/parity")
CAPABILITIES = [
    ("audio.transcription", "parity_whisper_pooled_stt.py", "parity"),
    ("audio.vad", "release_assert_vad_no_parity.py", "smoke"),
    (
        "audio.speaker.embedding",
        "release_assert_speaker_no_parity.py",
        "smoke",
    ),
]


def utc_iso() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def host_block() -> dict[str, Any]:
    return {
        "arch": platform.machine(),
        "os": platform.system().lower(),
        "python_version": platform.python_version(),
        "machine": platform.platform(),
    }


def resolve_runtime_repo() -> Path:
    env = os.environ.get("OCTOMIL_RUNTIME_REPO")
    if env:
        p = Path(env).resolve()
        if not p.is_dir():
            raise SystemExit(f"OCTOMIL_RUNTIME_REPO={env} is not a directory")
        return p
    # Sibling checkout layout: ../octomil-runtime relative to this script.
    here = Path(__file__).resolve().parent.parent
    sibling = (here / ".." / "octomil-runtime").resolve()
    if sibling.is_dir() and (sibling / "scripts" / "parity_whisper_pooled_stt.py").is_file():
        return sibling
    raise SystemExit(
        "Cannot resolve octomil-runtime checkout. Set OCTOMIL_RUNTIME_REPO or "
        "place a sibling checkout at ../octomil-runtime."
    )


def resolve_build_dir(runtime_repo: Path) -> Path:
    env = os.environ.get("OCTOMIL_RUNTIME_BUILD_DIR")
    if env:
        return Path(env).resolve()
    return (runtime_repo / "build").resolve()


def maybe_build(build_dir: Path) -> None:
    """Best-effort build. We only `cmake --build` if a configured
    build dir already exists (CMakeCache.txt). Configuring is left
    to the user — too many options to be safe to default."""
    if not (build_dir / "CMakeCache.txt").is_file():
        print(
            f"[release_run_parity] {build_dir} not configured (no CMakeCache.txt). "
            "Skipping auto-build; runtime scripts will SKIP if binaries are missing.",
            file=sys.stderr,
        )
        return
    cmake = shutil.which("cmake")
    if not cmake:
        print("[release_run_parity] cmake not on PATH; skipping auto-build", file=sys.stderr)
        return
    targets = [
        "parity_helper_n_transcribe",
        "test_silero_vad_smoke",
        "test_sherpa_onnx_smoke",
    ]
    for t in targets:
        try:
            subprocess.run(
                [cmake, "--build", str(build_dir), "--target", t, "--parallel"],
                check=False,
            )
        except OSError as exc:
            print(f"[release_run_parity] build {t} failed: {exc}", file=sys.stderr)


def run_runtime_script(
    runtime_repo: Path,
    build_dir: Path,
    script_name: str,
    capability: str,
    output_dir: Path,
    extra_args: list[str],
) -> tuple[int, str, str]:
    script = runtime_repo / "scripts" / script_name
    if not script.is_file():
        return (-1, "", f"runtime script missing: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--runtime-build-dir",
        str(build_dir),
        "--output-dir",
        str(output_dir / capability),
    ] + extra_args
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.returncode, out.stdout, out.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output dir (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--whisper-bin",
        type=Path,
        default=os.environ.get("OCTOMIL_WHISPER_BIN", ""),
        help="ggml-tiny.bin path (env: OCTOMIL_WHISPER_BIN)",
    )
    ap.add_argument(
        "--jfk-wav",
        type=Path,
        default=os.environ.get("OCT_WHISPER_JFK_WAV", ""),
        help="jfk.wav path (env: OCT_WHISPER_JFK_WAV)",
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="skip auto-build of runtime targets",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "release-CI mode: forwards --strict to all runtime parity "
            "scripts. Missing pywhispercpp / smoke binaries / unset env "
            "become hard fails instead of skips."
        ),
    )
    args = ap.parse_args()

    runtime_repo = resolve_runtime_repo()
    build_dir = resolve_build_dir(runtime_repo)
    print(f"[release_run_parity] runtime_repo: {runtime_repo}", file=sys.stderr)
    print(f"[release_run_parity] build_dir:    {build_dir}", file=sys.stderr)

    if not args.no_build:
        maybe_build(build_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_ok = True
    capabilities_summary: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sdk_release_parity_summary",
        "timestamp_utc": utc_iso(),
        "host": host_block(),
        "runtime_repo": str(runtime_repo),
        "build_dir": str(build_dir),
        "capabilities": capabilities_summary,
    }

    for capability, script_name, kind in CAPABILITIES:
        cap_dir = args.output_dir / capability
        cap_dir.mkdir(parents=True, exist_ok=True)
        extra: list[str] = []
        if args.strict:
            extra.append("--strict")
        if capability == "audio.transcription":
            if not args.whisper_bin or not str(args.whisper_bin):
                extra.extend(["--whisper-bin", "/dev/null"])
            else:
                extra.extend(["--whisper-bin", str(args.whisper_bin)])
            if not args.jfk_wav or not str(args.jfk_wav):
                extra.extend(["--jfk-wav", "/dev/null"])
            else:
                extra.extend(["--jfk-wav", str(args.jfk_wav)])

        rc, stdout, stderr = run_runtime_script(
            runtime_repo,
            build_dir,
            script_name,
            capability,
            args.output_dir,
            extra,
        )

        # Mirror the runtime artifact into a SDK-prefixed file. Read
        # the artifact the runtime script just wrote (if any) and
        # re-emit with kind=sdk_<original> for traceability.
        runtime_artifact = None
        if kind == "parity":
            runtime_artifact_path = cap_dir / f"{capability}.parity.json"
        else:
            runtime_artifact_path = cap_dir / f"{capability}.smoke.json"
        if runtime_artifact_path.is_file():
            try:
                runtime_artifact = json.loads(runtime_artifact_path.read_text())
            except json.JSONDecodeError:
                runtime_artifact = None

        sdk_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "capability": capability,
            "kind": "smoke_no_python_reference" if kind == "smoke" else "python_parity",
            "timestamp_utc": utc_iso(),
            "host": host_block(),
            "runtime": {"build_dir": str(build_dir)},
            "gate_pass": rc == 0,
            "gate_failures": (runtime_artifact or {}).get("gate_failures", []) if rc != 0 else [],
            "notes": (
                f"SDK runner — invoked runtime/{script_name} (rc={rc}). "
                f"Runtime artifact at {runtime_artifact_path.name}; this file "
                f"is the SDK-perspective mirror."
            ),
        }
        if runtime_artifact:
            for key in ("measurements", "smoke", "bound", "artifacts"):
                if key in runtime_artifact:
                    sdk_payload[key] = runtime_artifact[key]

        sdk_path = cap_dir / f"{SDK_PREFIX}{capability}.json"
        sdk_path.write_text(json.dumps(sdk_payload, indent=2, sort_keys=True) + "\n")

        capabilities_summary[capability] = {
            "rc": rc,
            "gate_pass": sdk_payload["gate_pass"],
            "sdk_artifact": str(sdk_path),
            "runtime_artifact": str(runtime_artifact_path) if runtime_artifact_path.is_file() else None,
        }

        # Echo runtime stdout for operator visibility.
        sys.stdout.write(f"--- {capability} (rc={rc}) ---\n")
        sys.stdout.write(stdout)
        if stderr.strip():
            sys.stderr.write(stderr)

        if rc != 0:
            overall_ok = False

    summary["overall_pass"] = overall_ok
    (args.output_dir / "sdk_release_parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\nSDK release parity summary: "
        f"{'PASS' if overall_ok else 'FAIL'}\n"
        f"summary written to {args.output_dir / 'sdk_release_parity_summary.json'}"
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
