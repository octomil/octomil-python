"""Slice 2B — Moshi/MLX viability probe.

Runs against a local artifact directory containing the pinned Moshiko
+ Mimi weights, exercises a streaming inference loop, maps the
outputs onto the existing ``oct_session_*`` event shapes, and writes
measurements to ``probe-results.json``.

This script is INTENTIONALLY runnable standalone (no pytest, no
octomil-runtime dylib dependency on the import path). It validates
the MLX path FIRST, then asserts the events it produces would round-
trip through the C ABI without any new fields. If the harness has
to invent a field that doesn't exist on ``oct_event_t``, the probe
exits non-zero with a descriptive ABI-delta diagnostic — the correct
response is to stop and debate, NOT to grow the ABI in this PR.

Acceptance gates (mirrored from manifest.toml):

  1. ``mlx`` + ``moshi-mlx`` import cleanly on macOS-arm64.
  2. Pinned weight hashes match (manifest.toml AS-DECLARED OR
     manifest.lock.toml AS-RUN); package version matches.
  3. Moshi LM + rustymimi.Tokenizer instantiate from the local
     artifact root (`models.config_v0_1()` fallback if no staged
     `config.json`).
  4. The harness produces ≥ N AUDIO_CHUNK events whose decoded
     PCM fits ``oct_event_t.data.audio_chunk`` field-for-field
     (float32, 80 ms × 24 kHz × channels samples, all finite).
  5. GATING budgets: cold_open / warm_open / first_audio_ms /
     real_time_factor / compute_per_chunk_max_ms / peak_rss_mb /
     gpu_active_pct (operator-supplied). Informational only:
     compute_per_chunk_{ms,p99_ms}, cancel_to_silent_python_proxy_ms,
     audio_chunk_validation signal stats — recorded, NOT gating.
  6. Every Moshi output maps to a defined ``oct_event_*`` payload
     with no leftover fields.

Usage::

    uv run python probes/moshi_mlx/probe.py \\
        --artifact-root ~/octomil-artifacts/moshi-v0.2 \\
        --output probe-results.json

Set ``OCTOMIL_PROBE_OFFLINE=1`` to forbid the probe from reaching out
to Hugging Face if the artifact root is incomplete (default fails
loud rather than silently downloading multi-gigabyte weights).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Manifest is the source of truth for everything pinned. Read once at
# startup; fail fast on parse errors.
try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    print(
        "error: tomllib (Python 3.11+) or tomli (Python 3.10) is required. "
        "Run `uv sync --extra moshi-probe` to install dependencies.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProbeResult:
    """The on-disk shape written to probe-results.json. Every consumer
    (the strategy doc, the debate artifact, the Slice 2C planning step)
    reads from here rather than parsing logs."""

    verdict: str  # "GREEN" | "RED" | "ABI_DELTA_REQUIRED"
    started_at_iso: str
    finished_at_iso: str
    host: dict[str, Any]
    pinned: dict[str, Any]
    measurements: dict[str, Any]
    abi_mapping: dict[str, Any]
    notes: list[str]
    errors: list[str]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Host fingerprint — every measurement is host-conditional
# ---------------------------------------------------------------------------


def _host_fingerprint() -> dict[str, Any]:
    uname = platform.uname()
    sysctl_keys = (
        "hw.model",
        "hw.physicalcpu",
        "hw.memsize",
        "machdep.cpu.brand_string",
    )
    sysctl_values: dict[str, str] = {}
    if uname.system == "Darwin":
        for key in sysctl_keys:
            try:
                out = subprocess.run(
                    ["sysctl", "-n", key],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                sysctl_values[key] = out.stdout.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                sysctl_values[key] = "unavailable"
    return {
        "system": uname.system,
        "machine": uname.machine,
        "release": uname.release,
        "python": sys.version.split()[0],
        "sysctl": sysctl_values,
    }


# ---------------------------------------------------------------------------
# Manifest loader — pinned values
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found at {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Acceptance #1 — clean MLX install
# ---------------------------------------------------------------------------


def acceptance_1_imports(manifest: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    notes: list[str] = []
    errors: list[str] = []
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError as exc:
        errors.append(f"acceptance-1: mlx.core import failed: {exc}")
        return False, {}, errors
    try:
        import moshi_mlx  # type: ignore[import-not-found]
    except ImportError as exc:
        errors.append(f"acceptance-1: moshi_mlx import failed: {exc}")
        return False, {"mlx_version": getattr(mx, "__version__", "unknown")}, errors

    pinned = manifest["upstream"]["moshi_mlx_pkg"]["version"]
    actual = getattr(moshi_mlx, "__version__", "unknown")
    if actual != pinned:
        errors.append(f"acceptance-1: moshi-mlx version drift: pinned={pinned}, actual={actual}")
        return False, {"mlx_version": getattr(mx, "__version__", "unknown"), "moshi_mlx_version": actual}, errors
    notes.append(f"mlx={getattr(mx, '__version__', 'unknown')}, moshi_mlx={actual}")
    return True, {"mlx_version": getattr(mx, "__version__", "unknown"), "moshi_mlx_version": actual}, notes


# ---------------------------------------------------------------------------
# Acceptance #2 — hash verification of local artifacts
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _load_lock_pins(lock_path: Path) -> dict[tuple[str, str], str]:
    """Read manifest.lock.toml and return {(art_name, role): sha256}.
    Empty mapping if the lock file doesn't exist yet.

    Codex R11 missed-case fix: a corrupt lock file used to be silently
    swallowed and treated as "no pins found", which would let a
    bootstrap run regenerate against arbitrary on-disk weights. Parse
    errors now raise — a corrupt lock IS a hard provenance failure.
    Operators recover by deleting the lock and re-running with
    OCTOMIL_PROBE_BOOTSTRAP=1 explicitly."""
    if not lock_path.is_file():
        return {}
    with lock_path.open("rb") as fh:
        lock = tomllib.load(fh)  # raises tomllib.TOMLDecodeError on corrupt
    pins: dict[tuple[str, str], str] = {}
    for art_name, art in lock.get("artifacts", {}).items():
        for entry in art.get("files", []):
            if "role" in entry and "sha256" in entry:
                pins[(art_name, entry["role"])] = entry["sha256"]
    return pins


def acceptance_2_artifact_hashes(
    manifest: dict[str, Any],
    artifact_root: Path,
    lock_writer: "LockWriter | None" = None,
    lock_path: Path | None = None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Walk every artifact file declared in the manifest, sha256 it,
    and compare to the AS-DECLARED pin OR the lock file's pin.

    Codex R3 blocker fix: previously this function only consulted
    `manifest.toml`'s empty pins, never reading `manifest.lock.toml`.
    A run with a populated lock file but empty manifest pin would
    incorrectly require BOOTSTRAP=1. Resolution order is now:
      1. manifest.toml `[artifacts.*.files].sha256` (explicit AS-DECLARED)
      2. manifest.lock.toml `[[artifacts.*.files]].sha256` (AS-RUN)
      3. None — requires OCTOMIL_PROBE_BOOTSTRAP=1 to record."""
    notes: list[str] = []
    errors: list[str] = []
    artifacts_section = manifest["artifacts"]
    observed: dict[str, dict[str, Any]] = {}
    overall_ok = True
    bootstrap = os.environ.get("OCTOMIL_PROBE_BOOTSTRAP") == "1"
    lock_pins = _load_lock_pins(lock_path) if lock_path is not None else {}
    for art_name, art in artifacts_section.items():
        art_dir = artifact_root / art_name
        if not art_dir.is_dir():
            errors.append(f"acceptance-2: artifact dir missing: {art_dir}")
            overall_ok = False
            continue
        observed[art_name] = {"files": []}
        for entry in art["files"]:
            file_path = art_dir / entry["path"]
            if not file_path.is_file():
                errors.append(f"acceptance-2: missing {entry['role']}: {file_path}")
                overall_ok = False
                continue
            digest = _sha256_file(file_path)
            expected = entry.get("sha256", "") or lock_pins.get((art_name, entry["role"]), "")
            pin_source = (
                "manifest.toml"
                if entry.get("sha256")
                else ("manifest.lock.toml" if (art_name, entry["role"]) in lock_pins else "none")
            )
            file_record = {
                "path": entry["path"],
                "role": entry["role"],
                "size_bytes": file_path.stat().st_size,
                "sha256": digest,
                "pinned_sha256": expected,
                "pin_source": pin_source,
            }
            observed[art_name]["files"].append(file_record)
            if expected and expected != digest:
                errors.append(
                    f"acceptance-2: hash drift on {file_path} "
                    f"(pin_source={pin_source}): expected={expected}, observed={digest}"
                )
                overall_ok = False
            elif not expected:
                if not bootstrap:
                    errors.append(
                        f"acceptance-2: no pin for {entry['role']} ({file_path}); "
                        "set OCTOMIL_PROBE_BOOTSTRAP=1 to record AS-RUN sha256 into "
                        "manifest.lock.toml on the first run"
                    )
                    overall_ok = False
                else:
                    notes.append(f"acceptance-2: pinning {entry['role']} sha256={digest} (bootstrap)")
            # Codex R4 blocker fix: ONLY record into the lock writer
            # when we are explicitly bootstrapping AND the file's
            # current digest is not in conflict with an existing pin.
            # Without this gate, a non-bootstrap RED run that detects
            # drift would overwrite the previous trusted lock with
            # the drifted hash, laundering the drift on the next run.
            if lock_writer is not None and bootstrap and (not expected or expected == digest):
                lock_writer.record_artifact(art_name, entry["role"], file_path, digest)
    return overall_ok, observed, errors + notes


# ---------------------------------------------------------------------------
# Lock-file writer — records AS-RUN values for reproducibility
# ---------------------------------------------------------------------------


class LockWriter:
    """Accumulates AS-RUN values across the probe's acceptance steps
    and serializes them to ``manifest.lock.toml`` on completion.
    Codex+Gemini R1 missed-case: README/RESULTS claim auto-generation
    but the previous version never wrote the file."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self._lock: dict[str, Any] = {
            "_generator": "probes/moshi_mlx/probe.py",
            "_warning": "Auto-generated. Do not edit by hand. Re-run probe to regenerate.",
            "manifest_version_seen": manifest.get("probe", {}).get("name", "unknown"),
            "upstream": {},
            "artifacts": {},
        }

    def record_upstream(self, key: str, value: dict[str, Any]) -> None:
        self._lock["upstream"][key] = value

    def record_artifact(self, art_name: str, role: str, path: Path, sha256: str) -> None:
        self._lock["artifacts"].setdefault(art_name, {"files": []})
        self._lock["artifacts"][art_name]["files"].append(
            {"role": role, "path": str(path.name), "size_bytes": path.stat().st_size, "sha256": sha256}
        )

    def _format_value(self, v: Any) -> str:
        if v is None:
            return '""'
        if isinstance(v, str):
            # Basic escaping for double quotes and backslashes
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return f'"{v}"'  # Fallback to quoted string

    def write(self, path: Path) -> None:
        # Hand-rolled minimal TOML emitter to avoid pulling tomli-w
        # as an extra dep. The lock file is read-only from probe-the-
        # tooling's POV; the manifest is the editable source.
        lines: list[str] = [
            f"# {self._lock['_generator']}",
            f"# {self._lock['_warning']}",
            "",
            f"manifest_version_seen = {self._format_value(self._lock['manifest_version_seen'])}",
            "",
        ]
        for upstream_key, upstream_val in self._lock["upstream"].items():
            lines.append(f"[upstream.{upstream_key}]")
            for k, v in upstream_val.items():
                lines.append(f"{k} = {self._format_value(v)}")
            lines.append("")
        for art_name, art_val in self._lock["artifacts"].items():
            for file_entry in art_val["files"]:
                lines.append(f"[[artifacts.{art_name}.files]]")
                for k, v in file_entry.items():
                    lines.append(f"{k} = {self._format_value(v)}")
                lines.append("")
        path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Acceptance #3-#5 — initialize Moshi/Mimi, run streaming, measure
# ---------------------------------------------------------------------------


def acceptance_run_streaming(
    manifest: dict[str, Any],
    artifact_root: Path,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Initialize Moshi + Mimi from local artifacts, run a short
    streaming inference loop, capture cold/warm-open + first-audio +
    chunk-cadence + peak RSS + cancel-to-silent."""
    notes: list[str] = []
    errors: list[str] = []
    measurements: dict[str, Any] = {
        "cold_open_ms": None,
        "warm_open_ms": None,
        "first_audio_ms": None,
        # Codex R3 honesty fix: rename to make the semantics explicit.
        # `compute_per_chunk_ms` is the wall-clock cost of producing
        # one chunk in a tight loop (max-throughput cadence). The
        # real-time consumer cadence is fixed at 80 ms; what matters
        # is whether the model can keep up — the `real_time_factor`
        # below is the gating signal.
        "compute_per_chunk_ms": [],
        "real_time_factor": None,  # avg_compute_ms / 80; < 1.0 means we keep up
        "peak_rss_mb": None,
        "gpu_active_pct": None,
        # Python-proxy only — moshi-mlx exposes no public cancel verb.
        # Kept for completeness, NOT a gating metric. Real cancel-to-
        # silent measurement lands in Slice 2C with the C++ adapter.
        "cancel_to_silent_python_proxy_ms": None,
        "n_audio_chunks": 0,
        "n_transcript_chunks": 0,
    }

    try:
        import json as _json
        import resource

        import mlx.core as mx  # type: ignore[import-not-found]
        import mlx.nn as nn  # type: ignore[import-not-found]
        import rustymimi  # type: ignore[import-not-found]
        import sentencepiece  # type: ignore[import-not-found]
        from moshi_mlx import models  # type: ignore[import-not-found]
        from moshi_mlx.utils.sampling import Sampler  # type: ignore[import-not-found]
    except ImportError as exc:
        errors.append(f"streaming: import failure: {exc}")
        return False, measurements, errors

    moshiko_dir = artifact_root / "moshiko_mlx_q4"

    # Codex R1 blocker fix: take filenames from the manifest's
    # [artifacts.*.files] table instead of glob-matching. A wrong
    # filename now fails acceptance #2 with a clear "missing X"
    # diagnostic rather than crashing with StopIteration here.
    # Codex R12 fix: mimi_weights lives inside moshiko_mlx_q4 (NOT a
    # separate `mimi/` artifact); upstream bundles them in one HF repo.
    def _resolve(art_name: str, role: str, base_dir: Path) -> Path | None:
        for entry in manifest["artifacts"][art_name]["files"]:
            if entry["role"] == role:
                return base_dir / entry["path"]
        return None

    weights_file = _resolve("moshiko_mlx_q4", "weights", moshiko_dir)
    tokenizer_file = _resolve("moshiko_mlx_q4", "tokenizer", moshiko_dir)
    mimi_weights = _resolve("moshiko_mlx_q4", "mimi_weights", moshiko_dir)
    # Codex R13 fix: config.json is NOT shipped in kyutai/moshiko-mlx-q4
    # (live HF tree). Optional override: if the operator stages a
    # config.json in the artifact dir, parse it; otherwise fall back
    # to the canonical models.config_v0_1() helper.
    config_json_optional = moshiko_dir / "config.json"
    for label, path in (
        ("weights", weights_file),
        ("tokenizer", tokenizer_file),
        ("mimi-weights", mimi_weights),
    ):
        if path is None:
            errors.append(f"streaming: manifest missing {label} entry for moshiko_mlx_q4")
            return False, measurements, errors
        if not path.is_file():
            errors.append(f"streaming: missing {label} file: {path}")
            return False, measurements, errors

    assert weights_file is not None
    assert tokenizer_file is not None and mimi_weights is not None

    # --- cold open (mirrors moshi_mlx.run_inference.main) ---------------
    t0 = time.perf_counter()
    try:
        # Codex R13 fix: prefer staged config.json, fall back to the
        # canonical models.config_v0_1() helper.
        if config_json_optional.is_file():
            lm_config_dict = _json.loads(config_json_optional.read_text())
            lm_config = models.LmConfig.from_config_dict(lm_config_dict)  # type: ignore[attr-defined]
            notes.append(f"streaming/cold-open: using staged config.json at {config_json_optional}")
        else:
            lm_config = models.config_v0_1()  # type: ignore[attr-defined]
            notes.append("streaming/cold-open: using moshi_mlx.models.config_v0_1() (no staged config.json)")
        lm = models.Lm(lm_config)  # type: ignore[attr-defined]
        lm.set_dtype(mx.bfloat16)  # type: ignore[attr-defined]
        nn.quantize(lm, bits=4, group_size=32)  # type: ignore[attr-defined]
        lm.load_weights(str(weights_file), strict=True)  # type: ignore[attr-defined]
        # Codex R5 fix: codebook counts come from lm_config; needed for
        # both Mimi tokenizer init AND streaming-loop tensor reshape.
        generated_codebooks = lm_config.generated_codebooks
        other_codebooks = lm_config.other_codebooks

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_file))

        # Codex R3 blocker fix + R5 mirror upstream: canonical inference
        # path in moshi_mlx.run_inference uses rustymimi.Tokenizer with
        # num_codebooks = max(generated, other) (NOT a manifest constant
        # — the model config decides). Mimi class is for training/export.
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        audio_tokenizer = rustymimi.Tokenizer(str(mimi_weights), num_codebooks=mimi_codebooks)
        mx.eval(lm.parameters())  # type: ignore[attr-defined]
        # Codex R5 fix: upstream calls model.warmup(ct) where ct is a
        # conditioning tensor (or None for unconditioned models like
        # Moshiko). Without warmup, first-audio measurements include
        # the warmup cost AND the streaming loop may behave inconsistently.
        if lm.condition_provider is not None:  # type: ignore[attr-defined]
            ct = lm.condition_provider.condition_tensor("description", "very_good")  # type: ignore[attr-defined]
        else:
            ct = None
        lm.warmup(ct)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — probe-level catch-all is the contract
        errors.append(f"streaming/cold-open: {type(exc).__name__}: {exc}")
        return False, measurements, errors
    measurements["cold_open_ms"] = (time.perf_counter() - t0) * 1000.0

    # --- warm open (LmGen instantiation; no weight re-read) -------------
    t0 = time.perf_counter()
    try:
        gen = models.LmGen(  # type: ignore[attr-defined]
            lm,
            max_steps=200,
            text_sampler=Sampler(temp=0.8),
            audio_sampler=Sampler(temp=0.8),
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"streaming/warm-open: {type(exc).__name__}: {exc}")
        return False, measurements, errors
    measurements["warm_open_ms"] = (time.perf_counter() - t0) * 1000.0

    # --- streaming inference loop ----------------------------------------
    # Mirrors moshi_mlx.run_inference.main streaming loop verbatim:
    #   other_audio_tokens = audio_tokenizer.encode_step(pcm[None, 0:1])
    #   other_audio_tokens = mx.array(other).transpose(0, 2, 1)[:, :, :other_codebooks]
    #   text_token = gen.step(other[0], ct)
    #   audio_tokens = gen.last_audio_tokens()
    #   if audio_tokens is not None and generated_codebooks > 0:
    #       audio_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
    #       out_pcm = audio_tokenizer.decode_step(audio_tokens)
    # (Codex R5 blocker fix.)
    import numpy as np  # type: ignore[import-not-found]

    # 80 ms of silence per Mimi step at 24 kHz mono = 1920 frames.
    # Shape (C=1, T=1920) — mirrors upstream `pcm_data` slice from
    # in_pcms (which is (C, T) from sphn.read). The streaming loop
    # then does `silence_step[None, 0:1]` to add the batch dim AND
    # select the first channel, yielding (1, 1, 1920) into
    # encode_step. Codex R6 blocker fix: previous version made
    # silence_step already (1, 1, 1920), then `[None, 0:1]` produced
    # (1, 1, 1, 1920) which broke encode_step immediately.
    silence_step = np.zeros((1, 1920), dtype=np.float32)
    inter_chunk_ms: list[float] = []
    t_send = time.perf_counter()
    try:
        first_audio_t: float | None = None
        prev_chunk_t: float | None = None
        for _ in range(40):  # 40 × 80 ms = 3.2 s of streaming
            other_audio_tokens = audio_tokenizer.encode_step(silence_step[None, 0:1])
            # Transpose to (B, T, C) and slice to other_codebooks per upstream.
            other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :other_codebooks]  # type: ignore[attr-defined]
            text_token = gen.step(other_audio_tokens[0], ct)  # type: ignore[attr-defined]
            generated_audio = gen.last_audio_tokens()  # type: ignore[attr-defined]
            if generated_audio is None or generated_codebooks <= 0:
                continue
            # Decode shape: (codebooks, T, 1) per upstream — NOT (1, codebooks, 1).
            decoded_input = np.array(generated_audio[:, :, None]).astype(np.uint32)
            decoded_pcm = audio_tokenizer.decode_step(decoded_input)
            # Codex R7 blocker fix: validate the decoded PCM matches the
            # `oct_event_t.audio_chunk` payload contract. Acceptance #4
            # ("field-for-field fit") requires float32 PCM at the
            # advertised sample rate; without this assertion the probe
            # could GREEN on garbage payloads.
            # Codex R15 blocker: validate EVERY decoded chunk, not
            # just the first. A backend that produced one valid
            # chunk and 11 NaN chunks would have GREEN'd before.
            # Per-chunk validation is cheap; only the first chunk's
            # full validation record is stored in measurements
            # (record-once for diff-friendliness).
            pcm_arr = np.asarray(decoded_pcm)
            expected_sr = int(manifest["audio"]["output_sample_rate_hz"])
            expected_channels = int(manifest["audio"]["output_channels"])
            expected_samples_per_step = 1920  # 80 ms × 24 kHz
            expected_n_bytes = expected_samples_per_step * 4 * expected_channels
            samples_per_channel = pcm_arr.size // max(expected_channels, 1)
            all_finite = bool(np.all(np.isfinite(pcm_arr))) if pcm_arr.size > 0 else False
            chunk_fits = (
                pcm_arr.dtype == np.float32
                and pcm_arr.size > 0
                and all_finite
                and samples_per_channel == expected_samples_per_step
            )
            if not chunk_fits:
                errors.append(
                    f"streaming/audio-chunk-validation: chunk #{measurements['n_audio_chunks']} "
                    f"dtype={pcm_arr.dtype}, shape={pcm_arr.shape}, "
                    f"samples_per_channel={samples_per_channel} "
                    f"(expected {expected_samples_per_step}), all_finite={all_finite} "
                    "— does not fit oct_event_t.audio_chunk"
                )
                return False, measurements, errors
            if measurements["n_audio_chunks"] == 0:
                # Codex R10 missed-case: record signal energy for the
                # caller of probe-results.json. The probe feeds silence
                # in, so the EXPECTED output is silence too — energy is
                # informational, NOT a gate. A separate non-silence
                # prompt run (operator-driven, post-merge) is what
                # gates "produces speech." See RESULTS.md.
                output_rms = float(np.sqrt(np.mean(np.square(pcm_arr.astype(np.float64))))) if pcm_arr.size > 0 else 0.0
                output_peak_abs = float(np.max(np.abs(pcm_arr))) if pcm_arr.size > 0 else 0.0
                pcm_validation: dict[str, Any] = {
                    "dtype": str(pcm_arr.dtype),
                    "shape": list(pcm_arr.shape),
                    "n_bytes_per_chunk": int(pcm_arr.nbytes),
                    "samples_per_channel": int(samples_per_channel),
                    "expected_sample_rate_hz": expected_sr,
                    "expected_channels": expected_channels,
                    "expected_sample_format": "PCM_F32LE",
                    "expected_samples_per_step": expected_samples_per_step,
                    "expected_n_bytes_per_chunk": expected_n_bytes,
                    "all_samples_finite": all_finite,
                    "output_rms": output_rms,
                    "output_peak_abs": output_peak_abs,
                    "fit_audio_chunk_payload": (
                        pcm_arr.dtype == np.float32
                        and pcm_arr.size > 0
                        and all_finite
                        and samples_per_channel == expected_samples_per_step
                    ),
                }
                measurements["audio_chunk_validation"] = pcm_validation
                # The fit assertion already ran for THIS chunk above
                # (chunk_fits) — record-once means we only stash the
                # full validation dict, not re-test fit here.
            t_chunk_end = time.perf_counter()
            if first_audio_t is None:
                first_audio_t = t_chunk_end
                measurements["first_audio_ms"] = (first_audio_t - t_send) * 1000.0
            else:
                assert prev_chunk_t is not None
                inter_chunk_ms.append((t_chunk_end - prev_chunk_t) * 1000.0)
            prev_chunk_t = t_chunk_end
            measurements["n_audio_chunks"] += 1
            # Compute-per-chunk wall clock: same boundary as inter-chunk
            # delta in this tight loop, but we record it under an
            # explicit name so the consumer of probe-results.json can't
            # mistake it for "real-time inter-arrival cadence" (Codex R3).
            if text_token is not None:
                tok = int(text_token.item()) if hasattr(text_token, "item") else int(text_token)
                # Codex R2 blocker fix: zero_token / ungenerated_token
                # are @property accessors in moshi-mlx 0.2.6, not
                # methods. Calling them as methods raises TypeError.
                if tok != gen.ungenerated_token and tok != gen.zero_token:  # type: ignore[attr-defined]
                    decoded = text_tokenizer.id_to_piece(tok)
                    if decoded:
                        measurements["n_transcript_chunks"] += 1
            if measurements["n_audio_chunks"] >= 12:
                break
    except Exception as exc:  # noqa: BLE001
        errors.append(f"streaming/inference: {type(exc).__name__}: {exc}")
        return False, measurements, errors

    measurements["compute_per_chunk_ms"] = inter_chunk_ms
    if inter_chunk_ms:
        avg_compute = sum(inter_chunk_ms) / len(inter_chunk_ms)
        # 80 ms is the Mimi frame rate (12.5 Hz). real_time_factor < 1.0
        # means the model finishes a chunk before the next 80 ms wall-
        # clock tick — i.e. we can keep up with a microphone.
        measurements["real_time_factor"] = avg_compute / 80.0

    # --- peak RSS --------------------------------------------------------
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in BYTES; Linux reports kilobytes. Probe
    # is macOS-only per the slice spec, so bytes → MB.
    measurements["peak_rss_mb"] = rusage.ru_maxrss / (1024 * 1024)

    # --- cancel-to-silent (Python-best-effort) --------------------------
    # The real cancel path is C++ atomic-flag-flip at the next 80ms
    # frame boundary; the Python probe can only measure generator
    # teardown wall-clock. Documented limitation.
    try:
        gen2 = models.LmGen(  # type: ignore[attr-defined]
            lm,
            max_steps=200,
            text_sampler=Sampler(temp=0.8),
            audio_sampler=Sampler(temp=0.8),
        )
        for _ in range(3):
            other = audio_tokenizer.encode_step(silence_step[None, 0:1])
            other = mx.array(other).transpose(0, 2, 1)[:, :, :other_codebooks]  # type: ignore[attr-defined]
            gen2.step(other[0], ct)  # type: ignore[attr-defined]
        t_cancel = time.perf_counter()
        del gen2
        measurements["cancel_to_silent_python_proxy_ms"] = (time.perf_counter() - t_cancel) * 1000.0
        notes.append(
            "cancel-to-silent: moshi-mlx exposes no public cancel verb. "
            "The recorded value is Python GC teardown wall-clock — "
            "informational only, NOT a gating budget. Slice 2C C++ "
            "adapter implements real cancellation via std::atomic<bool> "
            "checked at every 80 ms Mimi frame boundary; the budget "
            "evaluates against the C++ value."
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"streaming/cancel: {type(exc).__name__}: {exc}")

    return True, measurements, errors + notes


# ---------------------------------------------------------------------------
# Acceptance #6 — ABI mapping check
# ---------------------------------------------------------------------------


# Required ABI mapping coverage — every probe run MUST declare these
# rows so coverage cannot regress by omission. Codex R10 missed-case
# fix: previous version accepted any manifest with zero
# `delta_required` rows including an empty mapping section. Extended
# at v0.4 step 2 (octomil-python#521) so future removal of an
# operational-envelope or runtime-scope-event row also fails the probe.
_REQUIRED_ABI_MAPPING_ROWS = frozenset(
    {
        # Slice 2A baseline.
        "session_open",
        "audio_chunk_event",
        "transcript_chunk_event",
        "cancel",
        "input_dropped",
        # ABI v0.4 step 2 additions.
        "operational_envelope",
        "model_loaded",
        "model_evicted",
        "cache_hit_kv_prefix",
        "queued_preempted",
        "memory_pressure",
        "thermal_state",
        "watchdog_timeout",
        "metric",
        "error_code",
    }
)


def acceptance_6_abi_mapping(
    manifest: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate that the manifest declares every required ABI mapping
    row AND that no row sets ``delta_required = true``. If a delta IS
    required, the probe returns RED + ABI_DELTA_REQUIRED; the operator
    runs the debate workflow before any code change."""
    notes: list[str] = []
    errors: list[str] = []
    abi_section = manifest.get("abi_mapping", {})
    declared_rows = set(abi_section.keys())
    missing_rows = _REQUIRED_ABI_MAPPING_ROWS - declared_rows
    if missing_rows:
        errors.append(f"acceptance-6: required ABI mapping rows missing from manifest: {sorted(missing_rows)}")
        return False, {"missing": sorted(missing_rows), "rows": sorted(declared_rows)}, errors
    deltas: list[str] = []
    for row_name, row in abi_section.items():
        if row.get("delta_required", False):
            deltas.append(row_name)
    if deltas:
        errors.append(f"acceptance-6: ABI deltas required for: {deltas}")
        return False, {"deltas": deltas, "rows": sorted(declared_rows)}, errors
    notes.append(f"acceptance-6: {len(abi_section)} rows mapped (all required present), 0 deltas")
    return True, {"rows": sorted(declared_rows), "deltas": []}, notes


# ---------------------------------------------------------------------------
# Verdict + budget evaluation
# ---------------------------------------------------------------------------


def evaluate_budgets(
    manifest: dict[str, Any],
    measurements: dict[str, Any],
) -> tuple[str, list[str]]:
    """Compare measured values against the budgets in manifest.toml.
    Returns (verdict, breach_list). Verdict is GREEN if every budget
    is satisfied AND the streaming run produced ≥ 8 audio chunks;
    otherwise RED."""
    breaches: list[str] = []
    budgets = manifest["probe"]["budgets"]

    def _budget_max(key: str, value: Any, limit: int | float) -> None:
        if value is None:
            breaches.append(f"{key}: not measured")
            return
        if value > limit:
            breaches.append(f"{key}: {value:.1f} > budget {limit}")

    # _budget_min is reserved for the gpu_active_pct check once the
    # IOReport sampler lands (Slice 2C). Kept inline for symmetry
    # with _budget_max; not currently called.

    _budget_max("cold_open_ms", measurements.get("cold_open_ms"), budgets["cold_open_ms"])
    _budget_max("warm_open_ms", measurements.get("warm_open_ms"), budgets["warm_open_ms"])
    _budget_max("first_audio_ms", measurements.get("first_audio_ms"), budgets["first_audio_ms"])
    _budget_max("peak_rss_mb", measurements.get("peak_rss_mb"), budgets["peak_rss_mb"])
    # Real-time factor: avg compute-per-chunk / 80ms. Below 1.0 means
    # the model can keep up with a real-time microphone feed.
    _budget_max("real_time_factor", measurements.get("real_time_factor"), budgets["real_time_factor_max"])

    # Codex R1 blocker fix: gpu_active_pct is part of the GREEN gate
    # per manifest.toml [probe.budgets]. Until the Slice-2C IOReport
    # sampler lands, the probe cannot measure it — so we surface
    # "NOT MEASURED" as an explicit breach rather than letting the
    # gate auto-pass on a None value. Operator can override by
    # setting OCTOMIL_PROBE_GPU_PCT=<value> after measuring with
    # `sudo powermetrics --samplers gpu_power -i 1000 -n 5` during
    # the streaming run.
    gpu_pct_env = os.environ.get("OCTOMIL_PROBE_GPU_PCT")
    if gpu_pct_env is not None:
        try:
            measurements["gpu_active_pct"] = float(gpu_pct_env)
            # Codex R14 missed-case: the env var is operator-supplied
            # out-of-band, so a stale sample from a prior run could
            # satisfy the gate. Record provenance explicitly so the
            # consumer of probe-results.json sees the value did NOT
            # come from this probe's own measurement.
            measurements["gpu_active_pct_source"] = "OCTOMIL_PROBE_GPU_PCT (operator out-of-band)"
        except ValueError:
            breaches.append(f"gpu_active_pct: OCTOMIL_PROBE_GPU_PCT={gpu_pct_env!r} not a float")
    gpu_pct = measurements.get("gpu_active_pct")
    if gpu_pct is None:
        breaches.append("gpu_active_pct: NOT MEASURED — set OCTOMIL_PROBE_GPU_PCT after sampling with powermetrics")
    elif gpu_pct < budgets["gpu_active_pct_min"]:
        breaches.append(f"gpu_active_pct: {gpu_pct:.1f} < {budgets['gpu_active_pct_min']}")

    # Compute-per-chunk distribution stats (informational; gating is
    # via real_time_factor above). p99 + max surface jitter / thermal
    # throttling that average alone would mask.
    cadences = measurements.get("compute_per_chunk_ms") or []
    if cadences:
        p99 = sorted(cadences)[int(len(cadences) * 0.99)] if len(cadences) > 1 else cadences[0]
        max_c = max(cadences)
        measurements["compute_per_chunk_p99_ms"] = p99
        measurements["compute_per_chunk_max_ms"] = max_c
        # Hard cap: any single chunk above 2x the real-time budget
        # (160 ms) is a buffer-underrun risk in a real-time consumer.
        if max_c > 160.0:
            breaches.append(f"compute_per_chunk_max_ms peak {max_c:.1f} > 160 (real-time underrun risk)")

    if measurements.get("n_audio_chunks", 0) < 8:
        breaches.append(f"n_audio_chunks {measurements.get('n_audio_chunks')} < 8 — streaming aborted early")

    return ("GREEN" if not breaches else "RED"), breaches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    # Codex R8 missed-case fix: PEP 508 markers skip-not-fail on
    # unsupported Python versions, so `uv sync --extra moshi-probe`
    # returns success with an empty install on 3.14. Pre-flight fails
    # fast with a clear actionable message instead of a downstream
    # ImportError.
    if not (sys.version_info >= (3, 10) and sys.version_info < (3, 14)):
        print(
            f"error: probe requires Python 3.10–3.13 (mlx 0.24.x has no wheel for "
            f"3.14). Active interpreter is {sys.version.split()[0]}. Create the "
            f"isolated probe venv with `uv venv probes/moshi_mlx/.venv-probe "
            f"--python 3.12`.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description="Slice 2B Moshi/MLX viability probe.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Local directory containing the moshiko_mlx_q4/ subdir (LM weights + tokenizer + Mimi tokenizer; bundled in one HF repo).",
    )
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.toml")
    parser.add_argument("--output", type=Path, required=True, help="Where to write probe-results.json")
    args = parser.parse_args()

    if os.environ.get("OCTOMIL_PROBE_OFFLINE", "1") != "1":
        print("warning: OCTOMIL_PROBE_OFFLINE != 1 — probe may fetch weights from HuggingFace", file=sys.stderr)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = _load_manifest(args.manifest)
    lock_writer = LockWriter(manifest)
    lock_path = args.manifest.parent / "manifest.lock.toml"

    notes: list[str] = []
    errors: list[str] = []
    pinned: dict[str, Any] = {}
    measurements: dict[str, Any] = {}
    abi_mapping: dict[str, Any] = {}

    ok1, info1, msgs1 = acceptance_1_imports(manifest)
    pinned.update(info1)
    notes.extend(msgs1)
    # Record upstream pkg versions into the lock file regardless of
    # whether downstream gates pass — the lock file is the AS-RUN
    # truth even on a RED run.
    lock_writer.record_upstream(
        "moshi_mlx_pkg",
        {"pypi": "moshi-mlx", "version_seen": info1.get("moshi_mlx_version", "unknown")},
    )
    lock_writer.record_upstream(
        "mlx_core",
        {"pypi": "mlx", "version_seen": info1.get("mlx_version", "unknown")},
    )
    if not ok1:
        # Codex R4 blocker fix: do NOT write manifest.lock.toml on
        # early failure. The trusted lock from previous successful
        # runs must survive an import-failure rerun.
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = ProbeResult(
            verdict="RED",
            started_at_iso=started,
            finished_at_iso=finished,
            host=_host_fingerprint(),
            pinned=pinned,
            measurements=measurements,
            abi_mapping=abi_mapping,
            notes=notes,
            errors=errors + msgs1,
        )
        args.output.write_text(result.to_json())
        print(result.to_json())
        return 1

    ok2, info2, msgs2 = acceptance_2_artifact_hashes(
        manifest, args.artifact_root, lock_writer=lock_writer, lock_path=lock_path
    )
    pinned["artifacts"] = info2
    notes.extend(msgs2)
    if not ok2:
        errors.extend(msgs2)

    ok_run, m_run, msgs_run = acceptance_run_streaming(manifest, args.artifact_root)
    measurements.update(m_run)
    notes.extend(msgs_run)
    if not ok_run:
        errors.extend(msgs_run)

    ok6, info6, msgs6 = acceptance_6_abi_mapping(manifest)
    abi_mapping.update(info6)
    notes.extend(msgs6)
    if not ok6:
        verdict = "ABI_DELTA_REQUIRED"
        errors.extend(msgs6)
    else:
        verdict, breaches = evaluate_budgets(manifest, measurements)
        if breaches:
            errors.extend([f"budget breach: {b}" for b in breaches])

    # Top-level acceptance #2 result feeds back into the verdict —
    # if hashes/files were missing we cannot be GREEN regardless of
    # streaming success. (Codex R1 missed-case.)
    if not ok2 and verdict == "GREEN":
        verdict = "RED"

    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Codex R4 + R7 blocker fix: ONLY write the lock file when explicitly
    # bootstrapping AND artifact verification passed (acceptance #2).
    # The lock pins file hashes; budget breaches (gpu_active_pct,
    # real_time_factor) are gating signals for verdict GREEN/RED but
    # do NOT poison hashes. Tying lock-write to "zero errors" created
    # a chicken/egg: first bootstrap run breached on
    # gpu_active_pct=NOT_MEASURED → no lock written → no AS-RUN
    # hashes ever recorded. Codex R7 missed-case: scope the gate to
    # acceptance #2 specifically.
    bootstrap = os.environ.get("OCTOMIL_PROBE_BOOTSTRAP") == "1"
    if bootstrap and ok2:
        lock_writer.write(lock_path)
        notes.append(f"manifest.lock.toml written to {lock_path} (bootstrap; acceptance #2 passed)")
    elif bootstrap and not ok2:
        notes.append("manifest.lock.toml NOT written: bootstrap mode but acceptance #2 (artifact verification) failed")
    else:
        notes.append("manifest.lock.toml NOT written: not in bootstrap mode (verify-only run)")
    result = ProbeResult(
        verdict=verdict,
        started_at_iso=started,
        finished_at_iso=finished,
        host=_host_fingerprint(),
        pinned=pinned,
        measurements=measurements,
        abi_mapping=abi_mapping,
        notes=notes,
        errors=errors,
    )
    args.output.write_text(result.to_json())
    print(result.to_json())
    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
