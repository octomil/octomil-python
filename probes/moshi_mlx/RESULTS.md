# Slice 2B — probe results and decision log

This document captures the AS-RUN evidence the probe produces. The
debate workflow reads from this file (plus `probe-results.json`,
which the probe writes machine-readably). When the dev-box operator
re-runs the probe, they overwrite this doc with fresh measurements
in the same shape.

## Acceptance #1 — MLX deps install cleanly

**Status: PASSED** (verified during this PR's authoring run).

- Resolver succeeded with markers `sys_platform == "darwin" and
platform_machine == "arm64" and python_version >= "3.10"`.
- Wheel constraint surfaced: `mlx>=0.24,<0.25` (transitive from
  `moshi-mlx==0.2.6`). `mlx==0.24.2` ships wheels for CPython
  3.10–3.13 only — Python 3.14 has no wheel as of probe authoring.
- The probe runs in an isolated `probes/moshi_mlx/.venv-probe/`
  (Python 3.12.13) so the workspace's resolver-default venv (3.14)
  doesn't see the install pressure.
- AS-RUN versions: `mlx=0.24.2`, `moshi_mlx=0.2.6` on
  Apple M5 / Mac17,2 / 16 GB / Darwin 25.1.0.

## Acceptance #2 — pinned hashes

**Status: PARTIAL** at PR-author time (no weights downloaded in
this dev environment); fully verified by the dev-box operator on
the first real run, which fills `manifest.lock.toml` with AS-RUN
SHA-256 values for every weight file.

The probe's `acceptance_2_artifact_hashes` step:

- Walks every `[artifacts.*]` row in `manifest.toml`.
- `sha256`s the on-disk file.
- Compares against the AS-DECLARED `sha256` field (empty on first
  run; populated on subsequent runs).
- Records `pinned_sha256` and observed `sha256` side-by-side in
  `probe-results.json` so any future drift fails loudly.

## Acceptance #3 — Moshi/Mimi initialize from local artifact path

**Status: SCAFFOLD VERIFIED** (the probe's import + config-parse
pre-flight passes; the actual `Lm.load_weights` + `Mimi.load_pytorch_weights`
require weights and run on the dev box).

The canonical API path mirrors `moshi_mlx.run_inference.main`:

```python
import rustymimi
lm_config = models.LmConfig.from_config_dict(json.loads(config_json))
lm = models.Lm(lm_config)
lm.set_dtype(mx.bfloat16)
nn.quantize(lm, bits=4, group_size=32)         # for q4 weights
lm.load_weights(str(weights_file), strict=True)

# Codebook count comes from lm_config — NOT a manifest constant.
mimi_codebooks = max(lm_config.generated_codebooks, lm_config.other_codebooks)

text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_file))
audio_tokenizer = rustymimi.Tokenizer(str(mimi_weights), num_codebooks=mimi_codebooks)

# Conditioning + warmup before the streaming loop.
ct = (lm.condition_provider.condition_tensor("description", "very_good")
      if lm.condition_provider is not None else None)
lm.warmup(ct)
```

Note: the probe uses `rustymimi.Tokenizer` (the canonical inference path),
NOT `moshi_mlx.models.mimi.Mimi` (which is for training/export). Earlier
draft used the latter; Codex R3 caught it. Slice 2C's C++ adapter MUST
mirror this rustymimi path.

This is the path the Slice 2C C++ adapter must mirror via the MLX
C/C++ bindings (or a thin pybind11 shim, debated in 2C).

## Acceptance #4 — streaming events through `oct_session_*` ABI

**Status: DESIGN VERIFIED, RUNTIME PENDING.**

The Python probe runs the streaming inference loop directly against
moshi-mlx (no C++ runtime in the loop; the slice-2A C++ runtime still
returns `OCT_STATUS_UNSUPPORTED` for `oct_session_*`). The probe
asserts each Python-side output shape maps onto a defined
`oct_event_t` payload (see `event_mapping.md` for the row-by-row
defense), then records timing.

Exercising the **literal** `oct_session_*` C ABI with a real Moshi
implementation is Slice 2C — that's the C++ adapter that wraps the
MLX path. The probe's job is to prove the wrap is feasible without
ABI deltas; the C ABI itself is exercised by the slice-2A NativeSession
stub-behavior tests today.

## Acceptance #5 — measurements

**Status: PENDING dev-box run.** Schema below; values filled when
`probe-results.json` is committed.

| Metric                                   | Gate?         | AS-RUN                                                  |
| ---------------------------------------- | ------------- | ------------------------------------------------------- |
| `cold_open_ms`                           | ≤ 8000        | _pending_                                               |
| `warm_open_ms`                           | ≤ 1500        | _pending_                                               |
| `first_audio_ms`                         | ≤ 1200        | _pending_                                               |
| `real_time_factor`                       | ≤ 1.0         | _pending (avg compute / 80 ms)_                         |
| `compute_per_chunk_ms` (raw deltas)      | informational | _pending_                                               |
| `compute_per_chunk_p99_ms`               | informational | _pending_                                               |
| `compute_per_chunk_max_ms`               | hard cap 160  | _pending_                                               |
| `peak_rss_mb`                            | ≤ 6000        | _pending_                                               |
| `gpu_active_pct`                         | ≥ 30          | _NOT MEASURED in-probe; set OCTOMIL_PROBE_GPU_PCT_      |
| `cancel_to_silent_python_proxy_ms`       | informational | _Python GC teardown only; real measurement is Slice 2C_ |
| `audio_chunk_validation.output_rms`      | informational | _pending — silence input means near-zero is expected_   |
| `audio_chunk_validation.output_peak_abs` | informational | _pending — same posture as output_rms_                  |

**Probe limitations:**

- **`gpu_active_pct` not measured.** Apple's GPU active-time
  exposure requires Metal Performance HUD (powermetrics) or
  IOReport — both need elevated privileges or a separate sampler
  process. Out of scope for v1 of the probe; the budget is
  deferred to Slice 2C where the C++ adapter can integrate with
  IOReport directly.
- **`cancel_to_silent_ms` is Python-proxy only.** moshi-mlx
  exposes no public cancel verb; the Python probe measures
  `del LmGen()` wall-clock, which is not representative. The
  real cancel path lives in Slice 2C (C++ atomic flag + Mimi
  next-frame-boundary check). The Python value is recorded for
  completeness, not gating.

## Acceptance #6 — ABI sufficiency

**Status: PASSED.** Mapping table grew from 5 rows (slice-2A) to 14
rows (v0.4 step 2). All declare `delta_required = false`:

### Slice 2A baseline (5 rows)

| Mapping row              | Result                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `session_open`           | every input fits `oct_session_config_t`                                                                 |
| `audio_chunk_event`      | float32 @ 24kHz mono fits `audio_chunk` payload (`OCT_SAMPLE_FORMAT_PCM_F32LE` is in the slice-2A enum) |
| `transcript_chunk_event` | UTF-8 + n_bytes is the existing `transcript_chunk` shape                                                |
| `cancel`                 | atomic-flag-flip on next 80ms boundary maps to `oct_session_cancel` semantics                           |
| `input_dropped`          | encoder backpressure maps to existing `input_dropped` payload                                           |

### ABI v0.4 step 2 additions (9 new rows)

Per octomil-python#521 (merged), the production-debugging surface
landed and Moshi/MLX has explicit fits documented in
`event_mapping.md`:

| Mapping row            | Result                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `operational_envelope` | request_id/route_id/trace_id/engine_version/adapter_version/accelerator/artifact_digest/cache_was_hit echoed on every event                 |
| `model_loaded`         | engine + model_id + artifact_digest + load_ms + warm_ms + policy_preset + user_data + source ∈ {bench-cache-recommended, engine-hint, auto} |
| `model_evicted`        | engine + model_id + freed_bytes + reason ∈ {memory_pressure, ttl, manual}                                                                   |
| `cache_hit_kv_prefix`  | layer ∈ {kv-prefix, phoneme, voice, phrase, route} + saved_tokens                                                                           |
| `queued_preempted`     | queue_position/queue_depth + preempted_by_priority/reason (Slice 3b daemon scheduler emits)                                                 |
| `memory_pressure`      | ram_available_bytes + severity ∈ {warn, critical}                                                                                           |
| `thermal_state`        | state ∈ {nominal, fair, serious, critical}                                                                                                  |
| `watchdog_timeout`     | timeout_ms + phase ∈ {load, warm, first_audio, session_step}                                                                                |
| `metric`               | name (closed runtime_metric.json enum) + value (double); free-form names FORBIDDEN by-construction                                          |
| `error_code`           | OCT*ERR*\* taxonomy bound to OCT_EVENT_ERROR.error_code (slice-2A code+message strings stay for human context)                              |

**No ABI delta required for Moshi.** The v0.4 step 2 header is
sufficient for Slice 2C. The probe re-asserts no row flips on at
startup. If a future Moshi-specific requirement forces an ABI delta
(e.g., Moshika voice change mid-session needs a SPEAKER_CHANGED
event), the slice halts with `ABI_DELTA_REQUIRED` and the gap is
debated before any code change.

## Verdict (provisional)

- **Acceptance 1, 6: GREEN** (verified at PR time).
- **Acceptance 2, 3, 4, 5: PENDING dev-box run with weights staged.**

Final verdict (`GREEN` / `RED` / `ABI_DELTA_REQUIRED`) is set by the
dev-box operator's `probe-results.json`. If GREEN → prepare Slice 2C.
If RED → assemble fallback measurements per `manifest.toml [fallback.*]`
and bring to debate.

## Operator runbook (dev box)

```bash
# 0. Slice-2B probe runs in its own isolated venv.
cd ~/Developer/Octomil/octomil-python
uv venv probes/moshi_mlx/.venv-probe --python 3.12
source probes/moshi_mlx/.venv-probe/bin/activate
uv pip install "moshi-mlx==0.2.6" "huggingface-hub>=0.24"

# 1. Stage the pinned weights. Codex R12 fix: kyutai/moshiko-mlx-q4
#    bundles the Mimi tokenizer checkpoint AND the LM config in the
#    same HF repo as the LM weights — no separate kyutai/mimi
#    artifact directory.
mkdir -p ~/octomil-artifacts/moshi-v0.2/moshiko_mlx_q4
huggingface-cli download kyutai/moshiko-mlx-q4 \
    --local-dir ~/octomil-artifacts/moshi-v0.2/moshiko_mlx_q4

# 2. Run. First run requires the bootstrap flag to record AS-RUN
#    sha256 values into manifest.lock.toml; subsequent runs verify
#    against the lock and reject drift loudly.
OCTOMIL_PROBE_BOOTSTRAP=1 python probes/moshi_mlx/probe.py \
    --artifact-root ~/octomil-artifacts/moshi-v0.2 \
    --output probes/moshi_mlx/probe-results.json

# 2b. (Optional) Sample GPU active % out-of-band, then re-run with the
#     measurement so the gpu_active_pct gate evaluates instead of
#     marking "NOT MEASURED" as a breach. Slice 2C lifts this when
#     the C++ adapter integrates with IOReport directly.
sudo powermetrics --samplers gpu_power -i 1000 -n 5 \
    | grep "GPU active" | head -1
OCTOMIL_PROBE_GPU_PCT=42.0 python probes/moshi_mlx/probe.py \
    --artifact-root ~/octomil-artifacts/moshi-v0.2 \
    --output probes/moshi_mlx/probe-results.json

# 3. Commit the results, update this doc, push, run the debate.
git add probes/moshi_mlx/probe-results.json probes/moshi_mlx/RESULTS.md
git commit -m "wip(slice-2b): probe results — <verdict>"
git push
```
