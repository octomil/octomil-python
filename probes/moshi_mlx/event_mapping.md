# ABI mapping — Moshi outputs ↔ `oct_event_t` payloads

This document is the long-form companion to the `[abi_mapping]`
table in `manifest.toml`. It exists so the engineering-debate review
can verify each row by reading prose, not just a key/value pair.

The hinge claim of Slice 2B: **the existing slice-2A ABI is
sufficient for a working Moshi/MLX adapter.** Every row below
defends that claim. If any row reveals a gap, the probe verdict is
`ABI_DELTA_REQUIRED` and the slice stops before it touches code.

## 1. Session lifecycle: `oct_session_open` ↔ Moshi/Mimi load

| ABI input field   | Moshi/Mimi consumer                                                                                                                                                                                                                                                                                                                                                            | Required? |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| `model_uri`       | Resolves to local artifact directory containing `moshiko_mlx_q4/` (bundles LM weights + tokenizer + Mimi tokenizer + LM config; no separate `mimi/` dir per Codex R12)                                                                                                                                                                                                         | yes       |
| `capability`      | Strict-reject: must be `"audio.realtime.session"`                                                                                                                                                                                                                                                                                                                              | yes       |
| `locality`        | Must be `"on_device"`; the adapter never opens a network socket                                                                                                                                                                                                                                                                                                                | yes       |
| `policy_preset`   | Informational only on the input side. NOT carried as a dedicated field on `OCT_EVENT_SESSION_STARTED` (the slice-2A `session_started` payload is `engine + model_digest + locality + streaming_mode + runtime_build_tag` only). The policy_preset value is reflected through the `oct_telemetry_sink_fn` event stream instead, where Layer 2b code stamps it on each emission. | no        |
| `speaker_id`      | Currently unused by Moshiko (single-voice); reserved for future Moshika voice profiles                                                                                                                                                                                                                                                                                         | no        |
| `sample_rate_in`  | If 0, default to 24000 (Mimi's native rate); other values resampled by adapter (slice 2C scope)                                                                                                                                                                                                                                                                                | no        |
| `sample_rate_out` | Same posture as `sample_rate_in`                                                                                                                                                                                                                                                                                                                                               | no        |
| `priority`        | Routed to scheduler; foreground vs prefetch affects evict policy when multiple sessions run                                                                                                                                                                                                                                                                                    | no        |
| `user_data`       | Echoed verbatim on every event per slice-2A contract                                                                                                                                                                                                                                                                                                                           | no        |

**No new fields needed.** The entire session config is carried by
existing `oct_session_config_t` slots.

## 2. Output: `OCT_EVENT_SESSION_STARTED`

Moshi has a one-shot "model loaded + ready" signal — the moment
`Lm.load_weights` and `Mimi.load_pytorch_weights` return and the
first KV cache slot is initialized. The adapter emits
SESSION_STARTED with:

```
session_started {
    engine            = "moshi-mlx"
    model_digest      = sha256(moshiko_mlx_q4/model.safetensors)
    locality          = "on_device"
    streaming_mode    = "duplex"   // moshi processes audio + text in lockstep
    runtime_build_tag = "octomil-runtime/<git-sha>"
}
```

**No new fields needed.** Every value fits an existing `const char*`
slot on the `session_started` payload.

## 3. Output: `OCT_EVENT_AUDIO_CHUNK`

Mimi decodes 80 ms of float32 PCM per output frame at 24 kHz mono.
The adapter copies the decoded buffer into a runtime-owned arena and
emits one `OCT_EVENT_AUDIO_CHUNK` per frame:

```
audio_chunk {
    pcm           = <runtime-owned float32 buffer>
    n_bytes       = 1920 * 4 = 7680                  // 80 ms × 24 kHz × 4 bytes
    sample_rate   = 24000
    sample_format = OCT_SAMPLE_FORMAT_PCM_F32LE
    channels      = 1
    is_final      = 0  // 1 only on terminal flush
}
```

The slice-2A header committed `OCT_SAMPLE_FORMAT_PCM_F32LE` exactly
because Moshi/Mimi works in float32. **No new fields needed.**

## 4. Output: `OCT_EVENT_TRANSCRIPT_CHUNK`

Moshi emits text tokens in lockstep with audio frames via its text
head. The adapter detokenizes each non-pad token and emits a
transcript chunk:

```
transcript_chunk {
    utf8     = <runtime-owned UTF-8 buffer>
    n_bytes  = strlen(utf8)
}
```

**No new fields needed.** The `utf8 + n_bytes` pair is already the
canonical streaming-text shape on `oct_event_t`.

## 5. Cancellation: `oct_session_cancel` ↔ Moshi step boundary

Moshi's MLX generator does not expose a public cancel verb in
moshi-mlx 0.2.x — the C++ adapter implements cancellation by
holding an `std::atomic<bool> cancelled` and checking it at every
Mimi step boundary (every 80 ms). On flip:

1. Stop calling `gen.step()` and `mimi.decode_step()`.
2. Emit `OCT_EVENT_SESSION_COMPLETED` with
   `terminal_status = OCT_STATUS_CANCELLED`.
3. Subsequent `oct_session_poll_event` calls return
   `OCT_STATUS_CANCELLED`.

This is exactly the contract in slice 2A's `runtime.h`. **No new
fields needed.**

## 6. Backpressure: `OCT_EVENT_INPUT_DROPPED`

When Mimi's encoder backlogs (input frames arrive faster than
12.5 Hz), the adapter drops the oldest queued frames and emits an
`input_dropped` event:

```
input_dropped {
    n_frames_dropped = <int>
    sample_rate      = 24000
    channels         = 1
    reason           = "engine_lagging"  // runtime-owned static string
    dropped_at_ns    = <monotonic ns>
}
```

**No new fields needed.**

## 7. Telemetry: `oct_telemetry_sink_fn`

The adapter emits one structured event per state transition (load
start, load end, first chunk, cancel, completed). The
`runtime_config_t.telemetry_sink` callback is the existing channel.
**No new fields needed.**

## Result

All seven mapping rows fit the slice-2A ABI without modification.
The manifest's `delta_required = false` markers reflect this in
machine-readable form; the probe asserts no row flips on at startup.

## What the probe DOES NOT prove

- Cross-platform compatibility (Linux/Windows) — out of scope
  for slice 2B (macOS-arm64 only).
- Multi-session concurrency limits — slice 3b daemon territory.
- Voice profile / speaker_id integration — Moshiko is single-voice;
  Moshika support comes later.
- Real cancel-to-silent latency — the probe runs in Python (no
  atomic flag flip; teardown via Python GC). The C++ adapter measures
  this for real in slice 2C.

If a future requirement forces any of these into the ABI, slice 2A's
versioning rules apply: additive minor bump, the C ABI never
silently drops a contract.

## v0.4 step 2 — operational envelope + 10 runtime-scope events

ABI v0.4 step 2 (octomil-python#521, merged) ADDS the production-
debugging surface every Moshi/MLX adapter needs:

### 8. Operational envelope on every event

Every `OCT_EVENT_*` (existing audio/transcript/error/etc. PLUS the
10 new runtime-scope events) carries the v0.4 step 2 envelope
APPENDED after the union:

```
event {
    ... existing payload ...

    request_id      = <Layer-2b-stamped at session_open; NULL→"" echo>
    route_id        = <Layer-2b-stamped at session_open; NULL→"" echo>
    trace_id        = <W3C trace_id if present; NULL→"" echo>
    engine_version  = "moshi-mlx@<ver>"
    adapter_version = "octomil-runtime/<git-sha>"
    accelerator     = "metal"
    artifact_digest = "sha256:<hex>"
    cache_was_hit   = 0|1
}
```

The runtime ALWAYS writes non-NULL pointers (empty-string sentinels
for uncorrelated slots); bindings can `strlen()` safely.

**Layer 2a NEVER mints** request_id/route_id/trace_id — Layer 2b
sets them at `oct_session_open` and the runtime echoes verbatim.
This preserves end-to-end correlation through traces/dashboards
without the runtime inventing identity.

**No new fields needed for Moshi.** The envelope is exactly the
correlation surface the realtime adapter consumes.

### 9. Model lifecycle events

```
OCT_EVENT_MODEL_LOADED { engine, model_id, artifact_digest,
                         load_ms, warm_ms, policy_preset,
                         config_user_data, source }
OCT_EVENT_MODEL_EVICTED { engine, model_id, artifact_digest,
                          freed_bytes, reason, config_user_data }
```

Slice 2C emits MODEL_LOADED right after `lm.warmup(ct)` returns,
with `load_ms` = cold-open duration and `warm_ms` = warmup
duration. MODEL_EVICTED fires when `oct_model_evict()` runs OR
when the scheduler evicts under memory pressure.

### 10. Cache events

```
OCT_EVENT_CACHE_HIT  { layer ∈ closed enum, saved_tokens }
OCT_EVENT_CACHE_MISS { layer, saved_tokens=0 }
```

Slice 2C wires the kv-prefix cache hit signal when MLX prefix-cache
support lands. `layer` MUST be from the closed enum
{kv-prefix, phoneme, voice, phrase, route} — free-form labels
forbidden.

### 11. Scheduler events

```
OCT_EVENT_QUEUED    { queue_position, queue_depth }
OCT_EVENT_PREEMPTED { preempted_by_priority, reason }
```

Slice 2C single-session adapter doesn't emit these (no scheduling
needed for one session). Slice 3b daemon scheduler emits when
foreground sessions supplant prefetch/speculative.

### 12. Resource pressure events

```
OCT_EVENT_MEMORY_PRESSURE { ram_available_bytes, severity }
OCT_EVENT_THERMAL_STATE   { state }
OCT_EVENT_WATCHDOG_TIMEOUT { timeout_ms, phase }
```

Slice 2C watchdog samples macOS `NSProcessInfo.thermalState` and
RAM availability; emits when crossing thresholds. WATCHDOG_TIMEOUT
fires when load/warmup/first-audio/session-step exceeds the
budget defined in `manifest.toml [probe.budgets]`.

### 13. Bounded metrics

```
OCT_EVENT_METRIC { name (closed enum), value (double) }
```

Names drawn from `runtime_metric.json` closed enum (15 metrics
across cache._, gpu._, kv*prefix.*, mimi._, model._, scheduler.\_).
**Free-form names FORBIDDEN by-construction** — Slice 2C cannot
emit a name not in the canonical assignment table because the
emission helper takes a constant from the table, not an arbitrary
string.

### 14. Bounded error taxonomy

```
OCT_EVENT_ERROR { code, message, error_code }
```

`error_code` (v0.4 step 2 APPENDED inside the existing error
payload) is the bounded form for telemetry labels. Slice 2C maps:

- weight load fail → OCT_ERR_MODEL_LOAD_FAILED
- sha256 mismatch → OCT_ERR_ARTIFACT_DIGEST_MISMATCH
- engine init fail → OCT_ERR_ENGINE_INIT_FAILED
- OOM → OCT_ERR_RAM_INSUFFICIENT
- Metal context fail → OCT_ERR_ACCELERATOR_UNAVAILABLE
- > budget timeout → OCT_ERR_TIMEOUT
- internal invariant → OCT_ERR_INTERNAL

`code` and `message` (slice-2A free-form strings) stay for human-
readable context but should NOT be used as metric labels.

## v0.4 step 2 result

All 14 mapping rows fit the v0.4 ABI without modification. The
manifest's `delta_required = false` markers reflect this in
machine-readable form. Slice 2C builds against this contract; the
probe re-asserts no row flips on at startup.

If a future Moshi-specific requirement forces an ABI delta (e.g.,
Moshika voice change mid-session needs a SPEAKER_CHANGED event),
the slice halts with `ABI_DELTA_REQUIRED` and the gap is debated
before any code change.
