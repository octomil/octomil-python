# Slice 2C — candle-metal perf finding, 2026-05-04

## Stop point

Hit per the directive's fallback rule: "If either spike fails hard,
stop with exact failing command, exact missing symbol/API,
recommended fallback."

The C ABI works. The Rust shim (`engines/moshi-rs/`) compiles +
links + exposes 9 stable extern-"C" entrypoints. The streaming
pipeline runs end-to-end — `engine_open` loads Mimi + LM +
SentencePiece; `session_open` spawns a worker; the worker runs
`mimi.encode_step` → `state.step_` → `mimi.decode_step` and emits
an `OCTOMIL_MOSHI_EVENT_AUDIO_CHUNK` with 1920 samples through
`session_pop_event`.

The blocker is **candle-metal performance**, not correctness.

## Measured (Apple M5 / 16 GB / Darwin 25.1.0)

| Stage                                  | candle-metal observed | probe (MLX-Python) target |
| -------------------------------------- | --------------------- | ------------------------- |
| `mimi::load`                           | 0.4 – 0.7 s           | 0.07 s                    |
| `lm::load_lm_model` (bf16, 14.7 GB)    | 66 s                  | n/a (MLX-q4 path)         |
| `lm::load_lm_model` (q8 GGUF, 8 GB)    | 3.4 s                 | n/a                       |
| `mimi.encode_step` (80 ms frame)       | 1 – 100 ms            | ~10 ms                    |
| `state.step_` (LM forward) **bf16**    | **91 s / step**       | ~80 ms                    |
| `state.step_` (LM forward) **q8 GGUF** | **28 – 54 s / step**  | ~80 ms                    |
| `mimi.decode_step`                     | not reached           | ~10 ms                    |

LM step is **350× to 1100× slower than the realtime budget** on
both quantization formats. The probe's RTF=0.979 result was
Python+MLX+q4 — MLX has hand-tuned Metal kernels for the
transformer ops Moshi uses. Candle 0.9.1's Metal backend doesn't
hit MPS for many ops at bf16 or q8; per-op dispatch + readback
overhead compounds across ~32 layers + 6 depformer layers per
80 ms step.

## Exact failing command

```
OCTOMIL_TEST_MOSHI_ARTIFACTS=$HOME/octomil-artifacts/moshi-v0.2/moshiko_candle_bf16 \
  cargo test --release --test smoke_metal_load -- --nocapture
```

Result: AudioChunk eventually emerges, but `state.step_` per call
exceeds the 160 ms hard cap by ~570×. Cancellation works, frames
flow, audio chunks decode — the path is wired correctly.

## Why MLX C++ from Spike 1 doesn't save us

Spike 1 confirmed `libmlx.dylib` C++ surface is sufficient (loads
1247 q4 arrays, runs Metal matmul). But MLX has no Moshi LM
topology in C++; Moshi lives only in Python (`moshi_mlx`) and
Rust (`moshi-core`). Re-implementing the Moshi LM forward pass
in C++ on top of MLX is weeks of work and a parity-bug surface
area. The Rust+candle path was supposed to skip that work; it
got us a working pipeline but at unusable speed.

## Recommended fallbacks (per directive #4)

1. **mlx-rs (community Rust bindings to MLX)**: would give MLX
   kernel speed via Rust, no Python. Investigate maturity; if
   `swiftide-rs/mlx-rs` or similar is production-ready and exposes
   `quantize` + `load_safetensors` + transformer ops, retarget
   the Rust shim to use it instead of candle.
2. **Reimplement Moshi LM in C++ on libmlx**: Spike 1 verified the
   C++ surface; Moshi topology is ~200 lines of model.lm.Lm.forward
   with depformer. Multi-week effort but would get probe-equivalent
   speed.
3. **Wait for candle-metal perf**: candle's Apple Silicon backend
   is improving rapidly; tracking issues at huggingface/candle for
   Moshi-class workloads. Out of our control.
4. **Different model path**: Whisper or Cactus on candle-metal may
   have better perf characteristics. But Slice 2C target is
   `audio.realtime.session` which Moshi specifically implements.

## What doesn't change

- The public Octomil C ABI is unchanged. v0.4 step 2 is still the
  contract. **No ABI_DELTA_REQUIRED.**
- The C++ adapter base, capability registry, capability-honesty
  rule, and the Rust shim's C ABI surface ALL stay. They are
  correct; only the engine backend choice underneath the shim
  needs to change.
- The scaffolding PR's invariant — `audio.realtime.session` is
  NOT advertised — is preserved. `MoshiRsAdapter::is_loadable_now()`
  still returns false because the perf is unusable; capability
  honesty is structural.
