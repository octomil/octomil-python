# Moshi-rs adapter — Slice 2C

This adapter implements `audio.realtime.session` for Octomil's
v0.4 step 2 C ABI on darwin-arm64 by wrapping the upstream
`moshi-core` Rust crate (kyutai-labs/moshi) which runs on
`candle-rs` with the `metal` feature.

## Why moshi-rs, not moshi-mlx

The slice-2B probe (PR #523, GREEN on Apple M5) measured Python +
moshi-mlx + MLX viability — first_audio=229 ms, RTF=0.979, peak
RSS=1.5 GB. That confirmed the **algorithmic** feasibility of
streaming Moshi on Apple Silicon, not a specific engine choice.

For Layer 2a (the native runtime dylib, no Python in the realtime
path), `moshi_mlx` is unusable: it ships only as Python classes
binding MLX. Slice 2C's spike findings (see PR description on the
Slice 2C PR):

- `mlx==0.24.2` C++ surface IS sufficient for LM weight loading +
  q4 dequant + Metal matmul (verified by a 30-line smoke against
  the staged moshiko_mlx_q4 file). But there is no C++ Moshi LM
  topology — building one would be weeks of work and a parity-bug
  surface area.
- `rustymimi` (the canonical Mimi codec the probe used) exposes
  no stable C ABI; only `_PyInit_rustymimi`. It is a PyO3 wrapper
  over a Rust crate.
- The wrapped Rust crate IS `moshi-core`. It provides the entire
  streaming pipeline: `moshi::lm::load_lm_model`,
  `moshi::mimi::load`, `lm_generate_multistream::State::step`,
  with a `metal` candle backend. `moshi-cli/src/gen.rs` is a
  ~200-line reference streaming impl using that surface.

So Slice 2C wraps `moshi-core` directly via a small internal Rust
shim crate (`engines/moshi-rs/`) that exposes a stable
`extern "C"` surface. The C++ adapter (this directory) consumes
that surface and adapts it to the public Octomil C ABI.

This is exactly the fallback the user directive contemplated:

> If no [stable rustymimi C ABI], create a small internal Rust
> shim crate exposing only the tokenizer functions Slice 2C needs.
> This is below Layer 2a and is not an Octomil ABI delta.

## Two-commit landing plan

Slice 2C lands in two commits to keep the diffs reviewable and
to honor the user's "no false-positive capability advertisement"
rule:

### Commit 1 — internal adapter plumbing (this commit)

- Rust shim crate scaffolding (`engines/moshi-rs/`) with the full
  C ABI surface (`engine_open`, `session_open`, `session_send_audio`,
  `session_pop_event`, `session_cancel`, `session_close`,
  `engine_check_artifacts`).
- The shim's session worker does NOT call into `moshi-core` yet.
  It accepts audio frames, drops them, and emits a single
  `SESSION_COMPLETED { status = InitFailed, message = "scaffolding ..." }`
  event when the input channel closes.
- C++ adapter base (`IEngineAdapter`, `CapabilityRegistry`) +
  `MoshiRsAdapter` implementation.
- `MoshiRsAdapter::is_loadable_now()` returns `false`
  unconditionally with a documented reason. Therefore
  `audio.realtime.session` does NOT appear in
  `oct_runtime_capabilities`.
- CMake: `OCT_ENABLE_ENGINE_MOSHI_RS` option (default ON for
  darwin-arm64); cargo invocation as a custom target;
  `OCT_HAVE_MOSHI_RS_SHIM` compile definition for the C++ side
  to detect the link-in.
- Tests: capability is NOT advertised; engine_id is registered
  in the engine list IFF `is_loadable_now()` had been true
  (so the capability-honesty path is exercised); shim symbols
  are linkable.

This commit is **internal adapter plumbing**, not "Slice 2C
complete". Per the user directive:

> A scaffolding/stub PR is acceptable only if it never advertises
> `audio.realtime.session` and is explicitly labeled as internal
> adapter plumbing, not Slice 2C complete.

### Commit 2 — real inference path (follow-up)

- Wire `moshi::lm::load_lm_model` + `moshi::mimi::load` +
  `sentencepiece::SentencePieceProcessor::open` in the Rust shim's
  `engine_open`.
- Wire `lm_generate_multistream::State::step` in the session
  worker. Audio in: 1920 float32 @ 24 kHz mono per 80 ms frame.
  Audio out: same shape via Mimi decoder. Transcript out:
  per-step text token decoded by SentencePiece.
- Flip `MoshiRsAdapter::is_loadable_now()` to `true` when the
  artifact paths are present + the eager check passes.
- Replace the placeholder C++ adapter `oct_session_open` returning
  `OCT_STATUS_UNSUPPORTED` with a real session that delegates
  to the Rust shim.
- IOReport in-process GPU-active sampling (replaces the
  out-of-band `sample_gpu_active.sh`).
- Watchdog timeouts via `OCT_EVENT_WATCHDOG_TIMEOUT`.
- Conformance tests: 12-chunk streaming run with all gating
  budgets respected; cancel within 160 ms of the call;
  envelope echoed verbatim.

Until Commit 2 lands, the public-facing surface for
`audio.realtime.session` is unchanged from slice-2A: every
`oct_session_open` call returns `OCT_STATUS_UNSUPPORTED`.
