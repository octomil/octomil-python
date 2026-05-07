# Native runtime cutover matrix

Canonical, evidence-cited record of which on-device capabilities have been
hard-cut over to `octomil-runtime` (Layer 2a, native C ABI) and which are
still served by Python product code paths.

The matrix is the source of truth for two invariants enforced in
`tests/test_native_cutover_invariants.py`:

1. The runtime advertises **only** capabilities that have a real native
   adapter — no fake advertisement.
2. For every Python local backend that has a native equivalent, the SDK
   product path constructs the native backend exclusively. The legacy
   Python backend is unreachable except in explicit benchmark / reference
   tests.

A row is `DONE_NATIVE_CUTOVER` only when **all** of the following hold:

1. The runtime advertises the capability honestly via
   `oct_runtime_capabilities`.
2. The Python SDK binds the capability through `octomil.runtime.native`.
3. Product routing constructs the native backend exclusively.
4. The legacy Python execution path is unreachable on the product flow
   (defense-in-depth tests pin this).
5. Manual parity / speed gate has been run against staged artifacts.
6. Engineering-debate consensus reached on the cutover PR.

Anything missing one of these stays `BLOCKED_WITH_PROOF` until it is
completed. `BLOCKED_WITH_PROOF` is **not** a green-light to fabricate
native advertisement or product routing.

## Pinned runtime version

`scripts/fetch_runtime_dev.py` pins `DEFAULT_VERSION = "v0.1.4"`
(`scripts/fetch_runtime_dev.py:42`). The native loader resolves the
dylib from this dev cache or from `OCTOMIL_RUNTIME_DYLIB`.

## Capability matrix

| Capability                                                   | Status                | Native runtime adapter                                                                                                                                                                                                                                                                                                                                               | Python SDK binding                                                                                        | Product wiring                                                                                                                          | Defense-in-depth                                                                                                                                                   |
| ------------------------------------------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `chat.completion` (incl. streaming events)                   | `DONE_NATIVE_CUTOVER` | `octomil-runtime/src/adapters/llama_cpp/llama_cpp_adapter.h:60`. Streaming is delivered as `OCT_EVENT_TRANSCRIPT_CHUNK` events on the same session — there is no separate `chat.stream` runtime capability and the literal string is rejected with `OCT_STATUS_UNSUPPORTED` at `oct_session_open`.                                                                   | `octomil/runtime/native/chat_backend.py:224` `NativeChatBackend` (`streaming_supported=True` at line 248) | `octomil/runtime/engines/llamacpp/engine.py:160` `create_backend`                                                                       | `tests/test_native_chat_backend.py:71` traps legacy `LlamaCppBackend.__init__`; `test_native_chat_backend_generate_stream_end_to_end` exercises the streaming path |
| `embeddings.text`                                            | `DONE_NATIVE_CUTOVER` | `octomil-runtime/src/adapters/llama_cpp/llama_cpp_adapter.h:61` + `LlamaCppEmbeddingsSession` (`octomil-runtime/src/runtime.cpp:654`)                                                                                                                                                                                                                                | `octomil/runtime/native/embeddings_runtime.py:103` `NativeEmbeddingsRuntime`                              | `octomil/runtime/__init__.py:37` `_connect_native_embeddings`                                                                           | `tests/test_native_embeddings_cutover.py` (520 lines)                                                                                                              |
| `chat.stream`                                                | `BLOCKED_WITH_PROOF`  | **Not a separate runtime capability.** Streaming is the event mode of `chat.completion`. The literal `chat.stream` capability string is rejected at `oct_session_open` with `OCT_STATUS_UNSUPPORTED` (`octomil-runtime/src/runtime.cpp:738`). The contract enum reserves the string for callers who want to negotiate streaming explicitly in a future ABI revision. | None on product path                                                                                      | None on product path                                                                                                                    | Parametrized in `test_blocked_capability_rejects_with_bounded_unsupported`                                                                                         |
| `audio.tts.batch`, `audio.tts.stream`                        | `BLOCKED_WITH_PROOF`  | **No native adapter.** `oct_session_open` returns `OCT_STATUS_UNSUPPORTED` for any capability outside the llama.cpp set (`octomil-runtime/src/runtime.cpp:738`).                                                                                                                                                                                                     | Python sherpa-onnx engine still in tree at `octomil/runtime/engines/sherpa/engine.py:1`                   | Serve-layer Python path: `octomil/serve/app.py:1298` calls `state.sherpa_tts_backend.synthesize_stream` directly; no native runtime hop | Endpoint fails closed when no Python backend is loaded (`octomil/serve/app.py:1333` raises `MODEL_LOAD_FAILED`) — there is no silent native masquerade             |
| `audio.stt.batch`, `audio.stt.stream`, `audio.transcription` | `BLOCKED_WITH_PROOF`  | **No native adapter.** Whisper.cpp is checked out at `research/engines/whisper.cpp` but not wired as a runtime adapter.                                                                                                                                                                                                                                              | Python whisper.cpp engine at `octomil/runtime/engines/whisper/engine.py:59` (`pywhispercpp` wrapper)      | Serve-layer Python path: `octomil/serve/app.py:1262` calls `state.whisper_backend.transcribe` directly; no native runtime hop           | Endpoint fails closed when no Python backend is loaded (`octomil/serve/app.py:1271` raises `MODEL_LOAD_FAILED`)                                                    |
| `audio.realtime.session`                                     | `BLOCKED_WITH_PROOF`  | **No native adapter.** Moshi-rs adapter scaffolding exists (`octomil-runtime/src/runtime.cpp:296-298`) but `is_loadable_now()` returns false until the streaming pipeline lands; the capability is not advertised.                                                                                                                                                   | None on product path                                                                                      | None on product path                                                                                                                    | Parametrized in `test_blocked_capability_rejects_with_bounded_unsupported`                                                                                         |
| `audio.vad`                                                  | `BLOCKED_WITH_PROOF`  | **No native adapter.** Capability constant defined in contract (`octomil/runtime/native/capabilities.py:41`) but no engine implements it; `oct_runtime_capabilities` does not advertise it.                                                                                                                                                                          | None on product path                                                                                      | None on product path                                                                                                                    | Guard test asserts the capability does **not** appear in `oct_runtime_capabilities` output                                                                         |
| `audio.diarization`                                          | `BLOCKED_WITH_PROOF`  | **No native adapter.** Same as VAD.                                                                                                                                                                                                                                                                                                                                  | None on product path                                                                                      | None on product path                                                                                                                    | Guard test asserts non-advertisement                                                                                                                               |
| `audio.speaker.embedding`                                    | `BLOCKED_WITH_PROOF`  | **No native adapter.** Reference encoder pinned in `octomil-contracts/fixtures/runtime_bench/` is bench-only.                                                                                                                                                                                                                                                        | None on product path                                                                                      | None on product path                                                                                                                    | Guard test asserts non-advertisement                                                                                                                               |
| `embeddings.image`                                           | `BLOCKED_WITH_PROOF`  | **No native adapter.** Contract enum reserves the string; no image-embedding engine is wired into `octomil-runtime`.                                                                                                                                                                                                                                                 | None on product path                                                                                      | None on product path                                                                                                                    | Parametrized in `test_blocked_capability_rejects_with_bounded_unsupported`                                                                                         |
| `index.vector.query`                                         | `BLOCKED_WITH_PROOF`  | **Out of Layer 2a scope.** Vector storage / retrieval is a host concern, not an inference-engine capability. The contract enum reserves the string for future use.                                                                                                                                                                                                   | None on product path                                                                                      | None on product path                                                                                                                    | Guard test asserts non-advertisement                                                                                                                               |
| Runtime selection / bench                                    | `NOT_LAYER_2A`        | The bench cache + ranker (`octomil/runtime/bench/runner.py`, `octomil/commands/bench.py:60`) is a planner-host concern in v0.5. It selects which engine + adapter to dispatch to; it is not itself a runtime capability and is not subject to native cutover.                                                                                                        | n/a                                                                                                       | n/a                                                                                                                                     | n/a                                                                                                                                                                |

The matrix is partition-complete: every member of `RUNTIME_CAPABILITIES`
appears as either a `DONE_NATIVE_CUTOVER` capability (the row above
`chat.stream`) or a `BLOCKED_WITH_PROOF` row. The
`test_capabilities_partitioning_is_complete` guard asserts
`DONE_CAPABILITIES ∪ BLOCKED_CAPABILITIES == RUNTIME_CAPABILITIES`, so
adding a contract-enum entry without classifying it fails CI.

## Why TTS / STT / VAD / diarization / speaker-embedding are blocked

Each of the audio capabilities needs:

1. A native engine adapter (whisper.cpp / sherpa-onnx / silero-vad / pyannote-onnx
   / ECAPA) wrapped in a `octomil::adapters::*` C++ module that registers
   with the runtime's `CapabilityRegistry`.
2. An ABI-side session shape — for streaming audio that means
   `oct_session_send_audio` plus event types beyond the v0.1.4
   transcript / embedding-vector pair.
3. A signed runtime release cut (cffi binding compat: `OCT_RUNTIME_ABI_VERSION_*`,
   `oct_runtime_capabilities` size handshake).
4. A staged artifact for the manual parity gate (cold open, warm,
   first-output latency, total latency, throughput, peak RSS, output
   format, no control-token leakage).

Each step is real work. None of it is in v0.1.4. Promoting any of these
rows requires a separate cutover PR per capability, with engineering-debate
consensus.

## What `BLOCKED_WITH_PROOF` does **not** mean

- It does not mean we should advertise the capability and reject at
  session-open. The v0.1.4 runtime's allowlist is the truth: only
  `chat.completion` + `embeddings.text` appear in
  `oct_runtime_capabilities`. Anything else is `OCT_STATUS_UNSUPPORTED` at
  `oct_session_open` (`octomil-runtime/src/runtime.cpp:738`).
- It does not mean the existing Python sherpa-onnx / whisper.cpp paths
  in `octomil/serve/app.py` are forbidden. They are the **only** path
  that exists for those capabilities today; deleting them would remove
  working functionality without replacing it. They are not silent
  fallbacks behind a native attempt — there is no native attempt.
- It does mean: when a native adapter for one of these lands, that PR
  must move the capability to `DONE_NATIVE_CUTOVER` (all six conditions),
  delete or hard-deprecate the Python backend with a defense-in-depth
  test, and update this matrix.
