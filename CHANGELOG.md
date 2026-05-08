# Changelog

## Unreleased

## v0.1.9

### Features

- **progressive_during_synthesis flip**: `X-Octomil-Streaming-Honesty` response header on
  `/v1/audio/speech/stream` updated from `coalesced_after_synthesis` to
  `progressive_during_synthesis`, backed by measured proof artifact
  (`/tmp/v019-progressive-proof-20260508T185426Z.json`, `first_audio_ratio=0.5909`,
  gate < 0.75, `gate_pass=true`).

- **tts.first_audio_ms surfaced in verbose**: `TtsAudioChunk.streaming_mode` default flipped
  from `"coalesced"` to `"progressive"`. `OCT_EVENT_METRIC` events carrying
  `tts.first_audio_ms` (the `TTS_FIRST_AUDIO_MS_METRIC_NAME` constant) are now captured
  from the runtime event stream and surfaced in the verbose run metadata block when present.
  The field is omitted defensively if the metric is absent (env gate
  `OCTOMIL_TTS_FIRST_AUDIO_MS_EMIT=1` or contracts Lane 2 merged — PR #116 merged).

- **Measured delivery characteristics**: `first_audio_ratio=0.5909` (first chunk arrived
  at 59% of total synthesis wall-clock), `RTF=0.105` (faster than real-time),
  sentence-bounded chunks, sub-sentence cancel granularity ~150-200ms.

### Honest framing

- "first audio" = `open→first-chunk-dequeued`, NOT a streaming-latency floor.
- "progressive" means `delivery_timing=progressive_during_synthesis` — first audio before
  synthesis completes. Not "instantaneous" or "zero-delay."
- `realtime_streaming_claim=true` per contracts means `RTF < 1.0` (measured 0.105).

### Test changes

- Renamed `test_tts_stream_no_premature_progressive_claim.py` →
  `test_tts_stream_progressive_claim_requires_proof.py`. Inverted guard direction:
  test now asserts that progressive claims ARE present and always paired with a proof
  reference (proof artifact path or `first_audio_ratio`). Cites proof artifact and
  `proof_artifact.measured_first_audio_ratio` from contracts YAML.

## 4.17.5 (2026-05-05)

### Fixes

- flush shadow registration on shutdown (#541)
- report battery in telemetry resource (#543)
- report thermal state in telemetry (#545)

## 4.17.4 (2026-05-05)

### Fixes

- flush shadow registration on shutdown (#541)
- report battery in telemetry resource (#543)

## 4.17.3 (2026-05-05)

### Fixes

- flush shadow registration on shutdown (#541)

## 4.17.2 (2026-05-05)

### Fixes

- Register full device profiles in the background when telemetry indicates the server only has a stub device.

## 4.17.1 (2026-05-05)

### Fixes

- surface local runtime TTFT and tokens-per-second telemetry in route events so app monitoring can populate latency and throughput metrics (#536)
- harden telemetry and CLI tests that were crashing or hanging under xdist CI (#536)

## 4.17.0 (2026-05-04)

### Features

- runtime selection bench cache R/W skeleton (v0.5 PR A) (#497)
- C ABI header stub — slice-1 of runtime-architecture-v2 (#508)
- runtime selection bench cache R/W skeleton (v0.5 PR A) (#507)
- TTS bench harness skeleton — v0.5 PR B (#509)
- bench scheduler + env-var gate (v0.5 PR C) (#511)
- octomil bench CLI verbs (v0.5 PR D) (#512)
- build system + ABI stubs (slice-2 dependency) (#510)
- cffi loader for liboctomil-runtime (slice 3 PR1) (#515)
- capability-aware conformance harness (slice 3 PR2) (#516)
- slice 2A — runtime ABI closure before Moshi (#517)
- slice 2B — Moshi/MLX viability probe (after R16 consensus) (#519)
- ABI v0.4 step 1 — model lifecycle + error_code + 6 capabilities (#520)
- ABI v0.4 step 2 — operational envelope + 10 runtime-scope events (#521)
- probe verdict GREEN — Moshi/MLX viable on Apple M5 (#523)
- extract Layer 2a runtime to private octomil-runtime repo (#525)
- conformance against octomil-runtime v0.1.0 chat.completion (#526)
- SDK conformance against octomil-runtime v0.1.1 (#527)
- v0.1.2 SDK conformance + hard-cutover local chat to native runtime (#528)
- map UNSUPPORTED_MODALITY + peer codes to clean 4xx (#529)
- BackendCapabilities replaces isinstance(LlamaCppBackend) (#530)
- native chat.stream capability (#72) (#531)
- cache + latency telemetry to InferenceMetrics + verbose metadata (#73) (#532)
- configurable per-request deadline (#74) (#533)

### Fixes

- drop misleading fail-fast comment from nightly-floor (#506)
- worker-exception backoff in scheduler (#513)
- cap pytest-xdist + fail-fast diagnostics for worker death (#514)
- replace cryptic "No ModelRuntime registered" with NoRuntimeAvailableError (#534)

## 4.16.1 (2026-05-02)

### Fixes

- two text-normalize false positives in espeak_compat profile (#495). Currency normalization no longer duplicates an existing unit (`$1200 dollars` stays `1200 dollars`, not `1200 dollars dollars`); `St.` is dropped from the default abbreviation safe set so street addresses (`Meet me on St. John St.`) are not rewritten to `Saint John Saint`.

## 4.16.0 (2026-05-02)

### Features

- automatic backend-aware text normalization for TTS (#493) — Kokoro / Piper requests now have `$1200`, `Mr.`, `50%`, `°F`, etc. normalized in the dispatch path before the espeak-ng frontend sees them. No client-side normalizer needed. Pocket and other LM-based backends declare `text_normalization_profile() == "none"` and remain untouched. Opt-out via `text_normalization="off"`.
- cross-model perf cuts: voice catalog cache (mtime-keyed) + ORT thread default (cores capped at 4) + opt-in `OCTOMIL_SHERPA_PROVIDER=coreml` (#492) — measurable warm-dispatch `setup_ms` reduction across all sherpa-family TTS backends.

## 4.15.1 (2026-05-02)

### Fixes

- warmup cache key under planner candidate, not substituted candidate (#490) — fixes Eternum-reported cold-load regression where `client.warmup(model, capability='tts')` reported `loaded=True` but every subsequent `audio.speech.stream(...)` paid ~1.5–1.7s `setup_ms` (full ONNX cold load) instead of ~30–60ms warm. Substituted-static-recipe candidate's artifact identity didn't match the planner-original identity dispatch looks up under, so the cache entry was unreachable.

## 4.15.0 (2026-05-02)

### Features

- SDK scheduler + speaker resolver + Pocket engine + planner profiles (#485)
- TTS streaming observability collector + optional OTel sink (#479)
- tighten progressive TTS verification + integration acceptance (#488)

### Fixes

- hard cutover from v4.13 streaming compat surface (#486)

### Docs

- comment nit — sentence_chunk is realtime under the observability contract, not "progressive" (#487)

## 4.14.0 (2026-04-29)

### Features

- make streaming truthful and honest metrics (#480)

## 4.13.0 (2026-04-28)

### Features

- bring Python lifecycle parity to 100% (#473)
- client.audio.voices.list + shared catalog resolver (#476)
- general-purpose streaming TTS (SDK + sherpa engine + HTTP route) (#477)

## 4.12.4 (2026-04-28)

### Fixes

- static-recipe substitution rules in prepare() + warmup() (#471)

## 4.12.3 (2026-04-28)

### Fixes

- drop wave/audioop dep; surface engine-import failures (#469)

## 4.12.2 (2026-04-28)

### Fixes

- split local_tts_runtime_unavailable into 3 specific failure modes + e2e roundtrip pin (#467)

## 4.12.1 (2026-04-28)

### Fixes

- preserve source / recipe_id through both live + cached parse paths (#465)

## 4.12.0 (2026-04-28)

### Features

- manifest_uri + source='static_recipe' planner contract (PR C-followup) (#459)

### Fixes

- resolve catalog.py:523 mypy error (#460)

## 4.11.0 (2026-04-27)

Bundles the prepare-lifecycle / embedded-Python reliability work from PRs #451–457.

### Breaking

- **TTS dispatch cuts over to PrepareManager's artifact cache (PR D, #457).** The legacy `OCTOMIL_SHERPA_MODELS_DIR` / `~/.octomil/models/sherpa/<model>/` resolution path is removed from `_SherpaTtsBackend._resolve_model_dir`; calling `engine.create_backend(...)` without an injected `model_dir=` now raises with a message pointing at `client.prepare(model, capability='tts')`. `is_sherpa_tts_model_staged()` is removed (no replacement — the kernel consults PrepareManager's artifact cache directly). `local_tts_runtime_unavailable` text references `octomil prepare` / `client.prepare` rather than the legacy staging dirs. Callers who hand-staged Kokoro under the legacy paths must run `octomil prepare kokoro-82m --capability tts` once; subsequent `speech.create` calls reuse the cache.

### Features

- **Chat / responses prepare wiring (PR 10c, #451).** `MLXBackend` and `LlamaCppBackend` accept `model_dir` and load from PrepareManager's prepared dir (mlx_lm reads it like an HF repo id; llama_cpp opens the `<dir>/artifact` sentinel by GGUF magic bytes). Kernel threads `prepared_model_dir` through `_build_router` for `create_response` / `stream_response` / `stream_chat_messages`. Capability remains gated in `_PREPAREABLE_CAPABILITIES` until OctomilResponses goes through the kernel and PrepareManager grows multi-file snapshot support — both tracked as follow-ups.

- **Unified `client.warmup()` (PR 11, #452).** Strict superset of `prepare()`: pulls bytes on disk, constructs the local backend with the prepared `model_dir`, and caches the loaded instance. The next inference dispatch in the same process skips the cold-start `engine.create_backend` + `backend.load_model` loop. Cache key is `(capability, runtime_model, digest, format, quantization)` so distinct artifact identities can't alias. Strict-on-current-candidate lookup: an exact-key miss is a real cache miss — never falls back to a stale entry from a prior version. Transcription path force-loads `backend.load_model` before caching (`_WhisperBackend.load_model` was lazy and would otherwise pay the cold load on first transcribe). New `Octomil.warmup()` async facade and `octomil warmup` CLI.

- **Routing controls on the public facade (PR B, #454).** `client.audio.speech.create(..., policy=, app=)` now accepted; same kwargs already on transcription / chat / embeddings. `policy='local_only'` is now a valid preset (no longer rejected by `_normalise_preset`). Explicit `policy='private'` / `'local_only'` forces `cloud_available=False` so a planner outage cannot leak the request to a hosted backend.

- **Private `@app` refusal on planner outage (PR B, #454).** When the caller passes an `@app/<slug>/<capability>` ref (or `app=` with a concrete model) AND the planner returned no selection AND no explicit policy was given, the SDK now raises `OctomilError(RUNTIME_UNAVAILABLE)` naming the failed planner and suggesting `policy='local_only'` / fixing `OCTOMIL_SERVER_KEY`. Replaces the earlier silent cloud fallback that surfaced confusing 403s to local-first callers.

- **Cloud dispatch under canonical app identity (PR B, #454).** When `app=` is explicit (or `model=` is `@app/...`), the cloud branch sends the synthesized `@app/<app>/<canonical-capability>` ref to hosted inference instead of the resolved underlying model id. The canonical capability resolution maps `chat → responses` so the planner endpoint and the server-side app identity stay in agreement (`@app/eternum/responses`, never `@app/eternum/chat`).

- **Local TTS bootstrap (PR C, #455).**

  - New `[tts]` extra: `pip install "octomil[tts]"` pulls `sherpa-onnx>=1.12`.
  - Static offline recipe catalog. `octomil prepare kokoro-82m --capability tts` works without a server planner round-trip — the SDK ships canonical Kokoro v0.19 metadata (URL + verified GitHub-release SHA-256) and the kernel falls back to the recipe when the planner returns no candidate. Recipes are deliberately narrow: only canonical public bundles, never a public-mirror substitute for what was meant to be a private artifact.
  - Generic `MaterializationPlan` + `Materializer`. Recipes are _data_ (download metadata + materialization plan); the kernel does `outcome = PrepareManager.prepare(candidate); Materializer().materialize(outcome.artifact_dir, recipe.materialization)`. Adding the next model is a data row, not a copy of the Kokoro path. Plans declare `kind='none'` / `'archive'`, `archive_format`, `strip_prefix` (allowlist boundary), `required_outputs`, and a `MaterializationSafetyPolicy` (refuses traversal, symlink/hardlink escapes, zip/tar bombs by default).

- **`octomil doctor` (PR C, #455).** New diagnostic command that prints a structured report covering Python runtime, auth env vars, planner cache backend, artifact cache + free space, installed local engines, and registered static recipes. OK / WARN / ERROR rows; exits 0 on OK or WARN, 1 on ERROR. Never prints key material. The actionable one-liner for embedded callers.

- **`audio.speech.create` honors the prepared static-recipe cache (PR D, #457).** New `ExecutionKernel._prepared_local_artifact_dir(capability, model)` consults the static-recipe table, derives the same `<cache>/artifacts/<key>` path PrepareManager wrote to via `artifact_dir_for(...)`, and runs `Materializer().materialize(...)` idempotently. `_has_local_tts_backend` now returns true iff that layout is on disk AND the sherpa-onnx runtime is loadable; `synthesize_speech` threads the dir into `SherpaTtsEngine.create_backend(model_dir=...)`. Identity-and-scope-gated short circuit: the cache only substitutes a planner candidate when (a) artifact_id+digest match the static recipe exactly, (b) the request is direct (no `@app/...`, no `app=`) and the candidate is missing or echo-only. App-scoped requests with synthetic / mismatched candidates raise `local_tts_app_planner_unresolved` pointing at dashboard / planner config — the SDK never silently substitutes the public Kokoro for a private app artifact.

### Reliability

- **Embedded-Python compatibility (Ren'Py / sandboxed CPython / PyInstaller) (PRs A + C, #453, #455).** `import octomil` no longer crashes on environments where the `_sqlite3` extension or pandas is missing.

  - `RuntimePlannerStore` split into protocol + `SQLiteRuntimePlannerStore` / `MemoryRuntimePlannerStore` / `NullRuntimePlannerStore`. New `build_runtime_planner_store()` factory auto-falls-back to memory when sqlite3 is unavailable, with one WARNING ("`runtime planner sqlite cache unavailable; using in-memory planner cache`" — deliberately not "planner disabled").
  - `pandas`, `pyarrow`, `numpy`, `torch` removed from core dependencies. New `[analytics]` and `[fl]` extras carry them. `octomil/__init__.py` lazy-loads legacy / FL surfaces (`FederatedClient`, `ModelRegistry`, `SecAggClient`, `data_loader`, `federated_client`, …) via module-level `__getattr__`; the inner `octomil.python.octomil` package's `__init__` is itself lazy. Plain `import octomil` in a fresh subprocess shows pandas / pyarrow / torch absent from `sys.modules`.
  - Identity-preserving import hook: `import octomil.secagg`, `from octomil.secagg import ECKeyPair`, and `importlib.import_module('octomil.api_client')` all return the **same** module object. Class identity (`m1.OctomilClientError is m2.OctomilClientError`) holds across import shapes so `try / except` round-trips work cleanly.

- **Production planner cache key includes auth/API/runtime context (PR A, #453).** `RuntimePlanner.resolve()` now feeds `_make_cache_key` `api_base`, hashed `org_id`, `key_type`, `chip`, and `_installed_runtimes_hash` so a plan cached under org A doesn't leak to org B on the same machine, a plan cached against staging doesn't survive a switch to production `OCTOMIL_API_BASE`, and uninstalling `mlx-lm` invalidates the cached benchmark recommendations.

- **`OCTOMIL_API_BASE` normalization (PR B, #454).** `_normalize_api_base` strips trailing `/api/vN` (or `/vN`) so `OCTOMIL_API_BASE=https://api.octomil.com/api/v1` no longer produces `…/api/v1/api/v2/runtime/plan`.

- **Bootstrap-vs-HTTP log levels (PR A, #453).** `_resolve_planner_selection` now logs bootstrap / import / cache-construction failures at WARNING (once per process) and HTTP misses at DEBUG. Previously every failure logged DEBUG, hiding actionable problems behind transient HTTP misses.

- **Hard veto on unpreparable planner candidates (TTS + transcription).** A synthetic `prepare_required=True` candidate (no digest / url, traversal in `required_files`, NUL bytes, etc.) cannot win local routing even when a backend is staged on disk. The kernel runs `PrepareManager.can_prepare` as a dry-run before committing to local, so `local_first` falls back to cloud instead of crashing in `prepare()`.

### Safety

- **Generic materializer rejects every probe we knew to throw at it (PR C, #455).** Resolved-containment check via `_safe_join_under` (mirrors `durable_download._safe_join`) catches pre-existing-symlink escapes (`artifact_dir/linkdir → /tmp/outside`) on both ZIP and tar paths — the safe member extracts, the escape file never lands at the symlink target. `strip_prefix` is now an allowlist boundary: a malformed archive with root-level `model.onnx` cannot satisfy `required_outputs=('model.onnx',)` for a plan that declared `strip_prefix='kokoro-en-v0_19/'`. Symlinks, hardlinks, and uncompressed-bomb sizes are refused per the plan's `MaterializationSafetyPolicy`. Marker is written LAST and a partial extraction (interrupted before all required outputs landed) is detected and re-extracted instead of silently treated as complete.

### Documentation

- `octomil/cli.py --help` now lists `prepare`, `warmup`, and `doctor` alongside the existing commands.
- `client.audio.speech.create` docstring documents `policy=` / `app=` and the cloud-disabled forcing for private / local_only.

### Tracked follow-ups

- **Multi-file recipes via `manifest_uri`.** Today's recipes are single-file (Kokoro ships a tarball). PrepareManager only carries one artifact-level digest; per-file digests need `manifest_uri` support. Once that lands, the Kokoro recipe switches to per-asset downloads without changing callers.
- **OctomilResponses through the kernel.** Flips `chat` / `responses` capabilities from "kernel threading wired" to "publicly preparable" in `_PREPAREABLE_CAPABILITIES`.
- **MLX snapshot materialization.** `mlx_lm.load(<dir>)` needs a directory with `config.json` + tokenizer + safetensors; PrepareManager today only writes a single-file artifact. Once snapshot-shape support lands, MLX joins the inference-consumes-prepared rung.
- **CI + suite hygiene.** PR D's CI tuning (xdist parallelism + matrix prune to 3.9/3.11 + pyproject-keyed pip cache) unmasked ~134 pre-existing test failures across compression / grammar / decomposer / client / client_telemetry / facade_wiring suites — these were hidden behind the historical `tests/test_device_auth.py` ImportError on main and require triage independently of this release. PR D's own contract is verified by 48 targeted regressions plus a live `prepare → speech.create` dry run.

## 4.10.1 (2026-04-26)

### Features

- **Transcription joins the prepare lifecycle.** `await client.prepare(model="@app/<slug>/transcription", capability="transcription")` and `octomil prepare <model> --capability transcription` are now supported. `_WhisperBackend` honors the prepared `model_dir`: it loads from PrepareManager's `<dir>/artifact` sentinel (or matching `.bin`/`.gguf`/`.ggml`) instead of triggering pywhispercpp's HuggingFace download path. PRs #447, #448. Lifecycle support fixture in `octomil-contracts` cites `tests/test_transcription_prepare_adapter.py::test_transcription_prepare_threads_artifact_dir_into_whisper_backend` as the evidence test.

### Reliability

- **Hard veto on unpreparable planner candidates.** A synthetic `prepare_required=True` candidate (no digest/url, traversal in `required_files`, NUL bytes, etc.) no longer wins local routing even when a backend is staged on disk. New shared `ExecutionKernel._local_candidate_is_unpreparable(selection)` returns true iff `PrepareManager.can_prepare` rejects the metadata; both `synthesize_speech` and `transcribe_audio` apply it as a hard veto BEFORE the staging/runtime check, so `local_first` falls back to cloud instead of crashing in `prepare()`. PR #449.
- **Whisper resolver prefers PrepareManager's sentinel file.** When `required_files` is empty, PrepareManager writes the artifact to `<dir>/artifact` (no extension). The earlier resolver only matched `.bin`/`.gguf`/`.ggml` and silently fell back to pywhispercpp's download even when prepared bytes were on disk. Fixed in PR #448.
- **Transcription routing pre-flight.** Mirror of the earlier TTS clean-device fix: `_can_prepare_local_transcription` gates on `PrepareManager.can_prepare` so synthetic candidates are treated as local-unavailable before the locality decision rather than failing in prepare.

### Documentation

- `ExecutionKernel.prepare()` and `octomil prepare` module docs now describe both wired capabilities (TTS via SherpaTtsEngine, transcription via `_WhisperBackend`) and point at `_PREPAREABLE_CAPABILITIES` as the single source of truth.

## 4.10.0 (2026-04-26)

### Features

- **Prepare lifecycle.** New `await client.prepare(model="@app/<slug>/tts", capability="tts")` and `octomil prepare <model>` CLI command. Pre-warms an on-device TTS artifact via the runtime planner: lazy candidates short-circuit on second call, `explicit_only` candidates succeed, and `disabled` raises with an actionable message. Dashboard quickstarts surface the same one-liner.
- **Durable artifact downloader** (`octomil.runtime.lifecycle.durable_download`): multi-URL fallback, HTTP-Range resume from `.parts/*.part`, SQLite progress journal flushed every 4 MiB and clamped against on-disk size at open, signed-URL header forwarding, idempotent on already-verified files. Shared filesystem-key helper (`_fs_key.safe_filesystem_key`) keeps `PrepareManager` artifact dirs and `FileLock` lock files NAME_MAX-safe and Windows-safe under non-ASCII inputs.
- **`PrepareManager`** validates planner metadata before any disk/network: locality, delivery_mode, prepare_policy, digest + download_urls, multi-file rejection (per-file manifest is a follow-up), required-files path traversal/dot/abs/backslash/NUL, artifact_id sanitization. `can_prepare(candidate)` is a pure dry-run for the routing layer.
- TTS dispatch threads the prepared `model_dir` into `SherpaTtsEngine.create_backend(model_dir=...)`. First-run lazy prepare on a clean device now actually runs: runtime/package availability is split from artifact staging, and synthetic planner candidates are rejected before routing commits to local.

### Notes

- `client.prepare()` is TTS-only today. Transcription, embedding, and chat will be added one at a time as their backends consume the prepared `model_dir`. The CLI mirrors the same single-choice contract.
- The Node SDK ships its own `Octomil.prepare(...)` planner-introspection endpoint in `@octomil/sdk@1.5.0`. Nothing in the Node tree downloads bytes yet — host processes shell out to `octomil prepare` (Python) when materialization is needed.

## 4.9.0 (2026-04-25)

### Features

- Unified TTS routing: `client.audio.speech.create(model="@app/<slug>/tts", input=..., voice=..., response_format="wav", speed=...)` on `Octomil.from_env()` routes through the execution kernel. Local apps run on-device via sherpa-onnx; cloud apps wrap the existing `HostedClient` transport. Application code keeps `@app/<slug>/tts` regardless of policy.
- `octomil tts` CLI now routes through the same facade. `--voice`, `--speed`, and `--out` map 1:1 to the SDK kwargs.
- New `is_sherpa_tts_model_staged()` helper for pre-flight detection of an installed and runnable Kokoro/Piper model. Callers get `local_tts_runtime_unavailable` before a late backend load failure.
- Pre-flight voice validation against the Kokoro voice catalog: cloud voices like `alloy`/`onyx` against a local Kokoro model raise `voice_not_supported_for_locality`.
- `client.audio` raises `OctomilNotInitializedError` before `await client.initialize()`, symmetric with `responses` and `embeddings`.

### Notes

- Local TTS execution emits route telemetry but does not write `cloud_usage_logs` and does not increment `cloud_inference_monthly`.
- `response_format="wav"` is the only locally-supported format until local transcoding ships.

## 4.8.0 (2026-04-25)

### Features

- Hosted text-to-speech via `octomil.hosted.HostedClient.audio.speech.create(...)` against `api.octomil.com/v1/audio/speech`.
- Local sherpa-onnx TTS engine and `octomil tts` CLI for on-device synthesis with the Kokoro/Piper voice catalog.

## 4.7.6 (2026-04-24)

### Fixes

- emit canonical `route.decision` telemetry for successful Responses API requests, including local app-ref executions
- prefer live server planner resolutions for non-private `@app/...` refs so routing policy updates do not wait on a week-long client cache
- automatically drain queued telemetry events on process exit so short-lived scripts still publish route telemetry

## 4.6.0 (2026-03-25)

### Features

- verbose runtime event emitter for all backends (#334)
- first-party cloud routing via OpenAI-compatible APIs (#336)
- add Ollama cloud model support for minimax-m2.5 and kimi-k2.5 (#337)
- catalog-driven cloud model resolution (#340)
- gateway-first cloud mode + remove deprecated v1 telemetry (#341)

### Fixes

- fetch all platforms — CLI is a deployment tool, not a runtime
- show search hint in models footer
- hide empty parens for repo-level GGUF entries in octomil list (#331)
- wire telemetry reporter into OctomilResponses + hard-fail pip-audit
- resolve models with s3:// URIs from CLI push (#339)
- remove v1 registry check, use v2 catalog only
- read streaming response body before raising HTTP errors (#346)
- support native tool calling in OpenAI-compatible API messages

## 4.5.0 (2026-03-25)

### Features

- verbose runtime event emitter for all backends (#334)
- first-party cloud routing via OpenAI-compatible APIs (#336)
- add Ollama cloud model support for minimax-m2.5 and kimi-k2.5 (#337)
- catalog-driven cloud model resolution (#340)
- gateway-first cloud mode + remove deprecated v1 telemetry (#341)

### Fixes

- fetch all platforms — CLI is a deployment tool, not a runtime
- show search hint in models footer
- hide empty parens for repo-level GGUF entries in octomil list (#331)
- wire telemetry reporter into OctomilResponses + hard-fail pip-audit
- resolve models with s3:// URIs from CLI push (#339)
- remove v1 registry check, use v2 catalog only
- read streaming response body before raising HTTP errors (#346)

## 4.4.0 (2026-03-25)

### Features

- verbose runtime event emitter for all backends (#334)
- first-party cloud routing via OpenAI-compatible APIs (#336)
- add Ollama cloud model support for minimax-m2.5 and kimi-k2.5 (#337)
- catalog-driven cloud model resolution (#340)
- gateway-first cloud mode + remove deprecated v1 telemetry (#341)

### Fixes

- fetch all platforms — CLI is a deployment tool, not a runtime
- show search hint in models footer
- hide empty parens for repo-level GGUF entries in octomil list (#331)
- wire telemetry reporter into OctomilResponses + hard-fail pip-audit
- resolve models with s3:// URIs from CLI push (#339)

## 4.3.0 (2026-03-20)

### Features

- popularity sorting, pagination, and search for octomil models
- popularity sorting, pagination, and search for octomil models (#328)

## 4.2.0 (2026-03-20)

### Features

- text-based tool call extraction and input_schema normalization (#296)
- add RemoteToolExecutor, AgentSession, and CLI agent command
- add RemoteToolExecutor, AgentSession, and CLI agent command (#298)
- add ToolCallTier, strict tool-call parser, capability-aware adapter
- use presigned S3 upload for files >100MB
- add sync_embedded_catalog.py generation script
- regenerate embedded catalog with gemma-3 variants
- complete training module with tests
- add tests for all device agent core components
- add tests for telemetry store, uploader, policy engine, bandwidth budget
- add runtime updater, crash detector, and storage GC
- implement 4 core loop bodies
- add DeviceAgent top-level entrypoint
- support multimodal models with projector resources
- silent device registration with DeviceContext and AuthConfig (#307)
- emit locality and fallback span attributes on inference
- add report_observed_state() to OctomilControl
- sync generated enums for GAP-09, GAP-10, GAP-14
- sync generated enums for GAP-09, GAP-10, GAP-14
- Phase 1 manifest-driven runtime surface for Python SDK (#309)
- add heartbeat telemetry span (GAP-12)
- add PublishableKeyAuth class with restricted scopes and header generation
- wire artifact loop to report_observed_state after reconciliation
- wire routing.policy metadata from ResponseRequest to RouterModelRuntime
- generate and persist install_id on first SDK init
- complete benchmark regression gate for release CI
- add fetch_desired_state and report_observed_state methods (#315)
- sync embedded catalog with new models (#316)
- add multimodal support types and update catalog schema
- propagate multimodal fields through resolver and CLI
- complete agent wiring with tests
- POST device inventory with desired-state request
- GC handling, dynamic poll interval, startup sync
- store and pass engine policy constraints through activation
- add sync() to OctomilControl, try sync-first in get_desired_state
- add sync() to OctomilControl, try sync-first in get_desired_state (#321)
- replace prompt string with structured RuntimeRequest messages
- use model name as identifier for v2 catalog flow (#323)
- skip ensure_model, pass name directly to v2 upload flow (#324)
- expand embedded catalog to 57 families / 107 variants (#325)

### Fixes

- send org_id as query param on model create (#297)
- use query-param deep links matching SDK parsers (#299)
- check model versions before pairing, add deploy trigger, use octomil:// scheme
- increase upload timeout to 600s for large model files
- show upload progress message during push
- fix indentation of 'not in registry' message
- set network_type=wifi in artifact reconcile tests
- use typing.Union for runtime type aliases (Python 3.9 compat)
- add mcp importorskip and missing test dependencies
- remove mcp[cli] from test deps to avoid CI hang
- delegate chat.completions.create to responses.create per contract
- correct get_registry patch path in MCP platform and HTTP tests
- skip build config tests when release artifacts absent
- update telemetry event extraction to use OTLP envelope format
- update resource attribute keys in test_resource_fields
- add octomil.install.id to expected required keys
- update federated telemetry tests to parse OTLP envelope format
- mock responses API in facade wiring tests instead of stream_inference
- update MoE routing test to use prefixed OTLP resource attribute keys
- update manifest structure in HF resolver tests to v2 nested format
- update registry upload tests for presigned URL flow
- update deploy phone test for model list + QR pairing flow
- update deploy CLI tests for new defaults and strategy choices
- update ollama deploy tests for model list + versions flow
- mock browser login in federation create API key test
- make multi-turn integration test resilient to LLM variance
- fix 51 failing tests and omit \_generated/ from coverage
- update all tests for required multimodal fields
- hard cutover to models array, remove backwards compat
- resolve vendor from manifest, add missing family publishers
- strip /api/v1 from deep link host parameter

## 4.1.2 (2026-03-14)

### Fixes

- fall back to embedded manifest when server returns empty (#290)
- handle server's nested manifest format (#291)
- parse server's nested manifest format natively (#292)
- add HF checkpoint resolver for sharded + directory models (#293)
- iterate nested manifest directly, remove \_iter_manifest_models (#294)

## 4.1.1 (2026-03-14)

### Fixes

- use compact=True for smaller terminal QR output (#287)

## 4.1.0 (2026-03-14)

### Features

- switch to segno, add SVG export, path-based URLs (#285)

## 4.0.2 (2026-03-14)

### Fixes

- double QR module width for reliable phone scanning (#283)

## 4.0.1 (2026-03-14)

### Fixes

- resolve octomil.spec paths relative to repo root (#274)
- replace ValueError with OctomilError, add contract error responses (#276)
- improve terminal QR code scannability (#277)
- resolve auth module shadowing in PyInstaller builds (#278)
- look up models by name via list endpoint (#279)
- add auth re-export for PyInstaller, shrink QR URL (#280)

## 4.0.0 (2026-03-12)

### Breaking Changes

- Engine and runtime modules moved to octomil.runtime/

- octomil/engines/\* -> octomil/runtime/engines/{name}/engine.py
- octomil/responses/runtime/_ -> octomil/runtime/core/_
- Stable engines: mlx, llamacpp, ort, ollama, whisper, echo
- Experimental engines (gated by OCTOMIL_EXPERIMENTAL_ENGINES env var):
  cactus, samsung_one, mlc, mnn, executorch
- Backward-compatible shims at old import paths
- Version bumped to 3.0.0

### Features

- wire EngineRegistry as default ModelRuntime factory
- align SDK facade contract with Responses API
- add Layer 4 router/policy for local vs cloud routing
- add Layer 5 workflow orchestration
- add control namespace with register/heartbeat
- add canonical OctomilErrorCode enum with 19 codes
- add models namespace with status/load/unload/list/clearCache
- implement 5 contract directives (#266)
- import generated contract code and add conformance tests (#267)
- wire chat, capabilities, telemetry namespaces + model format/warmup + device_id (#268)
- restructure engine/runtime layer into octomil.runtime — v3.0.0 (#270)

### Fixes

- move fallback URL below QR box to fix layout
- move fallback URL below QR box to fix layout (#260)
- map platform to canonical DevicePlatform values (#269)
- sync octomil/python/pyproject.toml to v3.0.0 (#271)

## 2.11.0 (2026-03-10)

### Features

- expand server instructions to guide AI clients on when to use Octomil (#244)
- auto-trigger browser login when API key is missing
- auto-trigger browser login when API key is missing (#246)

### Fixes

- add branded header and styled output (#240)
- remove download status from model line (#241)

## 2.10.1 (2026-03-10)

### Fixes

- add branded header and styled output (#240)
- remove download status from model line
- remove download status from model line (#241)
- update 15 failing tests to match refactored source code

## 2.10.0 (2026-03-10)

### Features

- add MCP registration to octomil setup (#238)

## 2.9.0 (2026-03-10)

### Features

- register MCP server across all AI coding tools (#236)

## 2.8.0 (2026-03-10)

### Features

- integrate settle402 for batch payment settlement (#219)
- add Dockerfile and K8s manifests for MCP HTTP server (#220)
- add Streamable HTTP transport at /mcp for Smithery
- add parameter descriptions, annotations, prompts, and resources
- add smithery.yaml with configSchema for quality score
- Anthropic translation layer, color TUI, welcome redesign (#230)
- redesign CLI output, fix Ollama tag resolution and GGUF handling (#231)
- replace Ollama/registry with catalog in models command (#233)

### Fixes

- use 'instructions' kwarg instead of 'description' for FastMCP
- use lifespan-managed session manager for Streamable HTTP
- use ASGI wrapper for MCP session manager route
- improve tool names and add params to zero-param tools for Smithery score (#228)
- rename detect_hardware_profile → detect_hardware for consistent 2-word tool names (#229)
- server-side /resolve fallback for scrubbed catalog (#222)
- skip cloud registry for locally-resolvable models (#232)

## 2.7.0 (2026-03-09)

### Features

- add binary build workflow for Homebrew distribution (#88)
- add funnel event reporting to Python SDK (#91)
- add chat command, auto-select model, TUI picker, ollama:// deploy (#94)
- focused onboarding when no args passed (EDG-154)
- prefer downloaded models in auto-select, add warmup command (EDG-157)
- GPU cores, thermal, battery capture + ranking output (#97)
- add model name shell autocomplete to all model commands (#101)
- expand model autocomplete to 207 static names (#112)
- expand model catalog to 72 entries and resolver to 116 aliases (#117)
- add client-side training resilience (#119)
- add hf_onnx column for models with pre-built ONNX repos (#123)
- add shell completions command and auto-setup in install.sh (#124)
- migrate TelemetryReporter to v2 OTLP envelope format
- instrument Client.push, import_from_hf, rollback with v2 funnel events
- instrument FederatedClient with v2 funnel events
- add programmatic inference API and update push snippets
- add cloud streaming inference via SSE (#147)
- add embed() function and Client.embed() method (#148)
- replace complexity heuristic with thin policy-based client (#149)
- wire telemetry into Model.predict() and Client lifecycle (#131)
- Ollama fallback engine + alias resolution fix (#165)
- implement Phase 2 — ISO 8601 timestamps, deploy & experiment events (#172)
- wire experiment assignment and metric telemetry into call sites (#173)
- add resolve_model_experiment() and is_enrolled() for cross-SDK parity (#176)
- add ONNX in-graph sampling and FP16 scaling utilities (#180)
- add hardware profiling and quantization-aware memory estimation (#181)
- add max_entries pool capacity to KVCacheManager (#182)
- interactive agent picker when no agent specified
- add MCP server for Claude Code local inference integration
- add 9 new agent service tools (convert, optimize, hardware, benchmark, recommend, scan, compress, plan, embed)

### Fixes

- align 8 failing tests with current source code (#86)
- drop macos-13 runner, simplify homebrew update (#89)
- add workflow_dispatch trigger to build-binaries (#90)
- push --model-id, styled login, smart model resolution (#100)
- add cli_hw and sources.resolver to PyInstaller hidden imports (#104)
- update version string to 2.1.0 (#105)
- read org_id from credentials file, add copy button to login page (#106)
- PyInstaller crash + deploy auto-resolve + Python 3.14 compat
- add huggingface_hub dep, completions cmd, surface download errors (#109)
- remove duplicate \_get_org_id that shadows credentials lookup (#110)
- resolve 14 pre-existing test failures
- always show SDK snippets after successful push
- handle directory paths from HuggingFace snapshot downloads (#120)
- point phi-4-mini to ONNX repo (#122)
- strip variant suffix in resolve_model_id (#126)
- centralize **version** and add release automation (#127)
- replace PAT with GitHub App token for cross-repo dispatch (#129)
- add pyproject.toml to Knope versioned files (#130)
- sync homebrew formula and test_cli to 2.2.0
- auto-install shell completions on login
- add phi-3.5-mini and variant aliases
- update knope.toml to v0.22+ config format
- add branch creation steps to knope release workflow
- add homebrew formula and test_cli to knope versioned files
- add missing model.py and make SDK imports resilient
- only suppress ImportError in frozen binary, not normal use
- split git add and commit into separate knope steps
- consolidate release workflow and fix Homebrew formula (#146)
- add sonar-project.properties to knope versioned files (#150)
- read version from **version** and add missing hidden imports (#152)
- add automatic retries for transient HTTP failures (#153)
- add import and dependency lines to SDK snippets (#154)
- simplify push command on CLI Authenticated page (#155)
- also install Python SDK during setup (#156)
- fall back to latest release with binaries (#158)
- handle PEP 668 externally-managed Python (#159)
- add pydantic to install_requires (#160)
- Client.predict() builds GenerationRequest for Model.predict() (#162)
- auto-install mlx backend on Apple Silicon, suppress cold-start logs (#164)
- sync test_cli.py and sonar-project.properties to 2.5.3 (#167)
- align test suite with current model.py and client.py APIs (#169)
- skip redundant pull() for engines that manage own downloads (#170)
- cross-SDK naming — OctomilClient (#175)
- close(), join_round(), chat API for cross-SDK parity (#177)
- rename FederatedAnalyticsAPI to FederatedAnalyticsClient for cross-SDK parity (#178)
- sync requirements.txt with setup.py (#189)
- move hardcoded model routing data server-side (#198)
- move tuned constants server-side with safe fallbacks (#199)
- resolve 3 audit findings for cross-SDK consistency (#200)
- detect PyInstaller frozen binary when spawning serve subprocess
- resolve duplicate REDACTED_FIELD causing SyntaxError
- resolve all mypy errors across codebase
- fix test path validation and add python_version marker
- fix skipif pattern that skipped entire test module
- error when passing --model to agents with non-OpenAI API
- auto-run engine setup inline on first serve/launch

## 2.6.0 (2026-02-27)

### Features

- Ollama fallback engine + alias resolution fix (#165)

### Fixes

- auto-install mlx backend on Apple Silicon, suppress cold-start logs (#164)
- sync test_cli.py and sonar-project.properties to 2.5.3 (#167)

## 2.4.0 (2026-02-26)

### Features

- migrate TelemetryReporter to v2 OTLP envelope format
- instrument Client.push, import_from_hf, rollback with v2 funnel events
- instrument FederatedClient with v2 funnel events
- add programmatic inference API and update push snippets

### Fixes

- strip variant suffix in resolve_model_id (#126)
- centralize **version** and add release automation (#127)
- replace PAT with GitHub App token for cross-repo dispatch (#129)
- add pyproject.toml to Knope versioned files (#130)
- sync homebrew formula and test_cli to 2.2.0
- auto-install shell completions on login
- add phi-3.5-mini and variant aliases
- update knope.toml to v0.22+ config format
- add branch creation steps to knope release workflow
- add homebrew formula and test_cli to knope versioned files
- add missing model.py and make SDK imports resilient
- only suppress ImportError in frozen binary, not normal use
- split git add and commit into separate knope steps

## 2.3.0 (2026-02-26)

### Features

- migrate TelemetryReporter to v2 OTLP envelope format
- instrument Client.push, import_from_hf, rollback with v2 funnel events
- instrument FederatedClient with v2 funnel events
- add programmatic inference API and update push snippets

### Fixes

- strip variant suffix in resolve_model_id (#126)
- centralize **version** and add release automation (#127)
- replace PAT with GitHub App token for cross-repo dispatch (#129)
- add pyproject.toml to Knope versioned files (#130)
- sync homebrew formula and test_cli to 2.2.0
- auto-install shell completions on login
- add phi-3.5-mini and variant aliases
- update knope.toml to v0.22+ config format
