# Changelog

## Unreleased

- Added `DeviceAuthClient` runtime auth helper for device token bootstrap, refresh, and revoke flows.
- Added optional `auth` extra with `keyring` secure storage dependency.
- Fixed the published wheel surface so `from octomil import Octomil` resolves to the unified facade with `Octomil.from_env()`.

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
