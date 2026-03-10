# Changelog

## Unreleased

- Added `DeviceAuthClient` runtime auth helper for device token bootstrap, refresh, and revoke flows.
- Added optional `auth` extra with `keyring` secure storage dependency.

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
