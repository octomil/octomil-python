# AGENTS.md — octomil-python

## Purpose

Python SDK + CLI for local model serving, inference, federated learning, and device management. Provides an OpenAI-compatible API server (`octomil serve`), multi-engine auto-benchmarking, and an MCP server for AI tool integration.

## What This Repo Owns

### CLI (`octomil/cli.py` + `octomil/commands/`)

```
octomil setup              # Environment setup, engine detection
octomil serve <model>      # Local OpenAI-compatible server
octomil chat [model]       # Interactive chat REPL
octomil benchmark <model>  # Benchmark available engines
octomil deploy <model>     # Deploy to devices/fleet
octomil pull / push        # Model registry operations
octomil convert            # Convert to CoreML/TFLite
octomil models             # List available models (60+)
octomil login              # Authenticate with Octomil
octomil init               # Create organization
octomil mcp register       # Register MCP server with Claude/Cursor/VS Code
octomil mcp serve          # HTTP agent server (A2A + x402)
octomil launch [agent]     # Launch coding agent
octomil scan               # Security scanning
```

### Inference Stack (3 layers)

#### Layer 0: Engine Plugins (`octomil/runtime/engines/`)

| Engine       | Class              | Platform          | Status        |
| ------------ | ------------------ | ----------------- | ------------- |
| MLX-LM       | `MLXEngine`        | Apple Silicon     | Production    |
| llama.cpp    | `LlamaCppEngine`   | Mac/Linux/Windows | Production    |
| ONNX Runtime | `ORTEngine`        | All               | Production    |
| Whisper.cpp  | `WhisperEngine`    | All               | Production    |
| Ollama       | `OllamaEngine`     | Mac/Linux         | Production    |
| Echo         | `EchoEngine`       | All               | Test fallback |
| MLC-LLM      | `MLCEngine`        | Mac/Linux/Android | Experimental  |
| MNN          | `MNNEngine`        | All               | Experimental  |
| ExecuTorch   | `ExecutorchEngine` | Mobile            | Experimental  |
| Cactus       | `CactusEngine`     | —                 | Experimental  |
| Samsung ONE  | `SamsungOneEngine` | Mobile            | Experimental  |

#### Layer 1: ModelRuntime (`octomil/runtime/core/`)

- `ModelRuntime` protocol — abstract interface for any backend
- `ModelRuntimeRegistry` — global registry by model family
- `RouterModelRuntime` — device vs cloud routing
- `CloudModelRuntime` — remote API fallback
- `EngineBridge` — bridges EnginePlugin → ModelRuntime

#### Layer 2: OctomilResponses (`octomil/responses/`)

- `OctomilResponses.create(request)` → `Response`
- `OctomilResponses.stream(request)` → `AsyncIterator[ResponseStreamEvent]`
- Tool execution, prompt formatting, structured output

#### Server (`octomil/serve.py`)

- FastAPI with OpenAI-compatible endpoints
- `POST /v1/chat/completions` (streaming + non-streaming)
- `POST /v1/completions`
- `GET /v1/models`
- KV cache, prompt compression, early exit, grammar-constrained generation
- Multi-model serving with auto-routing

### SDK Client (`octomil/client.py`)

- `OctomilClient` — push/pull/deploy models, manage rollouts, experiments

### Federated Learning

- `FederatedClient` — main FL class with gradient caching
- `SecAggClient`, `SecAggPlusClient` — secure aggregation (cryptography)
- `DeltaFilter`, `FilterRegistry` — gradient processing
- `GradientCache` — offline gradient persistence

### MCP Integration (`octomil/mcp/`)

- stdio + HTTP MCP server
- Agent-to-agent (A2A) card support
- x402 micropayment protocol
- Registration with Claude Desktop, Cursor, VS Code

### Hardware Detection (`octomil/hardware/`)

- CPU, Metal (Apple GPU), CUDA (NVIDIA), ROCm (AMD)
- Unified detection with `HardwareInfo` ABC

### Model Catalog (`octomil/models/`)

- 60+ models with alias resolution
- Sources: HuggingFace, Kaggle, Ollama
- Quantization variants

### Telemetry (`octomil/telemetry.py`)

- OTLP v2 envelope format (custom, not standard OTel SDK)
- Events: INFERENCE_STARTED, INFERENCE_COMPLETED, INFERENCE_FAILED, DEPLOY_STARTED, etc.
- Best-effort background queue, non-blocking
- Sent to `{api_base}/v2/telemetry/events`

### Generated Code (`octomil/_generated/`)

- `error_code.py`, `auth_type.py`, `compatibility_level.py`, `device_class.py`, `finish_reason.py`, `model_status.py`, `principal_type.py`, `scope.py`, `otlp_resource_attributes.py`, `telemetry_events.py`

## What This Repo Must NOT Define

- **Contract enums** — import from `_generated/` (auto-generated from `octomil-contracts`)
- **Server-side control plane logic** — that belongs in `octomil-server`
- **Telemetry semantic conventions** — use definitions from contracts
- **Model format specifications** — those are defined in contracts (ArtifactFormat, ArtifactResourceKind)

## Public vs Internal

### Public

- CLI commands (`octomil serve`, `octomil chat`, etc.)
- `OctomilClient` class
- `OctomilResponses` class
- `ModelRuntime` protocol
- Engine plugin interface (`EnginePlugin` ABC)
- Server endpoints (`/v1/chat/completions`, `/v1/models`)

### Internal

- Engine implementations (MLXEngine, LlamaCppEngine, etc.)
- `InferenceBackend` implementations (MLXBackend, LlamaCppBackend)
- Hardware detection internals
- MCP backend implementation
- Gradient filters, secure aggregation internals

## Common Commands

```bash
# Install (development)
make install    # uv venv && uv pip install -e ".[dev]"

# Lint
ruff check . --fix
ruff format .
mypy .

# Test
pytest --cov=octomil --cov-report=term-missing --cov-fail-under=74

# Conformance tests only
pytest tests/conformance/

# Run server locally
octomil serve Qwen/Qwen3-0.6B --engine mlx-lm --port 8080

# Run CLI
octomil models
octomil benchmark Qwen/Qwen3-0.6B
```

## Testing Expectations

- 90+ test files in `tests/`
- Coverage floor: 74% (target 80%)
- Conformance tests in `tests/conformance/` (control, enums, errors, telemetry)
- Engine tests for each supported engine
- MCP server tests
- Routing and smart router tests
- Federated learning and SecAgg tests
- All new public API must have tests
- All bug fixes must include regression tests

## Release Rules

- Version managed by Knope (`knope.toml`): bumps `pyproject.toml`, `__init__.py`, homebrew formula, test_cli.py, sonar-project.properties
- Release flow: `knope release` → `release` branch → version bump + changelog → PR to main
- On merge: tag push → PyInstaller binaries (darwin-arm64, linux-amd64) → PyPI → Homebrew tap
- Binary builds: `release-binary.yml` GitHub Action
- Homebrew: `octomil/homebrew-octomil` repo

## Optional Dependencies

Install only what you need:

- `[serve]` — fastapi, uvicorn (for `octomil serve`)
- `[mlx]` — mlx-lm (Apple Silicon)
- `[llama]` — llama-cpp-python
- `[onnx]` — onnxruntime
- `[whisper]` — pywhispercpp
- `[ml]` — torch, numpy, pandas
- `[secagg]` — cryptography
- `[mcp]` — mcp[cli]
- `[x402]` — eth-account, httpx
- `[dev]` — all of the above + ruff, pytest

## Common Pitfalls

- **Do not hand-edit files in `octomil/_generated/`** — auto-generated from `octomil-contracts`
- **Do not add torch as a hard dependency** — it is optional (`[ml]`). Guard imports with try/except.
- **Do not break the zero-dep CLI** — `octomil models`, `octomil setup`, etc. must work without optional deps
- **Engine detection must be lazy** — don't import engine modules at startup; detect on demand
- **Do not add engines to the production list without benchmarks** — experimental engines stay in `engines/experimental/`
- **Telemetry must never block or raise** — best-effort only, failures logged as warnings
- **KV cache is opt-in** — do not enable by default in the server
- **Do not commit `.env` or API keys** — use macOS Keychain / env vars

## Agent Access Rules

### Primary Agents

- **Runtime Agent**: owns engine adapters, inference stack, performance, benchmarking, hardware detection. Primary code author for runtime/ and serve.py.
- **DX / Launch Agent**: owns CLI UX, README, examples, quickstarts, MCP registration, onboarding. Primary for commands/ and docs.
- **SDK Parity Agent**: verifies public API matches contracts, runs conformance tests

### Secondary Agents

- **Contracts Agent**: pushes generated code updates when contracts change — does not modify SDK logic
- **QA / Release Agent**: validates test coverage, conformance, release readiness, Knope config
- **Architect Agent**: reviews layering (Layer 0/1/2), engine plugin interface, module boundaries — advisory

### Approval Required

- New engine added to production list (out of experimental)
- Changes to `EnginePlugin` ABC or `ModelRuntime` protocol
- Changes to CLI command structure (new commands, renamed flags)
- New hard dependencies in `pyproject.toml`
- Server endpoint changes (`/v1/*`)
- Auth/security changes
- Release version bumps

### Auto-Approved

- Engine adapter improvements (internal)
- Test additions
- Benchmark instrumentation
- Generated code updates (from contracts)
- Documentation and example updates
- CLI help text improvements
- Hardware detection improvements
