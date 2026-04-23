# Architecture — octomil-python

## Repo Responsibility

Python SDK, CLI, local runtime, serve layer, and runner. Owns:

- **SDK client** — Hosted API client for chat, completions, embeddings, responses, model catalog
- **CLI** — `octomil` command-line tool (chat, run, serve, deploy, configure, scan, bench)
- **Local runtime** — On-device inference via pluggable engines (llama.cpp, MLX, MNN, ONNX RT, whisper.cpp, etc.)
- **Serve** — OpenAI-compatible HTTP server wrapping local inference (`octomil serve`)
- **Local runner** — Background process managing model lifecycle and inference sessions
- **Device agent** — Device-side control plane agent for fleet management
- **Unified facade** — Single entry point (`octomil/facade.py`) routing to hosted or local backends

## Module Layout

```
octomil/
├── _generated/          # Enum types from octomil-contracts — DO NOT HAND-EDIT
├── runtime/
│   ├── core/            # Engine registry, session management, kernel
│   ├── engines/         # Engine adapters (llama.cpp, MLX, MNN, ORT, whisper, etc.)
│   ├── routing/         # Smart routing (local vs cloud), query routing
│   ├── planner/         # Runtime planning and model selection
│   └── packaging/       # Model packaging and artifact handling
├── serve/               # OpenAI-compatible HTTP server (uvicorn + FastAPI)
├── local_runner/        # Background runner process (model lifecycle, sessions)
├── device_agent/        # Fleet management agent (desired state, heartbeat)
├── engines/             # Engine entry points
├── audio/               # Audio transcription/TTS
├── text/                # Text generation helpers
├── models/              # Model types and helpers
├── config/              # Configuration management
├── hardware/            # Hardware detection (Metal, CUDA, ROCm, CPU)
├── mcp/                 # MCP server integration
├── agents/              # Agent framework
├── commands/            # CLI command implementations
├── sources/             # Model source resolution
├── manifest/            # Engine manifest types
├── execution/           # Execution kernel
├── workflows/           # Multi-step workflow orchestration
├── benchmark/           # Benchmarking tools
├── facade.py            # Unified entry point
├── client.py            # Hosted API client
├── chat.py              # Chat completions
├── chat_client.py       # Chat client wrapper
├── embeddings.py        # Embedding API
├── responses/           # Responses API
├── cli.py               # CLI entry point
├── auth.py              # Auth configuration
├── routing.py           # High-level routing
├── smart_router.py      # Smart routing logic
├── streaming.py         # Streaming helpers
├── telemetry.py         # OpenTelemetry instrumentation
└── device_context.py    # Device capability detection

tests/
├── conformance/         # Contract conformance tests
├── device_agent/        # Device agent tests
├── integration/         # Integration tests
└── test_*.py            # Unit tests (mirror octomil/ structure)
```

## Boundary Rules

- **Runtime must not import CLI**: `octomil/runtime/` must never import from `octomil/cli.py` or `octomil/commands/`.
- **Serve must not import CLI**: `octomil/serve/` must never import from `octomil/cli.py` or `octomil/commands/`.
- **CLI can import anything**: CLI commands orchestrate client, runtime, and serve.
- **`_generated/` is read-only**: Never hand-edit. Run codegen from `octomil-contracts`.
- **Engine adapters are optional**: Each engine in `runtime/engines/` guards its imports; missing native libs raise clear errors, not import crashes.

## Public API Surfaces

- Python package: `import octomil` — client, facade, chat, embeddings, responses
- CLI: `octomil` binary (entry point: `octomil.cli:main`)
- Serve: `octomil serve` — OpenAI-compatible `/v1/chat/completions` endpoint

## Generated Code

Location: `octomil/_generated/`

Generated from `octomil-contracts/enums/*.yaml` via codegen. Version tracked in `_generated/.contract-version.json`.

**Do not hand-edit.** Run codegen from `octomil-contracts` to update.

## Source-of-Truth Dependencies

| Dependency        | Source                                                 |
| ----------------- | ------------------------------------------------------ |
| Enum definitions  | `octomil-contracts/enums/*.yaml`                       |
| Engine manifest   | `octomil-contracts/fixtures/core/engine_manifest.json` |
| API semantics     | `octomil-contracts/schemas/`                           |
| Conformance tests | `octomil-contracts/conformance/`                       |

## Test Commands

```bash
# All tests
pytest tests/ -v

# Tests with coverage
pytest --cov=octomil --cov-report=term-missing --cov-fail-under=80

# Single test
pytest tests/test_chat.py::test_function_name -v

# Conformance tests
pytest tests/conformance/ -v

# Lint + format (no venv needed)
uvx ruff check . --fix
uvx ruff format .

# Type check
mypy octomil/
```

## Review Checklist

- [ ] New enum value: was it added to `octomil-contracts` first, then regenerated?
- [ ] Boundary violation: does runtime import CLI? Does serve import CLI?
- [ ] Engine change: are native deps guarded with try/except ImportError?
- [ ] Facade change: does it handle both hosted and local paths?
- [ ] CLI change: does `octomil --help` still work?
- [ ] Conformance: do `tests/conformance/` tests still pass?
- [ ] Streaming: does it work for both SSE and iterator responses?
