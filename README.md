# Octomil

Run LLMs on your laptop, phone, or edge device. One command. OpenAI-compatible API.

[![CI](https://github.com/octomil/octomil-python/actions/workflows/ci.yml/badge.svg)](https://github.com/octomil/octomil-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/octomil)](https://pypi.org/project/octomil/)
[![License](https://img.shields.io/github/license/octomil/octomil-python)](https://github.com/octomil/octomil-python/blob/main/LICENSE)

## What is this?

Octomil is a CLI + Python SDK for running open-weight models locally behind an OpenAI-compatible API. It detects your hardware, picks the fastest available engine, and gives you a local-first replacement for cloud API calls on Mac, Linux, and Windows.

## Quick start

```bash
curl -fsSL https://get.octomil.com | sh
```

The installer runs `octomil setup` in the background — it creates a venv, installs the best engine for your hardware, downloads a recommended model, and registers the MCP server with your AI tools (Claude Code, Cursor, VS Code, Codex CLI).

Then start serving:

```bash
octomil serve gemma-1b
```

You now have an OpenAI-compatible server on `localhost:8080`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "gemma-1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or use any OpenAI client library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
r = client.chat.completions.create(
    model="gemma-1b",
    messages=[{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
)
print(r.choices[0].message.content)
```

## Native API

The `responses` API is the primary Octomil interface for new code. It gives you local inference, routing, multimodal inputs, and conversation threading without going through the OpenAI compatibility layer.

### responses.create

```python
import asyncio
from octomil.responses import OctomilResponses, ResponseRequest, text_input

responses = OctomilResponses()

async def main():
    result = await responses.create(ResponseRequest(
        model="gemma-1b",
        input=[text_input("Explain quantum computing in one sentence")],
    ))
    print(result.output[0].text)

asyncio.run(main())
```

Pass a plain string as shorthand:

```python
result = await responses.create(ResponseRequest.text("gemma-1b", "Hello"))
print(result.output[0].text)
```

### responses.stream

```python
import asyncio
from octomil.responses import OctomilResponses, ResponseRequest, TextDeltaEvent, DoneEvent, text_input

responses = OctomilResponses()

async def main():
    async for event in responses.stream(ResponseRequest(
        model="gemma-1b",
        input=[text_input("Write a haiku about the ocean")],
    )):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, DoneEvent):
            print()
            print(f"Tokens used: {event.response.usage.total_tokens}")

asyncio.run(main())
```

### With system instructions and conversation threading

```python
result1 = await responses.create(ResponseRequest(
    model="gemma-1b",
    input=[text_input("My name is Alice.")],
    instructions="You are a helpful assistant.",
))

# Continue the conversation by referencing the previous response
result2 = await responses.create(ResponseRequest(
    model="gemma-1b",
    input=[text_input("What's my name?")],
    previous_response_id=result1.id,
))
print(result2.output[0].text)  # "Your name is Alice."
```

The OpenAI-compatible `/v1/chat/completions` endpoint remains available for existing integrations. See [Migrating from OpenAI](#migrating-from-openai) if you are switching from the OpenAI SDK.

## Features

**Auto engine selection** -- benchmarks all available engines and picks the fastest:

```bash
octomil serve llama-3b
# => Detected: mlx-lm (38 tok/s), llama.cpp (29 tok/s), ollama (25 tok/s)
# => Using mlx-lm
```

**60+ models** -- Gemma, Llama, Phi, Qwen, DeepSeek, Mistral, Mixtral, and more:

```bash
octomil models                  # list all available models
octomil serve phi-mini          # Microsoft Phi-4 Mini (3.8B)
octomil serve deepseek-r1-7b    # DeepSeek R1 reasoning
octomil serve qwen3-4b          # Alibaba Qwen 3
octomil serve whisper-small     # Speech-to-text
```

**Interactive chat** -- one command from install to conversation:

```bash
octomil chat                        # auto-picks best model for your device
octomil chat qwen-coder-7b          # chat with a specific model
octomil chat llama-8b -s "You are a Python expert."
```

**Launch coding agents** -- power Codex, aider, or other agents with local inference:

```bash
octomil launch                  # pick an agent interactively
octomil launch codex            # launch OpenAI Codex CLI with local model
octomil launch codex --model codestral
```

**Deploy to phones** -- push models to iOS/Android devices:

```bash
octomil deploy gemma-1b --phone --rollout 10   # canary to 10% of devices
octomil status gemma-1b                        # monitor rollout
octomil rollback gemma-1b                      # instant rollback
```

**Benchmark your hardware**:

```bash
octomil benchmark gemma-1b
# Model: gemma-1b (4bit)
# Engine: mlx-lm
# Tokens/sec: 42.3
# Memory: 1.2 GB
# Time to first token: 89ms
```

**MCP server for AI tools** -- give Claude, Cursor, VS Code, and Codex access to local inference:

```bash
octomil mcp register                    # register with all detected AI tools
octomil mcp register --target claude    # register with Claude Code only
octomil mcp status                      # check registration status
```

**Model conversion** -- convert to CoreML (iOS) or TFLite (Android):

```bash
octomil convert model.pt --target ios,android
```

**Multi-model serving** -- load multiple models, route by request:

```bash
octomil serve --models smollm-360m,phi-mini,llama-3b
```

## Supported engines

| Engine                                                  | Platform            | Install                          |
| ------------------------------------------------------- | ------------------- | -------------------------------- |
| [MLX](https://github.com/ml-explore/mlx)                | Apple Silicon Mac   | `pip install 'octomil[mlx]'`     |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)     | Mac, Linux, Windows | `pip install 'octomil[llama]'`   |
| [ONNX Runtime](https://onnxruntime.ai/)                 | All platforms       | `pip install 'octomil[onnx]'`    |
| [MLC-LLM](https://llm.mlc.ai/)                          | Mac, Linux, Android | auto-detected                    |
| [MNN](https://github.com/alibaba/MNN)                   | All platforms       | auto-detected                    |
| [ExecuTorch](https://pytorch.org/executorch/)           | Mobile              | auto-detected                    |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | All platforms       | `pip install 'octomil[whisper]'` |
| [Ollama](https://ollama.com/)                           | Mac, Linux          | auto-detected if running         |

No engine installed? `octomil serve` tells you exactly what to install.

## Supported models

<details>
<summary>Full model list (60+ models)</summary>

| Model                  | Sizes             | Engines                        |
| ---------------------- | ----------------- | ------------------------------ |
| Gemma 3                | 1B, 4B, 12B, 27B  | MLX, llama.cpp, MNN, ONNX, MLC |
| Gemma 2                | 2B, 9B, 27B       | MLX, llama.cpp                 |
| Llama 3.2              | 1B, 3B            | MLX, llama.cpp, MNN, ONNX, MLC |
| Llama 3.1/3.3          | 8B, 70B           | MLX, llama.cpp                 |
| Phi-4 / Phi Mini       | 3.8B, 14B         | MLX, llama.cpp, MNN, ONNX      |
| Qwen 2.5               | 1.5B, 3B, 7B      | MLX, llama.cpp, MNN, ONNX      |
| Qwen 3                 | 0.6B - 32B        | MLX, llama.cpp                 |
| DeepSeek R1            | 1.5B - 70B        | MLX, llama.cpp                 |
| DeepSeek V3            | 671B (MoE)        | MLX, llama.cpp                 |
| Mistral / Nemo / Small | 7B, 12B, 24B      | MLX, llama.cpp                 |
| Mixtral                | 8x7B, 8x22B (MoE) | MLX, llama.cpp                 |
| Qwen 2.5 Coder         | 1.5B, 7B          | MLX, llama.cpp                 |
| CodeLlama              | 7B, 13B, 34B      | MLX, llama.cpp                 |
| StarCoder2             | 3B, 7B, 15B       | MLX, llama.cpp                 |
| Falcon 3               | 1B, 7B, 10B       | MLX, llama.cpp                 |
| SmolLM                 | 360M, 1.7B        | MLX, llama.cpp, MNN, ONNX      |
| Whisper                | tiny - large-v3   | Whisper.cpp                    |
| + many more            |                   |                                |

</details>

Use aliases: `octomil serve deepseek-r1` resolves to `deepseek-r1-7b`. Each model supports `4bit`, `8bit`, and `fp16` quantization variants.

## How it works

```
curl -fsSL https://get.octomil.com | sh
    │
    └── octomil setup (background)
         ├── 1. Find system Python with venv support
         ├── 2. Create ~/.octomil/engines/venv/
         ├── 3. Install best engine (mlx-lm on Apple Silicon, llama.cpp elsewhere)
         ├── 4. Download recommended model for your device
         └── 5. Register MCP server with AI tools (Claude, Cursor, VS Code, Codex)

octomil serve gemma-1b
    │
    ├── 1. Resolve model name → catalog lookup (aliases, quant variants)
    ├── 2. Detect engines     → MLX? llama.cpp? ONNX? Ollama running?
    ├── 3. Benchmark engines  → Run each, measure tok/s, pick fastest
    ├── 4. Download model     → HuggingFace Hub (cached after first pull)
    └── 5. Start server       → FastAPI on :8080, OpenAI-compatible API
                                 ├── POST /v1/chat/completions
                                 ├── POST /v1/completions
                                 └── GET  /v1/models
```

## CLI reference

| Command                     | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `octomil setup`             | Install engine, download model, register MCP servers |
| `octomil serve <model>`     | Start an OpenAI-compatible inference server          |
| `octomil chat [model]`      | Interactive chat (auto-starts server)                |
| `octomil launch [agent]`    | Launch a coding agent with local inference           |
| `octomil models`            | List available models                                |
| `octomil benchmark <model>` | Benchmark inference speed on your hardware           |
| `octomil warmup`            | Pre-download the recommended model for your device   |
| `octomil mcp register`      | Register MCP server with AI tools                    |
| `octomil mcp unregister`    | Remove MCP server from AI tools                      |
| `octomil mcp status`        | Show MCP registration status                         |
| `octomil mcp serve`         | Start the HTTP agent server (REST + A2A)             |
| `octomil deploy <model>`    | Deploy a model to edge devices                       |
| `octomil rollback <model>`  | Roll back a deployment                               |
| `octomil convert <file>`    | Convert model to CoreML / TFLite                     |
| `octomil pull <model>`      | Download a model                                     |
| `octomil push <file>`       | Upload a model to registry                           |
| `octomil status <model>`    | Check deployment status                              |
| `octomil scan <path>`       | Security scan a model or app bundle                  |
| `octomil completions`       | Print shell completion setup instructions            |
| `octomil pair`              | Pair with a phone for deployment                     |
| `octomil dashboard`         | Open the web dashboard                               |
| `octomil login`             | Authenticate with Octomil                            |
| `octomil init`              | Initialize an organization                           |

## Manifest

A **manifest** is a declarative YAML file (`octomil.yaml`) that describes which ML models your mobile or edge app needs, how they are delivered, and which capability each model serves. The iOS and Android SDKs read this file to manage model downloads and resolve the right model for each task — chat, transcription, keyboard prediction, and so on.

### Generate a manifest

```bash
octomil manifest init
```

This creates an `octomil.yaml` in the current directory, pre-populated with a starter configuration:

```yaml
# octomil.yaml — generated by `octomil manifest init`
version: "1"
models:
  - id: gemma-1b-chat
    capability: chat
    delivery: managed
    required: true
    download_url: https://models.octomil.com/gemma-1b-chat-4bit.onnx
    file_size_bytes: 800000000

  - id: whisper-small
    capability: transcription
    delivery: managed
    required: false
    download_url: https://models.octomil.com/whisper-small.onnx
    file_size_bytes: 240000000

  - id: smollm-360m
    capability: keyboard_prediction
    delivery: bundled
    bundled_path: models/smollm-360m.onnx
```

### Delivery modes

| Mode      | Behaviour                                                    |
| --------- | ------------------------------------------------------------ |
| `managed` | SDK downloads the model at runtime, caches it on device      |
| `bundled` | Model is included in the app binary at `bundled_path`        |
| `cloud`   | Inference runs remotely — no model artifact stored on device |

### Capabilities

Each manifest entry maps a model to a named capability the app requests at runtime:

| Capability            | Use case                            |
| --------------------- | ----------------------------------- |
| `chat`                | Conversational generation (chat UI) |
| `transcription`       | Speech-to-text (Whisper pipeline)   |
| `keyboard_prediction` | Next-word suggestion chips          |
| `embedding`           | Vector encoding for retrieval       |
| `classification`      | Text or image categorisation        |

### How iOS and Android SDKs consume it

**iOS** — parse the manifest and bootstrap the runtime:

```swift
import Octomil

let manifest = try AppManifest.load(from: Bundle.main.url(forResource: "octomil", withExtension: "yaml")!)
let catalog = ModelCatalogService(manifest: manifest)
catalog.bootstrap()

let runtime = catalog.runtime(for: .chat)
```

See the [iOS SDK README](https://github.com/octomil/octomil-ios) for full integration instructions.

**Android** — same pattern via the Kotlin SDK:

```kotlin
import ai.octomil.manifest.AppManifest
import ai.octomil.manifest.ModelCatalogService

val manifest = AppManifest.loadFromAssets(context, "octomil.yaml")
val catalog = ModelCatalogService(manifest)
catalog.bootstrap()

val runtime = catalog.runtimeFor(ModelCapability.CHAT)
```

See the [Android SDK README](https://github.com/octomil/octomil-android) for full integration instructions.

### Python SDK

Use the manifest programmatically for testing and fleet tooling:

```python
from octomil.manifest import AppManifest, AppModelEntry
from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.model_capability import ModelCapability
from octomil import Octomil

manifest = AppManifest(models=[
    AppModelEntry(
        id="gemma-1b-chat",
        capability=ModelCapability.CHAT,
        delivery=DeliveryMode.MANAGED,
        download_url="https://models.octomil.com/gemma-1b-chat-4bit.onnx",
    ),
])

client = Octomil(api_key="oct_...")
client.configure_manifest(manifest)
```

---

## vs. alternatives

|                                  | Octomil              | Ollama                  | llama.cpp (raw)        | Cloud APIs |
| -------------------------------- | -------------------- | ----------------------- | ---------------------- | ---------- |
| One-command serve                | yes                  | yes                     | no (build from source) | n/a        |
| OpenAI-compatible API            | yes                  | yes                     | partial                | native     |
| Auto engine selection            | yes (benchmarks all) | no (single engine)      | n/a                    | n/a        |
| Deploy to phones                 | yes                  | no                      | manual                 | no         |
| Fleet rollouts + rollback        | yes                  | no                      | no                     | n/a        |
| Model conversion (CoreML/TFLite) | yes                  | no                      | no                     | n/a        |
| A/B testing                      | yes                  | no                      | no                     | no         |
| Offline / on-device              | yes                  | yes                     | yes                    | no         |
| Cost per inference               | $0 (your hardware)   | $0                      | $0                     | $0.01-0.10 |
| 60+ models in catalog            | yes                  | yes (different catalog) | yes (manual download)  | varies     |
| Python SDK                       | yes                  | yes                     | community              | yes        |

## Migrating from OpenAI

Octomil is wire-compatible with the OpenAI API. Change two lines:

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After (local inference — no API key needed)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
```

That's it. `chat.completions.create`, streaming, tool calls, and audio transcriptions all work without further changes.

For a full guide including model name mapping, error code mapping, and a comparison of what's different: [docs/migration-from-openai.md](docs/migration-from-openai.md)

## SDKs

| SDK                                                   | Package                  | Status               | Inference Engine                                            |
| ----------------------------------------------------- | ------------------------ | -------------------- | ----------------------------------------------------------- |
| [Python](https://github.com/octomil/octomil-python)   | `octomil` (PyPI)         | Production (v2.10.1) | MLX, llama.cpp, ONNX, MLC, ExecuTorch, Whisper, MNN, Ollama |
| [Browser](https://github.com/octomil/octomil-browser) | `@octomil/browser` (npm) | Production (v1.0.0)  | ONNX Runtime Web (WebGPU + WASM)                            |
| [iOS](https://github.com/octomil/octomil-ios)         | Swift Package Manager    | Production (v1.1.0)  | CoreML + MLX                                                |
| [Android](https://github.com/octomil/octomil-android) | Maven (GitHub Packages)  | Production (v1.2.0)  | TFLite + vendor NPU                                         |
| [Node](https://github.com/octomil/octomil-node)       | `@octomil/sdk` (npm)     | WIP (v0.1.0)         | ONNX Runtime Node                                           |

### Python SDK

For fleet management, model registry, and A/B testing:

```python
from octomil import Octomil

client = Octomil(api_key="oct_...", org_id="org_123")

# Register and deploy a model
model = client.registry.ensure_model(name="sentiment", framework="pytorch")
client.rollouts.create(model_id=model["id"], version="1.0.0", rollout_percentage=10)

# Run an A/B test
client.experiments.create(
    name="v1-vs-v2",
    model_id=model["id"],
    control_version="1.0.0",
    treatment_version="1.1.0",
)
```

## MCP Server & AI Tool Integration

Octomil registers as an MCP server across your AI coding tools so they can use local inference. `octomil setup` does this automatically, or you can run it manually:

```bash
octomil mcp register                    # Claude Code, Cursor, VS Code, Codex CLI
octomil mcp register --target cursor    # single tool
octomil mcp status                      # check what's registered
octomil mcp unregister                  # remove from all tools
```

### HTTP Agent Server & x402 Payments

Octomil also exposes its tools over HTTP with an A2A agent card, OpenAPI docs, and optional micro-payments via the [x402 protocol](https://www.x402.org/).

```bash
octomil mcp serve                       # start HTTP agent server on :8402
octomil mcp serve --port 9000           # custom port

# With x402 payment gating (agents pay per call)
OCTOMIL_X402_ADDRESS=0xYourWallet \
OCTOMIL_SETTLER_TOKEN=s402_... \
octomil mcp serve --x402
```

**How it works:**

1. Agent calls an Octomil tool (e.g. `/api/v1/run_inference`)
2. Server returns `402 Payment Required` with x402 payment requirements
3. Agent signs an EIP-3009 `transferWithAuthorization` and retries with `x-payment` header
4. Server verifies the signature, serves the response, and accumulates the payment
5. When payments reach the settlement threshold ($1 USDC by default), the batch is submitted to [settle402](https://settle402.dev) for on-chain settlement via Multicall3

**Environment variables:**

| Variable                 | Default                     | Description                                        |
| ------------------------ | --------------------------- | -------------------------------------------------- |
| `OCTOMIL_X402_ADDRESS`   | —                           | Your wallet address (where you get paid)           |
| `OCTOMIL_X402_PRICE`     | `1000`                      | Price per call in base units (1000 = $0.001 USDC)  |
| `OCTOMIL_X402_NETWORK`   | `base`                      | Chain: base, ethereum, polygon, arbitrum, optimism |
| `OCTOMIL_X402_THRESHOLD` | `1.0`                       | Settlement threshold in USD                        |
| `OCTOMIL_SETTLER_URL`    | `https://api.settle402.dev` | settle402 batch settlement endpoint                |
| `OCTOMIL_SETTLER_TOKEN`  | —                           | settle402 API key                                  |

## Requirements

- Python 3.9+
- At least one inference engine (see [Supported engines](#supported-engines))
- macOS, Linux, or Windows

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
