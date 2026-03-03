# Octomil

Run LLMs on your laptop, phone, or edge device. One command. OpenAI-compatible API.

[![CI](https://github.com/octomil/octomil-python/actions/workflows/ci.yml/badge.svg)](https://github.com/octomil/octomil-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/octomil-sdk)](https://pypi.org/project/octomil-sdk/)
[![License](https://img.shields.io/github/license/octomil/octomil-python)](https://github.com/octomil/octomil-python/blob/main/LICENSE)

## What is this?

Octomil is a CLI + Python SDK that serves open-weight LLMs locally with an OpenAI-compatible API. It auto-detects your hardware, picks the fastest inference engine, and gives you a drop-in replacement for cloud API calls -- works on Mac (MLX), Linux/Windows (llama.cpp), and deploys to phones.

## Quick start

```bash
curl -fsSL https://get.octomil.com | sh
octomil serve gemma-1b
```

That's it. You now have an OpenAI-compatible server on `localhost:8080`:

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

## Features

**Auto engine selection** -- benchmarks all available engines and picks the fastest:

```bash
octomil serve llama-3b
# => Detected: mlx-lm (38 tok/s), llama.cpp (29 tok/s), ollama (25 tok/s)
# => Using mlx-lm
```

**60+ models** -- Gemma, Llama, Phi, Qwen, DeepSeek, Mistral, Mixtral, and more:

```bash
octomil serve phi-mini          # Microsoft Phi-4 Mini (3.8B)
octomil serve deepseek-r1-7b    # DeepSeek R1 reasoning
octomil serve qwen3-4b          # Alibaba Qwen 3
octomil serve whisper-small     # Speech-to-text
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

**Model conversion** -- convert to CoreML (iOS) or TFLite (Android):

```bash
octomil convert model.pt --target ios,android
```

**Multi-model serving** -- load multiple models, route by request:

```bash
octomil serve --models smollm-360m,phi-mini,llama-3b
```

## Supported engines

| Engine                                                  | Platform            | Install                              |
| ------------------------------------------------------- | ------------------- | ------------------------------------ |
| [MLX](https://github.com/ml-explore/mlx)                | Apple Silicon Mac   | `pip install 'octomil-sdk[mlx]'`     |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)     | Mac, Linux, Windows | `pip install 'octomil-sdk[llama]'`   |
| [ONNX Runtime](https://onnxruntime.ai/)                 | All platforms       | `pip install 'octomil-sdk[onnx]'`    |
| [MLC-LLM](https://llm.mlc.ai/)                          | Mac, Linux, Android | auto-detected                        |
| [MNN](https://github.com/alibaba/MNN)                   | All platforms       | auto-detected                        |
| [ExecuTorch](https://pytorch.org/executorch/)           | Mobile              | auto-detected                        |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | All platforms       | `pip install 'octomil-sdk[whisper]'` |
| [Ollama](https://ollama.com/)                           | Mac, Linux          | auto-detected if running             |

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

| Command                     | Description                                 |
| --------------------------- | ------------------------------------------- |
| `octomil serve <model>`     | Start an OpenAI-compatible inference server |
| `octomil benchmark <model>` | Benchmark inference speed on your hardware  |
| `octomil deploy <model>`    | Deploy a model to edge devices              |
| `octomil rollback <model>`  | Roll back a deployment                      |
| `octomil convert <file>`    | Convert model to CoreML / TFLite            |
| `octomil pull <model>`      | Download a model                            |
| `octomil push <file>`       | Upload a model to registry                  |
| `octomil status <model>`    | Check deployment status                     |
| `octomil scan <path>`       | Security scan a model or app bundle         |
| `octomil pair`              | Pair with a phone for deployment            |
| `octomil dashboard`         | Open the web dashboard                      |
| `octomil login`             | Authenticate with Octomil                   |
| `octomil init`              | Initialize an organization                  |

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

## Python SDK

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

## Requirements

- Python 3.9+
- At least one inference engine (see [Supported engines](#supported-engines))
- macOS, Linux, or Windows

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
