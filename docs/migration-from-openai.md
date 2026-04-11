# Migrating from OpenAI to Octomil

Octomil is wire-compatible with the OpenAI API. You can migrate in 2 minutes by changing your base URL and API key config — or you can adopt the native `responses` API to get local inference, routing policies, and manifest-driven delivery.

## Quick migration (2 minutes)

### Option A: Local server (recommended — no account needed)

Start a local OpenAI-compatible server and point your client at it:

```bash
octomil serve
```

Then change two lines in your existing OpenAI client code:

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After: local inference (no API key needed)
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
```

### Option B: Hosted API

Use the Octomil cloud API as a drop-in replacement:

```python
# After: hosted (use your server key)
from openai import OpenAI
client = OpenAI(base_url="https://api.octomil.com/v1", api_key="YOUR_SERVER_KEY")
```

Everything else — `chat.completions.create`, streaming, tool calls — works without changes.

---

## Side-by-side code examples

### Chat completions (non-streaming)

| OpenAI                                 | Octomil                                                            |
| -------------------------------------- | ------------------------------------------------------------------ |
| `base_url="https://api.openai.com/v1"` | `base_url="http://127.0.0.1:8080/v1"`                              |
| `api_key="sk-..."`                     | `api_key="unused"` (local) or `api_key="YOUR_SERVER_KEY"` (hosted) |
| `model="gpt-4o"`                       | `model="gemma-3-4b"` (see model mapping below)                     |

```python
# OpenAI
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print(response.choices[0].message.content)
```

```python
# Octomil (OpenAI-compatible path)
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print(response.choices[0].message.content)
```

### Chat completions (streaming)

```python
# OpenAI
from openai import OpenAI

client = OpenAI(api_key="sk-...")
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about the ocean."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

```python
# Octomil (OpenAI-compatible path — identical except base_url and model)
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
stream = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[{"role": "user", "content": "Write a haiku about the ocean."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

The SSE format is identical. No changes needed to your streaming loop.

### Audio transcription

```python
# OpenAI
from openai import OpenAI

client = OpenAI(api_key="sk-...")
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
    )
print(transcript.text)
```

```python
# Octomil (OpenAI-compatible path)
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-small",  # see model mapping below
        file=f,
    )
print(transcript.text)
```

---

## Model name mapping

| OpenAI model        | Octomil equivalent         | Notes                           |
| ------------------- | -------------------------- | ------------------------------- |
| `gpt-4o`            | `gemma-3-12b`              | Best quality available locally  |
| `gpt-4o-mini`       | `gemma-3-4b`               | Good quality, fast              |
| `gpt-3.5-turbo`     | `gemma-3-1b` or `phi-mini` | Fastest, smallest               |
| `gpt-4-turbo`       | `llama-3.3-70b`            | Large model, requires 40GB+ RAM |
| `o1` / `o1-mini`    | `deepseek-r1-7b`           | Reasoning model                 |
| `o3-mini`           | `deepseek-r1-1.5b`         | Small reasoning model           |
| `whisper-1`         | `whisper-small`            | Default transcription           |
| `whisper-1` (large) | `whisper-large-v3`         | Highest accuracy transcription  |
| `text-embedding-*`  | `nomic-embed-text-v1.5`    | `octomil embed` or SDK          |

Use `octomil models` to see the full catalog of 60+ available models.

---

## API key differences

|              | OpenAI           | Octomil (local)      | Octomil (hosted)            |
| ------------ | ---------------- | -------------------- | --------------------------- |
| Key format   | `sk-...`         | No key needed        | `oct_pub_...` (publishable) |
| Where to set | `OPENAI_API_KEY` | Any non-empty string | `OCTOMIL_SERVER_KEY`        |
| Key rotation | OpenAI dashboard | N/A                  | Octomil dashboard           |
| Rate limits  | Account-based    | None (your hardware) | Fleet-based                 |

For local inference, the `api_key` parameter is required by the OpenAI client library but is ignored by Octomil. Pass any non-empty string.

```python
# Local — no real key needed
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")

# Hosted — use your server key
client = OpenAI(base_url="https://api.octomil.com/v1", api_key="YOUR_SERVER_KEY")
```

---

## Base URL configuration

| Environment                 | Base URL                     |
| --------------------------- | ---------------------------- |
| Local server (default port) | `http://127.0.0.1:8080/v1`   |
| Local server (custom port)  | `http://127.0.0.1:<port>/v1` |
| Hosted API                  | `https://api.octomil.com/v1` |

Start a local server:

```bash
octomil serve gemma-3-4b          # starts on :8080 by default
octomil serve gemma-3-4b --port 9000
```

---

## Error code mapping

When an error occurs, Octomil returns an `ErrorCode` from the `octomil._generated.error_code` module. The table below maps common OpenAI error types to their Octomil equivalents.

| OpenAI error            | HTTP status | Octomil `ErrorCode`         | Description                                               |
| ----------------------- | ----------- | --------------------------- | --------------------------------------------------------- |
| `AuthenticationError`   | 401         | `INVALID_API_KEY`           | API key invalid or missing                                |
| `AuthenticationError`   | 401         | `AUTHENTICATION_FAILED`     | Token expired, revoked, or malformed                      |
| `PermissionDeniedError` | 403         | `FORBIDDEN`                 | Insufficient permissions                                  |
| `RateLimitError`        | 429         | `RATE_LIMITED`              | Too many requests                                         |
| `BadRequestError`       | 400         | `INVALID_INPUT`             | Malformed request body                                    |
| `BadRequestError`       | 400         | `CONTEXT_TOO_LARGE`         | Input exceeds model context window                        |
| `NotFoundError`         | 404         | `MODEL_NOT_FOUND`           | Requested model does not exist                            |
| `APIConnectionError`    | —           | `NETWORK_UNAVAILABLE`       | No connectivity to server                                 |
| `APITimeoutError`       | —           | `REQUEST_TIMEOUT`           | Server did not respond in time                            |
| `InternalServerError`   | 500         | `SERVER_ERROR`              | 5xx from server                                           |
| `APIError` (stream)     | —           | `STREAM_INTERRUPTED`        | Streaming response cut short                              |
| —                       | —           | `RUNTIME_UNAVAILABLE`       | No compatible engine for this model                       |
| —                       | —           | `INSUFFICIENT_MEMORY`       | OOM during inference                                      |
| —                       | —           | `POLICY_DENIED`             | Routing policy denied the request                         |
| —                       | —           | `CLOUD_FALLBACK_DISALLOWED` | Local inference failed, cloud fallback disabled by policy |

---

## What's different from OpenAI

**Local inference:** Models run on your hardware. There are no usage costs, no data sent to third-party servers, and no network latency once a model is loaded.

**Engine auto-selection:** Octomil benchmarks all available engines (MLX, llama.cpp, ONNX, etc.) and picks the fastest for your hardware. You don't choose an engine — you choose a model and Octomil handles the rest.

**Routing policies:** Octomil can route requests between local and cloud inference based on policies you define. The `ResponseRequest` supports a `metadata` field for routing hints.

```python
# Force local-only inference (no cloud fallback)
result = await responses.create(ResponseRequest(
    model="gemma-3-4b",
    input="Hello",
    metadata={"routing.policy": "local_only"},
))
```

**Manifest-driven delivery:** For mobile deployments, models are versioned and delivered via an `octomil.yaml` manifest. This enables controlled rollouts, A/B testing, and instant rollback across device fleets. The `/v1/chat/completions` endpoint is unaffected by manifests — they apply to the native control plane.

---

## What's the same as OpenAI

- `/v1/chat/completions` request and response schema
- `/v1/audio/transcriptions` request and response schema
- Server-sent events (SSE) format for streaming: `data: {...}\n\ndata: [DONE]\n\n`
- Tool calling format (function definitions, tool results in messages)
- `role` values: `system`, `user`, `assistant`, `tool`
- `finish_reason` values: `stop`, `tool_calls`, `length`
- Any OpenAI client library works — the official Python/Node SDKs, LangChain, LlamaIndex, etc.

---

## Native API (recommended for new code)

The OpenAI-compatible path is great for migration, but the native `responses` API gives you access to all Octomil features: routing, multimodal inputs, conversation threading, and structured output.

See the [Native API section in the README](../README.md#native-api) for usage examples.
