# Quickstart

Three ways to run inference with Octomil: the **CLI** (fastest start), the **local OpenAI-compatible server**, and the **hosted API**.

## Install

```bash
curl -fsSL https://get.octomil.com | sh
```

Or via pip:

```bash
pip install octomil
```

---

## Path 1: Local CLI (no server, no account needed)

The fastest way to get started. No server process, no API key, no account.

```bash
# Chat / responses
octomil run "What can you help me with?"

# Stream output (default when running in a terminal)
octomil run "Explain quantum computing in one sentence."

# Embeddings
octomil embed "On-device AI inference at scale" --json

# Transcription
octomil transcribe meeting.wav
```

### Options

```bash
octomil run --model gemma-1b "Hello"              # pick a model
octomil run --json "Return a haiku about SQLite"   # structured JSON output
cat prompt.txt | octomil run                       # pipe stdin
octomil run --no-stream "Hello"                    # disable streaming
```

---

## Path 2: OpenAI-Compatible Local Server

Start a persistent server and use any OpenAI-compatible client.

```bash
octomil serve
```

This starts an OpenAI-compatible server on `http://127.0.0.1:8080/v1`.

### Using curl

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

### Using the OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemma-1b",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")

stream = client.chat.completions.create(
    model="gemma-1b",
    messages=[{"role": "user", "content": "Write a haiku about the ocean."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

---

## Path 3: Hosted API

Use the Octomil cloud API for inference without running anything locally.

```bash
export OCTOMIL_SERVER_KEY=YOUR_SERVER_KEY

curl https://api.octomil.com/v1/responses \
  -H "Authorization: Bearer $OCTOMIL_SERVER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","input":"Hello"}'
```

Or via the Python SDK:

```bash
export OCTOMIL_SERVER_KEY=YOUR_SERVER_KEY
export OCTOMIL_ORG_ID=YOUR_ORG_ID
```

```python
import asyncio
from octomil import Octomil

async def main():
    client = Octomil.from_env()
    await client.initialize()
    response = await client.responses.create(model="phi-4-mini", input="Hello")
    print(response.output_text)

asyncio.run(main())
```

---

## Path 4: Native `responses.create()` (advanced)

The native Responses API gives you local inference, routing policies, multimodal inputs, structured output, tool use, and conversation threading.

### Basic completion

```python
import asyncio
from octomil.responses import OctomilResponses, ResponseRequest, text_input

responses = OctomilResponses()

async def main():
    result = await responses.create(ResponseRequest(
        model="gemma-1b",
        input=[text_input("Explain quantum computing in one sentence.")],
    ))
    print(result.output[0].text)

asyncio.run(main())
```

### Streaming

```python
import asyncio
from octomil.responses import (
    OctomilResponses, ResponseRequest, TextDeltaEvent, DoneEvent, text_input,
)

responses = OctomilResponses()

async def main():
    async for event in responses.stream(ResponseRequest(
        model="gemma-1b",
        input=[text_input("Write a haiku about the ocean.")],
    )):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, DoneEvent):
            print()

asyncio.run(main())
```

### System instructions and conversation threading

```python
result1 = await responses.create(ResponseRequest(
    model="gemma-1b",
    input=[text_input("My name is Alice.")],
    instructions="You are a helpful assistant.",
))

result2 = await responses.create(ResponseRequest(
    model="gemma-1b",
    input=[text_input("What's my name?")],
    previous_response_id=result1.id,
))
print(result2.output[0].text)  # "Your name is Alice."
```

### Routing policy via metadata

```python
result = await responses.create(ResponseRequest(
    model="gemma-1b",
    input=[text_input("Hello")],
    metadata={"routing.policy": "local_only"},
))
print(result.locality)  # "on_device"
```

---

## Which path should I use?

| Feature                         | CLI        | Local Server (OpenAI) | Hosted API | Native Responses API |
| ------------------------------- | ---------- | --------------------- | ---------- | -------------------- |
| Zero setup                      | Yes        | One command           | Key needed | Code needed          |
| Local inference                 | Yes        | Yes                   | No         | Yes                  |
| Streaming                       | Yes        | Yes                   | Yes        | Yes                  |
| Routing policies                | Via policy | No                    | No         | Yes                  |
| Conversation threading          | No         | Manual                | Manual     | Yes                  |
| Works with existing OpenAI code | No         | Yes                   | Yes        | No                   |
| Embeddings                      | Yes        | Yes                   | Yes        | Yes                  |
| Transcription                   | Yes        | Yes                   | Yes        | Yes                  |

Use the **CLI** for quick one-off tasks. Use the **local server** to migrate existing OpenAI code. Use the **hosted API** when you don't want to run models locally. Use the **native Responses API** for new projects that need full control.
