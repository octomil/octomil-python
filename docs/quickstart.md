# Quickstart

Two ways to run inference with Octomil: the **native Responses API** (recommended) and the **OpenAI-compatible** path. Both work against the same local model server.

## Install

```bash
curl -fsSL https://get.octomil.com | sh
```

## Start a model

```bash
octomil serve gemma-1b
```

This starts an OpenAI-compatible server on `http://localhost:8080/v1`.

---

## Path 1: Native `responses.create()` (recommended)

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

## Path 2: OpenAI-compatible `chat.completions.create()`

Drop-in replacement for any OpenAI client. Point the base URL at your local server.

### Basic completion

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemma-1b",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

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

### Using curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-1b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Which path should I use?

| Feature                         | Native Responses API | OpenAI-compatible   |
| ------------------------------- | -------------------- | ------------------- |
| Local inference                 | Yes                  | Yes                 |
| Streaming                       | Yes                  | Yes                 |
| Routing policies                | Yes                  | No                  |
| Conversation threading          | Yes                  | Manual              |
| Multimodal input                | Yes                  | Partial             |
| Structured output / JSON        | Yes                  | Via response_format |
| Tool calling                    | Yes                  | Yes                 |
| Works with existing OpenAI code | No                   | Yes                 |

Use the **native Responses API** for new projects. Use the **OpenAI-compatible** path to migrate existing code with minimal changes.
