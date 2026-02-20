# EdgeML Code Assistant

A fully on-device code assistant that runs through EdgeML. Zero cloud API calls, zero cost, complete data privacy.

## Quick Start

```bash
# Install EdgeML with inference backends
pip install edgeml[serve]

# Launch the demo (auto-starts edgeml serve)
edgeml demo code-assistant
```

That's it. The demo will:

1. Download and load a local LLM (Gemma 2B by default)
2. Start an OpenAI-compatible inference server
3. Open an interactive chat with live performance metrics

## What You'll See

```
  ╔══════════════════════════════════════════════════════╗
  ║         EdgeML Code Assistant                       ║
  ║         100% on-device · zero cloud · zero cost     ║
  ╚══════════════════════════════════════════════════════╝

  ✓ Connected to edgeml serve (gemma-2b on mlx-lm)

you> Write a Python function to merge two sorted lists

  def merge_sorted(a, b):
      result = []
      i = j = 0
      while i < len(a) and j < len(b):
          if a[i] <= b[j]:
              result.append(a[i])
              i += 1
          else:
              result.append(b[j])
              j += 1
      result.extend(a[i:])
      result.extend(b[j:])
      return result

  [142 tokens · 38.4 tok/s · 3694ms  TTFC: 412ms · cloud calls: 0 · saved: $0.0014]
```

Every response shows:

- **Token count** and **throughput** (tokens/second)
- **Latency** and **time to first chunk** (TTFC)
- **Cloud API calls**: always 0
- **Cost saved**: what this would have cost on GPT-4o

## Commands

| Command    | Description                  |
| ---------- | ---------------------------- |
| `/metrics` | Show full session statistics |
| `/clear`   | Clear conversation context   |
| `/quit`    | Exit and show final metrics  |

## Options

```bash
# Use a different model
edgeml demo code-assistant --model phi-3-mini

# Connect to an existing edgeml serve instance
edgeml demo code-assistant --url http://localhost:8080

# Use a different port for auto-started server
edgeml demo code-assistant --port 9000

# Don't auto-start; fail if no server is running
edgeml demo code-assistant --no-auto-start
```

## Run Directly

```bash
# Without the CLI wrapper
python examples/code-assistant/demo.py --model gemma-2b

# Or connect to an existing server
python examples/code-assistant/demo.py --url http://localhost:8080
```

## Architecture

```
┌────────────────────────────────┐
│  Terminal Chat UI (this demo)  │
│  - Rich markdown rendering     │
│  - Live metrics display        │
└────────────┬───────────────────┘
             │ HTTP (localhost only)
             ▼
┌────────────────────────────────┐
│  edgeml serve (local server)   │
│  - OpenAI-compatible API       │
│  - Auto-selects best engine    │
│  - KV cache for fast responses │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Inference Engine              │
│  - mlx-lm (Apple Silicon)     │
│  - llama.cpp (cross-platform) │
│  - Auto-benchmarked at start  │
└────────────────────────────────┘

Network calls: 0
Data sent to cloud: 0 bytes
```

## For Design Partners

This demo showcases the EdgeML inference pipeline. The same `edgeml serve` backend powers:

- **Local development**: Run models on your laptop
- **Phone deployment**: `edgeml deploy --phone` sends the model to your phone
- **Fleet deployment**: `edgeml deploy --rollout 10%` for production device fleets
- **Dashboard**: `edgeml dashboard` shows metrics across all devices

To start a pilot:

```bash
# 1. Run the demo locally
edgeml demo code-assistant

# 2. Deploy to your phone
edgeml deploy gemma-2b --phone

# 3. See both devices in the dashboard
edgeml dashboard
```

## Requirements

- Python 3.9+
- One of: Apple Silicon Mac (uses mlx-lm) or any platform (uses llama.cpp)
- 4GB+ RAM for 2B parameter models
- `pip install edgeml[serve]` installs inference backends automatically
