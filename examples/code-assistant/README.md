# Octomil Code Assistant

A fully on-device code assistant that runs through Octomil. Zero cloud API calls, zero cost, complete data privacy.

## Quick Start

```bash
# Install Octomil with inference backends
pip install octomil[serve]

# Launch the demo (auto-starts octomil serve)
octomil demo code-assistant
```

That's it. The demo will:

1. Download and load a local LLM (Gemma 2B by default)
2. Start an OpenAI-compatible inference server
3. Open an interactive chat with live performance metrics

## What You'll See

```
  ╔══════════════════════════════════════════════════════╗
  ║         Octomil Code Assistant                       ║
  ║         100% on-device · zero cloud · zero cost     ║
  ╚══════════════════════════════════════════════════════╝

  ✓ Connected to octomil serve (gemma-2b on mlx-lm)

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
octomil demo code-assistant --model phi-3-mini

# Connect to an existing octomil serve instance
octomil demo code-assistant --url http://localhost:8080

# Use a different port for auto-started server
octomil demo code-assistant --port 9000

# Don't auto-start; fail if no server is running
octomil demo code-assistant --no-auto-start
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
│  octomil serve (local server)   │
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

This demo showcases the Octomil inference pipeline. The same `octomil serve` backend powers:

- **Local development**: Run models on your laptop
- **Phone deployment**: `octomil deploy --phone` sends the model to your phone
- **Fleet deployment**: `octomil deploy --rollout 10%` for production device fleets
- **Dashboard**: `octomil dashboard` shows metrics across all devices

To start a pilot:

```bash
# 1. Run the demo locally
octomil demo code-assistant

# 2. Deploy to your phone
octomil deploy gemma-2b --phone

# 3. See both devices in the dashboard
octomil dashboard
```

## Requirements

- Python 3.9+
- One of: Apple Silicon Mac (uses mlx-lm) or any platform (uses llama.cpp)
- 4GB+ RAM for 2B parameter models
- `pip install octomil[serve]` installs inference backends automatically
