"""Ollama benchmark — same prompts as test.py for comparison."""
import asyncio
import json
import time

import httpx

BASE = "http://localhost:11434"
MODEL = "phi4-mini"

t0 = time.perf_counter()
# Warm check — ensure Ollama is running
httpx.get(f"{BASE}/api/tags", timeout=5)
print(f"Client init: {(time.perf_counter() - t0) * 1000:.0f}ms\n")

# --- 1. Single prediction (cold — forces model load) ---
print("=== Single Prediction (cold start) ===")
t1 = time.perf_counter()
resp = httpx.post(
    f"{BASE}/api/chat",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one sentence."}],
        "stream": False,
        "options": {"num_predict": 64},
    },
    timeout=120,
)
total_ms = (time.perf_counter() - t1) * 1000
data = resp.json()
text = data["message"]["content"]
tok_count = data.get("eval_count", 0)
eval_ns = data.get("eval_duration", 1)
load_ns = data.get("load_duration", 0)
prompt_ns = data.get("prompt_eval_duration", 0)
total_ns = data.get("total_duration", 0)
tps = tok_count / (eval_ns / 1e9) if eval_ns else 0
ttfc_ms = prompt_ns / 1e6
print(text)
print(f"  total: {total_ms:.0f}ms | {tps:.1f} tok/s | {tok_count} tokens")
print(f"  ttfc:  {ttfc_ms:.0f}ms | total_duration: {total_ns / 1e6:.0f}ms")
print(f"  overhead (load): {load_ns / 1e6:.0f}ms\n")

# --- 2. Single prediction (warm) ---
print("=== Single Prediction (warm) ===")
t2 = time.perf_counter()
resp2 = httpx.post(
    f"{BASE}/api/chat",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is 3+3? Answer in one sentence."}],
        "stream": False,
        "options": {"num_predict": 64},
    },
    timeout=120,
)
total_ms2 = (time.perf_counter() - t2) * 1000
d2 = resp2.json()
tok2 = d2.get("eval_count", 0)
eval_ns2 = d2.get("eval_duration", 1)
prompt_ns2 = d2.get("prompt_eval_duration", 0)
total_ns2 = d2.get("total_duration", 0)
tps2 = tok2 / (eval_ns2 / 1e9) if eval_ns2 else 0
print(d2["message"]["content"])
print(f"  total: {total_ms2:.0f}ms | {tps2:.1f} tok/s | {tok2} tokens")
print(f"  ttfc:  {prompt_ns2 / 1e6:.0f}ms | total_duration: {total_ns2 / 1e6:.0f}ms\n")

# --- 3. Streaming ---
print("=== Streaming ===")


async def stream():
    t3 = time.perf_counter()
    first_chunk_time = None
    full_text = ""
    chunks = 0
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Write a haiku about AI."}],
                "stream": True,
                "options": {"num_predict": 128},
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                token = obj.get("message", {}).get("content", "")
                if token:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    print(token, end="", flush=True)
                    full_text += token
                    chunks += 1
    elapsed = (time.perf_counter() - t3) * 1000
    ttfc = ((first_chunk_time or t3) - t3) * 1000
    print(f"\n  total: {elapsed:.0f}ms | ttfc: {ttfc:.0f}ms | {chunks} chunks | {len(full_text)} chars\n")


asyncio.run(stream())

# --- 4. Batch (sequential, same cached model) ---
print("=== Batch ===")
prompts = [
    "Name one planet.",
    "Name one color.",
    "Name one animal.",
]
t4 = time.perf_counter()
for prompt in prompts:
    tp = time.perf_counter()
    r = httpx.post(
        f"{BASE}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 32},
        },
        timeout=120,
    )
    ms = (time.perf_counter() - tp) * 1000
    rd = r.json()
    tok = rd.get("eval_count", 0)
    ev = rd.get("eval_duration", 1)
    tps_b = tok / (ev / 1e9) if ev else 0
    print(f"  Q: {prompt}")
    print(f"  A: {rd['message']['content'].strip()[:80]}")
    print(f"     {ms:.0f}ms | {tps_b:.1f} tok/s")
batch_ms = (time.perf_counter() - t4) * 1000
print(f"  batch total: {batch_ms:.0f}ms\n")

print("Done.")
