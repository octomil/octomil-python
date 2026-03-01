# pip install octomil-sdk
import asyncio
import time

import octomil

t0 = time.perf_counter()
client = octomil.Client(
    api_key="edg_6rHAx_bY1kmTwYbTh3BDRrmMbZy2Au4aUZ9wi2o_ogk",
    org_id="3c2eb004-2f28-44d1-9e85-10b3bd63a40c",
)
print(f"Client init: {(time.perf_counter() - t0) * 1000:.0f}ms\n")

# --- 1. Single prediction (cold — includes model load) ---
print("=== Single Prediction (cold start) ===")
t1 = time.perf_counter()
result = client.predict("phi-4-mini", [{"role": "user", "content": "What is 2+2? Answer in one sentence."}], max_tokens=64)
total_ms = (time.perf_counter() - t1) * 1000
print(result.text)
print(f"  total: {total_ms:.0f}ms | {result.metrics.tokens_per_second:.1f} tok/s | {result.metrics.total_tokens} tokens")
print(f"  ttfc:  {result.metrics.ttfc_ms:.0f}ms | total_duration: {result.metrics.total_duration_ms:.0f}ms")
print(f"  overhead (load+pull): {total_ms - result.metrics.total_duration_ms:.0f}ms\n")

# --- 2. Single prediction (warm — model cached) ---
print("=== Single Prediction (warm) ===")
t2 = time.perf_counter()
result2 = client.predict("phi-4-mini", [{"role": "user", "content": "What is 3+3? Answer in one sentence."}], max_tokens=64)
total_ms2 = (time.perf_counter() - t2) * 1000
print(result2.text)
print(f"  total: {total_ms2:.0f}ms | {result2.metrics.tokens_per_second:.1f} tok/s | {result2.metrics.total_tokens} tokens")
print(f"  ttfc:  {result2.metrics.ttfc_ms:.0f}ms | total_duration: {result2.metrics.total_duration_ms:.0f}ms\n")

# --- 3. Streaming ---
print("=== Streaming ===")


async def stream():
    t3 = time.perf_counter()
    first_chunk_time = None
    full_text = ""
    chunks = 0
    async for chunk in client.predict_stream(
        "phi-4-mini",
        [{"role": "user", "content": "Write a haiku about AI."}],
        max_tokens=128,
    ):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        print(chunk.text, end="", flush=True)
        full_text += chunk.text
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
    r = client.predict("phi-4-mini", [{"role": "user", "content": prompt}], max_tokens=32)
    ms = (time.perf_counter() - tp) * 1000
    print(f"  Q: {prompt}")
    print(f"  A: {r.text.strip()[:80]}")
    print(f"     {ms:.0f}ms | {r.metrics.tokens_per_second:.1f} tok/s")
batch_ms = (time.perf_counter() - t4) * 1000
print(f"  batch total: {batch_ms:.0f}ms\n")

client.dispose()
print("Done.")
