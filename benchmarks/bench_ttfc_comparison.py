"""Focused TTFC comparison — same prompt, 5 runs each."""
import asyncio
import json
import time
import httpx
import octomil

PROMPT = [{"role": "user", "content": "Write a haiku about AI."}]
MAX_TOKENS = 128
RUNS = 5

client = octomil.Client(
    api_key="edg_6rHAx_bY1kmTwYbTh3BDRrmMbZy2Au4aUZ9wi2o_ogk",
    org_id="3c2eb004-2f28-44d1-9e85-10b3bd63a40c",
)

# Warm both engines
client.predict("phi-4-mini", PROMPT, max_tokens=1)
httpx.post("http://localhost:11434/api/chat", json={
    "model": "phi4-mini", "messages": PROMPT, "stream": False, "options": {"num_predict": 1}
}, timeout=120)


async def octomil_ttfc():
    t0 = time.perf_counter()
    async for chunk in client.predict_stream("phi-4-mini", PROMPT, max_tokens=MAX_TOKENS):
        return (time.perf_counter() - t0) * 1000


async def ollama_ttfc():
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", "http://localhost:11434/api/chat", json={
            "model": "phi4-mini", "messages": PROMPT, "stream": True,
            "options": {"num_predict": MAX_TOKENS},
        }) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("message", {}).get("content"):
                    return (time.perf_counter() - t0) * 1000


async def main():
    print(f"{'Run':<5} {'Octomil TTFC':<15} {'Ollama TTFC':<15}")
    print("-" * 35)
    oct_times = []
    oll_times = []
    for i in range(RUNS):
        o = await octomil_ttfc()
        l = await ollama_ttfc()
        oct_times.append(o)
        oll_times.append(l)
        print(f"{i+1:<5} {o:<15.1f} {l:<15.1f}")

    print("-" * 35)
    print(f"{'avg':<5} {sum(oct_times)/len(oct_times):<15.1f} {sum(oll_times)/len(oll_times):<15.1f}")
    print(f"{'min':<5} {min(oct_times):<15.1f} {min(oll_times):<15.1f}")

asyncio.run(main())
client.dispose()
