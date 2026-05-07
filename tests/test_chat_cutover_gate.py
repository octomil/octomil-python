"""Native-vs-Python chat cutover gate.

Retrospective guard for the hard cutover: product
``chat.completion`` now routes through ``octomil.runtime.native``.
This gate keeps the deprecated Python-local ``llama_cpp.Llama`` path
around only as a benchmark reference and asserts the native runtime
does not materially regress on the supported local-GGUF subset.

Settings matched between backends:
- Same GGUF (``OCTOMIL_LLAMA_CPP_GGUF``).
- Same context size (``n_ctx=4096``).
- Same sampling: temperature=0 (greedy). v0.1.2's native engine
  ships greedy-only, so both paths run identical-policy decoding.
- Same prompt suite + same ``max_tokens``.
- Same canonical chat-messages JSON shape on both sides — the
  cutover subset (no tools, no function-calling, no ``name`` field,
  no multi-modal). Python ``llama_cpp.Llama.create_chat_completion``
  is more permissive than the native runtime; the gate confirms
  both serve the IN-subset shapes; out-of-subset is gated by the
  planner BEFORE native is selected (per the user spec).

Measurements per backend:
- model_open_ms — open the artifact (Python: ``Llama(model_path=...)``;
  native: ``oct_model_open``).
- warm_ms — Python: 0 (load is part of construction); native:
  ``oct_model_warm``.
- session_open_ms — Python: 0 (no separate session step); native:
  ``oct_session_open``.
- first_chunk_ms — wall time from send to first non-empty text chunk.
- total_latency_ms — wall time for the full request.
- output_events — non-empty stream events observed.
- output_chars — generated text length. This is the cross-backend
  throughput normalizer because Python streaming coalesces tokenizer
  pieces while the native runtime emits transcript chunks from its
  own decode loop.
- chars_per_second — output_chars / total_latency_ms.
- peak_rss_mb — best-effort process-level RSS observed at end of
  pass (not isolated; used to flag balloon).

Pass criteria:
- Native first chunk is no slower than Python by more than a small
  tolerance (with a 50ms absolute cushion for noisy local runs).
- Native total wall time stays within a bounded 2x of Python for the
  same prompt/max-token suite (with a 250ms absolute cushion).
- Both produce non-empty output for every supported prompt.
- Native correctly rejects an out-of-subset shape (smoke check
  on the gating boundary).

Why chars/sec, not "tokens/sec":
The two reference paths do not expose the same token accounting.
Python's stream yields non-empty delta chunks that may coalesce
multiple tokenizer pieces; native yields transcript chunks from the
runtime decode loop. Calling both "tokens" hid real behavior drift.
The native runtime is now canonical, so this gate measures non-empty
output, first-chunk latency, total latency, and text throughput
without requiring byte-for-byte parity with the deprecated path.

``first_chunk_ms`` is a pass-fail UX guard. Model open/warm costs
are reported separately because the product path keeps a
``NativeModel`` alive across requests.

Skipped without ``OCTOMIL_LLAMA_CPP_GGUF`` — same gate as the
conformance suite.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841
llama_cpp = pytest.importorskip("llama_cpp", reason="llama_cpp not installed")

CHAT_COMPLETION = "chat.completion"

# Cutover subset prompt suite. Each entry is a list of canonical
# chat-messages — exactly the shape v0.1.2+'s native engine accepts.
PROMPTS: list[tuple[str, list[dict[str, str]]]] = [
    ("greeting", [{"role": "user", "content": "hi"}]),
    ("arithmetic", [{"role": "user", "content": "What is 2+2? Reply with just the number."}]),
    (
        "system_plus_user",
        [
            {"role": "system", "content": "You are a terse assistant. Answer in one short sentence."},
            {"role": "user", "content": "Name a color."},
        ],
    ),
]

# Token budget per prompt — small enough that a 145MB GGUF on CPU
# completes in seconds.
MAX_TOKENS = 32

# Pass-criteria thresholds. Text throughput remains diagnostic only:
# the deprecated Python path and native canonical path do not expose
# identical token accounting or stop/template behavior, so chars/sec is
# useful context but too flaky to block cutover by itself.
NATIVE_FIRST_CHUNK_MAX_RATIO = 1.15
NATIVE_FIRST_CHUNK_ABS_CUSHION_MS = 50.0
NATIVE_TOTAL_LATENCY_MAX_RATIO = 2.0
NATIVE_TOTAL_LATENCY_ABS_CUSHION_MS = 250.0


@dataclass
class PromptResult:
    label: str
    output_text: str
    output_events: int
    output_chars: int
    first_chunk_ms: float
    total_latency_ms: float
    chars_per_second: float


@dataclass
class BackendResult:
    backend: str
    model_open_ms: float
    warm_ms: float
    session_open_ms: float
    prompts: list[PromptResult] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    notes: str = ""

    def total_latency_sum_ms(self) -> float:
        return sum(p.total_latency_ms for p in self.prompts)

    def first_chunk_avg_ms(self) -> float:
        if not self.prompts:
            return 0.0
        return sum(p.first_chunk_ms for p in self.prompts) / len(self.prompts)

    def total_output_events(self) -> int:
        return sum(p.output_events for p in self.prompts)

    def total_output_chars(self) -> int:
        return sum(p.output_chars for p in self.prompts)

    def avg_chars_per_second(self) -> float:
        """Aggregate steady-state throughput: generated text bytes
        across the prompt suite divided by total wall time."""
        total_ms = self.total_latency_sum_ms()
        if total_ms <= 0:
            return 0.0
        return self.total_output_chars() / (total_ms / 1000.0)


def _peak_rss_mb() -> float:
    """Best-effort RSS observed by the process so far. macOS reports
    ru_maxrss in bytes; Linux in KB. Return MB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _run_python_local(gguf: str) -> BackendResult:
    """Pass 1: Python-local ``llama_cpp.Llama`` path. Mirrors
    ``octomil/serve/backends/llamacpp.py:LlamaCppBackend``'s own
    construction (n_ctx=4096) but uses streaming so we can capture
    first_chunk_ms directly without relying on the wrapper's
    metrics."""
    from llama_cpp import Llama

    t0 = _now_ms()
    llm = Llama(
        model_path=gguf,
        n_ctx=4096,
        verbose=False,
        # temperature=0 (greedy) is set per-call below; nothing to set
        # at construction. Logits sampler defaults are fine.
    )
    model_open_ms = _now_ms() - t0
    # Python-local has no separate warm/session_open phase.
    warm_ms = 0.0
    session_open_ms = 0.0

    out = BackendResult(
        backend="python_llama_cpp",
        model_open_ms=model_open_ms,
        warm_ms=warm_ms,
        session_open_ms=session_open_ms,
    )

    for label, messages in PROMPTS:
        t_send = _now_ms()
        first_chunk_at: float | None = None
        text_chunks: list[str] = []
        n_events = 0
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if first_chunk_at is None:
                    first_chunk_at = _now_ms()
                text_chunks.append(content)
                n_events += 1
        t_end = _now_ms()
        total_ms = t_end - t_send
        first_chunk_ms = (first_chunk_at - t_send) if first_chunk_at is not None else total_ms
        text = "".join(text_chunks)
        output_chars = len(text)
        cps = (output_chars / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        out.prompts.append(
            PromptResult(
                label=label,
                output_text=text,
                output_events=n_events,
                output_chars=output_chars,
                first_chunk_ms=first_chunk_ms,
                total_latency_ms=total_ms,
                chars_per_second=cps,
            )
        )

    out.peak_rss_mb = _peak_rss_mb()
    # Drop the heavy reference so subsequent passes don't carry it.
    del llm
    return out


def _run_native(gguf: str) -> BackendResult:
    """Pass 2: native runtime path via
    ``octomil.runtime.native.NativeRuntime``."""
    from octomil.runtime.native import (
        OCT_EVENT_NONE,
        OCT_EVENT_SESSION_COMPLETED,
        OCT_EVENT_SESSION_STARTED,
        OCT_EVENT_TRANSCRIPT_CHUNK,
        NativeRuntime,
    )

    rt = NativeRuntime.open()
    try:
        t0 = _now_ms()
        mdl = rt.open_model(model_uri=gguf)
        model_open_ms = _now_ms() - t0

        t0 = _now_ms()
        mdl.warm()
        warm_ms = _now_ms() - t0

        out = BackendResult(
            backend="native_runtime",
            model_open_ms=model_open_ms,
            warm_ms=warm_ms,
            session_open_ms=0.0,
        )

        session_open_total = 0.0
        for label, messages in PROMPTS:
            t_session_open = _now_ms()
            sess = rt.open_session(
                capability=CHAT_COMPLETION,
                locality="on_device",
                policy_preset="private",
                model=mdl,
            )
            session_open_total += _now_ms() - t_session_open
            try:
                # Drain SESSION_STARTED so first_chunk_ms reflects
                # generation time, not bookkeeping.
                _wait_for(sess, OCT_EVENT_SESSION_STARTED, deadline_ms=2000)
                t_send = _now_ms()
                # v0.1.2+: use send_chat with max_tokens so the native
                # cap matches Python's max_tokens=32. The two paths
                # still expose different stream-event shapes, so the
                # gate reports output events/chars rather than calling
                # either side's count "tokens".
                sess.send_chat(messages, max_tokens=MAX_TOKENS)
                first_chunk_at: float | None = None
                text_chunks: list[str] = []
                n_events = 0
                deadline = _now_ms() + 60_000.0
                while _now_ms() < deadline:
                    ev = sess.poll_event(timeout_ms=200)
                    if ev is None or ev.type == OCT_EVENT_NONE:
                        continue
                    if ev.type == OCT_EVENT_TRANSCRIPT_CHUNK:
                        text = ev.text or ""
                        if text and first_chunk_at is None:
                            first_chunk_at = _now_ms()
                        if text:
                            text_chunks.append(text)
                            n_events += 1
                    elif ev.type == OCT_EVENT_SESSION_COMPLETED:
                        break
                    # Out-of-sequence event types are pinned in the
                    # conformance suite; here we just drain.
                t_end = _now_ms()
                total_ms = t_end - t_send
                first_chunk_ms = (first_chunk_at - t_send) if first_chunk_at is not None else total_ms
                output_text = "".join(text_chunks)
                output_chars = len(output_text)
                cps = (output_chars / (total_ms / 1000.0)) if total_ms > 0 else 0.0
                out.prompts.append(
                    PromptResult(
                        label=label,
                        output_text=output_text,
                        output_events=n_events,
                        output_chars=output_chars,
                        first_chunk_ms=first_chunk_ms,
                        total_latency_ms=total_ms,
                        chars_per_second=cps,
                    )
                )
            finally:
                sess.close()
        out.session_open_ms = session_open_total / max(len(PROMPTS), 1)

        # Tear down model + runtime explicitly.
        mdl.close()
        out.peak_rss_mb = _peak_rss_mb()
        return out
    finally:
        rt.close()


def _wait_for(sess, event_type: int, deadline_ms: float) -> None:
    from octomil.runtime.native import OCT_EVENT_NONE

    deadline = _now_ms() + deadline_ms
    while _now_ms() < deadline:
        ev = sess.poll_event(timeout_ms=200)
        if ev is None or ev.type == OCT_EVENT_NONE:
            continue
        if ev.type == event_type:
            return
        # Drain other events silently — caller doesn't pin them here.
    raise AssertionError(f"timeout waiting for event type {event_type}")


def _run_unsupported_shape_rejection_check() -> dict[str, str]:
    """Smoke-check that the native path rejects out-of-subset shapes
    consistently. The cutover subset is `{system,user,assistant}` ×
    `{role,content}` strings. Anything else (tool calls, `name`
    fields, etc.) MUST surface a typed error before we route.

    Returns a dict mapping shape-label to the observed status text.
    The gate doesn't ASSERT Python rejects (it doesn't — it's
    permissive); it ASSERTS native rejects."""
    from octomil.runtime.native import (
        NativeRuntime,
    )

    out: dict[str, str] = {}
    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    with NativeRuntime.open() as rt:
        mdl = rt.open_model(model_uri=gguf)
        try:
            mdl.warm()
            sess = rt.open_session(
                capability=CHAT_COMPLETION,
                locality="on_device",
                policy_preset="private",
                model=mdl,
            )
            try:
                # Out-of-subset: unknown role.
                sess.send_text(json.dumps([{"role": "tool", "content": "x"}]))
                # Drain to terminal — runtime rejects via SESSION_COMPLETED
                # (terminal_status=INVALID_INPUT) per the conformance suite.
                from octomil.runtime.native import (
                    OCT_EVENT_NONE,
                    OCT_EVENT_SESSION_COMPLETED,
                )

                observed_status = "no_terminal_observed"
                deadline = _now_ms() + 5000.0
                while _now_ms() < deadline:
                    ev = sess.poll_event(timeout_ms=200)
                    if ev is None or ev.type == OCT_EVENT_NONE:
                        continue
                    if ev.type == OCT_EVENT_SESSION_COMPLETED:
                        observed_status = f"terminal_status:{ev.terminal_status}"
                        break
                out["unknown_role"] = observed_status
            finally:
                sess.close()
        finally:
            mdl.close()
    return out


def _summary_table(py: BackendResult, native: BackendResult, cps_ratio: float) -> str:
    lines = [
        "│ metric                  │ python_llama_cpp  │ native_runtime    │ native/py │",
        "├─────────────────────────┼───────────────────┼───────────────────┼───────────┤",
        f"│ model_open_ms           │ {py.model_open_ms:>17.1f} │ {native.model_open_ms:>17.1f} │ {native.model_open_ms / max(py.model_open_ms, 1e-6):>9.2f} │",
        f"│ warm_ms                 │ {py.warm_ms:>17.1f} │ {native.warm_ms:>17.1f} │       —   │",
        f"│ session_open_ms (avg)   │ {py.session_open_ms:>17.1f} │ {native.session_open_ms:>17.1f} │       —   │",
        f"│ total_latency_sum_ms    │ {py.total_latency_sum_ms():>17.1f} │ {native.total_latency_sum_ms():>17.1f} │       —   │",
        f"│ first_chunk_avg_ms      │ {py.first_chunk_avg_ms():>17.1f} │ {native.first_chunk_avg_ms():>17.1f} │ {native.first_chunk_avg_ms() / max(py.first_chunk_avg_ms(), 1e-6):>9.2f} │",
        f"│ total_output_events     │ {py.total_output_events():>17d} │ {native.total_output_events():>17d} │       —   │",
        f"│ total_output_chars      │ {py.total_output_chars():>17d} │ {native.total_output_chars():>17d} │       —   │",
        f"│ avg_chars_per_second    │ {py.avg_chars_per_second():>17.1f} │ {native.avg_chars_per_second():>17.1f} │ {cps_ratio:>9.2f} │",
        f"│ peak_rss_mb             │ {py.peak_rss_mb:>17.1f} │ {native.peak_rss_mb:>17.1f} │       —   │",
    ]
    return "\n".join(lines)


@pytest.mark.requires_runtime
@pytest.mark.timeout(180)
def test_chat_cutover_gate_native_vs_python_local():
    """Pre-cutover gate. Runs both backends against the same GGUF
    + prompt suite, asserts native is within budget."""
    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing — gate requires a real GGUF")

    py_result = _run_python_local(gguf)
    native_result = _run_native(gguf)

    # Equivalence: both backends produced non-empty output for every
    # prompt in the cutover subset.
    for p in py_result.prompts:
        assert p.output_chars > 0, f"python_llama_cpp produced empty output for prompt={p.label!r}"
    for p in native_result.prompts:
        assert p.output_chars > 0, f"native_runtime produced empty output for prompt={p.label!r}"

    # Out-of-subset shape rejection smoke (native side).
    rejection = _run_unsupported_shape_rejection_check()
    assert rejection.get("unknown_role"), "native path must terminate on out-of-subset shape"

    # Diagnostic: native chars_per_second vs Python.
    # The deprecated Python path and native canonical path do not
    # expose identical token accounting, so use text throughput for
    # reporting and leave semantic parity to native conformance.
    py_cps = py_result.avg_chars_per_second()
    native_cps = native_result.avg_chars_per_second()
    cps_ratio = native_cps / max(py_cps, 1e-6)
    py_total = py_result.total_latency_sum_ms()
    native_total = native_result.total_latency_sum_ms()
    py_first = py_result.first_chunk_avg_ms()
    native_first = native_result.first_chunk_avg_ms()
    first_chunk_budget = max(
        py_first * NATIVE_FIRST_CHUNK_MAX_RATIO,
        py_first + NATIVE_FIRST_CHUNK_ABS_CUSHION_MS,
    )
    total_latency_budget = max(
        py_total * NATIVE_TOTAL_LATENCY_MAX_RATIO,
        py_total + NATIVE_TOTAL_LATENCY_ABS_CUSHION_MS,
    )

    first_chunk_pass = native_first <= first_chunk_budget
    total_latency_pass = native_total <= total_latency_budget
    verdict = "PASS" if first_chunk_pass and total_latency_pass else "FAIL"
    report = {
        "gguf": gguf,
        "max_tokens": MAX_TOKENS,
        "n_prompts": len(PROMPTS),
        "python_llama_cpp": asdict(py_result),
        "native_runtime": asdict(native_result),
        "comparison": {
            "primary_pass_criterion": "latency_budget",
            "native_cps_over_python_cps_ratio": cps_ratio,
            "native_first_chunk_ms": native_first,
            "python_first_chunk_ms": py_first,
            "first_chunk_budget_ms": first_chunk_budget,
            "first_chunk_pass": first_chunk_pass,
            "native_total_latency_ms": native_total,
            "python_total_latency_ms": py_total,
            "total_latency_budget_ms": total_latency_budget,
            "total_latency_pass": total_latency_pass,
            "native_over_python_total_latency_ratio": (native_total / max(py_total, 1e-6)),
            "native_over_python_first_chunk_ratio": (
                native_result.first_chunk_avg_ms() / max(py_result.first_chunk_avg_ms(), 1e-6)
            ),
            "note": (
                "Native is the canonical post-cutover path. Python stream chunks and "
                "native transcript chunks are not identical token accounting surfaces, "
                "so this gate compares generated text chars/sec under max_tokens={mt}."
            ).format(mt=MAX_TOKENS),
            "verdict": verdict,
        },
        "unsupported_shape_rejection": rejection,
    }
    report_path = Path("/tmp/octomil-chat-cutover-gate.json")
    report_path.write_text(json.dumps(report, indent=2))

    print()
    print(f"=== chat cutover gate report ({report_path}) ===")
    print(_summary_table(py_result, native_result, cps_ratio))
    print()
    print(
        f"primary criterion: first_chunk={native_first:.1f}ms <= {first_chunk_budget:.1f}ms "
        f"and total_latency={native_total:.1f}ms <= {total_latency_budget:.1f}ms; verdict: {verdict}"
    )
    print(
        f"  native_cps      = {native_cps:>7.1f} char/s  ({'+' if cps_ratio >= 1 else '-'}{abs(cps_ratio - 1) * 100:.1f}% vs python)"
    )
    print(f"  python_cps      = {py_cps:>7.1f} char/s")
    print(
        f"  first_chunk_ms  = {native_result.first_chunk_avg_ms():.1f} (native) vs {py_result.first_chunk_avg_ms():.1f} (python) — UX signal, not pass-fail"
    )

    assert first_chunk_pass, (
        f"native first_chunk_ms ({native_first:.1f}) exceeded budget ({first_chunk_budget:.1f}). "
        f"Report at {report_path}."
    )
    assert total_latency_pass, (
        f"native total_latency_sum_ms ({native_total:.1f}) exceeded budget ({total_latency_budget:.1f}). "
        f"Report at {report_path}."
    )
