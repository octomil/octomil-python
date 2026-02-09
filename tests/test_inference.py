import unittest
import time
from typing import Optional
from unittest.mock import patch

from edgeml.inference import (
    InferenceChunk,
    Modality,
    StreamingInferenceClient,
    StreamingInferenceResult,
)


class _StubApi:
    """Minimal API stub that records ``post`` calls for assertion."""

    def __init__(self, *, raise_on_post: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self._raise_on_post = raise_on_post

    def post(self, path: str, payload: dict) -> dict:
        if self._raise_on_post:
            raise RuntimeError("simulated network error")
        self.calls.append((path, payload))
        return {}


def _make_client(api: Optional[_StubApi] = None) -> StreamingInferenceClient:
    if api is None:
        api = _StubApi()
    return StreamingInferenceClient(api=api, device_id="dev-001", org_id="test-org")


# ------------------------------------------------------------------
# StreamingInferenceResult
# ------------------------------------------------------------------


class StreamingInferenceResultTests(unittest.TestCase):
    def test_to_dict_returns_correct_keys_and_values(self):
        result = StreamingInferenceResult(
            session_id="abc123",
            modality=Modality.TEXT,
            ttfc_ms=12.5,
            avg_chunk_latency_ms=3.2,
            total_chunks=10,
            total_duration_ms=100.0,
            throughput=100.0,
        )
        d = result.to_dict()
        self.assertEqual(d["session_id"], "abc123")
        self.assertEqual(d["modality"], "text")
        self.assertAlmostEqual(d["ttfc_ms"], 12.5)
        self.assertAlmostEqual(d["avg_chunk_latency_ms"], 3.2)
        self.assertEqual(d["total_chunks"], 10)
        self.assertAlmostEqual(d["total_duration_ms"], 100.0)
        self.assertAlmostEqual(d["throughput"], 100.0)

    def test_to_dict_modality_uses_enum_value(self):
        for mod in Modality:
            result = StreamingInferenceResult(
                session_id="x",
                modality=mod,
                ttfc_ms=0,
                avg_chunk_latency_ms=0,
                total_chunks=0,
                total_duration_ms=0,
                throughput=0,
            )
            self.assertEqual(result.to_dict()["modality"], mod.value)


# ------------------------------------------------------------------
# Modality enum
# ------------------------------------------------------------------


class ModalityTests(unittest.TestCase):
    def test_modality_values(self):
        self.assertEqual(Modality.TEXT.value, "text")
        self.assertEqual(Modality.IMAGE.value, "image")
        self.assertEqual(Modality.AUDIO.value, "audio")
        self.assertEqual(Modality.VIDEO.value, "video")

    def test_modality_is_str(self):
        # Modality inherits from str
        for mod in Modality:
            self.assertIsInstance(mod, str)


# ------------------------------------------------------------------
# _make_chunk
# ------------------------------------------------------------------


class MakeChunkTests(unittest.TestCase):
    def _fresh_state(self) -> dict:
        now = time.monotonic()
        return {
            "session_id": "sess-1",
            "session_start": now,
            "first_chunk_time": None,
            "previous_time": now,
            "latencies": [],
            "chunk_count": 0,
            "resolved_input": None,
        }

    def test_bytes_input(self):
        client = _make_client()
        state = self._fresh_state()
        chunk = client._make_chunk(b"\x00\x01", Modality.IMAGE, state)
        self.assertEqual(chunk.data, b"\x00\x01")
        self.assertEqual(chunk.modality, Modality.IMAGE)
        self.assertEqual(chunk.index, 0)
        self.assertEqual(state["chunk_count"], 1)

    def test_str_input_encoded_to_utf8(self):
        client = _make_client()
        state = self._fresh_state()
        chunk = client._make_chunk("hello", Modality.TEXT, state)
        self.assertEqual(chunk.data, b"hello")

    def test_other_type_converted_via_bytes(self):
        client = _make_client()
        state = self._fresh_state()
        raw = bytearray(b"\xab\xcd")
        chunk = client._make_chunk(raw, Modality.AUDIO, state)
        self.assertEqual(chunk.data, b"\xab\xcd")

    def test_first_chunk_time_set_once(self):
        client = _make_client()
        state = self._fresh_state()
        self.assertIsNone(state["first_chunk_time"])

        client._make_chunk(b"a", Modality.TEXT, state)
        first = state["first_chunk_time"]
        self.assertIsNotNone(first)

        client._make_chunk(b"b", Modality.TEXT, state)
        self.assertEqual(state["first_chunk_time"], first)

    def test_latencies_accumulated(self):
        client = _make_client()
        state = self._fresh_state()
        client._make_chunk(b"a", Modality.TEXT, state)
        client._make_chunk(b"b", Modality.TEXT, state)
        client._make_chunk(b"c", Modality.TEXT, state)
        self.assertEqual(len(state["latencies"]), 3)
        self.assertEqual(state["chunk_count"], 3)

    def test_chunk_index_increments(self):
        client = _make_client()
        state = self._fresh_state()
        c0 = client._make_chunk(b"a", Modality.TEXT, state)
        c1 = client._make_chunk(b"b", Modality.TEXT, state)
        c2 = client._make_chunk(b"c", Modality.TEXT, state)
        self.assertEqual(c0.index, 0)
        self.assertEqual(c1.index, 1)
        self.assertEqual(c2.index, 2)


# ------------------------------------------------------------------
# Sync generate() — all four modalities
# ------------------------------------------------------------------


class GenerateTextTests(unittest.TestCase):
    def test_generate_text_yields_chunks(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = list(
            client.generate("model-1", prompt="Hello world", modality=Modality.TEXT)
        )
        self.assertGreater(len(chunks), 0)
        for ch in chunks:
            self.assertIsInstance(ch, InferenceChunk)
            self.assertEqual(ch.modality, Modality.TEXT)
            self.assertIsInstance(ch.data, bytes)

    def test_generate_text_reports_events(self):
        api = _StubApi()
        client = _make_client(api)
        list(client.generate("model-1", prompt="Hello world", modality=Modality.TEXT))
        # Should have generation_started and generation_completed events
        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_completed", event_types)

    def test_generate_text_populates_last_result(self):
        client = _make_client()
        self.assertIsNone(client.last_result)
        list(client.generate("model-1", prompt="test", modality=Modality.TEXT))
        self.assertIsNotNone(client.last_result)
        self.assertIsInstance(client.last_result, StreamingInferenceResult)
        self.assertEqual(client.last_result.modality, Modality.TEXT)
        self.assertGreater(client.last_result.total_chunks, 0)

    def test_generate_text_result_metrics_reasonable(self):
        client = _make_client()
        list(client.generate("model-1", prompt="test", modality=Modality.TEXT))
        result = client.last_result
        self.assertGreater(result.total_duration_ms, 0)
        self.assertGreater(result.throughput, 0)
        self.assertGreaterEqual(result.ttfc_ms, 0)
        self.assertGreater(result.avg_chunk_latency_ms, 0)


class GenerateImageTests(unittest.TestCase):
    def test_generate_image_yields_20_chunks(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = list(
            client.generate("img-model", input="a photo", modality=Modality.IMAGE)
        )
        self.assertEqual(len(chunks), 20)
        for i, ch in enumerate(chunks):
            self.assertEqual(ch.index, i)
            self.assertEqual(ch.modality, Modality.IMAGE)
            self.assertEqual(len(ch.data), 64)

    def test_generate_image_events(self):
        api = _StubApi()
        client = _make_client(api)
        list(client.generate("img-model", input="a photo", modality=Modality.IMAGE))
        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertEqual(event_types[0], "generation_started")
        self.assertEqual(event_types[-1], "generation_completed")


class GenerateAudioTests(unittest.TestCase):
    def test_generate_audio_yields_80_chunks(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = list(
            client.generate("audio-model", input="audio input", modality=Modality.AUDIO)
        )
        self.assertEqual(len(chunks), 80)
        for ch in chunks:
            self.assertEqual(ch.modality, Modality.AUDIO)
            self.assertEqual(len(ch.data), 1024 * 2)

    def test_generate_audio_events(self):
        api = _StubApi()
        client = _make_client(api)
        list(client.generate("audio-model", input="audio input", modality=Modality.AUDIO))
        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_completed", event_types)


class GenerateVideoTests(unittest.TestCase):
    def test_generate_video_yields_30_chunks(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = list(
            client.generate("video-model", input="video input", modality=Modality.VIDEO)
        )
        self.assertEqual(len(chunks), 30)
        for ch in chunks:
            self.assertEqual(ch.modality, Modality.VIDEO)
            self.assertEqual(len(ch.data), 1024)

    def test_generate_video_events(self):
        api = _StubApi()
        client = _make_client(api)
        list(client.generate("video-model", input="video input", modality=Modality.VIDEO))
        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_completed", event_types)


# ------------------------------------------------------------------
# Async generate_async()
# ------------------------------------------------------------------


class GenerateAsyncTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_text_yields_chunks(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = []
        async for ch in client.generate_async(
            "model-1", prompt="Hello", modality=Modality.TEXT
        ):
            chunks.append(ch)
        self.assertGreater(len(chunks), 0)
        for ch in chunks:
            self.assertIsInstance(ch, InferenceChunk)
            self.assertEqual(ch.modality, Modality.TEXT)

    async def test_async_generate_reports_events(self):
        api = _StubApi()
        client = _make_client(api)
        chunks = []
        async for ch in client.generate_async(
            "model-1", prompt="Hello", modality=Modality.TEXT
        ):
            chunks.append(ch)
        self.assertGreater(len(chunks), 0)
        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_completed", event_types)

    async def test_async_generate_populates_last_result(self):
        client = _make_client()
        chunks = []
        async for ch in client.generate_async(
            "model-1", prompt="Hello", modality=Modality.TEXT
        ):
            chunks.append(ch)
        self.assertIsNotNone(client.last_result)
        self.assertEqual(client.last_result.modality, Modality.TEXT)
        self.assertGreater(client.last_result.total_chunks, 0)

    async def test_async_generate_failure_reports_failed_event(self):
        api = _StubApi()
        client = _make_client(api)

        def _exploding_backend(*args, **kwargs):
            raise ValueError("backend exploded")

        client._backend_generate = _exploding_backend

        with self.assertRaises(ValueError):
            chunks = []
            async for ch in client.generate_async(
                "model-1", prompt="Hello", modality=Modality.TEXT
            ):
                chunks.append(ch)

        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_failed", event_types)
        self.assertNotIn("generation_completed", event_types)


# ------------------------------------------------------------------
# _report_event
# ------------------------------------------------------------------


class ReportEventTests(unittest.TestCase):
    def test_report_event_posts_to_correct_path(self):
        api = _StubApi()
        client = _make_client(api)
        client._report_event(
            model_id="m1",
            version="v2",
            modality=Modality.TEXT,
            session_id="sess-1",
            event_type="generation_started",
        )
        self.assertEqual(len(api.calls), 1)
        path, payload = api.calls[0]
        self.assertEqual(path, "/inference/events")
        self.assertEqual(payload["model_id"], "m1")
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["modality"], "text")
        self.assertEqual(payload["session_id"], "sess-1")
        self.assertEqual(payload["event_type"], "generation_started")
        self.assertEqual(payload["device_id"], "dev-001")
        self.assertEqual(payload["org_id"], "test-org")
        self.assertIn("timestamp_ms", payload)

    def test_report_event_includes_metrics_when_provided(self):
        api = _StubApi()
        client = _make_client(api)
        metrics = {"ttfc_ms": 10.0, "total_chunks": 5}
        client._report_event(
            model_id="m1",
            version="v1",
            modality=Modality.IMAGE,
            session_id="sess-2",
            event_type="generation_completed",
            metrics=metrics,
        )
        payload = api.calls[0][1]
        self.assertEqual(payload["metrics"], metrics)

    def test_report_event_omits_metrics_key_when_none(self):
        api = _StubApi()
        client = _make_client(api)
        client._report_event(
            model_id="m1",
            version="v1",
            modality=Modality.TEXT,
            session_id="sess-3",
            event_type="generation_started",
        )
        payload = api.calls[0][1]
        self.assertNotIn("metrics", payload)

    def test_report_event_swallows_exceptions(self):
        api = _StubApi(raise_on_post=True)
        client = _make_client(api)
        # Should not raise even though the api.post raises
        client._report_event(
            model_id="m1",
            version="v1",
            modality=Modality.TEXT,
            session_id="sess-4",
            event_type="generation_started",
        )


# ------------------------------------------------------------------
# _report_failure
# ------------------------------------------------------------------


class ReportFailureTests(unittest.TestCase):
    def test_report_failure_sends_generation_failed(self):
        api = _StubApi()
        client = _make_client(api)
        client._report_failure("m1", "v1", Modality.TEXT, "sess-5")
        self.assertEqual(len(api.calls), 1)
        payload = api.calls[0][1]
        self.assertEqual(payload["event_type"], "generation_failed")
        self.assertEqual(payload["session_id"], "sess-5")
        self.assertEqual(payload["model_id"], "m1")


# ------------------------------------------------------------------
# generate() failure path
# ------------------------------------------------------------------


class GenerateFailureTests(unittest.TestCase):
    def test_backend_exception_reports_failure_and_reraises(self):
        api = _StubApi()
        client = _make_client(api)

        def _exploding_backend(*args, **kwargs):
            raise RuntimeError("backend crashed")

        client._backend_generate = _exploding_backend

        with self.assertRaises(RuntimeError) as ctx:
            list(client.generate("model-1", prompt="Hello", modality=Modality.TEXT))
        self.assertIn("backend crashed", str(ctx.exception))

        event_types = [call[1]["event_type"] for call in api.calls]
        self.assertIn("generation_started", event_types)
        self.assertIn("generation_failed", event_types)
        self.assertNotIn("generation_completed", event_types)

    def test_backend_exception_does_not_populate_last_result(self):
        client = _make_client()
        def _failing_backend(*a, **kw):
            yield from ()
            raise RuntimeError("fail")

        client._backend_generate = _failing_backend
        with self.assertRaises(RuntimeError):
            list(client.generate("model-1", prompt="Hello", modality=Modality.TEXT))
        self.assertIsNone(client.last_result)


# ------------------------------------------------------------------
# prompt vs input parameter resolution
# ------------------------------------------------------------------


class ParameterResolutionTests(unittest.TestCase):
    def test_prompt_takes_precedence_over_input(self):
        api = _StubApi()
        client = _make_client(api)
        list(
            client.generate(
                "model-1",
                input="ignored",
                prompt="used",
                modality=Modality.TEXT,
            )
        )
        # The started event is first; check that generate used prompt
        # We verify by inspecting _init_session's resolved_input indirectly:
        # the text backend produces output containing "used", not "ignored"
        result_text = b"".join(ch.data for ch in client.generate(
            "model-1", input="ignored", prompt="used", modality=Modality.TEXT
        ))
        self.assertIn(b"used", result_text)
        self.assertNotIn(b"ignored", result_text)

    def test_input_used_when_prompt_is_none(self):
        client = _make_client()
        result_text = b"".join(
            ch.data
            for ch in client.generate(
                "model-1", input="my input", modality=Modality.TEXT
            )
        )
        self.assertIn(b"my input", result_text)

    def test_both_none_uses_none(self):
        client = _make_client()
        # Should still work (text backend converts None to "")
        chunks = list(
            client.generate("model-1", modality=Modality.TEXT)
        )
        self.assertGreater(len(chunks), 0)


# ------------------------------------------------------------------
# _init_session
# ------------------------------------------------------------------


class InitSessionTests(unittest.TestCase):
    def test_init_session_returns_expected_state(self):
        api = _StubApi()
        client = _make_client(api)
        state = client._init_session("m1", "v1", Modality.TEXT, "hello")

        self.assertIn("session_id", state)
        self.assertIsInstance(state["session_id"], str)
        self.assertEqual(len(state["session_id"]), 32)  # uuid4().hex
        self.assertIsNone(state["first_chunk_time"])
        self.assertEqual(state["latencies"], [])
        self.assertEqual(state["chunk_count"], 0)
        self.assertEqual(state["resolved_input"], "hello")

    def test_init_session_reports_generation_started(self):
        api = _StubApi()
        client = _make_client(api)
        client._init_session("m1", "v1", Modality.TEXT, "hello")
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(api.calls[0][1]["event_type"], "generation_started")


# ------------------------------------------------------------------
# _finalize_session
# ------------------------------------------------------------------


class FinalizeSessionTests(unittest.TestCase):
    def test_finalize_session_stores_last_result(self):
        api = _StubApi()
        client = _make_client(api)

        now = time.monotonic()
        state = {
            "session_id": "sess-fin",
            "session_start": now - 0.1,
            "first_chunk_time": now - 0.09,
            "previous_time": now,
            "latencies": [10.0, 20.0, 30.0],
            "chunk_count": 3,
            "resolved_input": "test",
        }
        client._finalize_session("m1", "v1", Modality.TEXT, state)

        result = client.last_result
        self.assertIsNotNone(result)
        self.assertEqual(result.session_id, "sess-fin")
        self.assertEqual(result.modality, Modality.TEXT)
        self.assertEqual(result.total_chunks, 3)
        self.assertAlmostEqual(result.avg_chunk_latency_ms, 20.0)
        self.assertGreater(result.total_duration_ms, 0)
        self.assertGreater(result.throughput, 0)

    def test_finalize_session_reports_generation_completed(self):
        api = _StubApi()
        client = _make_client(api)

        now = time.monotonic()
        state = {
            "session_id": "sess-fin",
            "session_start": now - 0.05,
            "first_chunk_time": now - 0.04,
            "previous_time": now,
            "latencies": [5.0],
            "chunk_count": 1,
            "resolved_input": "test",
        }
        client._finalize_session("m1", "v1", Modality.TEXT, state)

        self.assertEqual(len(api.calls), 1)
        payload = api.calls[0][1]
        self.assertEqual(payload["event_type"], "generation_completed")
        self.assertIn("metrics", payload)
        self.assertIn("ttfc_ms", payload["metrics"])
        self.assertIn("total_chunks", payload["metrics"])
        self.assertIn("total_duration_ms", payload["metrics"])
        self.assertIn("throughput", payload["metrics"])


# ------------------------------------------------------------------
# Client construction
# ------------------------------------------------------------------


class ClientConstructionTests(unittest.TestCase):
    def test_default_org_id(self):
        api = _StubApi()
        client = StreamingInferenceClient(api=api, device_id="dev-1")
        self.assertEqual(client.org_id, "default")

    def test_custom_org_id(self):
        api = _StubApi()
        client = StreamingInferenceClient(api=api, device_id="dev-1", org_id="acme")
        self.assertEqual(client.org_id, "acme")

    def test_last_result_initially_none(self):
        client = _make_client()
        self.assertIsNone(client.last_result)


if __name__ == "__main__":
    unittest.main()
