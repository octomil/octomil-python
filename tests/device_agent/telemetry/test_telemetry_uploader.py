"""Tests for TelemetryUploader — batch creation, backoff, shutdown."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.telemetry.db_schema import TELEMETRY_SCHEMA_STATEMENTS
from octomil.device_agent.telemetry.events import TelemetryClass
from octomil.device_agent.telemetry.telemetry_store import TelemetryStore
from octomil.device_agent.telemetry.telemetry_uploader import TelemetryUploader


def _make_db() -> LocalDB:
    db = LocalDB(":memory:")
    for stmt in TELEMETRY_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


def _make_fixtures() -> tuple[LocalDB, TelemetryStore, TelemetryUploader]:
    db = _make_db()
    store = TelemetryStore(db, device_id="dev-1", boot_id="boot-1")
    uploader = TelemetryUploader(
        store=store,
        device_id="dev-1",
        boot_id="boot-1",
        api_base="https://api.example.com",
        api_key="test-key",
        batch_size=50,
        max_batch_bytes=65536,
        max_age_s=1.0,
    )
    return db, store, uploader


class TestBatchCreation:
    def test_upload_batch_sends_correct_envelope(self) -> None:
        _, store, uploader = _make_fixtures()
        store.append("e1", TelemetryClass.IMPORTANT, {"k": "v"})
        store.append("e2", TelemetryClass.MUST_KEEP, {"k2": "v2"})

        captured: list[dict] = []

        def _mock_post(url: str, json: dict, headers: dict) -> MagicMock:
            captured.append(json)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"ackedThroughSeq": 2}
            return resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post = _mock_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            uploader._upload_batch()

        assert len(captured) == 1
        envelope = captured[0]
        assert envelope["device_id"] == "dev-1"
        assert envelope["boot_id"] == "boot-1"
        assert "batch_id" in envelope
        assert len(envelope["events"]) == 2

    def test_upload_batch_marks_events_sent(self) -> None:
        _, store, uploader = _make_fixtures()
        store.append("e1", TelemetryClass.IMPORTANT, {})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"ackedThroughSeq": 1}
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            uploader._upload_batch()

        # After upload, events should be marked sent
        assert store.get_unsent() == []

    def test_no_upload_when_empty(self) -> None:
        _, store, uploader = _make_fixtures()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            uploader._upload_batch()

            mock_client.post.assert_not_called()


class TestBackoff:
    def test_backoff_on_http_error(self) -> None:
        _, store, uploader = _make_fixtures()
        store.append("e1", TelemetryClass.IMPORTANT, {})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 500
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with patch("time.sleep"):
                with pytest.raises(httpx.HTTPStatusError):
                    uploader._upload_batch()

        # Events should NOT be marked sent on failure
        assert len(store.get_unsent()) == 1


class TestSetPolicy:
    def test_set_policy_updates_batch_size(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader.set_policy({"max_batch_size": 25})
        assert uploader._batch_size == 25

    def test_set_policy_cellular_limits_bytes(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader.set_policy({"network_type": "cellular"})
        assert uploader._max_batch_bytes == 262_144


class TestTrimToBytes:
    def test_trims_large_batches(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader._max_batch_bytes = 200

        events = [{"event_id": f"e{i}", "sequence_no": i, "data": "x" * 50} for i in range(10)]

        trimmed = uploader._trim_to_bytes(events)
        total_size = sum(len(json.dumps(e, separators=(",", ":"))) for e in trimmed)
        assert total_size <= 200 or len(trimmed) == 1  # at least one event

    def test_includes_at_least_one_event(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader._max_batch_bytes = 1  # impossibly small

        events = [{"event_id": "e1", "sequence_no": 1, "data": "large" * 100}]
        trimmed = uploader._trim_to_bytes(events)
        assert len(trimmed) == 1  # first event always included


class TestLifecycle:
    def test_start_stop(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader.start()
        assert uploader._thread is not None
        assert uploader._thread.is_alive()
        uploader.stop()
        assert uploader._thread is None

    def test_double_start_is_idempotent(self) -> None:
        _, _, uploader = _make_fixtures()
        uploader.start()
        thread1 = uploader._thread
        uploader.start()  # should not create a new thread
        assert uploader._thread is thread1
        uploader.stop()
