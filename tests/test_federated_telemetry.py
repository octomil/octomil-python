"""Tests for FederatedClient telemetry instrumentation (train, weights, eligibility)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from octomil.telemetry import TelemetryReporter


def _make_fc_with_reporter():
    """Create a FederatedClient with mocked API and a captured reporter."""
    from octomil.python.octomil.federated_client import FederatedClient

    sent_events = []

    def mock_send(client, url, headers, payload):
        sent_events.append(payload)

    patcher = patch.object(TelemetryReporter, "_send", side_effect=mock_send)
    patcher.start()

    reporter = TelemetryReporter(
        api_key="test-key",
        api_base="https://api.test.com/api/v1",
        org_id="test-org",
        device_id="dev-1",
    )

    fc = FederatedClient(
        auth_token_provider=lambda: "test-key",
        org_id="test-org",
    )
    # Pre-register so register() doesn't call API
    fc.device_id = "device-123"

    # Mock out the API
    mock_api = MagicMock()
    fc.api = mock_api

    return fc, reporter, sent_events, patcher, mock_api


def _extract_funnel_events(sent_envelopes):
    """Extract all funnel events from sent envelopes."""
    events = []
    for envelope in sent_envelopes:
        for event in envelope.get("events", []):
            if event["name"].startswith("funnel."):
                events.append(event)
    return events


def _drain(reporter, sent, patcher):
    """Wait for events, close reporter, stop patcher."""
    time.sleep(0.15)
    reporter.close()
    patcher.stop()


# ---------------------------------------------------------------------------
# train() telemetry
# ---------------------------------------------------------------------------


class TestTrainTelemetry:
    def test_train_reports_started_upload_completed(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        mock_api.get.return_value = {"version": "1.0.0"}
        mock_api.post.return_value = {"status": "ok"}

        with patch("octomil.get_reporter", return_value=reporter):
            results = fc.train(
                model="test-model",
                data=b"fake-weights",
                rounds=2,
                version="1.0.0",
            )

        assert len(results) == 2
        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        event_stages = [e["name"] for e in funnel_events]

        # Should have: training_started, weight_upload x2, training_completed
        assert "funnel.training_started" in event_stages
        assert event_stages.count("funnel.weight_upload") == 2
        assert "funnel.training_completed" in event_stages

        # Verify training_completed has duration
        completed = [e for e in funnel_events if e["name"] == "funnel.training_completed"][0]
        assert completed["attributes"]["funnel.duration_ms"] >= 0

    def test_train_reports_failure(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        mock_api.get.return_value = {"version": "1.0.0"}
        mock_api.post.side_effect = RuntimeError("upload failed")

        with patch("octomil.get_reporter", return_value=reporter):
            with pytest.raises(RuntimeError, match="upload failed"):
                fc.train(model="test-model", data=b"fake", version="1.0.0")

        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        event_stages = [e["name"] for e in funnel_events]

        assert "funnel.training_started" in event_stages
        assert "funnel.training_failed" in event_stages
        # The weight_upload failure should also be reported
        weight_uploads = [e for e in funnel_events if e["name"] == "funnel.weight_upload"]
        assert len(weight_uploads) == 1
        assert weight_uploads[0]["attributes"]["funnel.success"] is False
        assert weight_uploads[0]["attributes"]["error.type"] == "upload_error"


# ---------------------------------------------------------------------------
# train_from_remote() telemetry
# ---------------------------------------------------------------------------


class TestTrainFromRemoteTelemetry:
    def test_train_from_remote_reports_lifecycle(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        mock_api.get.return_value = {"version": "1.0.0"}
        mock_api.get_bytes.return_value = b"fake-model-bytes"
        mock_api.post.return_value = {"status": "ok"}

        def fake_train_fn(state):
            return state, 10, {"loss": 0.5}

        with (
            patch("octomil.get_reporter", return_value=reporter),
            patch.object(fc, "_deserialize_weights", return_value={"w": 1}),
            patch.object(fc, "_serialize_weights", return_value=b"serialized"),
        ):
            results = fc.train_from_remote(
                model="test-model",
                local_train_fn=fake_train_fn,
                rounds=1,
                version="1.0.0",
            )

        assert len(results) == 1
        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        event_stages = [e["name"] for e in funnel_events]
        assert "funnel.training_started" in event_stages
        assert "funnel.weight_upload" in event_stages
        assert "funnel.training_completed" in event_stages


# ---------------------------------------------------------------------------
# join_round() telemetry
# ---------------------------------------------------------------------------


class TestParticipateInRoundTelemetry:
    def test_participate_reports_weight_upload(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        # api.get is called for: round status, then _resolve_model_id inside pull_model
        mock_api.get.side_effect = [
            # round status
            {
                "config": {"model_id": "model-1", "version": "1.0.0"},
                "model_id": "model-1",
                "version": "1.0.0",
            },
            # _resolve_model_id via pull_model
            {"models": [{"name": "model-1", "id": "model-1"}]},
        ]
        mock_api.get_bytes.return_value = b"model-bytes"
        mock_api.post.return_value = {"status": "ok"}

        def fake_train_fn(state):
            return state, 10, {"loss": 0.5}

        with (
            patch("octomil.get_reporter", return_value=reporter),
            patch.object(fc, "_deserialize_weights", return_value={"w": 1}),
            patch.object(fc, "_serialize_weights", return_value=b"serialized"),
            patch(
                "octomil.python.octomil.federated_client.compute_state_dict_delta",
                return_value={"w": 0},
            ),
        ):
            result = fc.join_round(
                round_id="round-1",
                local_train_fn=fake_train_fn,
            )

        assert result == {"status": "ok"}
        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        weight_uploads = [e for e in funnel_events if e["name"] == "funnel.weight_upload"]
        assert len(weight_uploads) == 1
        assert weight_uploads[0]["attributes"]["funnel.success"] is True
        assert weight_uploads[0]["attributes"]["funnel.duration_ms"] >= 0


# ---------------------------------------------------------------------------
# train_if_eligible() telemetry
# ---------------------------------------------------------------------------


class TestTrainIfEligibleTelemetry:
    def test_ineligible_reports_device_ineligible(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        with (
            patch("octomil.get_reporter", return_value=reporter),
            patch(
                "octomil.python.octomil.federated_client.get_battery_level",
                return_value=5,
            ),
            patch(
                "octomil.python.octomil.federated_client.is_charging",
                return_value=False,
            ),
        ):
            result = fc.train_if_eligible(
                round_id="round-1",
                local_train_fn=lambda s: (s, 0, {}),
                min_battery=15,
            )

        assert result["skipped"] is True
        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) >= 1
        started_events = [e for e in funnel_events if e["name"] == "funnel.training_started"]
        assert len(started_events) == 1
        attrs = started_events[0]["attributes"]
        assert attrs["funnel.success"] is False
        assert attrs["error.type"] == "device_ineligible"

    def test_cache_fallback_reports_upload_failed_cached(self):
        fc, reporter, sent, patcher, mock_api = _make_fc_with_reporter()

        mock_api.get.side_effect = [
            # round status
            {
                "config": {"model_id": "m1", "version": "1.0.0"},
            },
            # _resolve_model_id via pull_model
            {"models": [{"name": "m1", "id": "m1"}]},
        ]
        mock_api.get_bytes.return_value = b"model"
        mock_api.post.side_effect = RuntimeError("network down")

        mock_cache = MagicMock()

        def fake_train_fn(state):
            return state, 5, {}

        with (
            patch("octomil.get_reporter", return_value=reporter),
            patch(
                "octomil.python.octomil.federated_client.get_battery_level",
                return_value=80,
            ),
            patch(
                "octomil.python.octomil.federated_client.is_charging",
                return_value=True,
            ),
            patch.object(fc, "_deserialize_weights", return_value={"w": 1}),
            patch.object(fc, "_serialize_weights", return_value=b"data"),
            patch(
                "octomil.python.octomil.federated_client.compute_state_dict_delta",
                return_value={"w": 0},
            ),
        ):
            result = fc.train_if_eligible(
                round_id="round-1",
                local_train_fn=fake_train_fn,
                gradient_cache=mock_cache,
            )

        assert result["skipped"] is True
        assert result["reason"] == "upload_failed"
        _drain(reporter, sent, patcher)

        funnel_events = _extract_funnel_events(sent)
        # Should have weight_upload failure from join_round AND
        # the cached fallback event from train_if_eligible
        cached_events = [
            e for e in funnel_events
            if e["name"] == "funnel.weight_upload"
            and e["attributes"].get("error.type") == "upload_failed_cached"
        ]
        assert len(cached_events) == 1
