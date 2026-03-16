"""Tests for DeviceFederatedClient state machine and protocol."""

from __future__ import annotations

import json
import math
import unittest
from unittest.mock import MagicMock

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS
from octomil.device_agent.training.federated_client import (
    VALID_TRANSITIONS,
    DeviceFederatedClient,
    FederatedTransitionError,
    TrainingPlan,
)


def _make_db() -> LocalDB:
    db = LocalDB(":memory:")
    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


class TestFederatedClientStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.client = DeviceFederatedClient(self.db)

    def test_join_round_creates_participation(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        self.assertIsNotNone(pid)
        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "ACCEPTED")
        self.assertEqual(record["round_id"], "round_1")
        self.assertEqual(record["device_id"], "device_1")

    def test_valid_happy_path(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        transitions = [
            "PLAN_FETCHING",
            "PLAN_READY",
            "LOCAL_TRAINING",
            "LOCAL_EVAL",
            "UPDATE_PREPARING",
            "CLIPPING",
            "NOISING",
            "ENCRYPTING",
            "UPLOADING",
            "UPLOADED",
            "ACKNOWLEDGED",
            "COMPLETED",
        ]
        for state in transitions:
            self.client.transition(pid, state)
            record = self.client.get_participation(pid)
            assert record is not None
            self.assertEqual(record["state"], state)

    def test_invalid_transition_raises(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        with self.assertRaises(FederatedTransitionError):
            self.client.transition(pid, "UPLOADING")

    def test_transition_to_unknown_state_raises(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        with self.assertRaises(FederatedTransitionError):
            self.client.transition(pid, "BOGUS")

    def test_transition_on_missing_participation_raises(self) -> None:
        with self.assertRaises(FederatedTransitionError):
            self.client.transition("nonexistent", "PLAN_FETCHING")

    def test_error_states(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        self.client.transition(pid, "PLAN_FETCHING")
        self.client.transition(pid, "FAILED_RETRYABLE", last_error="timeout")

        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "FAILED_RETRYABLE")
        self.assertEqual(record["last_error"], "timeout")

        # Can retry
        self.client.transition(pid, "PLAN_FETCHING")
        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "PLAN_FETCHING")

    def test_terminal_states(self) -> None:
        terminal = {
            "COMPLETED",
            "DECLINED_POLICY",
            "ABORTED_POLICY",
            "REJECTED_LOCAL",
            "EXPIRED_ROUND",
        }
        for state in terminal:
            allowed = VALID_TRANSITIONS.get(state, set())
            self.assertEqual(
                allowed,
                set(),
                f"Terminal state {state} should have no transitions",
            )

    def test_expired_round_from_offered(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        # Manually set state to OFFERED for this test
        self.db.execute(
            "UPDATE federated_participation SET state = 'OFFERED' WHERE participation_id = ?",
            (pid,),
        )
        self.client.transition(pid, "EXPIRED_ROUND")
        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "EXPIRED_ROUND")


class TestFederatedClientClipNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.client = DeviceFederatedClient(_make_db())

    def test_clip_update_no_clipping_needed(self) -> None:
        update = [0.1, 0.2, 0.3]
        clipped = self.client.clip_update(update, clip_norm=10.0)
        self.assertEqual(clipped, update)

    def test_clip_update_clipping_applied(self) -> None:
        update = [3.0, 4.0]  # norm = 5.0
        clipped = self.client.clip_update(update, clip_norm=1.0)
        norm = math.sqrt(sum(x * x for x in clipped))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_clip_update_zero_vector(self) -> None:
        update = [0.0, 0.0, 0.0]
        clipped = self.client.clip_update(update, clip_norm=1.0)
        self.assertEqual(clipped, [0.0, 0.0, 0.0])

    def test_add_noise_changes_values(self) -> None:
        update = [1.0, 2.0, 3.0]
        noised = self.client.add_noise(update, sigma=0.1)
        # Very unlikely to be identical after noise
        self.assertEqual(len(noised), 3)
        # With sigma=0.1, values should be close but not identical
        for orig, noisy in zip(update, noised):
            self.assertAlmostEqual(orig, noisy, delta=1.0)

    def test_encrypt_update_returns_bytes(self) -> None:
        update = [1.0, 2.0, 3.0]
        envelope = self.client.encrypt_update(update)
        self.assertIsInstance(envelope, bytes)
        parsed = json.loads(envelope)
        self.assertEqual(parsed["format"], "plaintext_json")
        self.assertIn("sha256", parsed)
        self.assertEqual(parsed["data"], update)

    def test_encrypt_update_with_secagg(self) -> None:
        update = [1.0]
        config = {"session_id": "s1", "threshold": 2}
        envelope = self.client.encrypt_update(update, secure_agg_config=config)
        parsed = json.loads(envelope)
        self.assertEqual(parsed["secure_agg"]["session_id"], "s1")


class TestFederatedClientPrepareUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.client = DeviceFederatedClient(self.db)

    def test_prepare_update_pipeline(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        self.client.transition(pid, "PLAN_FETCHING")
        self.client.transition(pid, "PLAN_READY")
        self.client.transition(pid, "LOCAL_TRAINING")
        self.client.transition(pid, "LOCAL_EVAL")

        raw_update = [3.0, 4.0]  # norm = 5.0
        envelope = self.client.prepare_update(pid, raw_update, clip_norm=1.0, noise_sigma=0.001)

        self.assertIsInstance(envelope, bytes)
        record = self.client.get_participation(pid)
        assert record is not None
        # Should be at ENCRYPTING (last step of prepare_update)
        self.assertEqual(record["state"], "ENCRYPTING")


class TestFederatedClientRunLocalTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.client = DeviceFederatedClient(self.db)

    def test_run_local_training_success(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        self.client.transition(pid, "PLAN_FETCHING")
        self.client.transition(pid, "PLAN_READY")

        plan = TrainingPlan(
            plan_id="p1",
            round_id="round_1",
            model_id="m1",
            base_version="1.0",
            algorithm="fedavg",
            hyperparams={"lr": 0.01},
            clip_norm=1.0,
            noise_sigma=0.01,
        )

        def train_fn(ctx: dict) -> dict:
            return {"update": [0.1, 0.2, 0.3]}

        update = self.client.run_local_training(pid, plan, train_fn)
        self.assertEqual(update, [0.1, 0.2, 0.3])

        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "LOCAL_TRAINING")

    def test_run_local_training_failure(self) -> None:
        pid = self.client.join_round("round_1", "device_1")
        self.client.transition(pid, "PLAN_FETCHING")
        self.client.transition(pid, "PLAN_READY")

        plan = TrainingPlan(
            plan_id="p1",
            round_id="round_1",
            model_id="m1",
            base_version="1.0",
            algorithm="fedavg",
            hyperparams={},
            clip_norm=1.0,
            noise_sigma=0.01,
        )

        def train_fn(ctx: dict) -> dict:
            raise RuntimeError("Training failed")

        with self.assertRaises(RuntimeError):
            self.client.run_local_training(pid, plan, train_fn)

        record = self.client.get_participation(pid)
        assert record is not None
        self.assertEqual(record["state"], "FAILED_RETRYABLE")


class TestFederatedClientHTTP(unittest.TestCase):
    def test_check_offers_no_client(self) -> None:
        client = DeviceFederatedClient(_make_db())
        offers = client.check_offers("device_1")
        self.assertEqual(offers, [])

    def test_check_offers_with_mock(self) -> None:
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"round_id": "r1"}]
        mock_http.get.return_value = mock_resp

        client = DeviceFederatedClient(_make_db(), http_client=mock_http)
        offers = client.check_offers("device_1")
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0]["round_id"], "r1")


if __name__ == "__main__":
    unittest.main()
