"""Tests for LocalTrainer state machine and training functionality."""

from __future__ import annotations

import json
import unittest

from octomil.device_agent.db.local_db import LocalDB
from octomil.device_agent.training.db_schema import TRAINING_SCHEMA_STATEMENTS
from octomil.device_agent.training.local_trainer import (
    ALL_STATES,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    LocalTrainer,
    TrainingLimits,
)


def _make_db() -> LocalDB:
    """Create an in-memory DB with training schema applied."""
    db = LocalDB(":memory:")
    for stmt in TRAINING_SCHEMA_STATEMENTS:
        db.execute(stmt)
    return db


class TestLocalTrainerStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.trainer = LocalTrainer(self.db)

    def test_create_job_returns_id_and_persists(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="model_a::user_1",
            base_model_id="model_a",
            base_version="1.0",
        )
        self.assertIsNotNone(job_id)

        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "NEW")
        self.assertEqual(job["job_type"], "lora")
        self.assertEqual(job["binding_key"], "model_a::user_1")
        self.assertEqual(job["base_model_id"], "model_a")
        self.assertEqual(job["base_version"], "1.0")

    def test_create_job_with_limits(self) -> None:
        limits = TrainingLimits(max_steps=50, max_duration_sec=60.0)
        job_id = self.trainer.create_job(
            job_type="ia3",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
            limits=limits,
        )
        job = self.trainer.get_job(job_id)
        assert job is not None
        progress = json.loads(job["progress_json"])
        self.assertEqual(progress["max_steps"], 50)
        self.assertEqual(progress["max_duration_sec"], 60.0)

    def test_valid_happy_path_transitions(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        # Walk through the happy path
        transitions = [
            "ELIGIBLE",
            "QUEUED",
            "PREPARING_DATA",
            "WAITING_FOR_RESOURCES",
            "TRAINING",
            "CHECKPOINTING",
            "TRAINING",
            "EVALUATING",
            "CANDIDATE_READY",
            "STAGED",
            "ACTIVATING",
            "ACTIVE",
            "COMPLETED",
        ]
        for state in transitions:
            self.trainer.transition(job_id, state)
            job = self.trainer.get_job(job_id)
            assert job is not None
            self.assertEqual(job["state"], state)

    def test_invalid_transition_raises(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        with self.assertRaises(InvalidTransitionError):
            self.trainer.transition(job_id, "TRAINING")

    def test_transition_to_unknown_state_raises(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        with self.assertRaises(InvalidTransitionError):
            self.trainer.transition(job_id, "NONEXISTENT")

    def test_transition_on_missing_job_raises(self) -> None:
        with self.assertRaises(InvalidTransitionError):
            self.trainer.transition("nonexistent-id", "ELIGIBLE")

    def test_error_state_transition_blocked_policy(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        self.trainer.transition(job_id, "BLOCKED_POLICY")
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "BLOCKED_POLICY")

        # Can recover to ELIGIBLE
        self.trainer.transition(job_id, "ELIGIBLE")
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "ELIGIBLE")

    def test_error_state_paused_resume(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PAUSED")
        # Resume from PAUSED to QUEUED
        self.trainer.transition(job_id, "QUEUED")
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "QUEUED")

    def test_terminal_states_have_no_transitions(self) -> None:
        terminal = {"COMPLETED", "FAILED_FATAL", "REJECTED", "ROLLBACK", "SUPERSEDED"}
        for state in terminal:
            allowed = VALID_TRANSITIONS.get(state, set())
            self.assertEqual(
                allowed,
                set(),
                f"Terminal state {state} should have no transitions but has {allowed}",
            )

    def test_transition_kwargs_persist(self) -> None:
        job_id = self.trainer.create_job(
            job_type="lora",
            binding_key="b1",
            base_model_id="m1",
            base_version="1.0",
        )
        self.trainer.transition(
            job_id, "BLOCKED_POLICY", last_error="Budget exhausted"
        )
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["last_error"], "Budget exhausted")

    def test_list_jobs_all(self) -> None:
        self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.create_job("ia3", "b2", "m2", "2.0")
        jobs = self.trainer.list_jobs()
        self.assertEqual(len(jobs), 2)

    def test_list_jobs_by_state(self) -> None:
        j1 = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.create_job("ia3", "b2", "m2", "2.0")
        self.trainer.transition(j1, "ELIGIBLE")
        new_jobs = self.trainer.list_jobs(state="NEW")
        self.assertEqual(len(new_jobs), 1)
        eligible_jobs = self.trainer.list_jobs(state="ELIGIBLE")
        self.assertEqual(len(eligible_jobs), 1)


class TestLocalTrainerTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.trainer = LocalTrainer(self.db)

    def test_train_runs_steps(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PREPARING_DATA")
        self.trainer.transition(job_id, "WAITING_FOR_RESOURCES")

        steps_run: list[int] = []

        def train_fn(ctx: dict) -> dict:
            steps_run.append(ctx["step"])
            return {"loss": 1.0 / (ctx["step"] + 1), "done": ctx["step"] >= 4}

        metrics = self.trainer.train(job_id, train_fn, max_steps=10)
        self.assertEqual(len(steps_run), 5)  # steps 0-4
        self.assertIn("loss", metrics)
        self.assertIn("step", metrics)

    def test_train_with_checkpointing(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PREPARING_DATA")
        self.trainer.transition(job_id, "WAITING_FOR_RESOURCES")

        def train_fn(ctx: dict) -> dict:
            return {
                "loss": 0.5,
                "done": ctx["step"] >= 9,
                "checkpoint_data": b"model_weights_" + str(ctx["step"]).encode(),
            }

        metrics = self.trainer.train(
            job_id, train_fn, max_steps=10, checkpoint_every=5
        )
        # Should have checkpoint at step 5 and step 10
        checkpoints = self.db.execute(
            "SELECT * FROM training_checkpoints WHERE job_id = ? ORDER BY step",
            (job_id,),
        )
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(dict(checkpoints[0])["step"], 5)
        self.assertEqual(dict(checkpoints[1])["step"], 10)

    def test_train_failure_transitions_to_retryable(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PREPARING_DATA")
        self.trainer.transition(job_id, "WAITING_FOR_RESOURCES")

        def failing_train_fn(ctx: dict) -> dict:
            raise RuntimeError("OOM")

        with self.assertRaises(RuntimeError):
            self.trainer.train(job_id, failing_train_fn, max_steps=5)

        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "FAILED_RETRYABLE")
        self.assertIn("OOM", job["last_error"])

    def test_resume_from_checkpoint(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PREPARING_DATA")
        self.trainer.transition(job_id, "WAITING_FOR_RESOURCES")

        def train_fn(ctx: dict) -> dict:
            return {
                "loss": 0.5,
                "done": ctx["step"] >= 4,
                "checkpoint_data": b"ckpt",
            }

        self.trainer.train(job_id, train_fn, max_steps=5, checkpoint_every=5)
        ckpt = self.trainer.resume_from_checkpoint(job_id)
        self.assertIsNotNone(ckpt)
        assert ckpt is not None
        self.assertEqual(ckpt["step"], 5)

    def test_resume_no_checkpoint(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        ckpt = self.trainer.resume_from_checkpoint(job_id)
        self.assertIsNone(ckpt)


class TestLocalTrainerEvaluate(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.trainer = LocalTrainer(self.db)

    def _create_trained_job(self) -> str:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        for state in [
            "ELIGIBLE",
            "QUEUED",
            "PREPARING_DATA",
            "WAITING_FOR_RESOURCES",
            "TRAINING",
        ]:
            self.trainer.transition(job_id, state)
        return job_id

    def test_evaluate_accept(self) -> None:
        job_id = self._create_trained_job()

        def eval_fn(ctx: dict) -> dict:
            return {"accept": True, "score": 0.95}

        result = self.trainer.evaluate(job_id, eval_fn)
        self.assertTrue(result["accept"])

        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "CANDIDATE_READY")

    def test_evaluate_reject(self) -> None:
        job_id = self._create_trained_job()

        def eval_fn(ctx: dict) -> dict:
            return {"accept": False, "reason": "Score too low"}

        result = self.trainer.evaluate(job_id, eval_fn)
        self.assertFalse(result["accept"])

        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "REJECTED")

    def test_evaluate_failure(self) -> None:
        job_id = self._create_trained_job()

        def eval_fn(ctx: dict) -> dict:
            raise RuntimeError("Eval crash")

        with self.assertRaises(RuntimeError):
            self.trainer.evaluate(job_id, eval_fn)

        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "FAILED_RETRYABLE")


class TestLocalTrainerPauseCancel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _make_db()
        self.trainer = LocalTrainer(self.db)

    def test_pause(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.pause(job_id)
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "PAUSED")

    def test_cancel(self) -> None:
        job_id = self.trainer.create_job("lora", "b1", "m1", "1.0")
        self.trainer.transition(job_id, "ELIGIBLE")
        self.trainer.transition(job_id, "QUEUED")
        self.trainer.transition(job_id, "PREPARING_DATA")
        self.trainer.transition(job_id, "FAILED_RETRYABLE", last_error="test")
        self.trainer.cancel(job_id)
        job = self.trainer.get_job(job_id)
        assert job is not None
        self.assertEqual(job["state"], "FAILED_FATAL")


if __name__ == "__main__":
    unittest.main()
