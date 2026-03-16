"""DDL statements for training-related tables in the device agent local database.

These extend the base schema from device_agent.db.schema and cover:
- Training jobs and checkpoints
- Training datasets and snapshots
- Federated round participation
- Privacy accounting (epsilon/delta tracking)
"""

from __future__ import annotations

TRAINING_SCHEMA_STATEMENTS: list[str] = [
    # -- Training jobs --
    """
    CREATE TABLE IF NOT EXISTS training_jobs (
        job_id                   TEXT PRIMARY KEY,
        job_type                 TEXT NOT NULL,
        binding_key              TEXT NOT NULL,
        base_model_id            TEXT NOT NULL,
        base_version             TEXT NOT NULL,
        starting_adapter_version TEXT,
        state                    TEXT NOT NULL,
        dataset_id               TEXT,
        round_id                 TEXT,
        participation_id         TEXT,
        progress_json            TEXT,
        metrics_json             TEXT,
        last_error               TEXT,
        retry_count              INTEGER NOT NULL DEFAULT 0,
        created_at               TEXT NOT NULL,
        updated_at               TEXT NOT NULL
    )
    """,
    # -- Training checkpoints --
    """
    CREATE TABLE IF NOT EXISTS training_checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        job_id        TEXT NOT NULL,
        path          TEXT NOT NULL,
        step          INTEGER NOT NULL,
        bytes         INTEGER NOT NULL,
        sha256        TEXT,
        created_at    TEXT NOT NULL
    )
    """,
    # -- Training datasets (snapshot-based) --
    """
    CREATE TABLE IF NOT EXISTS training_datasets (
        dataset_id          TEXT PRIMARY KEY,
        source_scope        TEXT NOT NULL,
        example_count       INTEGER NOT NULL,
        byte_size           INTEGER NOT NULL,
        selection_policy_json TEXT,
        snapshot_created_at TEXT NOT NULL,
        expires_at          TEXT
    )
    """,
    # -- Federated round participation --
    """
    CREATE TABLE IF NOT EXISTS federated_participation (
        participation_id TEXT PRIMARY KEY,
        round_id         TEXT NOT NULL,
        device_id        TEXT NOT NULL,
        state            TEXT NOT NULL,
        accepted_at      TEXT,
        deadline_at      TEXT,
        upload_id        TEXT,
        update_sha256    TEXT,
        update_bytes     INTEGER,
        last_error       TEXT,
        updated_at       TEXT NOT NULL
    )
    """,
    # -- Privacy accounting per scope --
    """
    CREATE TABLE IF NOT EXISTS privacy_accounting (
        scope_key        TEXT PRIMARY KEY,
        epsilon_spent    REAL NOT NULL DEFAULT 0.0,
        delta_spent      REAL NOT NULL DEFAULT 0.0,
        last_updated_at  TEXT NOT NULL
    )
    """,
]
