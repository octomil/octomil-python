"""DDL statements for the device agent local database."""

from __future__ import annotations

SCHEMA_STATEMENTS: list[str] = [
    # -- Model artifact lifecycle tracking --
    """
    CREATE TABLE IF NOT EXISTS model_artifacts (
        artifact_id   TEXT PRIMARY KEY,
        model_id      TEXT NOT NULL,
        version       TEXT NOT NULL,
        status        TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        bytes_downloaded INTEGER NOT NULL DEFAULT 0,
        total_bytes   INTEGER NOT NULL,
        verified_at   TEXT,
        staged_at     TEXT,
        activated_at  TEXT,
        installed_at  INTEGER,
        activation_policy TEXT NOT NULL DEFAULT 'immediate',
        last_error    TEXT,
        retry_count   INTEGER NOT NULL DEFAULT 0,
        engine_benchmarks TEXT,
        updated_at    TEXT NOT NULL
    )
    """,
    # -- Per-chunk download progress --
    """
    CREATE TABLE IF NOT EXISTS download_chunks (
        artifact_id TEXT NOT NULL,
        file_path   TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        status      TEXT NOT NULL DEFAULT 'PENDING',
        attempts    INTEGER NOT NULL DEFAULT 0,
        last_error  TEXT,
        PRIMARY KEY (artifact_id, file_path, chunk_index)
    )
    """,
    # -- Lease-based operation queue --
    """
    CREATE TABLE IF NOT EXISTS operations (
        op_id          TEXT PRIMARY KEY,
        op_type        TEXT NOT NULL,
        resource_id    TEXT,
        state          TEXT NOT NULL DEFAULT 'PENDING',
        idempotency_key TEXT,
        attempt_count  INTEGER NOT NULL DEFAULT 0,
        lease_owner    TEXT,
        lease_expires_at TEXT,
        next_retry_at  TEXT,
        payload_json   TEXT,
        priority       INTEGER NOT NULL DEFAULT 100,
        updated_at     TEXT NOT NULL
    )
    """,
    # -- Active model version pointer --
    """
    CREATE TABLE IF NOT EXISTS active_model_pointer (
        model_id         TEXT PRIMARY KEY,
        active_version   TEXT NOT NULL,
        previous_version TEXT,
        updated_at       TEXT NOT NULL
    )
    """,
    # -- Registered base models --
    """
    CREATE TABLE IF NOT EXISTS base_models (
        model_id   TEXT NOT NULL,
        version    TEXT NOT NULL,
        status     TEXT NOT NULL DEFAULT 'REGISTERED',
        artifact_id TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (model_id, version)
    )
    """,
    # -- Personalization adapters (LoRA, etc.) --
    """
    CREATE TABLE IF NOT EXISTS personalization_adapters (
        adapter_id             TEXT PRIMARY KEY,
        base_model_id          TEXT NOT NULL,
        base_version           TEXT NOT NULL,
        adapter_scope          TEXT NOT NULL,
        version                TEXT NOT NULL,
        parent_adapter_version TEXT,
        status                 TEXT NOT NULL DEFAULT 'REGISTERED',
        artifact_path          TEXT NOT NULL,
        metrics_json           TEXT,
        created_at             TEXT NOT NULL
    )
    """,
    # -- Active binding: which adapter is active for a base model --
    """
    CREATE TABLE IF NOT EXISTS active_bindings (
        binding_key    TEXT PRIMARY KEY,
        base_model_id  TEXT NOT NULL,
        base_version   TEXT NOT NULL,
        adapter_id     TEXT,
        adapter_version TEXT,
        updated_at     TEXT NOT NULL
    )
    """,
    # -- Rollback audit log --
    """
    CREATE TABLE IF NOT EXISTS rollback_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id        TEXT NOT NULL,
        from_version    TEXT NOT NULL,
        to_version      TEXT NOT NULL,
        reason          TEXT NOT NULL,
        rolled_back_at  TEXT NOT NULL
    )
    """,
    # -- Runtime binary version tracking --
    """
    CREATE TABLE IF NOT EXISTS runtime_versions (
        runtime_id    TEXT PRIMARY KEY,
        version       TEXT NOT NULL,
        status        TEXT NOT NULL,
        artifact_path TEXT,
        downloaded_at TEXT,
        verified_at   TEXT,
        pending_since TEXT,
        activated_at  TEXT,
        updated_at    TEXT NOT NULL
    )
    """,
    # -- Per-engine benchmark results --
    """
    CREATE TABLE IF NOT EXISTS benchmark_results (
        model_id       TEXT NOT NULL,
        model_version  TEXT NOT NULL,
        device_class   TEXT NOT NULL,
        sdk_version    TEXT NOT NULL,
        engine         TEXT NOT NULL,
        latency_ms     REAL,
        throughput_tps REAL,
        memory_bytes   INTEGER,
        metadata_json  TEXT,
        recorded_at    TEXT NOT NULL,
        PRIMARY KEY (model_id, model_version, device_class, sdk_version, engine)
    )
    """,
    # -- Boot history for crash detection --
    """
    CREATE TABLE IF NOT EXISTS boot_history (
        boot_id              TEXT PRIMARY KEY,
        started_at           TEXT NOT NULL,
        active_model_id      TEXT,
        active_model_version TEXT,
        runtime_version      TEXT,
        clean_shutdown        INTEGER DEFAULT 0,
        crash_detected        INTEGER DEFAULT 0,
        duration_sec          REAL
    )
    """,
]
