"""DDL statements for telemetry tables in the device agent local database."""

from __future__ import annotations

TELEMETRY_SCHEMA_STATEMENTS: list[str] = [
    # -- Telemetry event WAL --
    """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        event_id        TEXT PRIMARY KEY,
        device_id       TEXT NOT NULL,
        boot_id         TEXT NOT NULL,
        session_id      TEXT,
        sequence_no     INTEGER NOT NULL,
        event_type      TEXT NOT NULL,
        telemetry_class TEXT NOT NULL,
        occurred_at     TEXT NOT NULL,
        model_id        TEXT,
        model_version   TEXT,
        payload_json    TEXT,
        uploaded        INTEGER NOT NULL DEFAULT 0,
        batch_id        TEXT
    )
    """,
    # -- Cursor tracking for at-least-once delivery --
    """
    CREATE TABLE IF NOT EXISTS telemetry_cursors (
        boot_id         TEXT PRIMARY KEY,
        last_acked_seq  INTEGER NOT NULL DEFAULT 0,
        updated_at      TEXT NOT NULL
    )
    """,
    # -- Index for efficient cursor-based batch selection --
    """
    CREATE INDEX IF NOT EXISTS idx_telemetry_boot_seq
        ON telemetry_events (boot_id, sequence_no)
    """,
    # -- Index for batch selection by upload status and class priority --
    """
    CREATE INDEX IF NOT EXISTS idx_telemetry_uploaded_class
        ON telemetry_events (uploaded, telemetry_class)
    """,
]
