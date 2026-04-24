"""
SQLite DDL for arch update guard (same DB file as persistence).

Call init_arch_guard_tables(conn) with an open connection; tables use arch_guard_ prefix.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1


def init_arch_guard_tables(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_schema_version (
            version INTEGER PRIMARY KEY
        )
    """)
    row = conn.execute("SELECT version FROM arch_guard_schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO arch_guard_schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_snapshots (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            kind TEXT NOT NULL,
            label TEXT,
            trigger_source TEXT,
            hostname TEXT,
            kernel_release TEXT,
            system_summary_json TEXT,
            deltai_summary_json TEXT,
            flags_json TEXT,
            rollback_notes TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_snapshot_blobs (
            snapshot_id TEXT NOT NULL,
            name TEXT NOT NULL,
            mime TEXT,
            storage TEXT NOT NULL,
            filesystem_path TEXT,
            sha256 TEXT,
            byte_size INTEGER,
            PRIMARY KEY (snapshot_id, name),
            FOREIGN KEY (snapshot_id) REFERENCES arch_guard_snapshots(id) ON DELETE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_update_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at REAL NOT NULL,
            mode TEXT,
            pending_json TEXT,
            errors_json TEXT,
            linked_snapshot_id TEXT,
            FOREIGN KEY (linked_snapshot_id) REFERENCES arch_guard_snapshots(id) ON DELETE SET NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_diffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_snapshot_id TEXT NOT NULL,
            to_snapshot_id TEXT NOT NULL,
            computed_at REAL NOT NULL,
            packages_added_json TEXT,
            packages_removed_json TEXT,
            packages_upgraded_json TEXT,
            configs_changed_json TEXT,
            breaking_flags_json TEXT,
            severity TEXT,
            UNIQUE (from_snapshot_id, to_snapshot_id),
            FOREIGN KEY (from_snapshot_id) REFERENCES arch_guard_snapshots(id) ON DELETE CASCADE,
            FOREIGN KEY (to_snapshot_id) REFERENCES arch_guard_snapshots(id) ON DELETE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arch_guard_rollback_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_at REAL NOT NULL,
            requested_by TEXT,
            target_snapshot_id TEXT NOT NULL,
            status TEXT NOT NULL,
            steps_json TEXT,
            error TEXT,
            FOREIGN KEY (target_snapshot_id) REFERENCES arch_guard_snapshots(id) ON DELETE CASCADE
        )
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_arch_guard_snapshots_created "
        "ON arch_guard_snapshots(created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_arch_guard_checks_checked "
        "ON arch_guard_update_checks(checked_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_arch_guard_diffs_pair "
        "ON arch_guard_diffs(from_snapshot_id, to_snapshot_id)"
    )
