"""SQLite connection and bootstrap helpers for DELTA."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection with sensible defaults."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_database(db_path: Path) -> None:
    """Create database file and apply core schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
