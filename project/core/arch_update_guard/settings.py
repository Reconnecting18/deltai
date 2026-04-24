"""Runtime settings for arch update guard (SQLite key/value)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

_DEFAULT_INTERVAL = 3600.0


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM arch_guard_settings WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return row[0]


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO arch_guard_settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def load_mode(conn: sqlite3.Connection) -> str:
    v = get_setting(conn, "mode", None)
    if v in ("auto", "manual"):
        return v
    env = os.getenv("ARCH_GUARD_MODE", "manual").strip().lower()
    return env if env in ("auto", "manual") else "manual"


def save_mode(conn: sqlite3.Connection, mode: str) -> None:
    if mode not in ("auto", "manual"):
        raise ValueError("mode must be auto or manual")
    set_setting(conn, "mode", mode)


def load_enabled(conn: sqlite3.Connection) -> bool:
    v = get_setting(conn, "enabled", None)
    if v is not None:
        return v.lower() in ("1", "true", "yes", "on")
    return os.getenv("ARCH_GUARD_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def load_interval_sec(conn: sqlite3.Connection) -> float:
    v = get_setting(conn, "auto_interval_sec", None)
    if v:
        try:
            return max(60.0, float(v))
        except ValueError:
            pass
    raw = os.getenv("ARCH_GUARD_INTERVAL_SEC", str(int(_DEFAULT_INTERVAL)))
    try:
        return max(60.0, float(raw))
    except ValueError:
        return _DEFAULT_INTERVAL


def save_interval_sec(conn: sqlite3.Connection, sec: float) -> None:
    set_setting(conn, "auto_interval_sec", str(max(60.0, float(sec))))


def load_last_check_at(conn: sqlite3.Connection) -> float | None:
    v = get_setting(conn, "last_check_at", None)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def save_last_check_at(conn: sqlite3.Connection, ts: float) -> None:
    set_setting(conn, "last_check_at", str(ts))


def public_settings(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "mode": load_mode(conn),
        "enabled": load_enabled(conn),
        "auto_interval_sec": load_interval_sec(conn),
        "last_check_at": load_last_check_at(conn),
    }
