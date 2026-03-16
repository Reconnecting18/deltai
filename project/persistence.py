"""
E3N Persistence Layer — SQLite backing for conversation history and cloud budget.

Provides durable storage so state survives server restarts.
In-memory data remains the primary source; SQLite is the backing store.
Reads happen once at startup, writes happen alongside in-memory updates.

DB location: configured via SQLITE_PATH in .env (default: C:\\e3n\\data\\sqlite\\e3n.db)
"""

import sqlite3
import os
import time
import logging

logger = logging.getLogger("e3n.persistence")

_db_path = os.getenv("SQLITE_PATH", r"C:\e3n\data\sqlite\e3n.db")


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection. Caller should use `with` or close manually."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    conn = sqlite3.connect(_db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_daily (
                date TEXT PRIMARY KEY,
                spent REAL NOT NULL DEFAULT 0.0
            )
        """)
        conn.commit()
    logger.info(f"Persistence DB initialized: {_db_path}")


# ── CONVERSATION HISTORY ─────────────────────────────────────────────


def load_history(max_turns: int) -> list[dict]:
    """
    Load the last N turns (N*2 rows) from the DB.
    Returns list of {"role": "user"|"assistant", "content": "..."}.
    """
    max_rows = max_turns * 2
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
            (max_rows,),
        ).fetchall()
    # Rows come back newest-first, reverse to get chronological order
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def save_history_pair(user_msg: str, assistant_msg: str):
    """Append one user+assistant pair to the DB."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversation_history (role, content, created_at) VALUES (?, ?, ?)",
            ("user", user_msg, now),
        )
        conn.execute(
            "INSERT INTO conversation_history (role, content, created_at) VALUES (?, ?, ?)",
            ("assistant", assistant_msg, now),
        )
        conn.commit()


def clear_history():
    """Delete all conversation history rows."""
    with _connect() as conn:
        conn.execute("DELETE FROM conversation_history")
        conn.commit()
    logger.info("Conversation history cleared from DB")


def trim_history(max_turns: int):
    """Keep only the last max_turns pairs in the DB (delete oldest)."""
    max_rows = max_turns * 2
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM conversation_history").fetchone()[0]
        if count > max_rows:
            conn.execute(
                "DELETE FROM conversation_history WHERE id NOT IN "
                "(SELECT id FROM conversation_history ORDER BY id DESC LIMIT ?)",
                (max_rows,),
            )
            conn.commit()


# ── BUDGET ───────────────────────────────────────────────────────────


def load_budget(date: str) -> float:
    """Load a specific day's spend. Returns 0.0 if no record."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT spent FROM budget_daily WHERE date = ?", (date,)
        ).fetchone()
    return row[0] if row else 0.0


def save_budget(date: str, spent: float):
    """Upsert a day's spend total."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO budget_daily (date, spent) VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET spent = excluded.spent",
            (date, spent),
        )
        conn.commit()
