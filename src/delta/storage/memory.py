"""Memory persistence primitives.

Implements storage operations for context snippets and learned memory items.
"""

from __future__ import annotations

from pathlib import Path

from delta.storage.db import connect


def store_memory_item(
    db_path: Path,
    session_id: str | None,
    kind: str,
    value: str,
    key: str | None = None,
    importance: float = 0.5,
) -> None:
    """Persist one memory item into SQLite."""
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO memory_items (session_id, kind, key, value, importance, created_at)
            VALUES (?, ?, ?, ?, ?, strftime('%s', 'now'))
            """,
            (session_id, kind, key, value, importance),
        )
        conn.commit()
