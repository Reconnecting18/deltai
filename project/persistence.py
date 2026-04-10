"""
deltai Persistence Layer — SQLite backing for conversation history and cloud budget.

Provides durable storage so state survives server restarts.
In-memory data remains the primary source; SQLite is the backing store.
Reads happen once at startup, writes happen alongside in-memory updates.

DB location: configured via SQLITE_PATH in .env (default: ~/.local/share/deltai/sqlite/e3n.db)
"""

import sqlite3
import os
import time
import json
import logging

logger = logging.getLogger("deltai.persistence")

_db_path = os.path.expanduser(os.getenv("SQLITE_PATH", "~/.local/share/deltai/sqlite/e3n.db"))


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection. Caller should use `with` or close manually."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    return sqlite3.connect(_db_path, timeout=10)


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                session_id TEXT DEFAULT NULL
            )
        """)
        # Migration: add session_id column if missing
        try:
            conn.execute("ALTER TABLE conversation_history ADD COLUMN session_id TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_daily (
                date TEXT PRIMARY KEY,
                spent REAL NOT NULL DEFAULT 0.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                domain TEXT,
                steps_json TEXT,
                final_summary TEXT,
                tool_sequence TEXT,
                success INTEGER,
                confidence TEXT,
                embedding BLOB,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                response_text_preview TEXT,
                score REAL,
                signals_json TEXT,
                tier INTEGER,
                domain TEXT,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT,
                classified_tier INTEGER,
                actual_model TEXT,
                domain TEXT,
                quality_score REAL,
                latency_ms REAL,
                tool_calls_count INTEGER,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                domain TEXT,
                quality_score REAL,
                gap_type TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_at REAL,
                created_at REAL
            )
        """)
        conn.commit()
    logger.info(f"Persistence DB initialized: {_db_path}")


# ── CONVERSATION HISTORY ─────────────────────────────────────────────


def load_history(max_turns: int, session_id: str = None) -> list[dict]:
    """
    Load the last N turns (N*2 rows) from the DB.
    Returns list of {"role": "user"|"assistant", "content": "..."}.
    """
    max_rows = max_turns * 2
    with _connect() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT role, content FROM conversation_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max_rows),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
                (max_rows,),
            ).fetchall()
    # Rows come back newest-first, reverse to get chronological order
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def save_history_pair(user_msg: str, assistant_msg: str, session_id: str = None):
    """Append one user+assistant pair to the DB."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversation_history (role, content, created_at, session_id) VALUES (?, ?, ?, ?)",
            ("user", user_msg, now, session_id),
        )
        conn.execute(
            "INSERT INTO conversation_history (role, content, created_at, session_id) VALUES (?, ?, ?, ?)",
            ("assistant", assistant_msg, now, session_id),
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


def export_session_history(session_id: str, output_path: str) -> dict:
    """Export a session's conversation history to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM conversation_history WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    if not rows:
        return {"status": "skipped", "reason": "no history for session"}
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for role, content, created_at in rows:
                f.write(json.dumps({"role": role, "content": content, "ts": created_at}) + "\n")
        return {"status": "ok", "turns": len(rows) // 2, "path": output_path}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def save_budget(date: str, spent: float):
    """Upsert a day's spend total."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO budget_daily (date, spent) VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET spent = excluded.spent",
            (date, spent),
        )
        conn.commit()


# ── REASONING TRACES ─────────────────────────────────────────────────


def save_reasoning_trace(query_text: str, domain: str, steps_json: str,
                         final_summary: str, tool_sequence: str,
                         success: bool, confidence: str = "unknown",
                         embedding: bytes = None):
    """Save a ReAct reasoning trace for future reference."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO reasoning_traces (query_text, domain, steps_json, final_summary, "
            "tool_sequence, success, confidence, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_text, domain or "general", steps_json, final_summary[:500],
             tool_sequence, int(success), confidence, embedding, now),
        )
        conn.commit()


def find_similar_traces(embedding: bytes, n: int = 3) -> list[dict]:
    """
    Find reasoning traces with similar embeddings.
    Returns list of {"query_text", "domain", "final_summary", "tool_sequence", "confidence", "steps_json"}.
    Uses brute-force cosine similarity on stored embeddings.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT query_text, domain, final_summary, tool_sequence, confidence, steps_json, embedding "
            "FROM reasoning_traces WHERE success = 1 AND embedding IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()

    if not rows or not embedding:
        return []

    # Deserialize query embedding
    query_emb = _deserialize_embedding(embedding)
    if not query_emb:
        return []

    # Score each trace by cosine similarity
    scored = []
    for query_text, domain, summary, tools, conf, steps, emb_bytes in rows:
        trace_emb = _deserialize_embedding(emb_bytes)
        if not trace_emb or len(trace_emb) != len(query_emb):
            continue
        sim = _cosine_similarity(query_emb, trace_emb)
        if sim > 0.7:  # similarity threshold
            scored.append((sim, {
                "query_text": query_text,
                "domain": domain,
                "final_summary": summary,
                "tool_sequence": tools,
                "confidence": conf,
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:n]]


def prune_old_traces(max_age_days: int = 30, max_count: int = 500):
    """Remove old reasoning traces beyond retention limits."""
    cutoff = time.time() - (max_age_days * 86400)
    with _connect() as conn:
        conn.execute("DELETE FROM reasoning_traces WHERE created_at < ?", (cutoff,))
        # Also cap total count
        count = conn.execute("SELECT COUNT(*) FROM reasoning_traces").fetchone()[0]
        if count > max_count:
            conn.execute(
                "DELETE FROM reasoning_traces WHERE id NOT IN "
                "(SELECT id FROM reasoning_traces ORDER BY created_at DESC LIMIT ?)",
                (max_count,),
            )
        conn.commit()


def _serialize_embedding(emb: list[float]) -> bytes:
    """Pack a float list into bytes for SQLite BLOB storage."""
    import struct
    return struct.pack(f'{len(emb)}f', *emb)


def _deserialize_embedding(data: bytes) -> list[float]:
    """Unpack bytes back to float list."""
    import struct
    if not data:
        return []
    try:
        n = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{n}f', data))
    except Exception:
        return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── QUALITY SCORES ───────────────────────────────────────────────────


def save_quality_score(query_text: str, response_preview: str, score: float,
                       signals_json: str, tier: int = 1, domain: str = "general"):
    """Persist a response quality score."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO quality_scores (query_text, response_text_preview, score, "
            "signals_json, tier, domain, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query_text, response_preview[:200], score, signals_json, tier, domain, now),
        )
        conn.commit()


# ── ROUTING FEEDBACK ─────────────────────────────────────────────────


def save_routing_feedback(query_hash: str, classified_tier: int, actual_model: str,
                          domain: str, quality_score: float, latency_ms: float,
                          tool_calls_count: int):
    """Record a routing outcome for adaptive feedback."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO routing_feedback (query_hash, classified_tier, actual_model, "
            "domain, quality_score, latency_ms, tool_calls_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (query_hash, classified_tier, actual_model, domain, quality_score,
             latency_ms, tool_calls_count, now),
        )
        conn.commit()


def get_routing_stats(domain: str, limit: int = 100) -> list[dict]:
    """Get recent routing outcomes for a domain."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT classified_tier, actual_model, quality_score, latency_ms "
            "FROM routing_feedback WHERE domain = ? ORDER BY created_at DESC LIMIT ?",
            (domain, limit),
        ).fetchall()
    return [{"tier": r[0], "model": r[1], "score": r[2], "latency": r[3]} for r in rows]


# ── KNOWLEDGE GAPS ───────────────────────────────────────────────────


def save_knowledge_gap(query_text: str, domain: str, quality_score: float, gap_type: str):
    """Log a knowledge gap when deltai fails to answer well."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO knowledge_gaps (query_text, domain, quality_score, gap_type, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (query_text, domain or "general", quality_score, gap_type, now),
        )
        conn.commit()


def count_unresolved_knowledge_gaps() -> int:
    """Return number of unresolved knowledge gap rows (for daily training report)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM knowledge_gaps WHERE resolved = 0"
        ).fetchone()
    return int(row[0]) if row else 0


def get_unresolved_gaps(limit: int = 50) -> list[dict]:
    """Get unresolved knowledge gaps."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, query_text, domain, quality_score, gap_type, created_at "
            "FROM knowledge_gaps WHERE resolved = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r[0], "query_text": r[1], "domain": r[2], "score": r[3],
             "gap_type": r[4], "created_at": r[5]} for r in rows]


def resolve_knowledge_gap(gap_id: int):
    """Mark a knowledge gap as resolved."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE knowledge_gaps SET resolved = 1, resolved_at = ? WHERE id = ?",
            (now, gap_id),
        )
        conn.commit()
