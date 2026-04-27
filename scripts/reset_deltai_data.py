#!/usr/bin/env python3
"""
Destructive maintenance: optional ChromaDB tree removal and SQLite analytics wipe.

Requires explicit confirmation flag. Does not start the FastAPI app.
Run from repo root: python scripts/reset_deltai_data.py --help
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def _expand(p: str | None, default: str) -> Path:
    raw = (p or default).strip()
    return Path(os.path.expanduser(raw)).resolve()


def wipe_sqlite_analytics(
    db_path: Path,
    *,
    include_budget: bool,
) -> None:
    if not db_path.is_file():
        print(f"SQLite file not found (skip): {db_path}", file=sys.stderr)
        return
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        for table in (
            "reasoning_traces",
            "quality_scores",
            "routing_feedback",
            "knowledge_gaps",
            "conversation_history",
        ):
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        if include_budget:
            try:
                conn.execute("DELETE FROM budget_daily")
            except sqlite3.OperationalError:
                pass
        try:
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
    print(f"Wiped analytics tables in {db_path}")


def wipe_chromadb_dir(chroma_path: Path) -> None:
    if not chroma_path.exists():
        print(f"Chroma path not found (skip): {chroma_path}", file=sys.stderr)
        return
    if chroma_path.is_dir():
        shutil.rmtree(chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        print(f"Removed and recreated directory: {chroma_path}")
    else:
        chroma_path.unlink()
        print(f"Removed file: {chroma_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset deltai local Chroma/SQLite analytics data.")
    parser.add_argument(
        "--i-understand-this-deletes-data",
        action="store_true",
        required=True,
        help="Required. Refuses to run without this flag.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("DELTA_SQLITE_PATH", os.getenv("SQLITE_PATH", "~/.local/share/deltai/delta.db")),
        help="SQLite database path",
    )
    parser.add_argument(
        "--chromadb-path",
        default=os.getenv("CHROMADB_PATH", "~/.local/share/deltai/chromadb"),
        help="ChromaDB directory to delete and recreate",
    )
    parser.add_argument(
        "--wipe-sqlite-analytics",
        action="store_true",
        help="DELETE analytics/chat tables and VACUUM",
    )
    parser.add_argument(
        "--wipe-chromadb",
        action="store_true",
        help="Remove Chroma directory (or file) and recreate empty dir",
    )
    parser.add_argument(
        "--include-budget",
        action="store_true",
        help="Also DELETE FROM budget_daily when wiping SQLite",
    )
    args = parser.parse_args()

    if not args.wipe_sqlite_analytics and not args.wipe_chromadb:
        print("Nothing to do: pass --wipe-sqlite-analytics and/or --wipe-chromadb", file=sys.stderr)
        return 2

    db = _expand(args.sqlite_path, "~/.local/share/deltai/delta.db")
    ch = _expand(args.chromadb_path, "~/.local/share/deltai/chromadb")

    if args.wipe_sqlite_analytics:
        wipe_sqlite_analytics(db, include_budget=args.include_budget)
    if args.wipe_chromadb:
        wipe_chromadb_dir(ch)

    print("Done. Clear in-memory chat via DELETE /chat/history if the server was running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
