"""
Create and query update snapshots (metadata + on-disk blobs).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from typing import Any

from persistence import get_sqlite_path

from .collector import build_snapshot_payload
from .schema import init_arch_guard_tables


def _connect_rw() -> sqlite3.Connection:
    path = get_sqlite_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    init_arch_guard_tables(conn)
    return conn


def _blobs_root() -> str:
    base = os.path.expanduser(os.getenv("DELTA_DATA_DIR", "~/.local/share/deltai"))
    root = os.path.join(base, "arch_guard", "blobs")
    os.makedirs(root, exist_ok=True)
    return root


def _snapshot_dir(sid: str) -> str:
    d = os.path.join(_blobs_root(), sid)
    os.makedirs(d, exist_ok=True)
    return d


def create_snapshot(
    *,
    kind: str,
    label: str | None,
    trigger: str,
    include_reverse_deps: bool = False,
    backup_sqlite: bool = True,
) -> dict[str, Any]:
    if kind not in ("pre_update", "post_update", "manual", "scheduled", "cli"):
        kind = "manual"

    payload = build_snapshot_payload(include_reverse_deps=include_reverse_deps)
    sid = str(uuid.uuid4())
    now = time.time()
    snap_dir = _snapshot_dir(sid)

    pq_text = payload["blobs"].get("pacman_Q") or ""
    pq_path = os.path.join(snap_dir, "pacman_Q.txt")
    with open(pq_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(pq_text)
    pq_sha = hashlib.sha256(pq_text.encode("utf-8", errors="replace")).hexdigest()

    cfg_json = payload["blobs"].get("configs_json") or "{}"
    cfg_path = os.path.join(snap_dir, "configs.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_json)
    cfg_sha = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    sqlite_backup_rel = None
    if backup_sqlite:
        db_src = os.path.expanduser(get_sqlite_path())
        if os.path.isfile(db_src):
            dst = os.path.join(snap_dir, "delta.db.snapshot")
            try:
                shutil.copy2(db_src, dst)
                sqlite_backup_rel = "delta.db.snapshot"
            except OSError:
                sqlite_backup_rel = None

    system_summary = payload["system_summary"]
    deltai_summary = payload["deltai_summary"]
    if sqlite_backup_rel:
        deltai_summary = dict(deltai_summary)
        deltai_summary["sqlite_snapshot_file"] = sqlite_backup_rel

    with _connect_rw() as conn:
        conn.execute(
            """
            INSERT INTO arch_guard_snapshots (
                id, created_at, kind, label, trigger_source, hostname, kernel_release,
                system_summary_json, deltai_summary_json, flags_json, rollback_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                now,
                kind,
                label or "",
                trigger,
                payload["hostname"],
                payload["kernel_release"],
                json.dumps(system_summary, ensure_ascii=False),
                json.dumps(deltai_summary, ensure_ascii=False),
                json.dumps(payload["flags"], ensure_ascii=False),
                "",
            ),
        )
        conn.execute(
            """
            INSERT INTO arch_guard_snapshot_blobs (
                snapshot_id, name, mime, storage, filesystem_path, sha256, byte_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sid, "pacman_Q", "text/plain", "file", pq_path, pq_sha, len(pq_text.encode("utf-8"))),
        )
        conn.execute(
            """
            INSERT INTO arch_guard_snapshot_blobs (
                snapshot_id, name, mime, storage, filesystem_path, sha256, byte_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                "configs_json",
                "application/json",
                "file",
                cfg_path,
                cfg_sha,
                len(cfg_json.encode("utf-8")),
            ),
        )
        if sqlite_backup_rel:
            sp = os.path.join(snap_dir, sqlite_backup_rel)
            st = os.stat(sp)
            conn.execute(
                """
                INSERT INTO arch_guard_snapshot_blobs (
                    snapshot_id, name, mime, storage, filesystem_path, sha256, byte_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    "delta_db_snapshot",
                    "application/octet-stream",
                    "file",
                    sp,
                    "",
                    st.st_size,
                ),
            )
        conn.commit()

    return {
        "snapshot_id": sid,
        "created_at": now,
        "kind": kind,
        "flags": payload["flags"],
        "blob_dir": snap_dir,
    }


def record_update_check(
    *,
    mode: str,
    pending: dict[str, Any],
    linked_snapshot_id: str | None = None,
) -> int:
    with _connect_rw() as conn:
        cur = conn.execute(
            """
            INSERT INTO arch_guard_update_checks (checked_at, mode, pending_json, errors_json, linked_snapshot_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                mode,
                json.dumps(pending, ensure_ascii=False),
                json.dumps(pending.get("errors") or [], ensure_ascii=False),
                linked_snapshot_id,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_snapshots(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with _connect_rw() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, kind, label, trigger_source, hostname, kernel_release, flags_json
            FROM arch_guard_snapshots
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    out = []
    for r in rows:
        flags: list[Any] = []
        try:
            flags = json.loads(r["flags_json"] or "[]")
        except json.JSONDecodeError:
            pass
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "kind": r["kind"],
                "label": r["label"],
                "trigger_source": r["trigger_source"],
                "hostname": r["hostname"],
                "kernel_release": r["kernel_release"],
                "flags": flags,
            }
        )
    return out


def get_snapshot_meta(snapshot_id: str) -> dict[str, Any] | None:
    with _connect_rw() as conn:
        r = conn.execute(
            "SELECT * FROM arch_guard_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if not r:
            return None
        blobs = conn.execute(
            "SELECT name, mime, storage, filesystem_path, sha256, byte_size "
            "FROM arch_guard_snapshot_blobs WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchall()
    meta = dict(r)
    for k in ("system_summary_json", "deltai_summary_json", "flags_json"):
        if meta.get(k):
            try:
                meta[k.replace("_json", "")] = json.loads(meta[k])
            except json.JSONDecodeError:
                meta[k.replace("_json", "")] = None
    meta["blobs"] = [dict(b) for b in blobs]
    return meta


def read_blob_text(snapshot_id: str, name: str) -> str | None:
    with _connect_rw() as conn:
        row = conn.execute(
            "SELECT filesystem_path, storage FROM arch_guard_snapshot_blobs "
            "WHERE snapshot_id = ? AND name = ?",
            (snapshot_id, name),
        ).fetchone()
    if not row or row["storage"] != "file" or not row["filesystem_path"]:
        return None
    p = row["filesystem_path"]
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None
