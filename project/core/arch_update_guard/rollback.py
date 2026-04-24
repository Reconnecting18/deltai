"""
Rollback planner: restore captured configs and stage SQLite restore (never overwrite live DB in-process).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from typing import Any

from persistence import get_sqlite_path

from .schema import init_arch_guard_tables
from . import snapshots as snap


def _connect() -> sqlite3.Connection:
    path = get_sqlite_path()
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    init_arch_guard_tables(conn)
    return conn


def _staging_root() -> str:
    base = os.path.expanduser(os.getenv("DELTA_DATA_DIR", "~/.local/share/deltai"))
    root = os.path.join(base, "arch_guard", "rollback_staging")
    os.makedirs(root, exist_ok=True)
    return root


def plan_rollback(snapshot_id: str) -> dict[str, Any]:
    meta = snap.get_snapshot_meta(snapshot_id)
    if not meta:
        return {"ok": False, "error": "snapshot not found"}

    steps: list[dict[str, Any]] = []
    raw_cfg = snap.read_blob_text(snapshot_id, "configs_json") or "{}"
    try:
        cfg = json.loads(raw_cfg)
    except json.JSONDecodeError:
        cfg = {}

    if isinstance(cfg, dict):
        for path, info in cfg.items():
            if not isinstance(info, dict) or "content" not in info:
                continue
            steps.append(
                {
                    "action": "restore_etc_file",
                    "path": path,
                    "bytes": len((info.get("content") or "").encode("utf-8")),
                    "requires_root": True,
                }
            )

    # delta db snapshot blob
    with _connect() as conn:
        row = conn.execute(
            "SELECT filesystem_path FROM arch_guard_snapshot_blobs "
            "WHERE snapshot_id = ? AND name = ?",
            (snapshot_id, "delta_db_snapshot"),
        ).fetchone()
    if row and row["filesystem_path"] and os.path.isfile(row["filesystem_path"]):
        live = os.path.expanduser(get_sqlite_path())
        candidate = live + f".restore_candidate.{snapshot_id[:8]}"
        steps.append(
            {
                "action": "stage_sqlite_restore",
                "source": row["filesystem_path"],
                "target_candidate": candidate,
                "note": "Stop deltai, then mv candidate over live DB if desired.",
            }
        )

    return {"ok": True, "snapshot_id": snapshot_id, "steps": steps}


def execute_rollback(
    snapshot_id: str,
    *,
    dry_run: bool,
    requested_by: str,
    apply_etc: bool = False,
) -> dict[str, Any]:
    plan = plan_rollback(snapshot_id)
    if not plan.get("ok"):
        return plan

    now = time.time()
    job_steps: list[dict[str, Any]] = []
    err: str | None = None
    status = "completed"
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    if dry_run:
        job_steps.append({"step": "dry_run", "detail": plan["steps"]})
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO arch_guard_rollback_jobs
                (requested_at, requested_by, target_snapshot_id, status, steps_json, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now, requested_by, snapshot_id, "completed", json.dumps(job_steps), None),
            )
            conn.commit()
        return {"ok": True, "dry_run": True, "plan": plan["steps"], "job_status": "completed"}

    raw_cfg = snap.read_blob_text(snapshot_id, "configs_json") or "{}"
    try:
        cfg = json.loads(raw_cfg)
    except json.JSONDecodeError:
        cfg = {}

    if apply_etc and is_root and isinstance(cfg, dict):
        for path, info in cfg.items():
            if not isinstance(info, dict):
                continue
            content = info.get("content")
            if content is None:
                continue
            p = path
            if not p.startswith("/etc/"):
                job_steps.append({"path": p, "skipped": True, "reason": "not under /etc"})
                continue
            try:
                parent = os.path.dirname(p)
                os.makedirs(parent, exist_ok=True)
                tmp = p + ".arch_guard_tmp"
                with open(tmp, "w", encoding="utf-8", errors="strict") as f:
                    f.write(content)
                os.replace(tmp, p)
                job_steps.append({"path": p, "restored": True})
            except OSError as e:
                job_steps.append({"path": p, "error": str(e)})
                err = str(e)
                status = "partial"
    elif apply_etc and not is_root:
        staging = os.path.join(_staging_root(), snapshot_id[:8] + "_" + str(int(now)))
        os.makedirs(staging, exist_ok=True)
        if isinstance(cfg, dict):
            for path, info in cfg.items():
                if not isinstance(info, dict) or "content" not in info:
                    continue
                if not path.startswith("/etc/"):
                    continue
                rel = path.lstrip("/").replace(os.sep, "_")
                out = os.path.join(staging, rel)
                try:
                    with open(out, "w", encoding="utf-8", errors="replace") as f:
                        f.write(info.get("content") or "")
                    job_steps.append(
                        {
                            "path": path,
                            "written_to": out,
                            "note": "Copy to target as root, or re-run with sudo and apply_etc.",
                        }
                    )
                except OSError as e:
                    job_steps.append({"path": path, "error": str(e)})
                    err = str(e)
                    status = "partial"
    else:
        job_steps.append(
            {
                "note": "Config restore skipped (apply_etc=false or not root). "
                "Use CLI with --apply-etc and root for in-place /etc restore."
            }
        )

    # Stage DB copy (safe: does not replace live)
    with _connect() as conn:
        row = conn.execute(
            "SELECT filesystem_path FROM arch_guard_snapshot_blobs "
            "WHERE snapshot_id = ? AND name = ?",
            (snapshot_id, "delta_db_snapshot"),
        ).fetchone()
    if row and row["filesystem_path"] and os.path.isfile(row["filesystem_path"]):
        live = os.path.expanduser(get_sqlite_path())
        candidate = live + f".restore_candidate.{snapshot_id[:8]}"
        try:
            shutil.copy2(row["filesystem_path"], candidate)
            job_steps.append({"sqlite_staged": candidate})
        except OSError as e:
            job_steps.append({"sqlite_stage_error": str(e)})
            if err is None:
                err = str(e)
            status = "partial"

    any_ok = any(
        s.get("restored") or s.get("written_to") or s.get("sqlite_staged") for s in job_steps
    )
    if err and not any_ok:
        final_status = "failed"
    else:
        final_status = status

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO arch_guard_rollback_jobs
            (requested_at, requested_by, target_snapshot_id, status, steps_json, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                requested_by,
                snapshot_id,
                final_status,
                json.dumps(job_steps, ensure_ascii=False),
                err,
            ),
        )
        conn.commit()

    return {
        "ok": final_status != "failed",
        "dry_run": False,
        "status": final_status,
        "steps": job_steps,
        "error": err,
    }
