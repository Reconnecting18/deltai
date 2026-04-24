"""
Compare snapshots and persist diffs; run update checks.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from persistence import get_sqlite_path

from .collector import parse_pacman_q_list
from .pacman_audit import get_pending_updates
from .schema import init_arch_guard_tables
from . import snapshots as snap


def _connect() -> sqlite3.Connection:
    path = get_sqlite_path()
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    init_arch_guard_tables(conn)
    return conn


def _configs_map_from_json(raw: str) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict) and "content" in v:
            out[k] = v
    return out


def compute_diff(from_snapshot_id: str, to_snapshot_id: str) -> dict[str, Any]:
    pq_a = snap.read_blob_text(from_snapshot_id, "pacman_Q") or ""
    pq_b = snap.read_blob_text(to_snapshot_id, "pacman_Q") or ""
    map_a = parse_pacman_q_list(pq_a)
    map_b = parse_pacman_q_list(pq_b)

    added: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []
    upgraded: list[dict[str, str]] = []

    for pkg, ver_b in map_b.items():
        if pkg not in map_a:
            added.append({"package": pkg, "version": ver_b})
        elif map_a[pkg] != ver_b:
            upgraded.append({"package": pkg, "from_version": map_a[pkg], "to_version": ver_b})

    for pkg, ver_a in map_a.items():
        if pkg not in map_b:
            removed.append({"package": pkg, "version": ver_a})

    raw_cfg_a = snap.read_blob_text(from_snapshot_id, "configs_json") or "{}"
    raw_cfg_b = snap.read_blob_text(to_snapshot_id, "configs_json") or "{}"
    cf_a = _configs_map_from_json(raw_cfg_a)
    cf_b = _configs_map_from_json(raw_cfg_b)

    configs_changed: list[dict[str, Any]] = []
    paths = set(cf_a) | set(cf_b)
    for path in sorted(paths):
        ha = (cf_a.get(path) or {}).get("sha256")
        hb = (cf_b.get(path) or {}).get("sha256")
        if path not in cf_a:
            configs_changed.append({"path": path, "change_type": "added"})
        elif path not in cf_b:
            configs_changed.append({"path": path, "change_type": "removed"})
        elif ha != hb:
            configs_changed.append({"path": path, "change_type": "modified", "sha256_before": ha, "sha256_after": hb})

    breaking: list[str] = []
    for u in upgraded:
        pkg = u["package"]
        if pkg.startswith("linux"):
            breaking.append("kernel_package_changed")
        if pkg == "glibc":
            breaking.append("glibc_changed")
    if len(upgraded) + len(added) + len(removed) > 40:
        breaking.append("large_change_set")

    severity = "info"
    if breaking:
        severity = "high" if "kernel_package_changed" in breaking or "glibc_changed" in breaking else "warn"

    computed_at = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO arch_guard_diffs (
                from_snapshot_id, to_snapshot_id, computed_at,
                packages_added_json, packages_removed_json, packages_upgraded_json,
                configs_changed_json, breaking_flags_json, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(from_snapshot_id, to_snapshot_id) DO UPDATE SET
                computed_at = excluded.computed_at,
                packages_added_json = excluded.packages_added_json,
                packages_removed_json = excluded.packages_removed_json,
                packages_upgraded_json = excluded.packages_upgraded_json,
                configs_changed_json = excluded.configs_changed_json,
                breaking_flags_json = excluded.breaking_flags_json,
                severity = excluded.severity
            """,
            (
                from_snapshot_id,
                to_snapshot_id,
                computed_at,
                json.dumps(added, ensure_ascii=False),
                json.dumps(removed, ensure_ascii=False),
                json.dumps(upgraded, ensure_ascii=False),
                json.dumps(configs_changed, ensure_ascii=False),
                json.dumps(breaking, ensure_ascii=False),
                severity,
            ),
        )
        conn.commit()

    return {
        "from_snapshot_id": from_snapshot_id,
        "to_snapshot_id": to_snapshot_id,
        "computed_at": computed_at,
        "packages_added": added,
        "packages_removed": removed,
        "packages_upgraded": upgraded,
        "configs_changed": configs_changed,
        "breaking_flags": breaking,
        "severity": severity,
    }


def get_cached_diff(from_snapshot_id: str, to_snapshot_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT * FROM arch_guard_diffs WHERE from_snapshot_id = ? AND to_snapshot_id = ?",
            (from_snapshot_id, to_snapshot_id),
        ).fetchone()
    if not r:
        return None
    return {
        "from_snapshot_id": r["from_snapshot_id"],
        "to_snapshot_id": r["to_snapshot_id"],
        "computed_at": r["computed_at"],
        "packages_added": json.loads(r["packages_added_json"] or "[]"),
        "packages_removed": json.loads(r["packages_removed_json"] or "[]"),
        "packages_upgraded": json.loads(r["packages_upgraded_json"] or "[]"),
        "configs_changed": json.loads(r["configs_changed_json"] or "[]"),
        "breaking_flags": json.loads(r["breaking_flags_json"] or "[]"),
        "severity": r["severity"],
    }


def run_check(
    *,
    mode: str,
    include_reverse_deps: bool = False,
    create_snapshot: bool = False,
    snapshot_label: str | None = None,
) -> dict[str, Any]:
    pending = get_pending_updates(include_reverse_deps=include_reverse_deps)
    linked: str | None = None
    snap_info: dict[str, Any] | None = None
    if create_snapshot:
        snap_info = snap.create_snapshot(
            kind="scheduled" if mode == "auto" else "manual",
            label=snapshot_label,
            trigger="auto_scheduler" if mode == "auto" else "check",
            include_reverse_deps=include_reverse_deps,
            backup_sqlite=True,
        )
        linked = snap_info["snapshot_id"]
    check_id = snap.record_update_check(mode=mode, pending=pending, linked_snapshot_id=linked)
    return {
        "check_id": check_id,
        "pending": pending,
        "snapshot": snap_info,
    }
