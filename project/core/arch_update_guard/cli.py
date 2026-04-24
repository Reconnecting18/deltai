"""CLI entrypoint: arch-update-guard (run from project/ or with PYTHONPATH=project)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    # Ensure project root is importable when installed as script without cwd
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)

    parser = argparse.ArgumentParser(prog="arch-update-guard", description="Arch update guard CLI")
    parser.add_argument("--mode", choices=("auto", "manual"), help="Persist scheduler mode in SQLite")
    parser.add_argument("--interval", type=float, help="Auto-check interval in seconds (min 60)")
    parser.add_argument("--check", action="store_true", help="Run pending update check")
    parser.add_argument("--snapshot", action="store_true", help="With --check, create a snapshot")
    parser.add_argument("--label", default="", help="Snapshot label")
    parser.add_argument("--list-snapshots", action="store_true", help="List recent snapshots")
    parser.add_argument("--show-snapshot", metavar="ID", help="Print snapshot metadata JSON")
    parser.add_argument("--diff", nargs=2, metavar=("FROM", "TO"), help="Compare two snapshot IDs")
    parser.add_argument("--rollback", metavar="ID", help="Rollback plan or execute")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Rollback dry run (default)")
    parser.add_argument("--no-dry-run", action="store_true", help="Execute rollback staging")
    parser.add_argument("--apply-etc", action="store_true", help="With rollback execute, write /etc (root only)")

    args = parser.parse_args()

    from persistence import init_db

    init_db()

    import sqlite3
    from persistence import get_sqlite_path

    from core.arch_update_guard.schema import init_arch_guard_tables
    from core.arch_update_guard import settings as ag_settings
    from core.arch_update_guard import tracker
    from core.arch_update_guard import snapshots as snap
    from core.arch_update_guard import rollback as rb

    path = get_sqlite_path()

    if args.mode is not None or args.interval is not None:
        conn = sqlite3.connect(path, timeout=15)
        init_arch_guard_tables(conn)
        if args.mode is not None:
            ag_settings.save_mode(conn, args.mode)
        if args.interval is not None:
            ag_settings.save_interval_sec(conn, args.interval)
        conn.commit()
        conn.close()
        print(json.dumps({"saved": True, "mode": args.mode, "interval": args.interval}, indent=2))

    if args.check:
        out = tracker.run_check(
            mode="manual",
            include_reverse_deps=False,
            create_snapshot=args.snapshot,
            snapshot_label=args.label or None,
        )
        conn = sqlite3.connect(path, timeout=15)
        init_arch_guard_tables(conn)
        ag_settings.save_last_check_at(conn, time.time())
        conn.commit()
        conn.close()
        print(json.dumps(out, indent=2))

    if args.list_snapshots:
        print(json.dumps(snap.list_snapshots(limit=100), indent=2))

    if args.show_snapshot:
        meta = snap.get_snapshot_meta(args.show_snapshot)
        print(json.dumps(meta, indent=2, default=str))

    if args.diff:
        d = tracker.compute_diff(args.diff[0], args.diff[1])
        print(json.dumps(d, indent=2))

    if args.rollback:
        dry = not args.no_dry_run
        if dry:
            print(json.dumps(rb.plan_rollback(args.rollback), indent=2))
        else:
            print(
                json.dumps(
                    rb.execute_rollback(
                        args.rollback,
                        dry_run=False,
                        requested_by="cli",
                        apply_etc=args.apply_etc,
                    ),
                    indent=2,
                    default=str,
                )
            )

    if not any(
        [
            args.mode,
            args.interval is not None,
            args.check,
            args.list_snapshots,
            args.show_snapshot,
            args.diff,
            args.rollback,
        ]
    ):
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
