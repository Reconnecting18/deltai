"""
Arch update guard — core module (Arch news/wiki, pacman evidence, snapshots, diffs, rollback).

Routes are registered from setup(); TOOLS lists LLM tool schemas (lightweight import).
"""

from __future__ import annotations

import json
import logging

from .tools_defs import ARCH_GUARD_TOOLS

logger = logging.getLogger("deltai.arch_update_guard")

TOOLS = ARCH_GUARD_TOOLS


def setup(app) -> None:
    from tools.executor import register_handler

    from .api import mount_all
    from . import news_wiki
    from . import pacman_audit
    from . import snapshots as snap_mod
    from . import tracker
    from . import rollback

    def _arch_pending_updates_report_handler(include_reverse_deps: bool = False) -> str:
        data = pacman_audit.get_pending_updates(
            include_reverse_deps=include_reverse_deps,
            reverse_deps_limit=5,
        )
        return json.dumps(data, indent=2)

    def _arch_refresh_news_digest_handler(wiki_query: str = "", force: bool = False) -> str:
        result = news_wiki.refresh_news_digest_to_rag(wiki_query=wiki_query, force=force)
        return json.dumps(result, indent=2, default=str)

    def _arch_create_snapshot_handler(label: str = "", include_reverse_deps: bool = False) -> str:
        out = snap_mod.create_snapshot(
            kind="manual",
            label=label or None,
            trigger="tool",
            include_reverse_deps=include_reverse_deps,
            backup_sqlite=True,
        )
        return json.dumps(out, indent=2)

    def _arch_compare_snapshots_handler(from_snapshot_id: str, to_snapshot_id: str) -> str:
        d = tracker.compute_diff(from_snapshot_id, to_snapshot_id)
        return json.dumps(d, indent=2)

    def _arch_rollback_plan_handler(snapshot_id: str, dry_run: bool = True) -> str:
        if dry_run:
            r = rollback.plan_rollback(snapshot_id)
        else:
            r = rollback.execute_rollback(
                snapshot_id, dry_run=False, requested_by="tool", apply_etc=False
            )
        return json.dumps(r, indent=2, default=str)

    register_handler("arch_pending_updates_report", _arch_pending_updates_report_handler)
    register_handler("arch_refresh_news_digest", _arch_refresh_news_digest_handler)
    register_handler("arch_create_update_snapshot", _arch_create_snapshot_handler)
    register_handler("arch_compare_snapshots", _arch_compare_snapshots_handler)
    register_handler("arch_rollback_plan", _arch_rollback_plan_handler)

    mount_all(app)
    logger.info("arch_update_guard: core module initialised")


def shutdown() -> None:
    logger.info("arch_update_guard: shutdown")
