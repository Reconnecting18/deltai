"""FastAPI routes for arch update guard."""

from __future__ import annotations

import logging
import sqlite3
import time

from fastapi import APIRouter, Request
from persistence import get_sqlite_path
from pydantic import BaseModel

from . import rollback, tracker
from . import settings as ag_settings
from . import snapshots as snap
from .news_wiki import refresh_news_digest_to_rag_async
from .pacman_audit import get_pending_updates
from .schema import init_arch_guard_tables

logger = logging.getLogger("deltai.arch_update_guard.api")

router = APIRouter(prefix="/arch-guard", tags=["arch_update_guard"])
# Backward-compatible alias
legacy_router = APIRouter(prefix="/ext/arch_update_guard", tags=["arch_update_guard"])


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(get_sqlite_path(), timeout=15)
    init_arch_guard_tables(c)
    return c


class RefreshNewsBody(BaseModel):
    wiki_query: str = ""
    force: bool = False


class CheckBody(BaseModel):
    create_snapshot: bool = False
    label: str = ""
    include_reverse_deps: bool = False


class SnapshotCreateBody(BaseModel):
    kind: str = "manual"
    label: str = ""
    include_reverse_deps: bool = False
    backup_sqlite: bool = True


class SettingsBody(BaseModel):
    mode: str | None = None
    enabled: bool | None = None
    auto_interval_sec: float | None = None


class CompareBody(BaseModel):
    from_snapshot_id: str
    to_snapshot_id: str


class RollbackBody(BaseModel):
    snapshot_id: str
    dry_run: bool = True
    apply_etc: bool = False


def _mount_routes(r: APIRouter) -> None:
    @r.get("/health")
    def health():
        return {"status": "ok", "module": "extensions.arch_update_guard"}

    @r.get("/settings")
    def get_settings():
        with _conn() as conn:
            return ag_settings.public_settings(conn)

    @r.put("/settings")
    def put_settings(body: SettingsBody):
        with _conn() as conn:
            if body.mode is not None:
                ag_settings.save_mode(conn, body.mode)
            if body.enabled is not None:
                ag_settings.set_setting(conn, "enabled", "true" if body.enabled else "false")
            if body.auto_interval_sec is not None:
                ag_settings.save_interval_sec(conn, body.auto_interval_sec)
            conn.commit()
            return ag_settings.public_settings(conn)

    @r.get("/pending")
    def pending(
        include_reverse_deps: bool = False,
        reverse_deps_limit: int = 5,
    ):
        return get_pending_updates(
            include_reverse_deps=include_reverse_deps,
            reverse_deps_limit=reverse_deps_limit,
        )

    @r.post("/check")
    def check(body: CheckBody):
        out = tracker.run_check(
            mode="manual",
            include_reverse_deps=body.include_reverse_deps,
            create_snapshot=body.create_snapshot,
            snapshot_label=body.label or None,
        )
        with _conn() as conn:
            ag_settings.save_last_check_at(conn, time.time())
            conn.commit()
        return out

    @r.post("/snapshots")
    def create_snapshot_route(body: SnapshotCreateBody):
        kind = body.kind if body.kind in (
            "pre_update",
            "post_update",
            "manual",
            "scheduled",
            "cli",
        ) else "manual"
        return snap.create_snapshot(
            kind=kind,
            label=body.label or None,
            trigger="api",
            include_reverse_deps=body.include_reverse_deps,
            backup_sqlite=body.backup_sqlite,
        )

    @r.get("/snapshots")
    def list_snapshots_route(limit: int = 50, offset: int = 0):
        return {"snapshots": snap.list_snapshots(limit=limit, offset=offset)}

    @r.get("/snapshots/{snapshot_id}")
    def get_snapshot_route(snapshot_id: str):
        from fastapi.responses import JSONResponse

        meta = snap.get_snapshot_meta(snapshot_id)
        if not meta:
            return JSONResponse({"detail": "snapshot not found"}, status_code=404)
        return meta

    @r.post("/snapshots/compare")
    def compare_snapshots(body: CompareBody):
        return tracker.compute_diff(body.from_snapshot_id, body.to_snapshot_id)

    @r.get("/diffs/{from_id}/{to_id}")
    def get_diff_cached(from_id: str, to_id: str):
        d = tracker.get_cached_diff(from_id, to_id)
        if not d:
            return {"cached": False, "diff": tracker.compute_diff(from_id, to_id)}
        return {"cached": True, "diff": d}

    @r.post("/rollback")
    def rollback_route(body: RollbackBody):
        if body.dry_run:
            return rollback.plan_rollback(body.snapshot_id)
        return rollback.execute_rollback(
            body.snapshot_id,
            dry_run=False,
            requested_by="api",
            apply_etc=body.apply_etc,
        )

    @r.get("/rollback/jobs")
    def rollback_jobs(limit: int = 20):
        limit = max(1, min(limit, 100))
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, requested_at, requested_by, target_snapshot_id, status, error "
                "FROM arch_guard_rollback_jobs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"jobs": [dict(r) for r in rows]}

    @r.post("/refresh-news")
    async def refresh_news(request: Request):
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}
        body = RefreshNewsBody.model_validate(raw)
        return await refresh_news_digest_to_rag_async(
            wiki_query=body.wiki_query,
            force=body.force,
        )


def mount_all(app) -> None:
    _mount_routes(router)
    _mount_routes(legacy_router)
    app.include_router(router)
    app.include_router(legacy_router)
    logger.info("arch_update_guard: routes at /arch-guard/ and /ext/arch_update_guard/")
