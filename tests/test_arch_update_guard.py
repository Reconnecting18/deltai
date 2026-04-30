"""Tests for project/extensions/arch_update_guard (non-Arch CI: mocks and imports)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip(
    "extensions.arch_update_guard",
    reason=(
        "optional extension not on main; use personal or git add -f "
        "project/extensions/arch_update_guard/"
    ),
)


@pytest.fixture()
def isolated_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("DELTA_SQLITE_PATH", path)
    monkeypatch.setenv("DELTA_DATA_DIR", tempfile.mkdtemp())

    import persistence

    monkeypatch.setattr(persistence, "_db_path", path)
    import importlib

    importlib.reload(persistence)
    persistence.init_db()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def test_health_route_via_setup():
    from extensions.arch_update_guard import setup

    app = FastAPI()
    setup(app)
    client = TestClient(app)
    r = client.get("/arch-guard/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    r2 = client.get("/ext/arch_update_guard/health")
    assert r2.status_code == 200


def test_parse_checkupdates_line():
    from extensions.arch_update_guard.pacman_audit import _parse_checkupdates_line

    assert _parse_checkupdates_line("alsa-lib  1:1.2.14-1 -> 1:1.2.15-1") == {
        "package": "alsa-lib",
        "from_version": "1:1.2.14-1",
        "to_version": "1:1.2.15-1",
    }
    assert _parse_checkupdates_line("") is None
    assert _parse_checkupdates_line("# comment") is None


def test_run_allowlisted_rejects_unknown_argv():
    from extensions.arch_update_guard.pacman_audit import _run_allowlisted

    with pytest.raises(ValueError, match="not allowlisted"):
        _run_allowlisted(["pacman", "-Syu"])


@pytest.mark.skipif(sys.platform != "linux", reason="pacman paths optional on non-Linux")
def test_get_pending_updates_when_no_pacman_tools():
    from extensions.arch_update_guard.pacman_audit import get_pending_updates

    with mock.patch("extensions.arch_update_guard.pacman_audit.shutil.which", return_value=None):
        out = get_pending_updates()
    assert out["pending_count"] == 0
    assert out["method"] == "none"
    assert any("Neither checkupdates nor pacman" in e for e in out["errors"])


def test_get_pending_updates_mock_checkupdates():
    from extensions.arch_update_guard import pacman_audit

    fake_out = "foo  1.0-1 -> 1.1-1\n"

    def fake_which(name):
        return "/usr/bin/" + name if name == "checkupdates" else None

    with (
        mock.patch.object(pacman_audit.shutil, "which", side_effect=fake_which),
        mock.patch.object(
            pacman_audit,
            "_run_allowlisted",
            return_value=(0, fake_out, ""),
        ),
    ):
        out = pacman_audit.get_pending_updates(include_reverse_deps=False)

    assert out["method"] == "checkupdates"
    assert out["pending_count"] == 1
    assert out["pending"][0]["package"] == "foo"


@pytest.mark.skipif(sys.platform != "linux", reason="pacman paths optional on non-Linux")
def test_get_pending_updates_pacman_qu_exit_1_empty_is_no_upgrades():
    from extensions.arch_update_guard import pacman_audit

    def fake_which(name):
        if name == "checkupdates":
            return None
        if name == "pacman":
            return "/usr/bin/pacman"
        return None

    with (
        mock.patch.object(pacman_audit.shutil, "which", side_effect=fake_which),
        mock.patch.object(
            pacman_audit,
            "_run_allowlisted",
            return_value=(1, "", ""),
        ),
    ):
        out = pacman_audit.get_pending_updates(include_reverse_deps=False)

    assert out["method"] == "pacman_-Qu"
    assert out["pending_count"] == 0
    assert out["errors"] == []


def test_refresh_news_skipped_when_rate_limited():
    from extensions.arch_update_guard import news_wiki

    with mock.patch.object(news_wiki, "_last_refresh_ts", news_wiki.time.monotonic()):
        r = news_wiki.refresh_news_digest_to_rag(force=False)
    assert r["status"] == "skipped"


def test_tool_handlers_json_roundtrip():
    from extensions.arch_update_guard import setup
    from tools.executor import execute_tool

    app = FastAPI()
    setup(app)

    with mock.patch(
        "extensions.arch_update_guard.pacman_audit.get_pending_updates",
        return_value={"pending": [], "pending_count": 0, "method": "test", "errors": []},
    ):
        s = execute_tool("arch_pending_updates_report", {"include_reverse_deps": False})
    data = json.loads(s)
    assert data["method"] == "test"

    with mock.patch(
        "extensions.arch_update_guard.news_wiki.refresh_news_digest_to_rag",
        return_value={"status": "ok", "items_fetched": 0, "ingest": {}},
    ):
        s2 = execute_tool(
            "arch_refresh_news_digest", {"wiki_query": "", "force": True}
        )
    assert json.loads(s2)["status"] == "ok"


def test_create_snapshot_and_diff(isolated_db, monkeypatch):
    from extensions.arch_update_guard import snapshots as snap
    from extensions.arch_update_guard import tracker

    def payload1(**kw):
        return {
            "hostname": "h",
            "kernel_release": "k",
            "system_summary": {"pending_updates": {"pending": [], "pending_count": 0}},
            "deltai_summary": {},
            "flags": [],
            "blobs": {"pacman_Q": "a 1-1\n", "configs_json": "{}"},
            "errors": {"pacman_Q": None},
        }

    def payload2(**kw):
        return {
            "hostname": "h",
            "kernel_release": "k",
            "system_summary": {"pending_updates": {"pending": [], "pending_count": 0}},
            "deltai_summary": {},
            "flags": [],
            "blobs": {"pacman_Q": "a 1-1\nb 2-1\n", "configs_json": "{}"},
            "errors": {"pacman_Q": None},
        }

    monkeypatch.setattr("extensions.arch_update_guard.snapshots.build_snapshot_payload", payload1)

    s1 = snap.create_snapshot(kind="manual", label="t1", trigger="test", backup_sqlite=False)
    sid1 = s1["snapshot_id"]

    monkeypatch.setattr("extensions.arch_update_guard.snapshots.build_snapshot_payload", payload2)
    s2 = snap.create_snapshot(kind="manual", label="t2", trigger="test", backup_sqlite=False)
    sid2 = s2["snapshot_id"]

    d = tracker.compute_diff(sid1, sid2)
    assert any(p["package"] == "b" for p in d["packages_added"])


def test_rollback_plan_unknown_snapshot(isolated_db):
    from extensions.arch_update_guard import rollback

    r = rollback.plan_rollback("00000000-0000-0000-0000-000000000000")
    assert r.get("ok") is False


def test_arch_rollback_tool_blocks_apply_etc_without_auto_approve(isolated_db, monkeypatch):
    monkeypatch.delenv("DELTAI_TOOL_AUTO_APPROVE", raising=False)
    from extensions.arch_update_guard import setup
    from tools.executor import execute_tool

    app = FastAPI()
    setup(app)
    out = execute_tool(
        "arch_rollback_plan",
        {
            "snapshot_id": "00000000-0000-0000-0000-000000000000",
            "dry_run": False,
            "apply_etc": True,
        },
    )
    data = json.loads(out)
    assert data.get("ok") is False
    assert "DELTAI_TOOL_AUTO_APPROVE" in (data.get("error") or "")


def test_arch_rollback_tool_apply_etc_when_auto_approve(isolated_db, monkeypatch):
    monkeypatch.setenv("DELTAI_TOOL_AUTO_APPROVE", "1")
    from extensions.arch_update_guard import setup
    from tools.executor import execute_tool

    app = FastAPI()
    setup(app)
    with mock.patch(
        "extensions.arch_update_guard.rollback.execute_rollback",
        return_value={"ok": True, "dry_run": False, "status": "completed", "steps": []},
    ) as ex:
        execute_tool(
            "arch_rollback_plan",
            {
                "snapshot_id": "11111111-1111-1111-1111-111111111111",
                "dry_run": False,
                "apply_etc": True,
            },
        )
    ex.assert_called_once()
    assert ex.call_args.kwargs.get("apply_etc") is True
