"""Tests for project/extensions/arch_update_guard (non-Arch CI: mocks and imports)."""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_health_route_via_setup():
    from extensions.arch_update_guard import setup

    app = FastAPI()
    setup(app)
    client = TestClient(app)
    r = client.get("/ext/arch_update_guard/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "extension": "arch_update_guard"}


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


def test_refresh_news_skipped_when_rate_limited():
    from extensions.arch_update_guard import news_wiki

    with mock.patch.object(news_wiki, "_last_refresh_ts", news_wiki.time.monotonic()):
        r = news_wiki.refresh_news_digest_to_rag(force=False)
    assert r["status"] == "skipped"


def test_tool_handlers_json_roundtrip():
    import extensions.arch_update_guard as ag

    with mock.patch.object(
        ag,
        "get_pending_updates",
        return_value={"pending": [], "pending_count": 0, "method": "test", "errors": []},
    ):
        s = ag._arch_pending_updates_report_handler(include_reverse_deps=False)
    data = json.loads(s)
    assert data["method"] == "test"

    with mock.patch.object(
        ag,
        "refresh_news_digest_to_rag",
        return_value={"status": "ok", "items_fetched": 0, "ingest": {}},
    ):
        s2 = ag._arch_refresh_news_digest_handler(wiki_query="", force=True)
    assert json.loads(s2)["status"] == "ok"
