"""Tests for on-disk AI report writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta.storage import reports


def test_write_ai_report_ok_creates_file(tmp_path: Path) -> None:
    root = tmp_path / "ai_reports"
    path = reports.write_ai_report(
        reports_dir=root,
        enabled=True,
        category="orchestrator",
        status="ok",
        fields={"query": "hi", "output": "hello", "agent": "workflow"},
    )
    assert path is not None
    assert path.exists()
    assert path.parent.parent.name == "orchestrator"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == reports.REPORT_SCHEMA_VERSION
    assert data["source"] == "orchestrator"
    assert data["status"] == "ok"
    assert data["query"] == "hi"
    err_dir = root / "errors" / path.parent.name
    assert not (err_dir / path.name).exists()


def test_write_ai_report_error_also_writes_errors_dir(tmp_path: Path) -> None:
    root = tmp_path / "ai_reports"
    path = reports.write_ai_report(
        reports_dir=root,
        enabled=True,
        category="chat",
        status="error",
        fields={"query": "x", "output": "fail", "error": {"detail": "test"}},
    )
    assert path is not None
    day = path.parent.name
    err_copy = root / "errors" / day / path.name
    assert err_copy.exists()
    assert json.loads(err_copy.read_text(encoding="utf-8"))["status"] == "error"


def test_write_ai_report_disabled_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "ai_reports"
    assert (
        reports.write_ai_report(
            reports_dir=root,
            enabled=False,
            category="orchestrator",
            status="ok",
            fields={"query": "q"},
        )
        is None
    )
    assert not root.exists()


def test_resolve_reports_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = tmp_path / "datahome"
    monkeypatch.setenv("DELTA_DATA_DIR", str(data))
    monkeypatch.delenv("DELTA_REPORTS_DIR", raising=False)
    assert reports.resolve_reports_dir_from_env() == data / "ai_reports"


def test_resolve_reports_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_reports"
    monkeypatch.setenv("DELTA_REPORTS_DIR", str(custom))
    assert reports.resolve_reports_dir_from_env() == custom


def test_reports_enabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELTA_AI_REPORTS", raising=False)
    assert reports.reports_enabled_from_env() is True
    monkeypatch.setenv("DELTA_AI_REPORTS", "0")
    assert reports.reports_enabled_from_env() is False


def test_write_chat_turn_report_respects_disable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "ai_reports"
    monkeypatch.setenv("DELTA_REPORTS_DIR", str(root))
    monkeypatch.setenv("DELTA_AI_REPORTS", "false")
    reports.write_chat_turn_report(
        user_message="u",
        assistant_response="a",
        chat_metadata={"model": "m"},
        status="ok",
    )
    assert not root.exists()
