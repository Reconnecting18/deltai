"""Smoke tests for deltai / delta package scaffolding."""

from pathlib import Path

from delta.config import load_settings
from delta.daemon.app import app
from delta.orchestrator.intents import classify_intent
from delta.platform.dbus_integration import DBusIntegration


def test_fastapi_app_name() -> None:
    assert app.title == "deltai daemon"


def test_settings_defaults() -> None:
    settings = load_settings()
    assert settings.ollama_fast_model
    assert Path(settings.data_dir).as_posix().endswith(".local/share/deltai")
    assert Path(settings.config_dir).as_posix().endswith(".config/deltai")
    assert Path(settings.cache_dir).as_posix().endswith(".cache/deltai")
    assert settings.sqlite_path.name == "delta.db"
    assert settings.daemon_socket_path.endswith("daemon.sock")
    assert settings.ipc_socket_path.endswith("ipc.sock")


def test_intent_classification_shell() -> None:
    assert classify_intent("run this shell command") == "shell"


def test_dbus_probe_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DELTA_DBUS_ENABLED", "false")
    probe = DBusIntegration().probe()
    assert probe.enabled is False
    assert probe.available is False
