"""Smoke tests for DELTA scaffolding."""

from delta.config import load_settings
from delta.daemon.app import app
from delta.orchestrator.intents import classify_intent


def test_fastapi_app_name() -> None:
    assert app.title == "DELTA Daemon"


def test_settings_defaults() -> None:
    settings = load_settings()
    assert settings.ollama_fast_model
    assert str(settings.data_dir).endswith(".local/share/deltai")
    assert str(settings.config_dir).endswith(".config/deltai")
    assert str(settings.cache_dir).endswith(".cache/deltai")
    assert settings.sqlite_path.name == "delta.db"
    assert settings.daemon_socket_path.endswith("daemon.sock")
    assert settings.ipc_socket_path.endswith("ipc.sock")


def test_intent_classification_shell() -> None:
    assert classify_intent("run this shell command") == "shell"
