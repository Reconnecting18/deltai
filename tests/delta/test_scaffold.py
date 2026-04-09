"""Smoke tests for DELTA scaffolding."""

from delta.config import load_settings
from delta.daemon.app import app
from delta.orchestrator.intents import classify_intent


def test_fastapi_app_name() -> None:
    assert app.title == "DELTA Daemon"


def test_settings_defaults() -> None:
    settings = load_settings()
    assert settings.ollama_fast_model
    assert settings.sqlite_path.name == "delta.db"


def test_intent_classification_shell() -> None:
    assert classify_intent("run this shell command") == "shell"
