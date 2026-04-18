"""Runtime configuration for DELTA.

This module centralizes environment-driven settings for daemon behavior,
IPC sockets, model selection, and storage paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIRNAME = "deltai"


def _xdg_data_home() -> Path:
    """Return XDG data home with Linux default fallback."""
    return Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))


def _xdg_config_home() -> Path:
    """Return XDG config home with Linux default fallback."""
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))


def _xdg_cache_home() -> Path:
    """Return XDG cache home with Linux default fallback."""
    return Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))


def _runtime_dir() -> Path:
    """Return a Linux-friendly runtime directory for Unix sockets."""
    xdg_runtime = os.getenv("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        return Path(xdg_runtime)
    return Path("/tmp")


def _env_flag_enabled(var_name: str, default: bool = True) -> bool:
    """Parse DELTA_AI_REPORTS-style env: unset defaults to True; 0/false/no/off disables."""
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class Settings:
    """Strongly-typed DELTA runtime settings."""

    daemon_host: str
    daemon_port: int
    daemon_socket_path: str
    ipc_socket_path: str
    data_dir: Path
    config_dir: Path
    cache_dir: Path
    sqlite_path: Path
    reports_dir: Path
    ai_reports_enabled: bool
    ollama_url: str
    ollama_dev_model: str
    ollama_fast_model: str
    anthropic_api_key: str
    anthropic_model_primary: str


def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    data_dir = Path(os.getenv("DELTA_DATA_DIR", str(_xdg_data_home() / APP_DIRNAME)))
    config_dir = Path(os.getenv("DELTA_CONFIG_DIR", str(_xdg_config_home() / APP_DIRNAME)))
    cache_dir = Path(os.getenv("DELTA_CACHE_DIR", str(_xdg_cache_home() / APP_DIRNAME)))
    sqlite_path = Path(os.getenv("DELTA_SQLITE_PATH", str(data_dir / "delta.db")))
    reports_dir = Path(
        os.getenv("DELTA_REPORTS_DIR", str(data_dir / "ai_reports"))
    ).expanduser()
    ai_reports_enabled = _env_flag_enabled("DELTA_AI_REPORTS", default=True)
    runtime_dir = _runtime_dir() / APP_DIRNAME

    return Settings(
        daemon_host=os.getenv("DELTA_DAEMON_HOST", "127.0.0.1"),
        daemon_port=int(os.getenv("DELTA_DAEMON_PORT", "8787")),
        daemon_socket_path=os.getenv("DELTA_DAEMON_SOCKET", str(runtime_dir / "daemon.sock")),
        ipc_socket_path=os.getenv("DELTA_IPC_SOCKET", str(runtime_dir / "ipc.sock")),
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
        sqlite_path=sqlite_path,
        reports_dir=reports_dir,
        ai_reports_enabled=ai_reports_enabled,
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_dev_model=os.getenv("DELTA_MODEL_DEV", "qwen2.5-coder:32b"),
        ollama_fast_model=os.getenv("DELTA_MODEL_FAST", "qwen2.5:7b"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model_primary=os.getenv("DELTA_CLOUD_MODEL", "claude-sonnet-4-20250514"),
    )
