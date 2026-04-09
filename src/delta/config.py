"""Runtime configuration for DELTA.

This module centralizes environment-driven settings for daemon behavior,
IPC sockets, model selection, and storage paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Strongly-typed DELTA runtime settings."""

    daemon_host: str
    daemon_port: int
    daemon_socket_path: str
    ipc_socket_path: str
    data_dir: Path
    sqlite_path: Path
    ollama_url: str
    ollama_dev_model: str
    ollama_fast_model: str
    anthropic_api_key: str
    anthropic_model_primary: str



def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    data_dir = Path(os.getenv("DELTA_DATA_DIR", str(Path.home() / ".local/share/delta")))
    sqlite_path = Path(os.getenv("DELTA_SQLITE_PATH", str(data_dir / "delta.db")))

    return Settings(
        daemon_host=os.getenv("DELTA_DAEMON_HOST", "127.0.0.1"),
        daemon_port=int(os.getenv("DELTA_DAEMON_PORT", "8787")),
        daemon_socket_path=os.getenv("DELTA_DAEMON_SOCKET", "/tmp/delta-daemon.sock"),
        ipc_socket_path=os.getenv("DELTA_IPC_SOCKET", "/tmp/delta-ipc.sock"),
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_dev_model=os.getenv("DELTA_MODEL_DEV", "qwen2.5-coder:32b"),
        ollama_fast_model=os.getenv("DELTA_MODEL_FAST", "qwen2.5:7b"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model_primary=os.getenv("DELTA_CLOUD_MODEL", "claude-sonnet-4-20250514"),
    )
