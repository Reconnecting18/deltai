"""Daemon launch entrypoint.

Runs the DELTA FastAPI app. On Linux, this should usually be started as a
systemd user service bound to a Unix domain socket.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn

from delta.config import load_settings



def main() -> None:
    """Start uvicorn bound to a Unix socket for local IPC-style HTTP access."""
    settings = load_settings()
    Path(settings.daemon_socket_path).parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        "delta.daemon.app:app",
        uds=settings.daemon_socket_path,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
