"""FastAPI daemon app for DELTA.

This app is the central process for interfaces (CLI, tray, overlay, API).
It exposes health and task execution endpoints and coordinates background
services such as Unix socket IPC and persistence.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from delta.config import Settings, load_settings
from delta.ipc.unix_socket import IPCServer
from delta.orchestrator.core import Orchestrator
from delta.platform.dbus_integration import DBusIntegration
from delta.storage.db import ensure_database


class ExecuteRequest(BaseModel):
    """User request sent by an interface to DELTA."""

    query: str = Field(..., min_length=1)
    source: str = Field(default="api")
    session_id: str | None = None


class ExecuteResponse(BaseModel):
    """Normalized orchestrator response."""

    status: str
    output: str
    agent: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down long-lived daemon components."""
    settings: Settings = load_settings()
    ensure_database(settings.sqlite_path)
    dbus = DBusIntegration()
    dbus_probe = dbus.probe()

    orchestrator = Orchestrator(settings=settings)
    ipc_server = IPCServer(socket_path=settings.ipc_socket_path, orchestrator=orchestrator)
    await ipc_server.start()

    app.state.settings = settings
    app.state.dbus = dbus
    app.state.dbus_probe = dbus_probe
    app.state.orchestrator = orchestrator
    app.state.ipc_server = ipc_server
    try:
        yield
    finally:
        await ipc_server.stop()


app = FastAPI(title="DELTA Daemon", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check for service managers and interfaces."""
    return {"status": "ok", "service": "delta-daemon"}


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Route incoming requests through the central orchestrator."""
    orchestrator: Orchestrator = app.state.orchestrator
    result = await orchestrator.handle_request(
        query=req.query,
        source=req.source,
        session_id=req.session_id,
    )
    return ExecuteResponse(**result)
