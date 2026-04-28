# deltai capability matrix (project app vs delta-daemon)

This table maps **features** to the two primary runtimes so forks (`main` vs `personal`) and operators do not confuse them.

| Feature | Project app (`project/main.py`, uvicorn TCP `:8000`) | Packaged daemon (`delta-daemon`, Unix socket) | Notes |
|--------|--------------------------------------------------------|-----------------------------------------------|--------|
| `POST /chat` (NDJSON stream) | Yes | No | Full chat, ReAct, tools, RAG injection, cloud/split paths. |
| `POST /v1/execute` (orchestrator JSON) | No | Yes | Intent → agent; not streaming chat. |
| `GET /health` (daemon liveness) | No | Yes | Project app uses `GET /api/health` for subsystem panel. |
| `POST /ingest`, RAG, Chroma | Yes | No | Ingest pipeline and memory live on the project app. |
| Training / adapters HTTP API | Yes | No | Under `project/` when training extension loads. |
| WebSocket alerts / health bus | Yes | No | `/ws/alerts`, `/ws/health`. |
| Extension routers (`/ext/...`) | Yes | No | `load_extensions(app)` on the FastAPI `app` instance. |
| Orchestrator agents (shell, file, dev, …) | No | Yes | `src/delta/orchestrator`, `src/delta/agents`. |
| Unix socket IPC | No | Yes | CLI / `delta.ipc` against `DELTA_DAEMON_SOCKET`. |
| Daemon plugin loader (`PluginManager`) | No | Yes | Under `$DELTA_DATA_DIR/plugins` + config. |
| AI reports (`ai_reports/`) | Both paths | Both paths | Orchestrator writes from daemon; chat reports when `delta` package importable from `project/`. |

**Implementation note:** Shared chat/ingest/RAG logic for the project app lives in `project/deltai_api/core.py`; `project/main.py` is the HTTP entrypoint (routing table, middleware, extension mount).
