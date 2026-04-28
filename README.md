<p align="center">
  <img src="https://img.shields.io/badge/status-early%20development-yellow?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/platform-Linux-FCC624?style=flat-square&logo=linux&logoColor=black" alt="Platform">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-local%20LLM-FF6F00?style=flat-square" alt="Ollama">
  <img src="https://img.shields.io/badge/philosophy-user%20space%20first-4CAF50?style=flat-square" alt="Philosophy">
</p>

# deltai

**A modular, configurable AI extension system for Linux — automate tasks and workflows, optimize system performance, and integrate AI into any application, all while respecting your user space and your choices.**

deltai is an open, user-controlled AI layer for Linux. Think of it as the answer to Copilot+Windows, but built around Linux philosophies: it runs as a systemd user service, exposes a clean HTTP/WebSocket API, never touches anything you didn't ask it to, and is heavily configurable. Local LLM inference via Ollama, RAG memory, structured reasoning, a plugin/tool system, and a systemd-native service architecture.

> **Status:** Early development. The codebase is actively being generalized for Linux-first deployment, modularity, and user choice. Contributions and feedback welcome.

---

## Philosophy

deltai is designed around Linux values:

- **User-space first** — deltai never requires root. It runs as a user service and only accesses what you grant it.
- **User choice** — bring your own models, swap out any component, opt into every feature.
- **Modularity** — plugins/extensions are first-class. The core stays small; capability is added through the plugin API.
- **Local extensions** — add optional daemon features as packages under [`project/extensions/`](project/extensions/README.md) (auto-discovered at startup; personal folders are git-ignored by default).
- **Transparency** — every decision (routing, tool call, RAG retrieval) is logged and observable.
- **No lock-in** — local models only by default. Cloud is optional, gated, and budget-controlled.
- **Config-file driven** — configure via `.env` / TOML / environment variables. No mandatory GUI.

---

## What deltai does

| Capability | Description |
|-----------|-------------|
| **Task automation** | Execute shell commands, scripts, and system operations via natural language |
| **Workflow orchestration** | Chain multi-step tasks with tool calls and structured reasoning (ReAct loop) |
| **System performance** | Monitor resources, identify bottlenecks, suggest and apply optimizations |
| **RAG memory** | Ingest your documents, notes, man pages, or any context — query it naturally |
| **Plugin API** | Any external service can push context in (`POST /ingest`) or register tools |
| **Local-first AI** | VRAM-aware model routing — uses the best model your hardware can run |

deltai is **not** a GUI assistant, not a desktop environment replacement, and not a telemetry or game integration. It is a backend intelligence service that other tools and scripts can talk to.

---

## Architecture

The full-stack dev server is `project/main.py` (routes and middleware) plus `project/deltai_api/core.py` (background work and chat implementation). The systemd-oriented **`delta-daemon`** is a separate, smaller surface ([docs/capability-matrix.md](docs/capability-matrix.md)).

```
┌──────────────────────────────────────────────────────────────┐
│  Clients (any of the following)                              │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ CLI      │ │ Browser  │ │ deltai     │ │ External     │  │
│  │ (curl /  │ │ dashboard│ │ status     │ │ scripts/apps │  │
│  │  delta)  │ │ :8000    │ │ (terminal) │ │ via HTTP     │  │
│  └──────────┘ └──────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────┬────────────────────────────────────┘
                          │ NDJSON streaming / REST
┌─────────────────────────┼────────────────────────────────────┐
│  deltai daemon          │  project/main.py  :8000            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ Router   │ │ RAG      │ │ Tools    │ │ Training        │ │
│  │ VRAM +   │ │ ChromaDB │ │ core +   │ │ QLoRA + GGUF    │ │
│  │ Quant    │ │ + Cold   │ │ plugins  │ │ + Adapters      │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ Persist  │ │ Session  │ │ ReAct    │ │ Resource        │ │
│  │ SQLite   │ │ History  │ │ Reasoning│ │ Self-Manager    │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────────┘ │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────┴────────────────────────────────────┐
│  Ollama (dynamic quantization + partial GPU offloading)      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Large model │  │  Small model │  │  Emergency chain │   │
│  │  Tier A/AB   │  │  Tier B/C    │  │  Last resort     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Plugin / ingest flow

```
External script ──POST /ingest──► Async Queue → Batch Embed → ChromaDB (RAG)
Any app/service   POST /ingest/batch
                  tags: ["alert"] ──► WebSocket → Dashboard / CLI notification
```

deltai does **not** know or care what pushed the context. Any service, script, or cron job can push text into its memory. RAG retrieves it when relevant.

### Security (short)

The daemon is built for **local trust**: bind to **127.0.0.1** in normal use. There is no default login on the API; the `run_shell` tool executes as your user (blocklist is not a security boundary). For shared networks or tunnels, use a reverse proxy and/or set **`DELTAI_INGEST_API_KEY`** and **`DELTAI_CORS_ORIGINS`** in `project/.env` (see [CLAUDE.md](CLAUDE.md) Security posture).

---

## Quick Start

### Prerequisites

- Linux (any modern distro)
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Python 3.11+
- SQLite with JSON1 enabled in Python's `sqlite3` build (required for reasoning trace JSON queries)
You can verify JSON1 quickly with:

```bash
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute(\"select json('{}')\"); print('sqlite json1: ok')"
```

### 1. Install

```bash
git clone https://github.com/Reconnecting18/deltai
cd deltai

# Backend
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

The core package metadata now includes ChromaDB, so a standard `pip install .`
is enough for runtime installs. Use `pip install -e .[dev]` when developing.

### 2. Pull a model

```bash
# A capable mid-size model (adjust to your GPU VRAM)
ollama pull qwen2.5:14b-instruct-q4_K_M   # ~9GB VRAM
ollama pull qwen2.5:3b-instruct-q4_K_M    # ~3GB VRAM (fallback)
```

### 3. Configure

```bash
cp project/.env.example project/.env
# Edit project/.env — set OLLAMA_URL, model names, and any features you want
```

See [Configuration](#configuration) below for all options.

### 4. Run

```bash
# Option A — run directly
cd project && source venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Option B — install the DELTA user daemon (optional; separate from uvicorn chat app)
cp systemd/user/delta-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now delta-daemon
```

**Two runtimes:** Option A is the **full project stack** in [`project/main.py`](project/main.py) — browser dashboard, `POST /chat`, ingest, and training on **TCP** `:8000`. Option B runs **`delta-daemon`** ([`src/delta/daemon/app.py`](src/delta/daemon/app.py)) on a **Unix socket** (`DELTA_DAEMON_SOCKET`), with `GET /health` and `POST /v1/execute`; it is not the same process or port as Option A. Use the `deltai` / `delta` CLI against Option B; use `curl` to `:8000` only for Option A.

### 5. Use it

```bash
# Optional: installed package CLI (talks to delta-daemon on the Unix socket — use Option B first)
deltai reference
deltai              # same as: deltai status — fastfetch-style health (daemon, Ollama, IPC, optional :8000)
deltai health       # JSON from GET /health only

# Project dev server (Option A) — browser dashboard
xdg-open http://localhost:8000

# Or curl the project app on :8000
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What processes are using the most memory?"}' \
  --no-buffer
```

---

## Configuration

deltai is configured via `project/.env`. All options have defaults and are optional unless marked required.

```env
# ── Models ──────────────────────────────────────────────────────────────────
OLLAMA_URL=http://localhost:11434
DELTAI_MODEL=qwen2.5:14b-instruct-q4_K_M   # Primary (large) model
DELTAI_SMALL_MODEL=qwen2.5:3b-instruct-q4_K_M  # Fallback (small) model

# ── Cloud (optional — completely dormant until key is set) ──────────────────
ANTHROPIC_API_KEY=
CLOUD_BUDGET_DAILY=5.00          # Hard daily spend cap in USD

# ── Intelligence ────────────────────────────────────────────────────────────
REACT_ENABLED=true               # Structured reasoning loop (Think/Act/Observe)
REACT_MAX_ITERATIONS=3
RAR_ENABLED=true                 # Two-round iterative RAG retrieval
SMART_HISTORY_ENABLED=true       # Intelligent context window compression
KNOWLEDGE_GAP_DETECTION=true     # Log queries deltai couldn't answer

# ── Resource management ──────────────────────────────────────────────────────
VRAM_TIER_AB_MIN_MB=5000         # Threshold for partial GPU offload
WARM_TO_COLD_AGE_SEC=86400       # Age before moving to cold storage (24h)
INGEST_QUEUE_MAX=500             # Max queued ingest items before 429

# ── Linux runtime paths (XDG-compliant defaults) ────────────────────────────
DELTA_DATA_DIR=~/.local/share/deltai
DELTA_CONFIG_DIR=~/.config/deltai
DELTA_CACHE_DIR=~/.cache/deltai
DELTA_SQLITE_PATH=~/.local/share/deltai/delta.db

# If you still have a legacy SQLite file under ~/.local/share/deltai/sqlite/e3n.db,
# either set DELTA_SQLITE_PATH to that path or: mv .../e3n.db ~/.local/share/deltai/delta.db
# (Legacy env SQLITE_PATH is still read if DELTA_SQLITE_PATH is unset.)

# ── Training (optional) ──────────────────────────────────────────────────────
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
DAILY_TRAIN_ENABLED=false        # Autonomous nightly fine-tuning cycle
SMART_CAPTURE_ENABLED=true       # Auto-save high-quality exchanges as training data
```

---

## Key Features

### VRAM-Aware Model Routing

The router detects available VRAM and selects the best model, quantization, and GPU layer count automatically:

| Tier | VRAM Free | Strategy |
|------|-----------|----------|
| A | > 9GB | Large model, full GPU, best quality |
| AB | 5–9GB | Large model, partial GPU offload (N layers) |
| B | 3–5GB | Small model on GPU |
| C | < 3GB | Small model, reduced quant or CPU |

Dynamic quantization (Q6 → Q4 → Q3 → Q2) and dynamic context window sizing prevent hard failures under memory pressure.

### ReAct Structured Reasoning

For complex, multi-step queries, deltai uses a structured **Think → Act → Observe → Confidence** loop:

1. **THINK** — identify what is needed
2. **ACT** — call tools, query RAG, run calculations
3. **OBSERVE** — analyse results, check for gaps
4. **CONFIDENCE** — HIGH → answer; MEDIUM → answer with caveats; LOW → ask for clarification

Past reasoning chains are stored in SQLite and retrieved by embedding similarity, so deltai learns from previous problem-solving.

### RAG Memory

ChromaDB vector store with three-tier hierarchical storage:

- **Hot** — in-memory, recent queries
- **Warm** — ChromaDB persistent, last 24 hours
- **Cold** — SQLite + zlib compression, older data

Features: multi-query expansion, source-grouped reranking, semantic deduplication, two-round iterative retrieval, watchdog auto-ingest from `data/knowledge/`.

### Plugin / Ingest API

Any script, application, or service can push context into deltai's memory:

```bash
# Push any text context into RAG
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "High CPU usage detected on core 3 at 14:32", "source": "monitor", "tags": ["alert"]}'

# Batch ingest
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"text": "...", "source": "..."}, ...]}'
```

Items tagged `"alert"` are forwarded to connected WebSocket clients as real-time notifications.

### Tool System

deltai has a structured tool system with domain-aware filtering (the catalog can grow with extensions; each query gets a small relevant subset, typically on the order of 5–8 tools):

| Category | Tools |
|----------|-------|
| Core | `read_file`, `write_file`, `list_directory`, `run_shell`, `get_system_info`, `search_knowledge`, `memory_stats` |
| Computation | `calculate`, `summarize_data`, `lookup_reference` |
| Diagnostic | `self_diagnostics`, `manage_models`, `repair_subsystem`, `resource_status` |
| Adapters | `manage_adapters` |

External tools/plugins can be registered via the plugin API (planned).

### Resource Self-Manager

Background loop (30s interval):

- VRAM lifecycle management (predictive trend analysis, thermal monitoring)
- Ollama health monitoring and auto-restart
- Warm→cold memory compaction
- Circuit breaker on Ollama inference (3 failures → exponential backoff → recovery)
- OS process priority adjustment under load

### Training Pipeline (optional)

Fine-tune models on your own data without leaving your machine:

- **Smart capture** — high-quality exchanges auto-saved as training data
- **QLoRA** — 4-bit training on small models (~3–7GB VRAM)
- **Domain adapters** — modular LoRA adapters per domain, TIES-mergeable
- **Knowledge distillation** — large model generates examples for small model
- **Daily autonomous cycle** — nightly training, weakness analysis, data collection (optional, disabled by default)

---

## Project Structure

```
deltai/
├── AGENTS.md                     Agent/Cursor onboarding (boundaries, verify commands)
├── CLAUDE.md                     Full architecture context for AI coding sessions
├── CONTRIBUTING.md               Issues and PR guidelines
├── .cursor/rules/                Cursor project rules (.mdc)
├── .github/                      Issue templates, workflows
├── .vscode/                      launch.json + tasks.json
├── project/                      deltai daemon (FastAPI)
│   ├── .env.example              Template copied to .env for local configuration
│   ├── main.py                   Chat, ingest pipeline, ReAct loop, resource manager
│   ├── router.py                 VRAM detection, dynamic quant, partial offload
│   ├── memory.py                 Hierarchical RAG (hot/warm/cold), iterative retrieval
│   ├── quality.py                Response quality scoring (6-signal heuristic)
│   ├── persistence.py            SQLite — history, budget, traces, quality, gaps
│   ├── anthropic_client.py       Cloud inference (dormant until API key set)
│   ├── training.py               Shim → extensions/training/pipeline.py
│   ├── collector.py              Web training data collection
│   ├── watcher.py                Watchdog for knowledge/ dir
│   ├── extensions/
│   │   └── training/             QLoRA, datasets, daily cycle, collection CLIs
│   ├── tools/
│   │   ├── definitions.py        Tool JSON schemas + filter_tools()
│   │   └── executor.py           Tool execution with retry + safety checks
│   ├── static/
│   │   └── index.html            Dashboard UI (single file — HTML + CSS + JS)
│   └── tests/
│       ├── verify_full.py        Core system tests
│       ├── verify_stress.py      Stress simulation tests
│       ├── verify_resource_mgmt.py  Resource management tests
│       └── verify_distill.py     Distillation pipeline tests
├── systemd/
│   └── user/
│       └── delta-daemon.service  systemd user unit (delta-daemon)
├── modelfiles/                   Ollama modelfiles
├── data/                         Runtime data (gitignored)
│   ├── chromadb/                 Vector store
│   ├── cold_memory.db            Cold tier archive
│   ├── sqlite/                   Persistent state
│   ├── knowledge/                Drop files here for RAG ingest
│   └── training/                 Datasets, adapters, GGUF exports
├── scripts/
│   ├── backup_s3.py              S3 backup (full/incremental/restore)
│   ├── daily_training.py         Wrapper → project/extensions/training/daily_training.py
│   └── collect_training_data.py  Wrapper → project/extensions/training/collect_training_data.py
└── docs/
    └── local-model-workflow.md   Operator guide: RAG, models, adapters
```

---

## API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Streaming chat (NDJSON) — auto-routes, ReAct for complex queries |
| GET | `/chat/history` | Conversation history |
| DELETE | `/chat/history` | Clear history |

### Ingest (plugin API)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Push context into RAG (async, non-blocking) |
| POST | `/ingest/batch` | Batch ingest (up to 100 items) |
| GET | `/ingest/pipeline/status` | Queue depth, throughput, latency |

### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memory/compact` | Trigger warm→cold compaction |
| GET | `/memory/cold/stats` | Cold tier statistics |
| GET | `/knowledge/gaps` | Unanswered queries |
| POST | `/knowledge/gaps/{id}/resolve` | Mark gap resolved |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Subsystem health check |
| GET | `/stats` | Model info, VRAM, budget |
| GET | `/resources/status` | VRAM prediction, GPU temp, circuit breaker |
| WS | `/ws/alerts` | Real-time alert notifications |
| WS | `/ws/health` | Real-time health events |

### Training (optional)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/training/datasets` | List datasets |
| POST | `/training/start` | Start fine-tuning |
| GET | `/training/status` | Progress, loss |
| GET | `/training/weaknesses` | Weak domains from quality feedback |

### MCP (Model Context Protocol)

Use the same tool catalog as chat from MCP-compatible clients (e.g. Cursor, Claude Desktop).

| Install | `pip install -e ".[mcp]"` from the repo root |
| Stdio | `deltai-mcp` (set `DELTAI_PROJECT_ROOT` to the repo if needed) or `cd project && python deltai_mcp_stdio.py` |
| HTTP | Set `DELTAI_MCP_HTTP_ENABLE=true` — MCP Streamable HTTP is mounted under `DELTAI_MCP_HTTP_PATH` (default `/mcp`). Optional `DELTAI_MCP_HTTP_KEY` + `Authorization: Bearer` or `X-Deltai-Mcp-Key`. |

Treat MCP like `/chat`: only expose on `127.0.0.1` unless you add your own auth and network controls.

## Stream Protocol

`POST /chat` returns NDJSON (one JSON object per line):

```
{"t":"route","backend":"local","model":"...","tier":1,"reason":"..."}
{"t":"rag","n":3}
{"t":"tool","n":"run_shell","a":{"cmd":"..."}}
{"t":"result","n":"run_shell","s":"..."}
{"t":"react","iteration":1,"phase":"think","c":"..."}
{"t":"text","c":"chunk of response text"}
{"t":"done","turns":5}
{"t":"error","c":"user-friendly error message"}
```

---

## systemd Integration

This repo ships a **user** unit for the `delta-daemon` CLI (`src/delta` FastAPI daemon), not the `project/` uvicorn app. Install and manage it like any user service:

```bash
# Install
cp systemd/user/delta-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload

# Enable on login
systemctl --user enable delta-daemon

# Start/stop/restart
systemctl --user start delta-daemon
systemctl --user stop delta-daemon
systemctl --user restart delta-daemon

# Logs
journalctl --user -u delta-daemon -f
```

No root required. The service starts after the user session begins.

---

## AI-assisted development

This repo is set up for **Cursor**, **Claude Code**, and any agent that reads markdown context:

| File | Role |
|------|------|
| [AGENTS.md](AGENTS.md) | Short onboarding for agents — boundaries, paths, verify commands |
| [CLAUDE.md](CLAUDE.md) | Full architecture, stream protocol, file map, development workflow |
| [`.cursor/rules/`](.cursor/rules/) | Cursor project rules (short `.mdc` files) |
| [`.vscode/launch.json`](.vscode/launch.json) | Debug/run FastAPI (uvicorn) from workspace |
| [docs/local-model-workflow.md](docs/local-model-workflow.md) | Operator guide for RAG, models, and adapters |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome — especially around the Linux integration layer, plugin system design, and systemd tooling.

---

## License

TBD.
