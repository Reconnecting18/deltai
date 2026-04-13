# deltai — Project Context

This is the deep context file for AI coding assistants (Cursor, Claude Code, Copilot, etc.). Keep it up to date when you change architecture, endpoints, file structure, or development workflow.

For short agent onboarding (Cursor entry point), see [AGENTS.md](AGENTS.md).

---

## What This Is

**deltai** is a modular, configurable AI extension system for Linux. It is the open, user-controlled answer to Copilot+Windows — built around Linux philosophies:

- **User-space first** — runs as a systemd user service; never requires root
- **User choice** — every feature is opt-in; bring your own models, swap any component
- **Modularity** — plugins/extensions are first-class; core stays minimal
- **Transparency** — all routing decisions, tool calls, and RAG retrievals are logged
- **No lock-in** — local models by default; cloud is optional and budget-gated
- **Config-file driven** — `.env` / environment variables; no mandatory GUI

### What deltai IS

- A local AI intelligence layer (reasoning, memory, tool execution, model routing)
- A systemd user service with an HTTP/WebSocket API
- A plugin API (`POST /ingest`, tool registration) for external services to push context in
- A task automation engine (natural language → shell/tool execution)
- A system performance advisor (monitors resources, suggests and applies optimizations)
- A RAG knowledge store (ingest any documents/notes, query them naturally)
- An optional voice interface (STT/TTS loop)
- An optional fine-tuning pipeline for local model improvement

### What deltai is NOT

- A desktop environment or GUI shell
- A telemetry or game integration
- A system daemon requiring elevated privileges
- A cloud service or SaaS product
- Anything that runs as root or modifies system files without explicit user authorization

### Architecture boundary

External services, scripts, and cron jobs push context into deltai via `POST /ingest`. deltai stores it in ChromaDB with TTL and source tags. RAG retrieves it when relevant to a query. This makes deltai pluggable — any service can feed it context without deltai knowing or caring about the domain.

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `project/` | deltai daemon (FastAPI backend) |
| `app/` | Electron desktop shell (optional) |
| `modelfiles/` | Ollama modelfiles |
| `systemd/user/` | systemd user service unit |
| `scripts/` | Standalone scripts (backup, training, data collection) |
| `docs/` | Operator guides |
| `data/` | Runtime data — gitignored (chromadb, sqlite, knowledge, training) |

**Important:** `data/` is gitignored. Never commit `.env`, credentials, or anything under `data/`.

---

## Architecture

### Backend (project/main.py)

FastAPI application served on `localhost:8000`. All features are implemented here. Entry point for the systemd service.

Key subsystems:
- **Chat endpoint** (`POST /chat`) — NDJSON streaming, three paths: local, cloud, split
- **ReAct reasoning loop** — Think/Act/Observe for complex queries (max 3 iterations)
- **Conversation history** — rolling in-memory + SQLite persistence, session-aware
- **Ingest pipeline** — async queue → batch embed → ChromaDB (non-blocking, backpressure via 429)
- **Resource self-manager** — background loop (30s): VRAM lifecycle, thermal, process priority, circuit breaker, memory compaction
- **WebSocket alerts** (`/ws/alerts`) — forwards tagged ingest items to connected clients
- **Training endpoints** — QLoRA, adapters, distillation, dataset CRUD

### Router (project/router.py)

VRAM-aware routing. Key functions:
- `_get_vram_info()` — consolidated GPU detection via pynvml
- `_calc_num_gpu()` — dynamic GPU layer count for partial offload
- `classify_tier()` — A / AB / B / C based on free VRAM
- `select_quant()` — Q6K → Q4KM → Q3KM → Q2K based on VRAM
- `resolve_adapter_model()` — Mixture-of-LoRA domain routing
- Cloud budget enforcement + adaptive feedback from quality scores

| Tier | VRAM Free | Strategy |
|------|-----------|----------|
| A | > 9GB | Large model, full GPU |
| AB | 5–9GB | Large model, partial GPU offload |
| B | 3–5GB | Small model on GPU |
| C | < 3GB | Small model, reduced quant or CPU |

### Memory (project/memory.py)

ChromaDB RAG with three-tier hierarchical storage:
- **Hot** — in-memory ChromaDB, recent queries
- **Warm** — ChromaDB persistent, < 24h
- **Cold** — SQLite + zlib compression, > 24h

Features: multi-query expansion, source-grouped reranking, recency bias, semantic dedup (cosine < 0.15), iterative two-round retrieval, batch ingest, TTL expiry.

### Tools (project/tools/)

- `definitions.py` — JSON schemas for all tools; `filter_tools()` pre-filters 19 tools to 5–8 per query
- `executor.py` — type coercion, safety checks, retry on error

Tool categories:
- **Core** (7): `read_file`, `write_file`, `list_directory`, `run_shell`, `get_system_info`, `search_knowledge`, `memory_stats`
- **Computation** (3): `calculate`, `summarize_data`, `lookup_reference`
- **Diagnostic** (4): `self_diagnostics`, `manage_models`, `repair_subsystem`, `resource_status`
- **Adapter** (1): `manage_adapters`

### Quality (project/quality.py)

6-signal heuristic scorer (0.0–1.0): `length_appropriateness`, `tool_success_rate`, `specificity`, `no_error_indicators`, `structural_match`, `no_repeat`. Drives: smart capture, adaptive routing feedback, knowledge gap detection.

### Persistence (project/persistence.py)

SQLite (WAL mode, short-lived connections). Path: **`DELTA_SQLITE_PATH`** (same default as `systemd/user/delta-daemon.service`), else legacy **`SQLITE_PATH`**, else `~/.local/share/deltai/delta.db`. Tables: `conversation_history`, `cloud_budget`, `reasoning_traces`, `quality_scores`, `routing_feedback`, `knowledge_gaps`.
Python 3.11 typically ships with SQLite 3.39+ (WAL-capable). JSON1 support must also be enabled in the Python `sqlite3` build for reasoning-trace JSON queries.

### Training (project/training.py)

Optional. QLoRA fine-tuning (Qwen2.5-3B via PEFT/TRL), adapter management (TIES merge, versioning), knowledge distillation, iterative distillation, DPO, smart auto-capture, daily cycle orchestrator.

### Voice (project/voice/)

Optional. STT: faster-whisper. TTS: edge-tts / Piper. Full loop: `POST /voice/chat`.

### Collector (project/collector.py)

Web training data collection: Wikipedia (HF datasets streaming), arXiv XML API, Semantic Scholar, general web via trafilatura. SHA256 dedup via SQLite.

---

## Key Files

| File | Purpose |
|------|---------|
| `project/main.py` | FastAPI app — all chat paths, ReAct loop, ingest pipeline, resource self-manager, WebSocket, training endpoints |
| `project/router.py` | VRAM detection, tier classification, quant selection, partial offload, cloud budget, adaptive routing |
| `project/memory.py` | ChromaDB RAG, hierarchical storage, iterative retrieval, dedup, ingest |
| `project/quality.py` | Response quality scorer, drives capture + routing feedback + gap detection |
| `project/persistence.py` | SQLite backing store — history, budget, traces, quality, routing, gaps |
| `project/tools/definitions.py` | Tool schemas, `filter_tools()` |
| `project/tools/executor.py` | Tool execution with retry + safety |
| `project/training.py` | QLoRA, adapters, distillation, dataset CRUD, auto-capture, daily cycle |
| `project/collector.py` | Web data collection for training |
| `project/voice/` | STT/TTS package |
| `project/watcher.py` | Watchdog file watcher for `data/knowledge/` |
| `project/anthropic_client.py` | Cloud inference (dormant until ANTHROPIC_API_KEY set) |
| `project/static/index.html` | Dashboard UI (single file — HTML + CSS + JS) |
| `project/.env.example` | Template for `project/.env` |
| `project/.env` | Runtime configuration (not committed) |
| `project/tests/verify_full.py` | 46-test core verification suite |
| `project/tests/verify_stress.py` | 30-test stress simulation suite |
| `project/tests/verify_resource_mgmt.py` | 29-test resource management suite |
| `project/tests/verify_distill.py` | 34-test distillation suite |
| `systemd/user/delta-daemon.service` | systemd user unit for `delta-daemon` |
| `scripts/backup_s3.py` | S3 backup (full/incremental/restore) |
| `scripts/daily_training.py` | Nightly autonomous training orchestrator |
| `scripts/collect_training_data.py` | Standalone web data collector |
| `docs/local-model-workflow.md` | Operator guide: RAG, models, adapters |

---

## Stream Protocol (frontend ↔ backend)

`POST /chat` returns NDJSON (one JSON object per line):

```
{"t":"route","backend":"local","model":"...","tier":1,"reason":"...","split":false,"query_category":"..."}
{"t":"session","active":true,"session_id":"..."}
{"t":"rag","n":3}
{"t":"split_phase","phase":1,"c":"..."}
{"t":"tool","n":"run_shell","a":{"cmd":"..."}}
{"t":"result","n":"run_shell","s":"summary"}
{"t":"retry","n":"run_shell","c":"..."}
{"t":"react","iteration":1,"phase":"think|act|observe","c":"..."}
{"t":"clarify","c":"..."}
{"t":"text","c":"chunk"}
{"t":"emergency","c":"..."}
{"t":"done","turns":5}
{"t":"error","c":"user-friendly message"}
```

---

## Configuration (.env reference)

```env
# Models
OLLAMA_URL=http://localhost:11434
DELTAI_MODEL=qwen2.5:14b-instruct-q4_K_M
DELTAI_SMALL_MODEL=qwen2.5:3b-instruct-q4_K_M

# Cloud (optional)
ANTHROPIC_API_KEY=
CLOUD_BUDGET_DAILY=5.00

# Voice
VOICE_ENABLED=false
TTS_VOICE=en-US-AndrewNeural

# Training
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct

# Intelligence
REACT_ENABLED=true
REACT_MAX_ITERATIONS=3
RAR_ENABLED=true
DEDUP_THRESHOLD=0.15
MIXTURE_LORA_ENABLED=false
SMART_HISTORY_ENABLED=true
KNOWLEDGE_GAP_DETECTION=true
REASONING_TRACE_ENABLED=true
REASONING_TRACE_MAX=500
QUALITY_CAPTURE_THRESHOLD=0.6
REACT_ALLOW_CLARIFY=true
SMART_CAPTURE_ENABLED=true

# Resource management
VRAM_TIER_AB_MIN_MB=5000
WARM_TO_COLD_AGE_SEC=86400
INGEST_QUEUE_MAX=500
INGEST_FLUSH_INTERVAL=2.0

# Training automation (disabled by default)
DAILY_TRAIN_ENABLED=false
DAILY_TRAIN_MIN_VRAM_MB=7000
DAILY_TRAIN_AUTO_PROMOTE=false
DPO_ENABLED=false
```

---

## Development Rules

- **Linux paths only** — no Windows-style `C:\...` paths anywhere in code.
- **User space only** — never require root, never write outside the project tree or XDG app dirs (`$XDG_DATA_HOME/deltai`, `$XDG_CONFIG_HOME/deltai`, `$XDG_CACHE_HOME/deltai`).
- **systemd user service** — deltai runs as a user service (`systemctl --user`), not system.
- **Single-file frontend** — all of `static/index.html` in one file (HTML + CSS + JS).
- **Shell tool** — use `run_shell` (bash/sh), not `run_powershell`.
- **No domain-specific ingest** — never add telemetry parsers, game data readers, or domain pipelines to core. External services push via `/ingest`.
- **Modelfile rebuilds** — after changes: `ollama create <name> -f modelfiles/<file>.modelfile`

---

## Development Workflow

After any feature, bug fix, or significant change:

### 1. Run verification

```bash
cd project && source venv/bin/activate
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
# If you touched training/distillation:
python tests/verify_distill.py
```

### 2. Update CLAUDE.md

Update every relevant section: Architecture, Key Files, Configuration, stream protocol, development rules, etc. This file is the primary context for future AI sessions — stale = wrong assumptions.

### 3. Update README.md

Update user-facing changes: new features, endpoints, configuration options, project structure.

### 4. Update AGENTS.md

If onboarding steps, boundaries, or verify commands changed, update AGENTS.md so Cursor agents stay aligned.

### 5. Commit and push

```bash
git add <relevant files>  # never .env, data/, credentials
git commit -m "feat|fix|refactor|docs: description"
git push
```

### 6. GitHub Issues

If the work corresponds to a GitHub issue, close it: `gh issue close <N> --comment "..."`.
The repo is `Reconnecting18/deltai`.

---

## Current Status

deltai is in early development. The FastAPI backend, router, RAG, tools, training, and voice modules are mature in features but still being **generalized for Linux-first, user-choice operation** (docs, defaults, and naming now target that story).

- [x] systemd user service unit in-repo (`systemd/user/delta-daemon.service`) and XDG-style env vars
- [ ] Linux-appropriate defaults everywhere (`run_shell` vs legacy PowerShell naming on the host)
- [ ] Plugin API design (tool registration from external services)
- [ ] Broader task automation examples beyond optional adapter domains (`racing`, `telemetry`, etc.)
- [ ] Documentation and onboarding kept aligned with root README / AGENTS.md
