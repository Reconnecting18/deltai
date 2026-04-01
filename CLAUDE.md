# E3N — Project Context

**Paths:** This file and others may use `C:\e3n\` as the canonical layout on the primary machine. **Clones and other machines should use paths relative to the git repository root** (e.g. `project/main.py`, not an absolute drive letter).

## Agent tooling (Cursor vs Claude Code)

- **Cursor:** Start from [AGENTS.md](AGENTS.md) and [`.cursor/rules/`](.cursor/rules/) for short, enforced conventions. Use this file (`CLAUDE.md`) for full architecture, endpoints, protocol, and status.
- **Claude Code:** [`.claude/launch.json`](.claude/launch.json) can launch app processes; paths there may need editing for your machine—[`.vscode/launch.json`](.vscode/launch.json) is the portable option for VS Code / Cursor.
- **Shared:** `CLAUDE.md` is the deep context for any assistant; keep it current when you change behavior.

## What This Is
E3N is a local AI system (the "brain") running on Windows 11. Named after E3N from COD: Infinite Warfare. Personality blends E3N (dry wit, loyal) and BT-7274 (precise, protocol-driven). Operator: Ethan, 17, incoming MechE student, sim racer (Le Mans Ultimate).

E3N is ONLY the AI intelligence layer — reasoning, memory, tool execution, and model routing. It does NOT contain telemetry processing, game integrations, or domain-specific data pipelines. External services (like a future Telemetry API for Le Mans Ultimate) connect to E3N via the /ingest endpoint to push context into its RAG memory.

## Hardware
- GPU: NVIDIA RTX 3060 12GB | CPU: i7-12700K | RAM: 34GB | Windows 11

## Architecture
- **Backend:** FastAPI at `C:\e3n\project\main.py`, served on localhost:8000
- **Frontend:** Single-file `C:\e3n\project\static\index.html` (~60KB, CSS+HTML+JS)
- **Electron:** `C:\e3n\app\main.js` (frameless window, IPC)
- **Models (primary):** Ollama — `e3n-qwen14b` (Qwen2.5-14B Q4_K_M, Tier A), `e3n-qwen3b` (Qwen2.5-3B Q4_K_M, Tier B/C). Dynamic quantization variants: Q6_K (best quality), Q3_K_M (moderate), Q2_K (emergency)
- **Models (emergency backup):** `e3n-nemo` (Mistral Nemo 12B), `e3n` (LLaMA 3.1 8B) — last resort only
- **Modelfiles:** `C:\e3n\modelfiles\` — 4 modelfiles (E3N-qwen14b, E3N-qwen3b, E3N-nemo, E3N), identical system prompts, only FROM/parameters differ. E3N-qwen3b has a condensed prompt variant.
- **RAG:** ChromaDB at `C:\e3n\data\chromadb`, nomic-embed-text, 0.75 cosine threshold, multi-query expansion + reranking, semantic deduplication, iterative retrieval-augmented reasoning
- **Knowledge:** Drop files in `C:\e3n\data\knowledge\` — watchdog auto-ingests
- **Ingest connector:** `POST /ingest` — external services push structured context into ChromaDB with source tags and TTL (async pipeline with queue-based batching)
- **Router:** `C:\e3n\project\router.py` — VRAM-aware GPU detection, sim process detection, tier classification (A/AB/B/C), dynamic GPU layer offloading (`_calc_num_gpu`), dynamic quantization tier selection, model cascade, emergency backup chain, cloud budget enforcement, split workload detection, session mode, telemetry query classification, adapter domain classification, Mixture-of-LoRA adapter routing
- **Tools:** 7 core tools + 3 computation delegation + 4 diagnostic/self-management tools + 1 adapter management + 4 conditional telemetry tools in `C:\e3n\project\tools\` (core: read_file, write_file, list_directory, run_powershell, get_system_info, search_knowledge, memory_stats; computation: calculate, summarize_data, lookup_reference; diagnostic: self_diagnostics, manage_ollama_models, repair_subsystem, resource_status; adapter: manage_adapters; telemetry: get_session_status, get_lap_summary, get_tire_status, get_strategy_recommendation — only loaded when TELEMETRY_API_URL is set). Tool relevance filtering (`filter_tools()`) pre-filters 19 tools to 5-8 per query based on domain.
- **Quality scoring:** `C:\e3n\project\quality.py` — heuristic 6-signal scorer (0.0-1.0), drives smart capture, routing feedback, and knowledge gap detection
- **Reasoning traces:** SQLite `reasoning_traces` table persists ReAct Think/Act/Observe chains, retrieved via embedding similarity as `[PRIOR REASONING]` context
- **Resource self-manager:** Background loop (30s interval) — auto VRAM lifecycle, predictive VRAM management (decline rate + thermal monitoring), OS process priority management, Ollama health monitoring + auto-restart, watcher recovery, TTL cleanup, warm-to-cold memory compaction
- **Circuit breaker:** Protects Ollama inference path — 3-failure threshold, exponential backoff (5s→60s max), half-open recovery testing
- **Anthropic client:** `C:\e3n\project\anthropic_client.py` — cloud inference with tool use, split workload mode, telemetry mode prompt injection, conversation history support (dormant — no API key set)
- **Conversation history:** In-memory rolling 10-turn window + SQLite persistence across all chat paths, session-aware tagging, conversation-aware smart history with tiered compression
- **Split workload:** Local 3B gathers data via tools → cloud reasons over enriched context
- **Cloud cost budget:** Daily spend tracking ($5 default), router gates cloud when exhausted, persisted to SQLite
- **Voice module:** `C:\e3n\project\voice\` (package) — STT (faster-whisper) + TTS (Piper/edge-tts), RVC voice conversion, playback, post-processing
- **Training pipeline:** `C:\e3n\project\training.py` — dataset management, QLoRA fine-tuning (Qwen2.5-3B), adapter surgery (4 domain slots with TIES merge, selective layer freezing, adapter registry + versioning), knowledge distillation, iterative distillation (weakness identification + targeted improvement), few-shot fallback, GGUF export + Ollama registration, A/B eval, smart auto-capture (quality-tiered with dedup + negative examples)
- **Hierarchical memory:** Hot (in-memory, <5min) → Warm (ChromaDB persistent, 5min-24hr) → Cold (SQLite + zlib, >24hr) at `C:\e3n\data\cold_memory.db`
- **ReAct reasoning:** Structured Think-Act-Observe loop for complex local queries (max 3 iterations), confidence-aware (HIGH/MEDIUM/LOW), reasoning trace memory for learning from past chains
- **Knowledge gap detection:** SQLite `knowledge_gaps` table logs failed queries from low quality scores, ReAct max iterations, RAG zero results
- **Venv:** `C:\e3n\project\venv\`

## Key Architecture Decision: E3N is a Pure AI Brain
E3N does NOT directly process telemetry, UDP packets, or game data. Instead:
- External services (Telemetry API, voice module, etc.) process their own data
- They push summarized context into E3N via `POST /ingest`
- E3N stores this in ChromaDB with TTL (auto-expiry) and source tags
- When the user asks a question, RAG pulls relevant context regardless of source
- This makes E3N pluggable — any service can feed it context

### What does NOT belong in E3N:
- UDP listeners, packet parsers, frame buffers
- Game-specific telemetry tools (get_live_telemetry, get_tire_analysis, etc.)
- Telemetry REST/WebSocket endpoints
- Telemetry dashboard widgets
- Any domain-specific data processing pipeline

### What DOES belong in E3N:
- LLM inference routing (local + cloud)
- VRAM-aware model management
- RAG memory (ChromaDB) + ingestion connector
- General-purpose tool system
- Smart routing with tier classification
- Split workload (local tools + cloud reasoning)
- Conversation history + persistence (session-aware)
- Cloud cost budget enforcement
- Training pipeline for fine-tuning (+ racing auto-capture)
- Emergency backup system
- Voice module (STT/TTS)
- Session mode + GPU protection for active racing
- Telemetry query classification and prompt injection
- Resource self-management (VRAM lifecycle, auto-recovery, circuit breaker, predictive VRAM, thermal monitoring)
- OS process priority management (yield CPU to sim during racing)
- Self-diagnostic tools (AI-driven inspection and repair of own subsystems)
- WebSocket alerts for proactive racing notifications
- Batch ingest for high-frequency data sources
- Hierarchical memory (hot/warm/cold tiering with compression)
- ReAct reasoning loop (structured Think-Act-Observe for complex queries)
- Dynamic quantization and GPU layer offloading
- Mixture-of-LoRA adapter routing
- Streaming async ingest pipeline
- Response quality scoring and routing feedback
- Reasoning trace memory (learn from past Think/Act/Observe chains)
- Tool relevance filtering (domain-aware prompt token savings)
- Confidence-aware reasoning (clarification protocol)
- Smart auto-capture with quality tiering and negative examples
- Iterative distillation (weakness detection + targeted improvement)
- Knowledge gap detection and tracking
- Conversation-aware smart history (tiered compression)

## Key Files
| File | Purpose |
|------|---------|
| `AGENTS.md` | Short agent onboarding (Cursor); points here for full context |
| `project/main.py` (~2500 lines) | FastAPI app — chat (3 paths: local, cloud, split), ReAct reasoning loop (confidence-aware, trace memory), conversation-aware smart history, conversation history, stats, health, budget, memory, ingest (async pipeline), batch ingest, session mode, WebSocket alerts, backup, training, voice endpoints, resource self-manager loop (predictive VRAM, process priority, thermal, cold compaction), circuit breaker, resource status endpoint, cold memory + ingest pipeline endpoints, knowledge gap endpoints |
| `project/router.py` | Smart routing: consolidated VRAM detection (_get_vram_info), sim detection, tier classification (A/AB/B/C), dynamic GPU layer offloading (_calc_num_gpu), dynamic quantization tier selection, split workload detection, cloud budget enforcement + persistence, emergency backup chain, session mode (GPU protection), telemetry query classification, Mixture-of-LoRA adapter routing (resolve_adapter_model), adaptive routing feedback (quality-driven tier adjustments), multi-domain classification |
| `project/quality.py` | Response quality scoring: 6-signal heuristic scorer (length_appropriateness, tool_success_rate, specificity, no_error_indicators, structural_match, no_repeat), SQLite persistence, drives smart capture + routing feedback + gap detection |
| `project/memory.py` | ChromaDB RAG: chunking, embedding, multi-query expansion, source-grouped reranking, recency bias, ingest with TTL, batch ingest, source/age filtering, semantic deduplication, iterative RAG (two-round retrieval), hierarchical memory (hot/warm/cold tiers with SQLite cold storage + zlib compression) |
| `project/watcher.py` | Watchdog file watcher for knowledge dir |
| `project/anthropic_client.py` | Anthropic API streaming with tool use, split_mode, telemetry_mode prompt injection, conversation history support (dormant — no API key) |
| `project/persistence.py` | SQLite backing store for conversation history (session-aware), cloud budget, session history export, reasoning traces, quality scores, routing feedback, knowledge gaps (WAL mode, short-lived connections) |
| `project/training.py` (~3200 lines) | Training pipeline: dataset CRUD, export (alpaca/sharegpt/chatml), QLoRA fine-tuning + GGUF export + Ollama registration, adapter surgery (registry, 6 domain slots — racing/engineering/personality/reasoning/telemetry/audio, selective layer freezing, TIES merge, eval/promotion/rollback), knowledge distillation, iterative distillation, DPO training (trl.DPOTrainer), session knowledge synthesis, daily training cycle orchestrator (run_daily_cycle), few-shot fallback, A/B eval, smart auto-capture (quality-tiered, dedup, negative examples) |
| `project/voice/` (~1780 lines) | Voice package: STT (faster-whisper), TTS (Piper/edge-tts), RVC voice conversion, playback, post-processing, voice config |
| `project/tools/executor.py` (~1270 lines) | Tool execution with type coercion, safety checks, tool retry on error, computation delegation (calculate, summarize_data, lookup_reference), self-diagnostics (7 subsystem deep checks), manage_ollama_models (status/unload/preload with sim guard), repair_subsystem (4 allowlisted repairs), resource_status, manage_adapters, conditional telemetry tools |
| `project/tools/definitions.py` (~420 lines) | 7 core + 3 computation + 4 diagnostic + 1 adapter + 4 conditional telemetry tool JSON schemas, `filter_tools()` domain-aware tool relevance filtering |
| `project/static/index.html` | Full dashboard UI (single file) — header health monitor, budget display, terminal with history CLR, WebSocket alert toasts |
| `project/.env` | Config (models, VRAM thresholds, backup, budget, history, paths, cloud settings, voice, session mode, telemetry, ReAct, RAR, dedup, quant tiers, MoLoRA, cold memory, ingest pipeline, reasoning traces, quality scoring, smart capture, smart history, knowledge gaps) |
| `project/tests/verify_full.py` | 46-test verification suite — training, safety guards, router stress, persistence, RAG, CUDA, computation delegation, anthropic client |
| `project/tests/verify_stress.py` | 30-test stress suite — review fix verification (10), high-stress simulations (20): low VRAM, backup cascade, concurrent routing, tool safety |
| `project/tests/verify_resource_mgmt.py` | 29-test resource management suite — circuit breaker (5), resource manager (3), diagnostic tools (11), stress simulations (10) |
| `project/tests/verify_distill.py` | 34-test distillation suite — config validation, teacher generation, blending, retention, distill mode integration, override parameters |
| `app/main.js` | Electron main process |
| `tools/llama.cpp/` | GGUF conversion toolchain — `convert_hf_to_gguf.py` + `build/bin/Release/llama-quantize.exe` |
| `scripts/backup_s3.py` | S3 nightly backup — full backup, incremental (MD5 skip), restore, retention cleanup |
| `scripts/setup_backup_task.ps1` | Windows Task Scheduler registration for nightly backup at 3 AM |
| `scripts/daily_training.py` | Daily autonomous training orchestrator — guard checks, web collection (Phase 0.5), weakness analysis, targeted distillation, QLoRA, memory consolidation, report. Standalone (no FastAPI needed). Flags: `--collect`, `--collect-only`. |
| `scripts/setup_daily_training_task.ps1` | Windows Task Scheduler registration for daily training at 2 AM (before S3 backup) |
| `scripts/collect_training_data.py` | Standalone web training data collector — Wikipedia HF streaming, arXiv, OpenF1, Semantic Scholar, motorsport web. Flags: `--source`, `--batch`, `--dry-run`. |
| `project/collector.py` | Web training data collection logic — Wikipedia (HF datasets streaming, checkpointed offset covering 6.7M articles), arXiv XML API (8 categories), OpenF1 race strategy API, Semantic Scholar paper abstracts, trafilatura + DDG motorsport pages. SHA256 dedup SQLite. run_collection_cycle() invoked as Phase 0.5 of nightly training cycle. |
| `project/training_build.py` | Dataset builder — 13 curated datasets: anti-hallucination, engineering, race engineering, data interpretation, personality, reasoning, MechE, simulations, telemetry analysis, audio analysis, engineering simulations, advanced strategy, CoT reasoning |

## How to Start
```powershell
# Terminal 1 — Backend
cd C:\e3n\project
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Electron
cd C:\e3n\app
npm start
```

## Stream Protocol (frontend ↔ backend)
JSON lines from `/chat`:
- `{"t":"route","backend":"...","model":"...","tier":N,"reason":"...","split":bool,"query_category":"..."}` — routing decision (includes split flag, query category)
- `{"t":"session","active":bool,"session_id":"..."}` — session state change
- `{"t":"rag","n":N}` — RAG context chunks found
- `{"t":"split_phase","phase":N,"c":"..."}` — split workload phase (1=gathering, 2=reasoning, 0=fallback)
- `{"t":"tool","n":"tool_name","a":{args}}` — tool call
- `{"t":"result","n":"tool_name","s":"summary"}` — tool result
- `{"t":"retry","n":"tool_name","c":"..."}` — tool retry (error recovery)
- `{"t":"react","iteration":N,"phase":"think|act|observe","c":"..."}` — ReAct reasoning loop event
- `{"t":"clarify","c":"..."}` — confidence-aware clarification request (LOW confidence in ReAct)
- `{"t":"text","c":"chunk"}` — response text
- `{"t":"emergency","c":"..."}` — backup model activated (rare)
- `{"t":"done","turns":N}` — stream complete (turns = conversation history count)
- `{"t":"error","c":"message"}` — error (user-friendly messages for connection/timeout issues)

## Development Rules
- **Single-file frontend:** Everything in `index.html`. CSS + HTML + JS in one file.
- **PowerShell:** No `&&` operator. Use semicolons or separate commands.
- **Python packages:** Always `pip install X --break-system-packages` outside venv.
- **Modelfile rebuilds:** After changes: `ollama create <model> -f C:\e3n\modelfiles\<modelfile>`
- **All modelfiles must stay in sync** — identical system prompts, only FROM line differs (except E3N-qwen3b which has a condensed variant).
- **Editing index.html:** Python scripts for surgical replacements work better than PowerShell string replacement.
- **pynvml warning:** Cosmetic FutureWarning, ignore it.
- **Ollama KEEP_ALIVE:** Models stay in VRAM 5min. Can tune with `OLLAMA_KEEP_ALIVE=2m`.
- **E3N boundary:** Never add domain-specific data processing (telemetry, game parsers, etc.) inside E3N. External services push data in via /ingest.

## Development Workflow (IMPORTANT — follow on every feature/fix)
After completing any feature, bug fix, or significant change, you MUST do ALL of the following before considering the task done:

### 1. GitHub Issues
- If the work corresponds to an existing GitHub issue, **close it** with a comment describing what was done: `gh issue close <N> --comment "..."`
- If implementing a new feature that doesn't have an issue, **create one first** with `gh issue create`, then close it when done.
- The repo is at `Reconnecting18/e3n`. The `gh` CLI is authenticated via keyring.

### 2. Update CLAUDE.md
- Update **every relevant section** of this file to reflect the changes: Architecture, Key Files, Current Status, Build Phases, Known Issues, feature-specific sections, env config blocks, etc.
- Add new sections for major new features.
- Keep the Current Status list up to date — mark new items as DONE.
- This file is the primary context for future AI sessions. If it's stale, future sessions will make incorrect assumptions.

### 3. Update README.md
- Update the README to reflect any user-facing changes: new features, new endpoints, new configuration, new scripts, changed project structure.
- The README should be high-quality and comprehensive — it's what people see on GitHub.
- Update the Build Phases table, API Endpoints tables, Project Structure tree, and feature descriptions as needed.

### 4. Commit and Push
- Stage only the relevant files (not `.env`, credentials, or data files).
- Write a clear commit message following the existing convention: `feat:`, `fix:`, `refactor:`, `docs:`.
- Include `Closes #N` for any issues being closed.
- Push to `origin main`.
- **Optional:** If the work was assisted by Claude, you may end the commit message with:
  ```
  Co-Authored-By: Claude <co-author> <noreply@anthropic.com>
  ```
  Other tools or human-only commits do not need this trailer.

### 5. AGENTS.md (Cursor and short onboarding)
- If onboarding steps, verification commands, or the E3N boundary description in [AGENTS.md](AGENTS.md) are affected, update that file so Cursor agents stay aligned with this document.

### Example Workflow
```
1. Implement the feature in code
2. gh issue create --title "feat: ..." --body "..."   (if no issue exists)
3. Edit CLAUDE.md — update all relevant sections
4. Edit README.md — update user-facing docs
5. Edit AGENTS.md — if onboarding or verification steps changed
6. git add <files>
7. git commit -m "feat: ... Closes #N ..."
8. git push origin main
9. gh issue close N --comment "Implemented in ..."
```

This ensures the repo, docs, and project board stay in sync across sessions.

## VRAM-Aware Model Tiers
| Tier | VRAM Free | Model | Ollama Name | Notes |
|------|-----------|-------|-------------|-------|
| A | > 9GB | Qwen2.5-14B Q4_K_M (~8.5GB) | `e3n-qwen14b` | No sim running, full VRAM |
| AB | 5-9GB | Qwen2.5-14B partial GPU offload (~4GB VRAM) | `e3n-qwen14b` | 50-70% GPU speed via `num_gpu` layer calc |
| B | 3-5GB | Qwen2.5-3B Q4_K_M (~2.5GB) | `e3n-qwen3b` | Sim running, GPU contested |
| C | < 3GB | Qwen2.5-3B on CPU (0 VRAM) | `e3n-qwen3b` | VRAM critical |

Router detects LMU sim process via psutil. On sim launch: unload large model, preload 3B. On sim close: swap back to 14B within 30 seconds.

**Dynamic GPU layer offloading:** `_calc_num_gpu()` in router.py calculates optimal partial GPU layers for Tier AB. Instead of fully loading or fully offloading, the 14B model runs with a fraction of layers on GPU and the rest on CPU. The `num_gpu` value and `num_ctx` (dynamic context window: 1024/2048/default based on VRAM) are passed through `RouteDecision` → `_try_ollama_inference` → Ollama API options.

**Config (.env):**
```
VRAM_TIER_AB_MIN_MB=5000
```

## Emergency Backup System
Last-resort failover — like a backup generator. Only engages after primary model fails BACKUP_MAX_RETRIES (default: 2) consecutive attempts with 3-second retry delays.

**Backup chain:**
- `e3n-qwen14b` → `e3n-nemo` → `e3n` → system down
- `e3n-qwen3b` → `e3n` → system down

**Infrastructure:**
- Background health loop pings backup models hourly (1-token generation test)
- Startup validates all primary + backup models exist in Ollama
- `GET /backup/status` endpoint for diagnostics
- `{"t":"emergency"}` stream event on backup activation
- Invisible to frontend during normal operation

**Config (.env):**
```
E3N_BACKUP_STRONG_MODEL=e3n-nemo
E3N_BACKUP_MODEL=e3n
BACKUP_HEALTH_INTERVAL_SEC=3600
BACKUP_ENABLED=true
BACKUP_MAX_RETRIES=2
```

## Conversation History
In-memory rolling window of recent exchanges so the model can reference prior conversation turns.

- **Max turns:** 10 user-assistant pairs (configurable via `CONVERSATION_HISTORY_MAX` in .env)
- **Injected into all 3 chat paths:** Ollama (prepended to messages array), Anthropic (prepended via `history` param), split workload (both phases)
- **What's stored:** Clean user text + final assistant response text only
- **What's NOT stored:** Greetings (canned), tool calls, RAG context, errors
- **Session-aware:** When a racing session is active, history pairs are tagged with `session_id`. On session end, tagged turns are exported to `C:\e3n\data\training\sessions\{session_id}.jsonl` for post-race review.
- **Smart history:** `_get_smart_history(max_tokens)` replaces `_get_history()`. Last 3 turns: full text. Turns 4-7: compressed (key facts — sentences with numbers, units, conclusions extracted via regex, no LLM). Turns 8-10: one-line summaries. Token budget from `decision.num_ctx` drives compression aggressiveness.
- **Endpoints:** `DELETE /chat/history` (clear), `GET /chat/history` (turns + max_turns metadata)
- **Frontend:** CLR button in terminal header, turn counter `[NT]` (e.g., `[3T]` = 3 turns)
- **Persistent:** Backed by SQLite (`persistence.py`) — survives server restarts. In-memory remains primary; DB is the backing store.

**Config (.env):**
```
CONVERSATION_HISTORY_MAX=10
SMART_HISTORY_ENABLED=true
```

## Smarter RAG
Multi-query retrieval with reranking for better knowledge base results.

- **Query expansion:** `_expand_query()` generates 2-3 search variants from keywords + domain synonyms (no LLM needed)
- **Multi-query search:** All variants searched in parallel, results deduplicated by chunk ID
- **Source grouping:** Documents with multiple matching chunks get a relevance boost (diminishing returns)
- **Recency bias:** Recently ingested context (within 5 minutes via /ingest) gets a slight boost — useful for live data
- **Source filtering:** `source_filter` parameter restricts results to a specific source (e.g., `ingest:lmu-telemetry`)
- **Age filtering:** `max_age_sec` parameter filters out results older than N seconds (based on `ingested_at` metadata) — useful for live session queries
- **Semantic deduplication:** `_check_semantic_duplicate()` prevents near-duplicate ingestion (cosine distance < 0.15 threshold)
- **Iterative RAG:** `iterative_query_knowledge()` performs two-round retrieval with gap analysis and sub-query generation (see dedicated section)
- **Backward compatible:** `query_knowledge()` API unchanged, optional parameters added

## Split Workload
Two-phase inference: local model gathers data (free), cloud model reasons over it (premium quality). Best of both worlds.

- **Activates when:** `decision.backend == "anthropic" AND decision.split == True`
- **Detection:** `SPLIT_PATTERNS` regex in `router.py` matches tasks like "analyze telemetry", "review code", "calculate from data"
- **Phase 1:** Local 3B model runs Ollama tool loop to gather data. Tool/result events stream to frontend. Local model's text is suppressed (intermediate, not final).
- **Phase 2:** Tool results packaged into `[SPLIT WORKLOAD — Local tool results]` context block (each result truncated to 8000 chars). Sent to Anthropic with `split_mode=True` (system prompt tells cloud to focus on analysis, not gather data).
- **Fallback:** If no tools are called in Phase 1 → falls back to standard cloud with tools enabled
- **Currently dormant:** Requires `ANTHROPIC_API_KEY` to be set. Router won't select `backend=anthropic` without it.

## Cloud Cost Budget
Daily spend tracking with automatic enforcement. Prevents surprise bills when the API key goes live.

- **Config:** `CLOUD_BUDGET_DAILY=5.00` in .env (default $5/day)
- **Enforcement:** `_check_budget()` in `router.py` gates `cloud_ready` — when budget exhausted, cloud routing is disabled
- **Tracking:** `record_cloud_usage()` estimates cost from token counts (Sonnet: $3/$15 per M in/out, Opus: $15/$75 per M in/out)
- **Endpoints:** `GET /budget/status` returns `{daily_budget, daily_spent, daily_remaining, budget_ok}`
- **Also in:** `/stats` response includes `budget` field for the frontend poll
- **Header display:** `$X.XX/$5.00` with color coding: green (normal) → amber (>70%) → red (>90%)
- **Resets daily** at midnight. Spend persists across restarts via SQLite (`persistence.py`).

## Voice Module
Integrated STT + TTS for voice interaction with E3N.

- **STT:** faster-whisper (CTranslate2 backend) — runs locally on GPU or CPU, VRAM-aware device selection
- **TTS:** edge-tts (Microsoft Edge free API, high quality) with Windows SAPI fallback
- **Lazy loading:** Whisper model loaded on first use, not at startup (saves VRAM)
- **Audio cleanup:** `_clean_for_tts()` strips code blocks, URLs, markdown, long text before synthesis
- **Endpoints:**
  - `POST /voice/stt` — Upload audio bytes → get transcription JSON
  - `POST /voice/tts` — Send text → get MP3/WAV audio response
  - `POST /voice/chat` — Full loop: audio in → transcribe → chat → synthesize → audio out (base64)
  - `GET /voice/status` — STT/TTS subsystem health
- **Health monitor:** Voice status included in `/api/health` response (8 subsystems now)

**Config (.env):**
```
VOICE_ENABLED=true
WHISPER_MODEL=base
WHISPER_DEVICE=auto
TTS_VOICE=en-US-GuyNeural
TTS_RATE=+0%
```

## Session Mode (GPU Protection)
Active racing session detection and GPU protection for frame rate preservation.

- **Auto-detection:** When `/ingest` receives data from a source matching `SESSION_SOURCE_PATTERN` (default: `lmu`), session mode auto-activates
- **Manual control:** `POST /session/start` and `POST /session/end` endpoints
- **GPU protection:** When session active + GPU protect enabled, ALL inference routes away from GPU:
  - Cloud available → route to Anthropic (Sonnet/Opus based on complexity)
  - No cloud → route to CPU-only local (e3n-qwen3b on CPU)
- **Auto-timeout:** Session deactivates after `SESSION_TIMEOUT_SEC` (default: 60s) without ingest activity
- **Session history:** Conversation turns tagged with `session_id`, exported to JSONL on session end
- **Endpoints:**
  - `POST /session/start` — manual activation, returns session_id
  - `POST /session/end` — deactivate, export session history to `C:\e3n\data\training\sessions\`
  - `GET /session/status` — current state (active, started_at, last_ingest, timeout, etc.)

**Config (.env):**
```
SESSION_GPU_PROTECT=true
SESSION_FORCE_CLOUD=true
SESSION_DETECT_MODE=ingest
SESSION_SOURCE_PATTERN=lmu
SESSION_TIMEOUT_SEC=60
```

## Telemetry Query Classification
Regex-based classifier in `router.py` that categorizes telemetry-related queries during active sessions.

- **4 categories** (checked in priority order — most specific first):
  - `telemetry_debrief` — race/session analysis, post-race review
  - `telemetry_strategy` — pit calls, tire strategy, fuel management, weather decisions
  - `telemetry_coaching` — driving improvement, braking points, lines, oversteer/understeer
  - `telemetry_lookup` — simple data queries (fuel, tire temps, lap times, gaps)
- **Activates when:** Session is active OR sim is running
- **Query category** propagated through `RouteDecision.query_category` to all downstream systems
- **Telemetry lookup short-circuit:** Simple data queries answered directly from RAG without model inference (zero latency)
- **Racing prompt templates:** Dynamic coach/engineer prompt injection based on category:
  - Coaching → structured OBSERVATION/IMPACT/ACTION format
  - Strategy/Debrief → structured RECOMMENDATION/NUMBERS/RISK format
  - Injected into Ollama (as `[SYSTEM NOTE]` prefix) and Anthropic (via `telemetry_mode` parameter)

## WebSocket Alerts
Real-time proactive alert system for racing notifications.

- **Endpoint:** `WebSocket /ws/alerts` with auto-reconnect client
- **Trigger:** `/ingest` with `"alert"` in tags automatically forwards to all connected WebSocket clients
- **Priority levels:** `normal`, `high` (amber), `critical` (red)
- **Recent history:** `GET /alerts/recent` returns last 20 alerts (in-memory deque)
- **Frontend:** Toast notifications slide in from bottom-right, auto-dismiss after 10 seconds
- **Connection:** Auto-reconnect with 5-second retry on disconnect

## Batch Ingest
High-throughput ingestion for multiple context items in a single call.

- **Endpoint:** `POST /ingest/batch` with `{"items": [...]}`
- **Efficiency:** Single `get_embeddings()` call + single `collection.add()` for all chunks
- **Rate limit:** Max 100 items per batch
- **Validation:** Items with empty source/context silently skipped, accurate `items_processed` count
- **Alert forwarding:** Items with `"alert"` in tags still forwarded to WebSocket clients

## Conditional Telemetry Tools
4 tools that only load when `TELEMETRY_API_URL` env var is set — keeps E3N clean when no telemetry service is running.

- **Tools:** `get_session_status`, `get_lap_summary`, `get_tire_status`, `get_strategy_recommendation`
- **API calls:** Each tool hits the external Telemetry API via httpx GET requests
- **Error handling:** Graceful failure with clear error messages when API is unreachable
- **Dynamic loading:** `TOOLS` list and `EXECUTORS` dict extended at import time only when configured

## Training Pipeline
Dataset management, QLoRA fine-tuning, and progressive model improvement.

- **Two training modes:**
  - `"lora"` — Real QLoRA fine-tuning via transformers/peft/trl on Qwen2.5-3B (~6-7GB VRAM)
  - `"fewshot"` — Legacy few-shot embedding into system prompt (always available, no deps needed)
  - `"auto"` (default) — Tries LoRA first, falls back to fewshot if deps missing
- **LoRA pipeline:** Load 4-bit base → apply LoRA (r=16, alpha=32) → train with SFTTrainer → save adapter → merge on CPU → export GGUF (Q4_K_M) → register in Ollama
- **Safety guards:** Blocks training during active racing sessions or when sim is running. SafetyCallback aborts mid-training if sim launches. Ollama models unloaded from VRAM before training starts.
- **GGUF conversion:** Uses llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` for Q4_K_M output
- **A/B evaluation:** `run_ab_eval()` compares two models on a dataset — measures latency and response quality, saves results to `eval/` directory
- **Dataset CRUD:** Create, list, add examples, remove examples, get all examples
- **Export formats:** Alpaca, ShareGPT, ChatML — ready for external training tools
- **Smart auto-capture:** `smart_auto_capture()` replaces basic `auto_capture()`. Quality-tiered: score >= 0.8 always captured, 0.6-0.8 at 50% probability, < 0.3 stored as negative example in `{dataset}-negative` for DPO training. Hash-based deduplication (recent 200 queries).
- **Iterative distillation:** `identify_weak_domains()` finds domains with avg quality < 0.6. `distill_targeted()` pulls worst queries for teacher data generation. `run_improvement_cycle()` orchestrates full loop: identify weaknesses → distill targeted data → (train → eval → promote).
- **DPO training:** `start_dpo_training()` uses `trl.DPOTrainer` to align on positive/negative example pairs. Runs after SFT. Needs matched pairs (same query, different quality responses).
- **Session knowledge synthesis:** `synthesize_session_knowledge()` — after session ends, teacher generates 200-400 word knowledge article from session turns. Returned for permanent RAG ingestion.
- **Daily cycle:** `run_daily_cycle()` — full autonomous improvement loop: guard checks → weakness analysis → targeted distillation → blend + train → gap review → report to `data/training/daily_reports/`.
- **Background execution:** Training runs in a background thread, status trackable via `/training/status` (includes loss, step progress, trainable params)
- **Progressive loop:** Use E3N → smart auto-capture → accumulate dataset → trigger LoRA → deploy fine-tuned model → repeat
- **Endpoints:**
  - `GET /training/datasets` — list datasets
  - `POST /training/datasets` — create dataset
  - `POST /training/datasets/{name}/add` — add example
  - `POST /training/datasets/{name}/export` — export to format
  - `POST /training/start` — start fine-tuning (accepts `mode`: lora/fewshot/auto)
  - `POST /training/stop` — cancel training
  - `GET /training/status` — progress, status, loss, errors
  - `GET /training/lora/status` — check if LoRA deps are installed
  - `POST /training/eval/ab` — A/B model comparison
  - `GET /training/weaknesses` — identify weak domains via quality scores
  - `POST /training/improve/{domain}` — trigger targeted iterative distillation for a domain

**LoRA config (.env):**
```
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_EPOCHS=3
LORA_BATCH_SIZE=2
LORA_GRAD_ACCUM=4
LORA_LR=2e-4
LORA_MAX_SEQ_LEN=1024
LORA_QUANT_METHOD=Q4_K_M
LLAMA_CPP_PATH=C:\e3n\tools\llama.cpp
SMART_CAPTURE_ENABLED=true
CAPTURE_DEDUP_THRESHOLD=0.15
# Phase 10 — Continuous intelligence
DAILY_TRAIN_ENABLED=true
DAILY_TRAIN_MIN_VRAM_MB=7000
DAILY_TRAIN_AUTO_PROMOTE=false
DAILY_TRAIN_AUTO_MERGE=false
SESSION_SYNTHESIS_ENABLED=true
SESSION_SYNTHESIS_MODEL=local14b
DPO_ENABLED=false
# Phase 11 — Web training data collector
WEB_COLLECT_ENABLED=true
WEB_COLLECT_WIKIPEDIA=true
WEB_COLLECT_ARXIV=true
WEB_COLLECT_OPENF1=true
WEB_COLLECT_MOTORSPORT=true
WEB_COLLECT_PAPERS=true
WEB_COLLECT_WIKIPEDIA_BATCH=2000
WEB_COLLECT_MAX_PER_SOURCE=200
```

## Adapter Surgery (Augmentation Slots)
Modular LoRA adapters: train, version, evaluate, and merge domain-specific "augmentations" independently.

### Six Augmentation Slots
| Domain | LoRA Rank | Freeze Depth | Purpose |
|--------|-----------|-------------|---------|
| Racing | r=16 | 18/36 layers (50%) | Tire strategy, telemetry, driving technique |
| Engineering | r=16 | 12/36 layers (33%) | Physics, calculus, statics, thermo, FEA/CFD |
| Personality | r=8 | 24/36 layers (67%) | E3N voice, response style |
| Reasoning | r=32 | 8/36 layers (22%) | Chain-of-thought, analysis, structured CoT |
| Telemetry | r=16 | 14/36 layers (39%) | Multi-sensor correlation, anomaly detection, degradation curves |
| Audio | r=16 | 16/36 layers (44%) | Engine knock, mechanical signatures, real-time audio diagnostics |

### How It Works
1. **Domain training:** `start_domain_training(domain)` trains a LoRA adapter with selective layer freezing — bottom layers frozen (universal knowledge), top layers trained (domain-specific)
2. **Adapter registry:** JSON at `C:\e3n\data\training\adapter_registry.json` tracks all adapters with version, eval score, status, and active slot assignment
3. **TIES merge:** Combines active adapters from all slots into a single production GGUF — CPU-only, zero GPU cost. Resolves weight conflicts via Trim, Elect Sign, Merge. Dynamic weighting based on actual query distribution across domains.
4. **Eval & promote:** `eval_adapter()` compares against baseline, auto-promotes winners to active slots
5. **Rollback:** Revert any domain to a previous adapter version
6. **Self-management:** E3N can call `manage_adapters` tool to inspect and manage its own augmentations
7. **Cross-domain classification:** `classify_adapter_domains()` returns ALL matching domains sorted by score (not just primary). Cross-domain trace retrieval in ReAct for multi-domain queries.

### Endpoints
```
GET  /adapters              — List all adapters (optional domain filter)
GET  /adapters/active       — Current active adapter map
GET  /adapters/{name}       — Adapter details
POST /adapters/train        — Start domain training
POST /adapters/merge        — TIES merge active adapters → production GGUF
POST /adapters/eval/{name}  — Evaluate adapter
POST /adapters/promote/{name} — Promote to active slot
POST /adapters/rollback     — Rollback domain
DELETE /adapters/{name}     — Remove adapter
```

### Config (.env)
```
ADAPTER_MERGE_METHOD=ties
ADAPTER_MERGE_DENSITY=0.5
ADAPTER_AUTO_MERGE=false
ADAPTER_AUTO_PROMOTE=false
```

## Daily Autonomous Training
Scheduled daily self-improvement cycle. Runs at 2:00 AM via Windows Task Scheduler (before the 3 AM S3 backup, so new adapters are included in backup). Standalone — does not require the FastAPI server to be running.

- **Script:** `scripts/daily_training.py` — standalone orchestrator
- **Scheduler:** `scripts/setup_daily_training_task.ps1` — registers Task Scheduler job
- **Log:** `data/training/daily_training.log`
- **Reports:** `data/training/daily_reports/YYYY-MM-DD.json`

**Phases:**
1. Guard checks: sim not running, VRAM ≥ `DAILY_TRAIN_MIN_VRAM_MB`, no active session, no training already running
2. Weakness analysis: calls `identify_weak_domains()` — domains with avg quality < 0.6
3. Targeted distillation: teacher (14B) generates examples for up to 2 weak domains
4. Curriculum: daily topic rotating by weekday (Mon=telemetry, Tue=strategy, Wed=engineering, Thu=simulations, Fri=audio, Sat=cross-domain, Sun=personality/retention)
5. QLoRA domain adapter training: blends curriculum datasets + distillation, trains one domain adapter
6. Knowledge gap review: counts unresolved gaps, surfaces patterns
7. Report: writes structured JSON report to `daily_reports/`

**Weekly curriculum:**
| Day | Primary | Secondary |
|-----|---------|-----------|
| Mon | Telemetry analysis | Audio analysis |
| Tue | Race strategy advanced | Race engineering |
| Wed | Engineering + MechE | — |
| Thu | Engineering simulations | CoT reasoning |
| Fri | Audio analysis | Telemetry analysis |
| Sat | General reasoning | Data interpretation |
| Sun | Personality calibration | Retention baseline |

**CLI usage:**
```powershell
# Dry run (no actual training, just reports)
python scripts/daily_training.py --dry-run

# Force Monday curriculum
python scripts/daily_training.py --day 0

# Skip QLoRA, distillation only
python scripts/daily_training.py --no-train

# View last report
python scripts/daily_training.py --report-only
```

**Config (.env):**
```
DAILY_TRAIN_ENABLED=true
DAILY_TRAIN_MIN_VRAM_MB=7000
DAILY_TRAIN_AUTO_PROMOTE=false
DAILY_TRAIN_AUTO_MERGE=false
```

## DPO Training
Direct Preference Optimization using positive/negative example pairs.

- **Function:** `start_dpo_training()` in `training.py`
- **Uses:** `trl.DPOTrainer` (same trl dependency already installed)
- **Input:** positive dataset + `{dataset}-negative` dataset (accumulated by smart_auto_capture)
- **Preference pairs:** matched by input text — same query, different quality responses
- **Minimum:** 10 matched pairs required
- **LoRA config:** r=8 (lighter than SFT) for preference tuning
- **Endpoint:** Same `/training/start` with `mode=dpo` (or direct function call)
- **Best used:** After SFT base is trained, to further align response quality
- **Status:** Functional, disabled by default until negative datasets accumulate (needs ~50+ pairs)

**Config (.env):**
```
DPO_ENABLED=false
```

## Session End Knowledge Synthesis
After a racing/work session ends, synthesizes key learnings into a durable knowledge article.

- **Function:** `synthesize_session_knowledge(session_id, session_turns)` in `training.py`
- **Trigger:** Called from `POST /session/end` handler in `main.py` (requires integration)
- **Teacher:** 14B local model (or 3B if VRAM constrained) synthesizes last 30 turns
- **Output:** 200-400 word knowledge article with structured headings
- **Storage:** Returned to caller for ingestion as permanent (TTL=0) RAG entry
- **Source tag:** `session-synthesis:{session_id}` — searchable in future sessions
- **Value:** Turns race sessions into durable, searchable knowledge, not just ephemeral chat logs

**Config (.env):**
```
SESSION_SYNTHESIS_ENABLED=true
SESSION_SYNTHESIS_MODEL=local14b
```

## Symbolic Math Engine (solve_math)
SymPy-powered computation tool for calculus, algebra, and linear algebra.

- **Operations:** solve, differentiate, integrate, limit, simplify, expand, factor, matrix, series, laplace, eigenvalues
- **Safety:** Controlled symbol namespace (x,y,z,t,s,a,b,c,n,r,theta,phi,omega,k,m,g), no raw eval, 500-char limit, same blocklist as calculate
- **Integration:** Models instructed via Protocol 6 to delegate all symbolic math to solve_math, arithmetic to calculate
- **Zero VRAM:** Runs on CPU via SymPy, no GPU impact

## Text-as-Tool Parser
Robust fallback parser for when models output tool calls as text instead of structured tool_calls.

- **Balanced brace extraction:** Handles nested JSON objects (not just flat `[^{}]` regex)
- **Multiple formats:** Direct JSON, markdown code blocks, Ollama-style arrays, embedded in text
- **Windows path fix:** `_fix_windows_paths()` handles invalid JSON escapes from `C:\e3n\...` paths
- **Python literal fix:** Converts `True/False/None` to `true/false/null`
- **Single-quote fallback:** Tries replacing `'` with `"` as last resort
- **First-only:** When multiple tool calls present in text, returns only the first (prevents loops)

## Error Recovery UX
User-friendly error handling and automatic retry logic.

- **Tool retry:** If a tool execution fails (read_file, write_file, run_powershell), automatically retries once with sanitized/normalized parameters
- **Friendly errors:** Connection timeouts, Ollama down, and model failures get human-readable messages instead of raw errors
- **`{"t":"retry"}` event:** Frontend notified when a retry is happening
- **Split fallback:** Phase 1 errors gracefully fall back to direct cloud inference

## Header Health Monitor
Live subsystem health monitor in the dashboard header.

- **Endpoint:** `GET /api/health` checks 8-9 subsystems: Ollama, FastAPI, ChromaDB/RAG, Router, Watcher, Tools, Anthropic, Voice, and optionally Telemetry (when `TELEMETRY_API_URL` is configured)
- **Display:** "X/8 ONLINE" (or "X/9 ONLINE" with telemetry) with colored dot (green = all nominal, orange/red = degraded)
- **Alert badges:** Non-nominal systems show warning/error badges (e.g., "CLOUD — STANDBY", "WATCHER — STOPPED")
- **Dynamic count:** Subsystem count adjusts based on whether telemetry is configured
- **Polled every 10 seconds** via JS `pollHealth()`

## Greeting Short-Circuit
Server-side pattern matcher in `main.py` intercepts obvious greetings/farewells BEFORE they reach any model. Fixes 3B over-triggering tools on simple messages. Model-agnostic, zero latency. Maps ~40 phrases to canned E3N-style responses (e.g., "hey" → "Operational."). Non-greeting messages pass through normally to the model. Greetings are NOT stored in conversation history.

## Ingest Connector Spec
```
POST /ingest
{
  "source": "lmu-telemetry",        // identifies the sender
  "context": "Lap 15: FL tire 98C",  // human-readable summary
  "ttl": 300,                        // seconds until auto-expiry (0 = permanent)
  "tags": ["race", "tires"]          // optional filtering tags
}

POST /ingest/batch
{
  "items": [
    {"source": "lmu-telemetry", "context": "...", "ttl": 300, "tags": ["race"]},
    ...  // max 100 items per batch
  ]
}
```
- Queued to async ingest pipeline (returns immediately, non-blocking)
- Worker batches items (every 2s or 10 items) for efficient embedding
- Chunks and embeds the context into ChromaDB
- Semantic deduplication: near-duplicates (cosine distance < 0.15) freshen existing entries instead of inserting
- Tagged with source + timestamp
- TTL-expired entries cleaned up periodically (and on every /chat request during active sessions)
- RAG queries pull from all sources (knowledge files + ingested context)
- Recent ingests get a relevance boost in query results
- Batch endpoint: single embedding + ChromaDB call for all items (efficient for high-frequency sources)
- Sources matching `SESSION_SOURCE_PATTERN` auto-activate session mode (GPU protection)
- Items with `"alert"` in tags forwarded to WebSocket clients as real-time notifications

## UI Design Direction
- Militaristic/tactical ops center aesthetic — NOT neon sci-fi
- Muted grey-green palette: accent `#5a8a7a`, amber `#c49a3a`, danger `#a03030`
- Dot-grid background, targeting reticle SVG, military designator codes
- Particle sphere = network node map with labeled subsystems
- Fonts: Rajdhani (sans) + Share Tech Mono (mono)
- Header: subsystem health monitor (X/8 ONLINE) + cloud budget display ($X/$X)
- Terminal: CLR button + turn counter for conversation history

## Resource Self-Manager
Background task in main.py that runs every 30 seconds, monitoring system resources and taking automatic corrective actions.

**Automatic actions (no AI reasoning needed):**
| Trigger | Action | Cooldown |
|---------|--------|----------|
| VRAM < 1500MB for 2+ readings | Unloads large models (14B during sim), or all-but-smallest under extreme pressure | 60s between actions |
| VRAM < 1000MB with multiple models loaded | Keeps only smallest model loaded | 60s |
| VRAM declining >100MB/s (predictive) | Preemptive model unload before hitting critical threshold | 60s |
| GPU temp >= 80°C | Auto-unload models to prevent thermal throttling | 60s |
| Sim running or VRAM pressure | Sets E3N + Ollama to BELOW_NORMAL_PRIORITY_CLASS (OS process priority) | Per-cycle |
| Idle (no sim, healthy VRAM) | Restores NORMAL_PRIORITY_CLASS | Per-cycle |
| 3 consecutive Ollama connectivity failures | Attempts `ollama serve` auto-restart | 2min between attempts |
| File watcher dies | Auto-restart (capped at 5 restarts/session) | Immediate |
| Session active | Periodic TTL cleanup of expired RAG entries | Rate-limited (10s) |
| Warm memory age > 24hr | Compacts aged ChromaDB entries to cold SQLite tier | ~10min interval |

**Predictive VRAM management:** Sliding 60s window of VRAM readings in `_resource_state["vram_history"]`. `_predict_vram_decline()` calculates rate-of-change. When VRAM is declining faster than 100MB/s, proactively unloads models before hitting critical thresholds. GPU thermal monitoring via pynvml triggers auto-unload at 80°C.

**OS process priority:** Uses psutil to adjust E3N and Ollama process priority classes. BELOW_NORMAL during sim or VRAM pressure ensures zero game FPS impact. Reverts to NORMAL when idle.

**State tracking:** `_resource_state` dict with vram_warnings, vram_history (60s sliding window), ollama_failures, watcher_restarts, actions_taken (last 50 entries).

**Endpoint:** `GET /resources/status` — exposes resource manager + circuit breaker state, plus `vram_decline_rate_mb_s`, `vram_prediction`, `gpu_temp_c`.

## Circuit Breaker (Ollama)
Protects the Ollama inference path from hammering a dead service. Integrated into `_try_ollama_inference`.

- **States:** closed (normal) -> open (blocking) -> half-open (testing) -> closed
- **Threshold:** 3 consecutive connectivity failures opens the circuit
- **Backoff:** 5s -> 10s -> 20s -> 40s -> 60s (max), doubles on each cycle
- **Recovery:** half-open allows 1 test call; success resets to closed with 5s backoff
- **Model errors** (wrong model name, etc.) do NOT trigger the circuit breaker — only connectivity failures do

## Self-Diagnostic Tools (AI-Driven)
4 tools the AI model can call to inspect and repair its own subsystems:

| Tool | Purpose |
|------|---------|
| `self_diagnostics(subsystem?)` | Full sweep or deep-check one of 7 subsystems (ollama, chromadb, gpu, voice, watcher, backup, paths). Returns actionable fix suggestions. |
| `manage_ollama_models(action, model?)` | status (list loaded + available + VRAM), unload (free VRAM), preload (with sim-guard blocking 14B during racing + VRAM sufficiency check) |
| `repair_subsystem(repair)` | 4 allowlisted repairs: restart_watcher, clear_vram, reindex_knowledge, check_ollama |
| `resource_status()` | Current VRAM pressure/tier, loaded models, circuit breaker state, auto-recovery action log, sim/session detection |

**AI reasoning chain example:**
1. E3N calls `self_diagnostics()` -> sees "GPU: Tier C, 1200MB free"
2. E3N calls `manage_ollama_models('status')` -> sees 14B loaded during sim
3. E3N calls `manage_ollama_models('unload', model='e3n-qwen14b')` -> frees VRAM
4. E3N calls `self_diagnostics(subsystem='gpu')` -> confirms Tier B restored

## Health Event Bus
Lightweight pub/sub for subsystem state changes. Pushes events to frontend via WebSocket.

- **Sync emitter:** `_record_health_event(type, data)` — safe for sync contexts (circuit breaker)
- **Async emitter:** `_emit_health_event(type, data)` — records + broadcasts to WebSocket clients
- **Event types:** `subsystem_changed`, `vram_tier_changed`, `circuit_breaker_changed`, `sim_state_changed`, `model_loaded`, `model_unloaded`, `repair_attempted`, `self_heal_action`
- **Storage:** `_health_events` deque (maxlen=100)
- **Endpoints:** `WebSocket /ws/health` (real-time push), `GET /health/events?limit=50` (REST)
- **Wired into:** circuit breaker state transitions, resource manager actions, model lifecycle, self-heal loop

## Proactive Model Lifecycle
Automatic model swap on sim start/stop transitions. Integrated into `_resource_manager_loop`.

- **Sim starts:** Unload 14B from VRAM, preload 3B (if VRAM >= 2500MB)
- **Sim stops:** Wait ~30s for cleanup, then preload 14B (if VRAM >= 9000MB / Tier A)
- **State tracking:** `_resource_state["last_sim_state"]`, `sim_stop_detected_at`, `pending_14b_preload`
- **Events emitted:** `sim_state_changed`, `model_loaded`, `model_unloaded`

## AI Self-Heal Loop
Background task that periodically runs diagnostics, has the LLM reason about issues, and executes repairs.

- **Interval:** 5 minutes (configurable via `SELF_HEAL_INTERVAL_SEC`)
- **Model:** Always uses smallest (e3n-qwen3b) to avoid VRAM contention
- **Skip conditions:** chat active, racing session, circuit breaker open
- **Short-circuit:** if diagnostics show "No issues detected", skips LLM call entirely
- **LLM prompt:** constrained to output `NO_ACTION` or `REPAIR:<name>` with fuzzy fallback
- **Available repairs:** `restart_watcher`, `clear_vram`, `reindex_knowledge`, `check_ollama`
- **Verification:** re-runs diagnostics after repair to confirm fix
- **Safety:** blocks `clear_vram` during sim, respects all existing safety guards
- **Endpoint:** `GET /self-heal/status` — enabled, interval, model, idle state, recent actions

**Config (.env):**
```
SELF_HEAL_ENABLED=true
SELF_HEAL_INTERVAL_SEC=300
```

## Dynamic Quantization Tiers
Router auto-selects the best quantization variant for each model based on available VRAM.

- **Lookup tables:** `_QUANT_TIERS_14B` and `_QUANT_TIERS_3B` in `router.py`
- **14B variants** (best to worst quality):
  - Idle (>10GB): Q6_K (~10GB) — best quality
  - Normal (>9GB): Q4_K_M (~8.5GB) — current default
  - Moderate (>6.5GB): Q3_K_M (~6.5GB) — good quality
  - Below: falls through to 3B tier
- **3B variants:**
  - Normal (>2.5GB): Q4_K_M (~2.5GB) — current fallback
  - Critical (<2.5GB): Q2_K (~1.2GB) — emergency
- **Prerequisite:** Quantization variants must be pre-registered in Ollama as separate models

**Config (.env):**
```
E3N_STRONG_MODEL_Q6=e3n-qwen14b-q6
E3N_STRONG_MODEL_Q3=e3n-qwen14b-q3
E3N_MODEL_Q2=e3n-qwen3b-q2
```

## ReAct Reasoning Loop
Structured Think-Act-Observe protocol for complex local queries that benefit from multi-step reasoning.

- **Protocol:** Max 3 iterations of THINK (reason about what to do) → ACT (call a tool) → OBSERVE (process result) → repeat or FINAL
- **Convergence:** Model outputs `FINAL:` when it has enough information to answer
- **Eligibility:** Activated for Tier 2/3 complexity queries via `_is_react_eligible(decision)` — simple lookups skip the loop
- **Integration:** Built into `stream_local()` in main.py
- **Stream events:** `{"t":"react","iteration":N,"phase":"think|act|observe","c":"..."}` plus standard tool/result events
- **Fallback:** If max iterations reached without convergence, synthesizes answer from gathered observations
- **Confidence-aware:** System prompt extended with `CONFIDENCE: HIGH/MEDIUM/LOW` protocol. HIGH → FINAL answer. MEDIUM → answer with confidence tag. LOW → `CLARIFY:` asks user for more info. `CLARIFY:` parsed into `{"t":"clarify","c":"..."}` stream event
- **Reasoning trace memory:** Past ReAct chains persisted in SQLite `reasoning_traces` table. `find_similar_traces()` retrieves relevant past reasoning via embedding similarity and injects as `[PRIOR REASONING]` block. 30-day auto-prune, max 500 traces
- **Tool filtering:** `filter_tools()` pre-filters 19 tools to 5-8 per query based on domain. Reduces prompt tokens by ~1000. Tier 1 queries capped at 6 tools

**Config (.env):**
```
REACT_ENABLED=true
REACT_MAX_ITERATIONS=3
REACT_ALLOW_CLARIFY=true
REASONING_TRACE_ENABLED=true
REASONING_TRACE_MAX=500
```

## Iterative RAG (Retrieval-Augmented Reasoning)
Two-round retrieval that identifies knowledge gaps and refines queries automatically.

- **Function:** `iterative_query_knowledge()` in memory.py
- **Round 1:** Standard multi-query expansion (existing `query_knowledge()`)
- **Gap analysis:** `generate_sub_queries()` compares query terms against result terms, identifies gaps, generates up to 3 refined sub-queries targeting missing information
- **Round 2:** Searches with refined sub-queries, deduplicates against Round 1 results, merges and re-sorts by relevance
- **Integration:** Used in `build_rag_context()` in main.py when enabled
- **Backward compatible:** Falls back to standard single-round retrieval when disabled

**Config (.env):**
```
RAR_ENABLED=true
```

## Semantic Deduplication for RAG
Prevents near-duplicate content from bloating ChromaDB during ingestion.

- **Function:** `_check_semantic_duplicate()` in memory.py
- **Mechanism:** Computes cosine distance between new embedding and existing entries; if distance < 0.15, considers it a duplicate
- **On duplicate:** Freshens metadata (timestamp, TTL, tags) on the existing entry instead of inserting a new one
- **Configurable threshold:** `DEDUP_THRESHOLD` env var (default: 0.15)
- **Scope:** Applied to single `/ingest` path; batch ingest processes items individually through the same check
- **Response:** Returns `deduplicated` count alongside `chunks_stored` in the ingest response

**Config (.env):**
```
DEDUP_THRESHOLD=0.15
```

## Mixture-of-LoRA Dynamic Adapter Routing
Routes queries to domain-specific adapter models instead of always using the TIES-merged production model.

- **Function:** `resolve_adapter_model()` in router.py
- **Mechanism:** Uses existing `classify_adapter_domain()` to determine query domain (racing/engineering/reasoning), then checks the adapter registry for an active adapter's Ollama model name for that domain
- **Fallback:** If no domain-specific adapter found or disabled, falls back to standard TIES-merged model
- **Integration:** Called within `route()` function to override model selection
- **Prerequisite:** Requires domain adapters to be trained and registered first

**Config (.env):**
```
MIXTURE_LORA_ENABLED=false
```

## Hierarchical Memory (Hot/Warm/Cold)
Three-tier memory system that balances query speed against storage efficiency.

- **Hot tier:** ChromaDB in-memory collection — entries less than 5 minutes old (live session data, recent ingests)
- **Warm tier:** ChromaDB persistent collection — entries 5 minutes to 24 hours old (recent knowledge, session context)
- **Cold tier:** SQLite with zlib compression — entries older than 24 hours (~80% space reduction vs raw storage)
- **Compaction:** `compact_warm_to_cold()` runs as a background job in the resource manager (~10min interval). Moves aged entries from ChromaDB to cold SQLite via `_demote_to_cold()`
- **Cold search:** `_search_cold_tier()` performs brute-force cosine similarity search as a fallback when hot/warm results are insufficient
- **Cold DB location:** `C:\e3n\data\cold_memory.db`
- **Endpoints:**
  - `POST /memory/compact` — manually trigger warm-to-cold compaction
  - `GET /memory/cold/stats` — cold tier statistics (entry count, DB size, compression ratio)

**Config (.env):**
```
COLD_MEMORY_DB=C:\e3n\data\cold_memory.db
WARM_TO_COLD_AGE_SEC=86400
```

## Streaming Async Ingest Pipeline
Non-blocking ingestion with queue-based batching for high-throughput external data sources.

- **Worker:** `_ingest_pipeline_worker()` — background asyncio task that processes queued ingest items
- **Non-blocking:** `/ingest` returns immediately after queuing the item (no waiting for embedding/storage)
- **Batching:** Worker flushes every 2 seconds or when 10 items accumulate, whichever comes first — uses efficient batch embedding
- **Backpressure:** Queue bounded at 500 items; returns HTTP 429 when full to signal the sender to slow down
- **Metrics endpoint:** `GET /ingest/pipeline/status` — returns `queue_depth`, `queued`, `processed`, `errors`, `avg_latency_ms`

**Config (.env):**
```
INGEST_QUEUE_MAX=500
INGEST_FLUSH_INTERVAL=2.0
INGEST_FLUSH_BATCH_SIZE=10
```

## S3 Nightly Backup
Automated backup of E3N's persistent data to AWS S3 with incremental uploads and retention.

- **Script:** `scripts/backup_s3.py` — full backup, dry-run, restore, list backups
- **Scheduler:** `scripts/setup_backup_task.ps1` — registers Windows Task Scheduler job at 3 AM daily
- **Incremental:** MD5 hash comparison — skips files that haven't changed since last backup
- **Retention:** Auto-deletes backups older than `E3N_S3_RETENTION` days (default 30)
- **What's backed up:** `data/chromadb/`, `data/sqlite/`, `data/knowledge/`, `data/training/`, `data/cold_memory.db`
- **Max file size:** 500MB per file (skips large GGUF exports)
- **Restore:** `python scripts/backup_s3.py --restore 2026-03-24` — downloads all files back to `C:\e3n\data\`

**Config (env vars):**
```
E3N_S3_BUCKET=your-bucket-name   # required
E3N_S3_PREFIX=e3n-backups
E3N_S3_REGION=us-east-1
E3N_S3_RETENTION=30              # days (0 = keep forever)
E3N_DATA_PATH=C:\e3n\data
```

## Response Quality Scoring
Heuristic multi-signal scorer that rates every response 0.0-1.0. Drives smart capture, routing feedback, and knowledge gap detection.

- **File:** `C:\e3n\project\quality.py`
- **6 signals:** `length_appropriateness` (response length vs query complexity), `tool_success_rate` (tools called vs succeeded), `specificity` (concrete details, numbers, names), `no_error_indicators` (absence of error/fallback language), `structural_match` (format matches query type), `no_repeat` (absence of repetitive phrases)
- **Persistence:** Scores stored in SQLite `quality_scores` table with query, model, domain, and timestamp
- **Consumers:** Smart auto-capture (quality threshold for dataset inclusion), adaptive routing feedback (quality per model/tier), knowledge gap detection (low scores trigger gap logging)

**Config (.env):**
```
QUALITY_CAPTURE_THRESHOLD=0.6
```

## Tool Relevance Filtering
Pre-filters the 19-tool inventory down to 5-8 tools per query based on domain classification.

- **Function:** `filter_tools()` in `tools/definitions.py`
- **Domain sets:** Racing (telemetry tools + search_knowledge + calculate), Engineering (calculate + solve_math + lookup_reference + search_knowledge), System (self_diagnostics + manage_ollama_models + repair_subsystem + resource_status), General (core tools only)
- **Tier cap:** Tier 1 (simple) queries capped at 6 tools maximum
- **Token savings:** ~1000 prompt tokens saved per query by omitting irrelevant tool schemas
- **Integration:** Called in `stream_local()` and ReAct loop before building the tool prompt

## Reasoning Trace Memory
Persists ReAct Think/Act/Observe chains so E3N can learn from past multi-step reasoning.

- **Storage:** SQLite `reasoning_traces` table — query, domain, full chain (Think/Act/Observe steps), confidence level, quality score, embedding
- **Retrieval:** `find_similar_traces()` computes embedding similarity against past traces, returns top matches
- **Injection:** Retrieved traces injected as `[PRIOR REASONING]` block into the ReAct system prompt, giving the model a head start on similar problems
- **Pruning:** 30-day auto-prune, max 500 traces (oldest removed first)
- **Confidence:** Traces store the confidence level (HIGH/MEDIUM/LOW) from the ReAct run that generated them

**Config (.env):**
```
REASONING_TRACE_ENABLED=true
REASONING_TRACE_MAX=500
```

## Confidence-Aware Reasoning
Extension to the ReAct system prompt that adds self-assessed confidence levels.

- **Protocol:** Model outputs `CONFIDENCE: HIGH`, `CONFIDENCE: MEDIUM`, or `CONFIDENCE: LOW` alongside its reasoning
- **HIGH:** Proceeds directly to `FINAL` answer
- **MEDIUM:** Answers but includes a confidence qualifier in the response
- **LOW:** Outputs `CLARIFY: <question>` to ask the user for more information before proceeding
- **Stream event:** `CLARIFY:` parsed into `{"t":"clarify","c":"..."}` for the frontend to display as a distinct clarification request
- **Trace storage:** Confidence level persisted in reasoning traces for trend analysis

**Config (.env):**
```
REACT_ALLOW_CLARIFY=true
```

## Adaptive Routing Feedback
Quality scores feed back into the router to adjust model/tier selection based on recent outcomes.

- **Storage:** SQLite `routing_feedback` table tracks quality scores per model, tier, and domain
- **Adjustment:** `get_routing_adjustment(domain)` analyzes recent quality scores and returns a tier shift of -1, 0, or +1
- **Integration:** `classify_complexity()` accepts the adjustment as a soft modifier — never forces Tier 3 (cloud)
- **Cache:** Adjustment results cached for 10 minutes to avoid repeated DB queries
- **Distillation link:** Routing feedback data feeds into iterative distillation to identify which domains need targeted improvement

## Smart Auto-Capture
Quality-tiered training data collection that replaces the basic `auto_capture()`.

- **Function:** `smart_auto_capture()` in `training.py`
- **Quality tiers:**
  - Score >= 0.8: Always captured (high-quality examples)
  - Score 0.6-0.8: 50% probability capture (decent examples, avoids dataset bloat)
  - Score < 0.3: Captured as negative example in `{dataset}-negative` dataset (for future DPO training)
- **Deduplication:** Hash-based dedup over recent 200 queries — prevents near-identical examples from flooding the dataset
- **Negative examples:** Stored separately for Directed Preference Optimization training workflows

**Config (.env):**
```
SMART_CAPTURE_ENABLED=true
CAPTURE_DEDUP_THRESHOLD=0.15
```

## Iterative Distillation Pipeline
Automated weakness detection and targeted knowledge distillation for continuous model improvement.

- **Weakness identification:** `identify_weak_domains()` queries quality scores to find domains with average quality below 0.6
- **Targeted distillation:** `distill_targeted()` pulls the worst-performing queries from a weak domain and generates teacher data (14B or cloud) specifically for those failure cases
- **Improvement cycle:** `run_improvement_cycle()` orchestrates the full loop: identify weaknesses → generate targeted distillation data → (optionally train → eval → promote)
- **Endpoints:**
  - `GET /training/weaknesses` — returns domains ranked by weakness (avg quality, query count, worst examples)
  - `POST /training/improve/{domain}` — triggers targeted distillation + optional training for a specific domain

## Knowledge Gap Detection
Tracks queries where E3N fails to provide good answers, enabling targeted knowledge base improvement.

- **Storage:** SQLite `knowledge_gaps` table — query text, domain, failure source, timestamp, resolution status
- **Gap sources:** Low quality scores (< 0.3), ReAct max iterations reached without convergence, RAG queries returning zero results
- **Endpoints:**
  - `GET /knowledge/gaps` — list unresolved knowledge gaps (filterable by domain)
  - `POST /knowledge/gaps/{id}/resolve` — mark a gap as resolved (e.g., after adding knowledge articles)
- **Future:** Background auto-fill with 14B-generated knowledge articles for common gap patterns

**Config (.env):**
```
KNOWLEDGE_GAP_DETECTION=true
```

## Cross-Domain Reasoning Transfer
Multi-domain query support that leverages adapters and reasoning traces across domain boundaries.

- **Multi-domain classification:** `classify_adapter_domains()` returns ALL matching domains sorted by relevance score (not just the primary). `classify_adapter_domain()` wraps it for backward compatibility (returns primary only).
- **Cross-domain traces:** ReAct loop retrieves reasoning traces from all relevant domains for multi-domain queries (e.g., a racing + engineering question pulls traces from both)
- **Dynamic TIES merge weighting:** Adapter merge weights adjusted based on actual query distribution — domains that are queried more frequently get higher merge weight in production GGUF builds

## Current Status
- Phase 1-4 complete (dashboard, RAG, tools, router, cloud client, split workload, budget, voice, training)
- Phase 5 complete (telemetry prep — all 11 issues implemented, verified 50/50, debugged)
- Phase 8 complete (advanced intelligence & resource optimization — 10 features)
- Phase 9 complete (learning & self-improvement — 10 features)
- Dashboard UI polish: DONE (tactical redesign deployed)
- Qwen model migration: DONE (14B primary, 3B sim/default)
- Emergency backup system: DONE (e3n-nemo + e3n as last resort)
- /ingest connector: DONE (external services push context, now with batch + alert forwarding)
- VRAM-aware routing: DONE (Tier A/B/C with sim detection)
- Greeting short-circuit: DONE (3B over-triggering fix)
- Header health monitor: DONE (8-9 subsystems, dynamic telemetry count)
- Split workload: DONE (local tools + cloud reasoning, dormant until API key)
- Conversation history: DONE (10-turn rolling, persistent, session-aware tagging)
- Cloud cost budget: DONE (daily tracking, header display, router enforcement, persistent)
- Cloud tool-use: DONE (anthropic_client.py fully built, dormant until API key)
- Persistence (SQLite): DONE (history + budget + session export survive restarts)
- Smarter RAG: DONE (multi-query expansion, source reranking, recency bias, source/age filtering)
- Training pipeline: DONE (dataset CRUD, export, QLoRA fine-tuning, GGUF export, Ollama registration, A/B eval, few-shot fallback, auto-capture, racing dataset)
- Text-as-tool parser: DONE (balanced brace extraction, Windows paths, Python literals, Ollama arrays)
- Error recovery UX: DONE (tool retry, friendly errors, retry events)
- Voice module: DONE (STT via faster-whisper, TTS via edge-tts, full voice chat loop)
- Session mode: DONE (GPU protection, auto-detect from ingest, manual start/end, session history export)
- Telemetry query classification: DONE (4 categories, priority ordering, prompt templates)
- Racing prompt templates: DONE (coach + engineer modes, injected into both Ollama + Anthropic paths)
- WebSocket alerts: DONE (real-time toast notifications, auto-reconnect, priority levels)
- Batch ingest: DONE (single embedding call, max 100 items, alert forwarding)
- Conditional telemetry tools: DONE (4 tools, loaded only when TELEMETRY_API_URL set)
- Session-aware history: DONE (tagged turns, JSONL export on session end)
- Racing auto-capture: DONE (e3n-racing dataset, RAG context in training input)
- Dynamic health monitor: DONE (conditional 9th telemetry subsystem)
- Resource self-manager: DONE (VRAM lifecycle, auto-unload during sim, Ollama auto-restart, watcher recovery)
- Circuit breaker: DONE (3-failure threshold, exponential backoff 5s-60s, half-open recovery)
- Self-diagnostic tools: DONE (self_diagnostics with 7 deep checks, manage_ollama_models with sim guard, repair_subsystem with 4 allowlisted repairs, resource_status)
- Health event bus: DONE (typed events, sync/async emitters, WebSocket /ws/health, REST /health/events, CB wiring)
- Proactive model lifecycle: DONE (auto-unload 14B on sim start, auto-preload on sim stop after 30s cooldown)
- AI self-heal loop: DONE (5min interval, LLM-driven diagnostics, repair execution, verification, idle/session/CB guards)
- Adapter surgery: DONE (4 domain slots — racing/engineering/personality/reasoning, selective layer freezing, TIES merge, adapter registry + versioning, eval/promotion/rollback, manage_adapters tool, 9 API endpoints)
- Computation delegation tools: DONE (calculate with sandboxed eval, summarize_data, lookup_reference)
- Knowledge distillation: DONE (teacher generation, dataset blending, replay ratio, retention verification)
- Stress tests: PASSED (46/46 verification + 30/30 stress + 29/29 resource mgmt + 34/34 distill = 139/139 total)
- OS process priority management: DONE (BELOW_NORMAL during sim/pressure, NORMAL when idle, zero FPS impact)
- Dynamic GPU layer offloading: DONE (Tier AB partial offload via _calc_num_gpu, num_ctx dynamic sizing)
- Predictive VRAM management: DONE (60s sliding window, decline rate calculation, preemptive unload at >100MB/s, thermal monitoring at 80°C)
- Semantic deduplication: DONE (cosine distance < 0.15, metadata freshening, configurable threshold)
- ReAct reasoning loop: DONE (Think-Act-Observe, max 3 iterations, FINAL convergence, stream events)
- Dynamic quantization tiers: DONE (Q6_K/Q4_K_M/Q3_K_M/Q2_K auto-selection based on VRAM)
- Iterative RAG: DONE (two-round retrieval, gap analysis, sub-query generation, result merging)
- Mixture-of-LoRA routing: DONE (domain-specific adapter selection via resolve_adapter_model, default off)
- Hierarchical memory: DONE (hot/warm/cold tiers, SQLite cold storage with zlib compression, background compaction)
- Streaming async ingest: DONE (queue-based batching, non-blocking /ingest, backpressure at 500 items, metrics endpoint)
- S3 nightly backup: DONE (scripts/backup_s3.py, Task Scheduler setup, incremental MD5, restore, 30-day retention)
- Reasoning trace memory: DONE (SQLite persistence, embedding similarity retrieval, [PRIOR REASONING] injection, 30-day prune, max 500 traces)
- Response quality scoring: DONE (quality.py, 6-signal heuristic scorer, SQLite persistence, drives smart capture + routing feedback + gap detection)
- Tool relevance filtering: DONE (filter_tools() in definitions.py, domain-specific tool sets, Tier 1 cap at 6, ~1000 token savings)
- Confidence-aware reasoning: DONE (HIGH/MEDIUM/LOW protocol in ReAct, CLARIFY stream event, confidence stored in traces)
- Adaptive routing feedback: DONE (SQLite routing_feedback table, quality-driven tier adjustments, 10-min cache, feeds into distillation)
- Smart auto-capture: DONE (quality-tiered capture, hash dedup over 200 queries, negative examples for DPO)
- Iterative distillation pipeline: DONE (identify_weak_domains, distill_targeted, run_improvement_cycle, /training/weaknesses + /training/improve endpoints)
- Cross-domain reasoning transfer: DONE (classify_adapter_domains multi-domain, cross-domain trace retrieval, dynamic TIES merge weighting)
- Conversation-aware smart history: DONE (_get_smart_history, tiered compression — full/compressed/summary, token-budget-driven)
- Knowledge gap detection: DONE (SQLite knowledge_gaps table, gap logging from low quality/ReAct max iter/RAG zero results, /knowledge/gaps endpoints)
- Phase 10 complete (continuous intelligence — daily training scheduler, telemetry/audio adapters, DPO, session synthesis, 5 new datasets, CoT modelfile protocol)
- Daily autonomous training: DONE (scripts/daily_training.py, Task Scheduler at 2 AM, rotating weekly curriculum)
- Phase 11 complete (web training data collector — Wikipedia HF streaming, arXiv, OpenF1, Semantic Scholar, motorsport web, fetch_url tool)
- Web data collector: DONE (project/collector.py, scripts/collect_training_data.py, Phase 0.5 in run_daily_cycle)
- fetch_url tool: DONE (trafilatura + httpx fallback, session guard, registered in CORE_TOOLS)
- 5 new web-collected datasets: DONE (e3n-general-knowledge, e3n-science-knowledge, e3n-arxiv-papers, e3n-openf1-strategy, e3n-web-motorsport)
- Wikipedia streaming: DONE (HF datasets, 2000 articles/night default, checkpointed offset, full 6.7M article coverage over ~3350 runs)
- Telemetry adapter domain: DONE (r=16, 14/36 layers, ADAPTER_DOMAINS expanded to 6)
- Audio adapter domain: DONE (r=16, 16/36 layers, engine knock/bearing/brake acoustic signatures)
- DPO training: DONE (start_dpo_training() via trl.DPOTrainer, consumes smart_auto_capture negatives)
- Session knowledge synthesis: DONE (synthesize_session_knowledge(), 14B teacher, permanent RAG ingest)
- New datasets (5): DONE (e3n-telemetry-analysis, e3n-audio-analysis, e3n-eng-simulations, e3n-strategy-advanced, e3n-cot-reasoning)
- Modelfile Protocol 7: DONE (structured CoT THINK/REASON/FINAL for complex problems)
- Modelfile telemetry/audio domains: DONE (E3N-qwen14b.modelfile updated with sensor correlation + acoustic diagnosis)
- Router domain patterns: DONE (telemetry + audio pattern sets in ADAPTER_DOMAIN_PATTERNS)
- Future (separate projects): Telemetry API for LMU

## Build Phases
| Phase | Status | Scope |
|-------|--------|-------|
| 1 | DONE | Dashboard, RAG, tools, router, cloud client |
| 2 | DONE | VRAM-aware routing, Qwen migration, sim detection, /ingest, emergency backup |
| 3 | DONE | Cloud tool-use, split workload, cost budget, conversation history, header health monitor, SQLite persistence |
| 4 | DONE | Smarter RAG, training pipeline execution, text-as-tool hardening, error recovery UX, voice module (STT/TTS) |
| 5 | DONE | Telemetry prep — session mode, GPU protection, query classification, racing prompts, RAG recency tuning, telemetry tools, WebSocket alerts, session-aware history, batch ingest, dynamic health, racing auto-capture |
| 6 | DONE | Self-management — resource self-manager, circuit breaker, self-diagnostic tools (4), auto-recovery watchdog, VRAM lifecycle management |
| 7 | DONE | Adapter surgery — modular domain LoRA (racing/engineering/personality/reasoning), selective layer freezing, TIES merge, adapter registry + versioning, eval/promotion/rollback, computation delegation tools, knowledge distillation |
| 8 | DONE | Advanced intelligence & resource optimization — OS process priority, dynamic GPU layer offloading (Tier AB), predictive VRAM management, semantic deduplication, ReAct reasoning loop, dynamic quantization tiers, iterative RAG, Mixture-of-LoRA routing, hierarchical memory (hot/warm/cold), streaming async ingest pipeline |
| 9 | DONE | Learning & self-improvement — reasoning trace memory, response quality scoring, tool relevance filtering, confidence-aware reasoning, adaptive routing feedback, smart auto-capture (quality-tiered + DPO negatives), iterative distillation pipeline, cross-domain reasoning transfer, conversation-aware smart history, knowledge gap detection |
| 10 | DONE | Continuous intelligence — daily autonomous training scheduler (2 AM Task Scheduler), telemetry + audio adapter domain slots (6 total), DPO training (trl.DPOTrainer), session end knowledge synthesis, 5 new training datasets, CoT Protocol 7 in modelfile, router domain patterns for telemetry + audio |
| 11 | DONE | Web training data collector — Wikipedia HF datasets streaming (6.7M articles, checkpointed), arXiv XML API (8 categories), OpenF1 race strategy API, Semantic Scholar papers, trafilatura web scraping. Phase 0.5 in nightly cycle. fetch_url chat tool. 5 new web datasets. |
| — | SEPARATE PROJECT | Telemetry API (LMU race engineer) — connects to E3N via /ingest |

## Known Issues
1. ~~Text-as-tool fallback needed~~ — FIXED: robust parser with balanced brace extraction, Windows path handling, multiple format support
2. 3B model occasionally fires unnecessary tools for non-greeting messages (greeting short-circuit covers the common cases)
3. No Anthropic API key — cloud tier fully built but dormant. **IMPORTANT:** Set `ANTHROPIC_API_KEY` in .env before first race session — cloud is the only safe inference path when GPU is protected during racing.
4. Ethan has $200/mo Claude Max but NO separate API key (obtain from console.anthropic.com)
5. VRAM tier dynamically shifts between A and B as Ollama loads/unloads models — correct behavior, not a bug
6. ~~Conversation history is ephemeral~~ — FIXED: now persisted to SQLite
7. ~~Cloud budget resets on restart~~ — FIXED: now persisted to SQLite
8. Voice STT loads Whisper model on first use — ~2-5 second cold start on first transcription
9. ~~LoRA training requires external deps~~ — INSTALLED: PyTorch 2.10.0+cu126, transformers 5.3, peft 0.18, trl 0.29, bitsandbytes 0.49, accelerate 1.13, datasets 4.8. llama.cpp cloned and built at `C:\e3n\tools\llama.cpp`. Full pipeline verified (46/46 tests). Only Qwen2.5-3B is trainable on 12GB VRAM.
10. Session mode without API key falls back to CPU-only local inference — functional but slow. Cloud is strongly recommended for racing sessions.
11. Dynamic quantization tiers require pre-registering Q6_K/Q3_K_M/Q2_K variants in Ollama — without them, router falls back to default Q4_K_M models.
12. Mixture-of-LoRA routing is disabled by default (`MIXTURE_LORA_ENABLED=false`) — requires domain adapters to be trained first via adapter surgery.
13. Cold tier search (`_search_cold_tier`) uses brute-force cosine similarity — acceptable for archive queries but slower than ChromaDB for large cold stores.
