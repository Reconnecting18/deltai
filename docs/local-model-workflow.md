# Local Qwen improvement workflow

Operator guide for pushing **local** Qwen quality without changing model architecture inside Ollama. E3N already wires **RAG**, **ingest**, **quality → capture → training**, and **modelfiles**; this doc sequences them into a repeatable cadence.

**Paths:** Replace `C:\e3n\` with your repo’s `data/` layout if you use a clone. Env defaults live in [project/memory.py](../project/memory.py) (`KNOWLEDGE_PATH`, `CHROMADB_PATH`, `OLLAMA_URL`).

---

## How the three pillars connect

1. **RAG + ingest** — Ground answers in files and live summaries (ChromaDB).
2. **Behavior / reliability** — System prompt, `.env` toggles, and inference hygiene.
3. **Domain adaptation** — Datasets from real chats, distillation, LoRA adapters, GGUF back to Ollama.

Low-quality turns feed **knowledge gaps** and **negative capture**; fixing knowledge or training closes the loop.

---

## Phase A — RAG and ingest (continuous)

### A.1 Knowledge files (`KNOWLEDGE_PATH`)

- **Location:** `KNOWLEDGE_PATH` (default `C:\e3n\data\knowledge` on the primary machine).
- **Watcher:** On FastAPI startup, `ingest_all` runs once, then [project/watcher.py](../project/watcher.py) watches the directory recursively (debounce ~1s per file).
- **Supported extensions** (from [project/memory.py](../project/memory.py)): `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.csv`, `.log`, `.bat`, `.ps1`, `.sh`, `.c`, `.cpp`, `.h`, `.rs`, `.go`, `.java`.
- **Limits:** Max **5 MB** per file; empty files skipped; unchanged files skipped via `file_hash`.
- **Chunking:** ~512 characters per chunk with overlap (tune in `memory.py` only if you know the tradeoff).

**Practice:** One topic per note, clear headings, concrete numbers and units. Large references split into multiple files so chunks stay coherent.

### A.2 Live context (`POST /ingest`, `POST /ingest/batch`)

External services (or scripts) push **short, human-readable** summaries — not raw telemetry dumps.

| Field | Role |
|--------|------|
| `source` | Stable id (e.g. `lmu-telemetry`, `session-notes`) |
| `context` | Text to embed and retrieve |
| `ttl` | Seconds until expiry; `0` = keep until manually managed / cold tier rules |
| `tags` | Optional; include `alert` to forward to WebSocket toasts |

**Single item:**

```http
POST http://127.0.0.1:8000/ingest
Content-Type: application/json

{
  "source": "lmu-session",
  "context": "Stint 3: FL/FR mediums, 12 laps, fuel +8s margin. Track temp dropping — watch rear grip T14.",
  "ttl": 600,
  "tags": ["race", "strategy", "stint"]
}
```

**Batch (max 100 items):** `POST /ingest/batch` with `{"items": [ {...}, ... ]}`.

**Operators:** Check queue health with `GET /ingest/pipeline/status`. TTL cleanup: `POST /ingest/cleanup`.

### A.3 Retrieval tuning (when answers miss context)

- **Iterative RAG:** Enable `RAR_ENABLED=true` in `project/.env` (two-round retrieval with sub-queries) — see CLAUDE.md.
- **Deduplication:** `DEDUP_THRESHOLD` for near-duplicate ingest (memory layer).
- **Code path:** `query_knowledge()` supports `source_filter` and `max_age_sec` for session-scoped or fresh-only context (used from internal call sites).

### A.4 Knowledge gaps (weekly or after bad sessions)

- **List:** `GET http://127.0.0.1:8000/knowledge/gaps`
- **Resolve after you add a knowledge article or fix ingest:** `POST http://127.0.0.1:8000/knowledge/gaps/{id}/resolve`

Gaps are logged from low quality scores and related signals (see [project/main.py](../project/main.py) and persistence layer).

### A.5 Hierarchical memory (optional maintenance)

- **Compaction:** `POST /memory/compact` (warm → cold).
- **Stats:** `GET /memory/cold/stats`.

---

## Phase B — Behavior and reliability (iterate as needed)

### B.1 Modelfiles and Ollama rebuild

- **Files:** [modelfiles/E3N-qwen14b.modelfile](../modelfiles/E3N-qwen14b.modelfile), [modelfiles/E3N-qwen3b.modelfile](../modelfiles/E3N-qwen3b.modelfile) (and backups: nemo / legacy as applicable).
- **Rule:** Keep system prompts aligned across modelfiles; only `FROM` / parameters differ (3B may use a condensed system prompt).
- **Rebuild after edits:**

```powershell
ollama create e3n-qwen14b -f modelfiles\E3N-qwen14b.modelfile
ollama create e3n-qwen3b -f modelfiles\E3N-qwen3b.modelfile
```

Use repo-relative paths on clones: `ollama create e3n-qwen14b -f .\modelfiles\E3N-qwen14b.modelfile`.

### B.2 `.env` toggles (project/.env)

| Area | Examples (see CLAUDE.md for full list) |
|------|----------------------------------------|
| ReAct / reasoning | `REACT_ENABLED`, `REACT_MAX_ITERATIONS`, `REACT_ALLOW_CLARIFY` |
| Traces | `REASONING_TRACE_ENABLED`, `REASONING_TRACE_MAX` |
| RAG | `RAR_ENABLED`, `DEDUP_THRESHOLD`, ingest pipeline vars |
| Quality / capture | `QUALITY_CAPTURE_THRESHOLD`, `SMART_CAPTURE_ENABLED` |
| Smart history | `SMART_HISTORY_ENABLED`, `CONVERSATION_HISTORY_MAX` |
| Session / telemetry prompts | `SESSION_*`, telemetry classification (router) |

Restart uvicorn after `.env` changes.

### B.3 Session and telemetry behavior

When a session is active or the sim is detected, the router applies **telemetry query categories** and injects racing/coach templates. No code change needed — ensure **ingest** keeps session context fresh so RAG + templates agree.

---

## Phase C — Domain adaptation (weekly / scheduled)

### C.1 Automatic capture from chat

After each turn, [project/main.py](../project/main.py) calls `smart_auto_capture()` into dataset **`e3n-auto`** (quality-tiered; poor turns may land in **`e3n-auto-negative`** for DPO). Ensure `SMART_CAPTURE_ENABLED` matches your intent.

### C.2 Weak domains and improvement cycle

1. `GET http://127.0.0.1:8000/training/weaknesses`
2. `POST http://127.0.0.1:8000/training/improve/{domain}` — runs improvement cycle for that domain (server must be up; training deps installed).

### C.3 Nightly / unattended cycle

- **Script:** [scripts/daily_training.py](../scripts/daily_training.py) — dry-run: `python scripts/daily_training.py --dry-run`
- **Scheduler:** [scripts/setup_daily_training_task.ps1](../scripts/setup_daily_training_task.ps1) (Windows).

### C.4 Adapter surgery (domain LoRA slots)

| Step | Endpoint / action |
|------|-------------------|
| List | `GET /adapters`, `GET /adapters/active` |
| Train | `POST /adapters/train` (body per API) |
| Merge | `POST /adapters/merge` (TIES → production GGUF path) |
| Eval | `POST /adapters/eval/{name}` |
| Promote | `POST /adapters/promote/{name}` |
| Rollback | `POST /adapters/rollback` |

Domains include `racing`, `engineering`, `personality`, `reasoning`, `telemetry`, `audio` (see `ADAPTER_DOMAINS` in training).

### C.5 DPO (when negatives exist)

When `e3n-auto-negative` (or paired negatives) has enough matched pairs, enable and run DPO per CLAUDE.md (`DPO_ENABLED`, `/training/start` with `mode=dpo`).

---

## Worked examples

### Example 1 — Good knowledge article outline (save as `KNOWLEDGE_PATH/strategy/stint-fuel.md`)

```markdown
# Stint fuel check — LMU GTE

## When to use
End of stint, before pit window opens.

## Rule of thumb
- Target: cross line + pit lane with ≥8s margin at current consumption.
- If margin <5s: short-shift or lift one sector before pit entry.

## Numbers (example)
- Consumption: 2.8 L/lap; 12 laps = 33.6 L; tank 90 L → safe if >40 L at lap 8.
```

### Example 2 — Ingest JSON (session snippet)

```json
{
  "source": "lmu-telemetry",
  "context": "Lap 24: P2, gap ahead +1.2s. FL inner 102C, FR outer 98C. Rear degradation +0.3s vs stint avg.",
  "ttl": 300,
  "tags": ["race", "tires", "session"]
}
```

### Example 3 — Promote adapter after eval (checklist)

1. `POST /adapters/eval/{name}` — confirm eval score beats baseline in saved output.
2. Backup current production model name in Ollama (note which GGUF is live).
3. `POST /adapters/promote/{name}` if promotion is manual in your flow.
4. If you use merged GGUF: `POST /adapters/merge`, then re-register Ollama model from exported GGUF per training pipeline docs.
5. Smoke-test 5–10 real queries on **14B** and **3B** tiers.

---

## Quick verification

- **Prereqs script:** From repo root, use the project venv so imports resolve (e.g. `project\venv\Scripts\python scripts\check_local_workflow_prereqs.py` on Windows, or activate `project/venv` then `python scripts/check_local_workflow_prereqs.py`). Creates no data; expects `KNOWLEDGE_PATH` to exist (create the folder if the check fails).
- **Backend tests (from `project/`):** `python tests/verify_full.py` after changing RAG or training code.

---

## Related docs

- [CLAUDE.md](../CLAUDE.md) — full architecture and env reference
- [AGENTS.md](../AGENTS.md) — agent onboarding and verify commands
- [README.md](../README.md) — setup and Quick Start
