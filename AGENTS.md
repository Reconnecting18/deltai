# E3N — Agent onboarding

Concise context for Cursor and other coding agents. **Deep architecture, endpoints, stream protocol, and current status live in [CLAUDE.md](CLAUDE.md).** Update that file when behavior or structure changes.

## What this is

E3N is a local AI stack (FastAPI + optional Electron UI) with VRAM-aware routing, RAG (ChromaDB), tools, and training-related code. Personality and product framing: a “race engineer brain” for Le Mans Ultimate, but **E3N is only the intelligence layer**—not a telemetry or game client.

## Non-negotiable boundary

- **Do not** add UDP/game parsers, live telemetry pipelines, or domain-specific ingest logic inside this repo’s core.
- **Do** expose integration via **`POST /ingest`** (and related APIs) so external services push summarized context into RAG.

## Paths (clones vs. author machine)

Docs may mention `C:\e3n\`; that is the primary author layout. **In this repo, use paths relative to the git root:**

| Area | Path |
|------|------|
| Backend (FastAPI) | `project/` |
| Desktop shell | `app/` |
| Ollama modelfiles | `modelfiles/` |
| Runtime data (gitignored) | `data/` |

## Run (typical)

From repo root, after venv and deps (see [README.md](README.md)):

**Terminal 1 — API**

```powershell
cd project
.\venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Electron (optional)**

```powershell
cd app
npm start
```

Or open `http://127.0.0.1:8000` in a browser for the dashboard without Electron.

## Verify after substantive backend changes

From `project/` with venv active:

```powershell
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
```

When you touch distillation-related code, also run `python tests/verify_distill.py` (see [CLAUDE.md](CLAUDE.md) Key Files for the full test list).

### Phase 10 — Daily Training Verification

After changing `training.py`, `router.py`, or anything in the training pipeline:

```powershell
# Verify new adapter domains are registered
python -c "from training import ADAPTER_DOMAINS; print(ADAPTER_DOMAINS)"
# Expected: ['racing', 'engineering', 'personality', 'reasoning', 'telemetry', 'audio']

# Verify router domain patterns include telemetry + audio
python -c "from router import ADAPTER_DOMAIN_PATTERNS; print(list(ADAPTER_DOMAIN_PATTERNS.keys()))"
# Expected: ['racing', 'engineering', 'reasoning', 'telemetry', 'audio']

# Dry-run the daily training cycle (no server required)
python scripts/daily_training.py --dry-run

# View the report
python scripts/daily_training.py --report-only
```

### Phase 11 — Web Collector Verification

After changing `collector.py` or any web collection source:

```powershell
# Verify new web-collected datasets are in DATASET_DOMAIN_MAP
python -c "from training import DATASET_DOMAIN_MAP; web = [k for k in DATASET_DOMAIN_MAP if 'general' in k or 'arxiv' in k or 'openf1' in k or 'science' in k or 'motorsport' in k]; print(web)"
# Expected: ['e3n-general-knowledge', 'e3n-science-knowledge', 'e3n-arxiv-papers', 'e3n-openf1-strategy', 'e3n-web-motorsport']

# Verify fetch_url is registered in the tool executor
python -c "from tools.executor import EXECUTORS; print('fetch_url' in EXECUTORS)"
# Expected: True

# Dry-run the full collector (no writes)
python scripts/collect_training_data.py --dry-run --report

# Run a single source dry-run
python scripts/collect_training_data.py --source arxiv --dry-run

# Dry-run training cycle with collection phase
python scripts/daily_training.py --dry-run --collect

# Collect-only (no training, uses .env source flags)
python scripts/daily_training.py --collect-only --dry-run
```

## Cursor-specific

- **Project rules:** [`.cursor/rules/`](.cursor/rules/) (short `.mdc` files). Keep them actionable; put narrative detail in `CLAUDE.md`, not in rules.
- **Debug / launch:** Prefer [`.vscode/launch.json`](.vscode/launch.json) (workspace-relative). [`.claude/launch.json`](.claude/launch.json) is for Claude Code; adjust paths there if defaults do not match your machine.

## Doc maintenance

When you change user-visible behavior, configuration, or architecture:

1. Update [CLAUDE.md](CLAUDE.md) (sections that apply).
2. Update [README.md](README.md) if contributors or users need to know.
3. Keep [AGENTS.md](AGENTS.md) accurate if onboarding steps or boundaries shift.
4. If you change RAG, ingest, training capture, modelfiles, or adapter flows, align [docs/local-model-workflow.md](docs/local-model-workflow.md) so the operator cadence stays accurate.

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues and PR expectations.
