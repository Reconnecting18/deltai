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

## Cursor-specific

- **Project rules:** [`.cursor/rules/`](.cursor/rules/) (short `.mdc` files). Keep them actionable; put narrative detail in `CLAUDE.md`, not in rules.
- **Debug / launch:** Prefer [`.vscode/launch.json`](.vscode/launch.json) (workspace-relative). [`.claude/launch.json`](.claude/launch.json) is for Claude Code; adjust paths there if defaults do not match your machine.

## Doc maintenance

When you change user-visible behavior, configuration, or architecture:

1. Update [CLAUDE.md](CLAUDE.md) (sections that apply).
2. Update [README.md](README.md) if contributors or users need to know.
3. Keep [AGENTS.md](AGENTS.md) accurate if onboarding steps or boundaries shift.

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues and PR expectations.
