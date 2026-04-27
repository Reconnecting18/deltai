# deltai — Agent Onboarding

Short context for Cursor, Copilot, and other coding agents. **Full architecture, endpoints, stream protocol, and development workflow live in [CLAUDE.md](CLAUDE.md).** Update that file when behavior or structure changes.

## What this is

deltai is a modular AI extension system for Linux. It runs as a **systemd user service** (no root required), exposes an HTTP/WebSocket API on `localhost:8000`, and provides: local LLM routing (Ollama), RAG memory (ChromaDB), a structured tool system, task automation, and a plugin/ingest API for external services.

It is the open, user-controlled Linux answer to Copilot+Windows — built around Linux philosophies: user-space first, user choice, modularity, transparency, no lock-in.

## Non-negotiable boundaries

- **Never require root.** deltai runs entirely in user space. No `sudo`, no system-wide writes.
- **No domain-specific ingest.** Never add telemetry parsers, game data readers, or domain pipelines to the core. External services push context via `POST /ingest`.
- **Linux paths only.** No `C:\...` Windows paths anywhere in code or config.
- **User choice.** Every feature is opt-in. Defaults must be safe and minimal.
- **No lock-in.** Local models are the default. Cloud is dormant until explicitly configured.
- **Network exposure:** Run the API on `127.0.0.1` unless you understand the risk: there is no default auth on `/chat` or tools. Optional **`DELTAI_INGEST_API_KEY`** locks ingest-related routes; see [CLAUDE.md](CLAUDE.md) Security posture.

## Repository paths

| Area | Path |
|------|------|
| Backend (FastAPI daemon) | `project/` |
| **Extensions** (personal or domain-specific features; keep core minimal) | `project/extensions/` — see [project/extensions/README.md](project/extensions/README.md) |
| Desktop shell (optional) | `app/` |
| Ollama modelfiles | `modelfiles/` |
| systemd user unit | `systemd/user/` |
| Runtime data (gitignored) | `data/` |
| Scripts | `scripts/` |

Domain-specific automation (e.g. distro maintenance assistants) belongs in `project/extensions/`, not in `project/main.py` or core tool definitions, unless it is a generic primitive reused everywhere.

## How to run (development)

```bash
# Backend (run from project/ — main.py lives here)
cd deltai
python -m venv venv && source venv/bin/activate
pip install -e .[dev]
cd project
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Desktop shell (optional)
cd app && npm install && npm start
```

The main package metadata now includes ChromaDB, so `pip install .` covers
runtime installs and `pip install -e .[dev]` covers local development.

Or as a systemd user service (unit file in-repo: `systemd/user/delta-daemon.service`):
```bash
cp systemd/user/delta-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now delta-daemon
journalctl --user -u delta-daemon -f
```

## Verify after substantive backend changes

```bash
cd project && source venv/bin/activate
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
# If you touched training/distillation:
python tests/verify_distill.py
```

## Doc maintenance

When you change user-visible behavior, configuration, or architecture:

1. Update [CLAUDE.md](CLAUDE.md) — the primary context for all AI sessions.
2. Update [README.md](README.md) — user-facing changes, endpoints, config.
3. Keep [AGENTS.md](AGENTS.md) accurate — onboarding, boundaries, verify commands.
4. If you change RAG, ingest, models, or adapters, update [docs/local-model-workflow.md](docs/local-model-workflow.md).
5. If you change maintainer branching or extension tracking policy, update [docs/git-workflow.md](docs/git-workflow.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues and PR expectations.
