# Contributing to E3N

## Issues

- Use [GitHub Issues](https://github.com/Reconnecting18/e3n/issues) with the provided templates when possible (bug report or feature / task).
- Describe reproduction steps for bugs, including OS, Python version, and whether Ollama is running.

## Pull requests

- Prefer a focused branch and a clear commit message (`feat:`, `fix:`, `refactor:`, `docs:`).
- Link related issues with `Closes #N` in the commit message or PR description when applicable.
- Do not commit secrets, `project/.env`, or generated data under `data/`.

## Checks before you open a PR

From `project/` with the virtual environment active:

```powershell
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
```

Run `python tests/verify_distill.py` when you change distillation or related training code.

## Documentation and agents

- **User-facing or structural changes:** update [README.md](README.md) and [CLAUDE.md](CLAUDE.md) in the same PR when appropriate.
- **Onboarding, boundaries, or verification commands:** update [AGENTS.md](AGENTS.md) so Cursor and other agents stay aligned.
- **New conventions agents must always follow:** add or adjust rules under [`.cursor/rules/`](.cursor/rules/) (keep them short; put narrative detail in `CLAUDE.md`).
