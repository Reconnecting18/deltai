# Contributing to deltai

## Issues

- Use [GitHub Issues](https://github.com/Reconnecting18/deltai/issues) with the provided templates (bug report or feature request).
- For bugs, include: OS/distro, Python version, Ollama version, whether you're running as a systemd user service, and full error output.

## Pull requests

- Prefer a focused branch and a clear commit message (`feat:`, `fix:`, `refactor:`, `docs:`).
- Link related issues with `Closes #N` in the commit message or PR description.
- Do not commit `project/.env`, credentials, or anything under `data/`.
- Keep changes consistent with the [Linux philosophy principles](AGENTS.md#non-negotiable-boundaries) described in AGENTS.md.
- **Core vs personal branches:** use `main` for upstream-shared work and PRs from `feature/*`. A maintainer-style `personal` branch is for private overlays (e.g. force-added extensions); never bulk-merge it into `main`. Details: [docs/git-workflow.md](docs/git-workflow.md).

## Checks before you open a PR

From `project/` with the virtual environment active:

```bash
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
```

Run `python tests/verify_distill.py` when you change distillation or related training code.

## Documentation and agents

- **User-facing or structural changes:** update [README.md](README.md) and [CLAUDE.md](CLAUDE.md) in the same PR.
- **Onboarding, boundaries, or verification commands:** update [AGENTS.md](AGENTS.md).
- **New agent conventions:** add or adjust rules under [`.cursor/rules/`](.cursor/rules/) (keep them short; put narrative detail in `CLAUDE.md`).
