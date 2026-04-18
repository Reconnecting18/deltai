# Training extension

QLoRA / few-shot training, datasets, adapters, distillation, and the **daily autonomous cycle**.

## Layout

| File | Role |
|------|------|
| `pipeline.py` | Main module (formerly `project/training.py`): datasets, LoRA, `run_daily_cycle()`, etc. |
| `build.py` | Dataset / prompt builders (formerly `training_build.py`) |
| `daily_training.py` | CLI entrypoint for the nightly scheduler (also runnable via `scripts/daily_training.py`) |
| `collect_training_data.py` | Standalone web collection runner |
| `tests/fixture_exports/` | Sample export JSON for stress tests |

Public imports use the shim `import training` (see `project/training.py`), which re-exports `pipeline.py`.

## Daily training sessions

Yes — when enabled, a full cycle is implemented in `run_daily_cycle()` in `pipeline.py`:

1. Guards (sim / focus workload, VRAM, training-in-progress)
2. Optional web collection (`collector.run_collection_cycle`)
3. Weakness analysis, targeted distillation, rotating curriculum
4. Domain QLoRA when `auto_train` and enough examples (≥20) — otherwise skipped with reason
5. Knowledge-gap summary, JSON report under `~/…/training/daily_reports/` (via `resolve_under`)

The **offline** runner `daily_training.py` adds pre-checks (`DAILY_TRAIN_ENABLED`, VRAM vs `DAILY_TRAIN_MIN_VRAM_MB`, session API) then calls `run_daily_cycle()`. Schedule it with cron or systemd (e.g. 2:00 AM) per `CLAUDE.md` / root `README.md`.

Do **not** import `pipeline` from `extensions.training.__init__` at extension load time — it would pull torch at startup. The extensions loader only imports this package’s `__init__.py`.
