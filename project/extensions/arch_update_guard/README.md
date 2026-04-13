# Arch update guard (deltai extension)

Optional extension for **Arch Linux** operators who want **evidence-backed** context before upgrades: official news in RAG, optional ArchWiki snippets, and read-only pacman/checkupdates output. It does **not** replace careful reading of Arch News or the official wiki.

## What this is not

- **Not a proof of zero breakage.** No process can guarantee every rolling update is safe. This extension surfaces **structured facts** (pending package list, reverse deps where `pactree` is available, announcement text) so you and the LLM reason from evidence.
- **Not an automatic privileged updater.** deltai stays user-space ([AGENTS.md](../../../AGENTS.md)): no `sudo pacman`, no silent system file edits. Mitigations are checklists, RAG context, and snippets **you** review.

## Two layers

1. **Deterministic** — allowlisted subprocess calls (`checkupdates`, `pacman -Qu`, optional `pactree -ru`) and HTTP fetches (Arch news RSS, ArchWiki API). See [`pacman_audit.py`](pacman_audit.py) and [`news_wiki.py`](news_wiki.py).
2. **Interpretive** — the normal deltai chat loop (local Ollama and/or cloud per your settings) summarizes ingested news and tool JSON. Use **`CLOUD_ENABLED=false`** or the UI/chat **force local** path if you want this workflow local-only.

## HTTP routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/ext/arch_update_guard/health` | Extension loaded |
| GET | `/ext/arch_update_guard/pending` | JSON pending upgrades (`include_reverse_deps`, `reverse_deps_limit`) |
| POST | `/ext/arch_update_guard/refresh-news` | JSON body `{"wiki_query":"","force":false}` — fetch RSS (+ optional wiki), `POST` digest to `http://127.0.0.1:8000/ingest` |

Set **`DELTAI_BASE_URL`** if the daemon listens elsewhere (default `http://127.0.0.1:8000`).

## Chat tools

- **`arch_pending_updates_report`** — JSON report from the same logic as `GET /pending`.
- **`arch_refresh_news_digest`** — same as `POST /refresh-news` (respects a short in-process rate limit unless `force`).

## Optional: systemd user timer

Example user timer that refreshes news daily (adjust paths):

```ini
# ~/.config/systemd/user/arch-news-refresh.service
[Unit]
Description=Refresh Arch news digest into deltai RAG

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -fsS -X POST http://127.0.0.1:8000/ext/arch_update_guard/refresh-news -H 'Content-Type: application/json' -d '{}'
```

## Optional: ALPM hook (example only — not installed by deltai)

If **you** choose to run a root-side hook that POSTs transaction info to localhost, keep it minimal and audit it yourself. Example pattern (install under `/etc/pacman.d/hooks/` only after review):

```ini
[Trigger]
Operation = Upgrade
Type = Package
Target = *

[Action]
Description = Notify deltai before upgrade (example)
When = PreTransaction
Exec = /usr/bin/curl -fsS -X POST http://127.0.0.1:8000/ext/arch_update_guard/refresh-news -H 'Content-Type: application/json' -d '{"force":true}'
```

This runs **as root** during pacman; use only if you accept that trust boundary.

## RSS source

Official Arch Linux news feed: `https://archlinux.org/feeds/news/`

## Dependencies

- **`httpx`** — already a deltai dependency.
- **`pacman-contrib`** (`checkupdates`) recommended on Arch for unprivileged pending-package listing.
