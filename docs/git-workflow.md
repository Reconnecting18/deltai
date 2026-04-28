# Git workflow: core vs personal

deltai keeps a **small upstream core** and pushes optional features into [`project/extensions/`](../project/extensions/). Git branches mirror that split.

## Branch roles

| Branch | Purpose |
|--------|---------|
| **`main`** | Shareable **lightweight** upstream: core daemon, router, RAG, shared tools, and only **`example_extension`** + **`training`** under `project/extensions/` (see [`.gitignore`](../.gitignore)). Homelab / distro / cloud-bridge packages (e.g. `server_network`, `arch_update_guard`, `appwrite_bridge`) are **not** tracked here — default clone stays minimal. |
| **`feature/*`** | Short-lived work you intend to merge into `main` via PR (e.g. `feature/workstation-cloud-arch`). |
| **`personal`** | Long-lived overlay: merge `main`, then add **your** integrations — optional extensions (`git add -f project/extensions/<name>/`), Cursor rules, etc. Examples: server inventory + SSH tools, Arch update guard, Appwrite bridge. Remove or stop tracking a directory to “uninstall” that overlay. Not merged wholesale into `main`. |

Do **not** merge `personal` → `main` in bulk. Promote changes with cherry-picks or a fresh branch from `main` and a focused PR.

## Sync rules

- **Update `personal` from core:** `git checkout personal && git merge origin/main` (or `git rebase origin/main` if you prefer a linear history—pick one style and keep it).
- **Land shared work:** PR from `feature/*` → `main`, not from `personal` → `main`.
- **Promote a private experiment:** branch from `main`, copy or cherry-pick only what is safe to share, open a PR.

## Where changes belong

| Kind of change | Branch / mechanism |
|----------------|-------------------|
| Core daemon, router, RAG, shared tools, `project/deltai_api/` | **`main`** (via PR) — merge into **`personal`** as usual; keep extensions in `project/extensions/` only |
| Extension you want **everyone** to get | **`main`**: add `!project/extensions/your_pkg/` in [`.gitignore`](../.gitignore) and commit the tree (prefer keeping `main` to core + training only). |
| Extension **only for you** (Arch guard, server network, Appwrite, …) | **`personal`**: `project/extensions/*/` is gitignored by default. On `personal`, `git add -f project/extensions/<name>/` so it versions with your branch; **`main`** stays free of those trees. |
| Secrets, local experiments you never push | [`.git/info/exclude`](https://git-scm.com/docs/gitignore) or local ignore; never commit |

See [project/extensions/README.md](../project/extensions/README.md) for extension authoring.

## Optional: local pre-push guard

A sample hook that blocks accidentally pushing **`personal`** to a remote named **`upstream`** (adjust names to match your remotes):

```bash
cp scripts/git-hooks/pre-push.sample .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Edit the script if your remotes differ. Hooks under `.git/hooks/` are not committed; each clone installs its own.

## Optional: sync alias

```bash
git config alias.sync-personal '!git checkout personal && git merge origin/main'
```
