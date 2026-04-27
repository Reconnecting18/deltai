# `delta.core` — optional daemon plugins

This package provides **`PluginManager`** ([`plugin_manager.py`](plugin_manager.py)): load user-supplied Python modules for **`delta-daemon`** only ([`delta.daemon.app`](../daemon/app.py)).

## How it works

- **Registry:** `extensions.toml` under your XDG config dir (default `~/.config/deltai/extensions.toml`).
- **Modules:** one `.py` file per plugin under `plugins/` in your XDG data dir (default `~/.local/share/deltai/plugins/`).
- **Contract:** each module exposes a **`Plugin`** class with `on_init(core_context)`, `on_shutdown()`, and `get_commands()` — see [`plugin_protocol.py`](plugin_protocol.py).
- **CLI:** `deltai plugin install|unload` updates the registry; restart the daemon so `on_init` runs.

## What this is not

- **Not** a replacement for [`project/extensions/`](../../../project/extensions/README.md) (those load inside the **full** FastAPI app in `project/main.py`).
- **Not** where [Arch Update Guard](../../../project/core/arch_update_guard/) lives; that stack mounts routes and schedulers on the full backend and is configured with `ARCH_GUARD_*` env vars.

Keeping these surfaces separate avoids merge conflicts and wrong assumptions about which process loads which code.
