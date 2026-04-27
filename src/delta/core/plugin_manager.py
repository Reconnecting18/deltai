"""Load, unload, and query optional plugins from a config file and plugin directory."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import tomli_w

from delta.core.plugin_protocol import DeltaPlugin, validate_plugin_instance

logger = logging.getLogger("delta.plugins")

_CONFIG_HEADER = """# Plugin registry for delta-daemon (XDG config).
# Manage with: deltai plugin install|unload <name>
#
# Example:
# [[plugin]]
# name = "example"
# enabled = true
# module = "example"
# auto_start = true

"""


@dataclass
class PluginCoreContext:
    """Stable handle passed to every plugin ``on_init``."""

    settings: object
    app: object
    orchestrator: object
    ipc_server: object


class PluginManager:
    """Explicitly loads plugins listed in ``extensions.toml`` under ``plugin_dir``."""

    def __init__(self, core_context: object, plugin_dir: Path, config_path: Path) -> None:
        self._core_context = core_context
        self._plugin_dir = plugin_dir
        self._config_path = config_path
        self._instances: dict[str, DeltaPlugin] = {}

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def plugin_dir(self) -> Path:
        return self._plugin_dir

    def read_config(self) -> list[dict[str, Any]]:
        """Return the list of ``[[plugin]]`` table rows (may be empty)."""
        if not self._config_path.is_file():
            return []
        raw = self._config_path.read_bytes()
        if not raw.strip():
            return []
        data = tomllib.loads(raw.decode("utf-8"))
        rows = data.get("plugin")
        if rows is None:
            return []
        if not isinstance(rows, list):
            logger.warning("extensions.toml: 'plugin' must be a list of tables — ignoring")
            return []
        return [r for r in rows if isinstance(r, dict)]

    def load_enabled_from_config(self) -> int:
        """Load every row with ``enabled: true``. Returns count of successfully loaded plugins."""
        loaded = 0
        for row in self.read_config():
            if not row.get("enabled", False):
                continue
            name = row.get("name")
            if not name or not isinstance(name, str):
                logger.warning("extensions.toml: plugin entry missing string 'name' — skipping")
                continue
            try:
                if self._load_plugin_entry(name, row):
                    loaded += 1
            except Exception as exc:
                logger.warning("Plugin %r failed to load — skipping: %s", name, exc)
        return loaded

    def _load_plugin_entry(self, name: str, row: dict[str, Any]) -> bool:
        if name in self._instances:
            return False
        module_stem = row.get("module") or name
        if not isinstance(module_stem, str) or not module_stem:
            raise ValueError("plugin entry needs non-empty 'module' or use 'name' as module stem")
        path = self._plugin_dir / f"{module_stem}.py"
        if not path.is_file():
            raise FileNotFoundError(f"plugin file not found: {path}")
        instance = _instantiate_plugin_from_file(path, unique_tag=f"delta_plugin_{name}")
        validate_plugin_instance(instance, label=f"plugin {name!r}")
        instance.on_init(self._core_context)
        self._instances[name] = instance
        logger.info("Plugin loaded: %s", name)
        return True

    def unload_plugin(self, name: str) -> None:
        inst = self._instances.pop(name, None)
        if inst is None:
            return
        try:
            inst.on_shutdown()
        except Exception as exc:
            logger.warning("Plugin %r on_shutdown failed: %s", name, exc)
        else:
            logger.info("Plugin unloaded: %s", name)

    def shutdown_all(self) -> None:
        for name in list(self._instances):
            self.unload_plugin(name)

    def get_all_commands(self) -> dict[str, Callable[..., Any]]:
        """Merge ``get_commands()`` from each loaded plugin with ``plugin_name:cmd`` keys."""
        merged: dict[str, Callable[..., Any]] = {}
        for pname, inst in self._instances.items():
            try:
                raw = inst.get_commands()
            except Exception as exc:
                logger.warning("Plugin %r get_commands failed: %s", pname, exc)
                continue
            for cmd_name, fn in raw.items():
                if not isinstance(cmd_name, str) or not callable(fn):
                    logger.warning(
                        "Plugin %r: skipping invalid command entry %r", pname, cmd_name
                    )
                    continue
                merged[f"{pname}:{cmd_name}"] = fn
        return merged


def _instantiate_plugin_from_file(path: Path, *, unique_tag: str) -> object:
    """Import ``path`` as a one-off module and return ``Plugin()`` if a ``Plugin`` class exists."""
    spec = importlib.util.spec_from_file_location(unique_tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_tag] = mod
    spec.loader.exec_module(mod)
    return _plugin_object_from_module(mod, path)


def _plugin_object_from_module(mod: ModuleType, path: Path) -> object:
    if hasattr(mod, "Plugin"):
        cls = mod.Plugin
        if isinstance(cls, type):
            return cls()
        raise TypeError(f"{path}: Plugin must be a class, got {type(cls).__name__}")
    raise TypeError(f"{path}: no 'Plugin' class found")


def upsert_plugin_enabled(
    config_path: Path,
    *,
    name: str,
    module_stem: str,
    enabled: bool = True,
    auto_start: bool = True,
) -> None:
    """Create or update one ``[[plugin]]`` row and write ``extensions.toml``."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if config_path.is_file() and config_path.stat().st_size > 0:
        raw = config_path.read_bytes()
        if raw.strip():
            data = tomllib.loads(raw.decode("utf-8"))
            existing = data.get("plugin")
            if isinstance(existing, list):
                rows = [r for r in existing if isinstance(r, dict)]
    found = False
    new_row = {
        "name": name,
        "enabled": enabled,
        "module": module_stem,
        "auto_start": auto_start,
    }
    for i, row in enumerate(rows):
        if row.get("name") == name:
            rows[i] = new_row
            found = True
            break
    if not found:
        rows.append(new_row)
    body = tomli_w.dumps({"plugin": rows})
    if not config_path.exists():
        config_path.write_text(_CONFIG_HEADER + body, encoding="utf-8")
    else:
        config_path.write_text(body, encoding="utf-8")


def set_plugin_enabled(config_path: Path, name: str, *, enabled: bool) -> bool:
    """Set ``enabled`` for the named plugin. Returns False if no row matched."""
    if not config_path.is_file():
        return False
    raw = config_path.read_bytes()
    if not raw.strip():
        return False
    data = tomllib.loads(raw.decode("utf-8"))
    rows = data.get("plugin")
    if not isinstance(rows, list):
        return False
    changed = False
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        copy = dict(row)
        if copy.get("name") == name:
            copy["enabled"] = enabled
            changed = True
        out.append(copy)
    if not changed:
        return False
    config_path.write_text(tomli_w.dumps({"plugin": out}), encoding="utf-8")
    return True
