"""Tests for delta.core.plugin_manager (no running daemon)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from delta.core.plugin_manager import PluginManager, set_plugin_enabled, upsert_plugin_enabled


def _good_plugin_source() -> str:
    return '''
class Plugin:
    def __init__(self):
        self.shutdown_called = False

    def on_init(self, core_context):
        self.ctx = core_context

    def on_shutdown(self):
        self.shutdown_called = True

    def get_commands(self):
        return {"ping": lambda: "pong"}
'''


def test_load_enabled_and_get_all_commands(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "demo.py").write_text(_good_plugin_source(), encoding="utf-8")
    config_path = tmp_path / "extensions.toml"
    upsert_plugin_enabled(
        config_path,
        name="demo",
        module_stem="demo",
        enabled=True,
        auto_start=True,
    )
    mgr = PluginManager(core_context=object(), plugin_dir=plugin_dir, config_path=config_path)
    n = mgr.load_enabled_from_config()
    assert n == 1
    cmds = mgr.get_all_commands()
    assert "demo:ping" in cmds
    assert cmds["demo:ping"]() == "pong"
    mgr.shutdown_all()


def test_load_skips_broken_plugin_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "bad.py").write_text("not_a_plugin = 1\n", encoding="utf-8")
    config_path = tmp_path / "extensions.toml"
    upsert_plugin_enabled(
        config_path,
        name="bad",
        module_stem="bad",
        enabled=True,
        auto_start=True,
    )
    caplog.set_level(logging.WARNING)
    mgr = PluginManager(core_context=object(), plugin_dir=plugin_dir, config_path=config_path)
    n = mgr.load_enabled_from_config()
    assert n == 0
    assert any("failed to load" in r.message.lower() for r in caplog.records)


def test_on_shutdown_called(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "demo.py").write_text(_good_plugin_source(), encoding="utf-8")
    config_path = tmp_path / "extensions.toml"
    upsert_plugin_enabled(
        config_path,
        name="demo",
        module_stem="demo",
        enabled=True,
        auto_start=True,
    )
    mgr = PluginManager(core_context=object(), plugin_dir=plugin_dir, config_path=config_path)
    mgr.load_enabled_from_config()
    inst = mgr._instances["demo"]
    mgr.shutdown_all()
    assert inst.shutdown_called is True


def test_set_plugin_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "extensions.toml"
    upsert_plugin_enabled(
        config_path,
        name="x",
        module_stem="x",
        enabled=True,
        auto_start=True,
    )
    assert set_plugin_enabled(config_path, "x", enabled=False) is True
    mgr = PluginManager(core_context=object(), plugin_dir=tmp_path / "p", config_path=config_path)
    rows = mgr.read_config()
    assert len(rows) == 1
    assert rows[0]["enabled"] is False
