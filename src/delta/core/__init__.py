"""Core utilities shared by the delta package (optional plugin system, etc.)."""

from delta.core.plugin_manager import (
    PluginCoreContext,
    PluginManager,
    set_plugin_enabled,
    upsert_plugin_enabled,
)
from delta.core.plugin_protocol import DeltaPlugin, validate_plugin_instance

__all__ = [
    "DeltaPlugin",
    "PluginCoreContext",
    "PluginManager",
    "set_plugin_enabled",
    "upsert_plugin_enabled",
    "validate_plugin_instance",
]
