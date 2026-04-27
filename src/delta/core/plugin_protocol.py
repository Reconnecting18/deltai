"""Contract for delta-daemon plugins loaded from user plugin directories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DeltaPlugin(Protocol):
    """Plugins are objects (typically a ``Plugin`` class instance) that expose these hooks."""

    def on_init(self, core_context: object) -> None:
        """Called once after the daemon core (DB, IPC, orchestrator) is ready.

        Use ``core_context`` to access ``settings``, ``app``, ``orchestrator``,
        and ``ipc_server`` (see ``PluginCoreContext`` in the daemon).
        """

    def on_shutdown(self) -> None:
        """Called when the daemon stops or when this plugin is unloaded.

        Release resources and cancel background work started in ``on_init``.
        """

    def get_commands(self) -> dict[str, Callable[..., Any]]:
        """Return command name → handler callables for in-daemon dispatch.

        Keys should be short names (e.g. ``\"ping\"``); the manager prefixes them
        as ``\"plugin_name:command_name\"`` when merging.
        """


def validate_plugin_instance(obj: object, *, label: str) -> None:
    """Raise ``TypeError`` if ``obj`` does not satisfy :class:`DeltaPlugin`."""
    if not isinstance(obj, DeltaPlugin):
        raise TypeError(f"{label}: object does not implement DeltaPlugin (missing methods)")
    cmds = obj.get_commands()
    if not isinstance(cmds, dict):
        raise TypeError(f"{label}: get_commands() must return dict, got {type(cmds).__name__}")
