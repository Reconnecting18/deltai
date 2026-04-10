"""D-Bus integration hooks for DELTA.

Provides optional runtime probing for desktop D-Bus services and a minimal
notification helper. All functionality degrades gracefully when D-Bus or
pydbus is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


def _parse_bool(value: str, default: bool) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "on"}:
		return True
	if normalized in {"0", "false", "no", "off"}:
		return False
	return default


@dataclass(frozen=True)
class DBusProbeResult:
	"""Snapshot of available D-Bus integration points."""

	enabled: bool
	available: bool
	session_bus: bool
	system_bus: bool
	notifications: bool
	login1: bool
	network_manager: bool
	error: str | None = None

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable representation."""
		return asdict(self)


class DBusIntegration:
	"""Optional D-Bus integration surface for Linux desktop services."""

	def __init__(self, enabled: bool | None = None) -> None:
		if enabled is None:
			enabled = _parse_bool(os.getenv("DELTA_DBUS_ENABLED", "true"), default=True)
		self.enabled = enabled

	def _load_pydbus(self):
		try:
			import pydbus  # type: ignore

			return pydbus
		except Exception:
			return None

	def _safe_bus(self, bus_factory):
		try:
			return bus_factory()
		except Exception:
			return None

	def _has_service(self, bus: Any, bus_name: str, object_path: str) -> bool:
		if bus is None:
			return False
		try:
			bus.get(bus_name, object_path)
			return True
		except Exception:
			return False

	def probe(self) -> DBusProbeResult:
		"""Probe D-Bus availability and commonly used desktop services."""
		if not self.enabled:
			return DBusProbeResult(
				enabled=False,
				available=False,
				session_bus=False,
				system_bus=False,
				notifications=False,
				login1=False,
				network_manager=False,
			)

		pydbus_module = self._load_pydbus()
		if pydbus_module is None:
			return DBusProbeResult(
				enabled=True,
				available=False,
				session_bus=False,
				system_bus=False,
				notifications=False,
				login1=False,
				network_manager=False,
				error="pydbus unavailable",
			)

		session_bus = self._safe_bus(pydbus_module.SessionBus)
		system_bus = self._safe_bus(pydbus_module.SystemBus)

		return DBusProbeResult(
			enabled=True,
			available=True,
			session_bus=session_bus is not None,
			system_bus=system_bus is not None,
			notifications=self._has_service(
				session_bus,
				"org.freedesktop.Notifications",
				"/org/freedesktop/Notifications",
			),
			login1=self._has_service(
				system_bus,
				"org.freedesktop.login1",
				"/org/freedesktop/login1",
			),
			network_manager=self._has_service(
				system_bus,
				"org.freedesktop.NetworkManager",
				"/org/freedesktop/NetworkManager",
			),
		)

	def send_desktop_notification(self, title: str, body: str, app_name: str = "deltai") -> bool:
		"""Send a desktop notification through org.freedesktop.Notifications."""
		if not self.enabled:
			return False

		pydbus_module = self._load_pydbus()
		if pydbus_module is None:
			return False

		session_bus = self._safe_bus(pydbus_module.SessionBus)
		if session_bus is None:
			return False

		try:
			notifications = session_bus.get(
				"org.freedesktop.Notifications",
				"/org/freedesktop/Notifications",
			)
			notifications.Notify(app_name, 0, "", title, body, [], {}, 5000)
			return True
		except Exception:
			return False
