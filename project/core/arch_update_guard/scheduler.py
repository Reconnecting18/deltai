"""Background auto-check loop for arch update guard."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

from persistence import get_sqlite_path

from . import settings as ag_settings
from .schema import init_arch_guard_tables
from .tracker import run_check

logger = logging.getLogger("deltai.arch_update_guard.scheduler")


async def scheduler_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            path = get_sqlite_path()
            conn = sqlite3.connect(path, timeout=10)
            try:
                init_arch_guard_tables(conn)
                if not ag_settings.load_enabled(conn):
                    await asyncio.sleep(30.0)
                    continue
                if ag_settings.load_mode(conn) != "auto":
                    await asyncio.sleep(30.0)
                    continue
                interval = ag_settings.load_interval_sec(conn)
                last = ag_settings.load_last_check_at(conn)
                now = time.time()
                if last is not None and (now - last) < interval:
                    wait = min(interval - (now - last), 60.0)
                    await asyncio.wait_for(stop_event.wait(), timeout=max(1.0, wait))
                    continue

                await asyncio.to_thread(
                    run_check,
                    mode="auto",
                    include_reverse_deps=False,
                    create_snapshot=False,
                )
                ag_settings.save_last_check_at(conn, time.time())
                conn.commit()
            finally:
                conn.close()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("arch_guard scheduler tick failed: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass
