"""
deltai File Watcher — monitors ~/deltai/data\\knowledge\\ for changes.
Auto-ingests new/modified files, removes deleted ones from ChromaDB.
Runs as a background thread inside the FastAPI process.
"""

import logging
import os
import time

from memory import KNOWLEDGE_PATH, SUPPORTED_EXTENSIONS, ingest_file, remove_file
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB — skip files larger than this

logger = logging.getLogger("deltai.watcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] WATCHER: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)


class KnowledgeHandler(FileSystemEventHandler):
    """Handle file system events in the knowledge directory."""

    def __init__(self):
        super().__init__()
        self._debounce = {}
        self._debounce_delay = 1.0  # seconds

    def _should_process(self, path: str) -> bool:
        """Check if file has a supported extension."""
        ext = os.path.splitext(path)[1].lower()
        return ext in SUPPORTED_EXTENSIONS

    def _debounced_ingest(self, path: str):
        """Debounce rapid file events (e.g., editors saving multiple times)."""
        now = time.time()
        last = self._debounce.get(path, 0)
        if now - last < self._debounce_delay:
            return
        self._debounce[path] = now

        # Prune stale debounce entries to prevent unbounded growth
        if len(self._debounce) > 500:
            cutoff = now - self._debounce_delay
            self._debounce = {k: v for k, v in self._debounce.items() if v > cutoff}

        # Skip files that are too large
        try:
            if os.path.getsize(path) > MAX_FILE_SIZE:
                logger.warning(
                    f"Skipped (>{MAX_FILE_SIZE // (1024 * 1024)}MB): {os.path.basename(path)}"
                )
                return
        except OSError:
            return  # file vanished between event and size check

        try:
            result = ingest_file(path)
            if result["status"] == "ok":
                logger.info(f"Ingested: {os.path.basename(path)} ({result['chunks']} chunks)")
            elif result["status"] == "skipped":
                logger.info(f"Skipped: {os.path.basename(path)} ({result.get('reason', '?')})")
            else:
                logger.warning(
                    f"Error ingesting {os.path.basename(path)}: {result.get('reason', '?')}"
                )
        except Exception as e:
            logger.error(f"Ingest failed for {os.path.basename(path)}: {e}")

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"New file detected: {os.path.basename(event.src_path)}")
            # Small delay to let the file finish writing
            time.sleep(0.3)
            self._debounced_ingest(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._debounced_ingest(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"File deleted: {os.path.basename(event.src_path)}")
            try:
                result = remove_file(event.src_path)
                if result["status"] == "ok":
                    logger.info(
                        f"Removed {result['removed']} chunks for {os.path.basename(event.src_path)}"
                    )
            except Exception as e:
                logger.error(f"Remove failed: {e}")

    def on_moved(self, event):
        if event.is_directory:
            return
        # Treat as delete old + create new
        if self._should_process(event.src_path):
            try:
                remove_file(event.src_path)
            except Exception:
                pass
        if self._should_process(event.dest_path):
            time.sleep(0.3)
            self._debounced_ingest(event.dest_path)


_observer = None


def start_watcher():
    """Start the file watcher in a background thread."""
    global _observer

    if _observer is not None:
        return  # Already running

    # Ensure knowledge directory exists
    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)

    handler = KnowledgeHandler()
    _observer = Observer()
    _observer.schedule(handler, KNOWLEDGE_PATH, recursive=True)
    _observer.daemon = True
    _observer.start()

    logger.info(f"Watching: {KNOWLEDGE_PATH}")
    return _observer


def stop_watcher():
    """Stop the file watcher."""
    global _observer
    if _observer:
        _observer.stop()
        _observer.join(timeout=3)
        _observer = None
        logger.info("Watcher stopped")


def watcher_running() -> bool:
    """Check if the file watcher is alive."""
    return _observer is not None and _observer.is_alive()
