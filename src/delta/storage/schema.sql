-- DELTA core SQLite schema
-- Tracks sessions, requests, context snapshots, and memory items.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  source TEXT NOT NULL,
  query TEXT NOT NULL,
  intent TEXT,
  selected_agent TEXT,
  response TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);

CREATE TABLE IF NOT EXISTS context_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  active_window_title TEXT,
  active_window_app TEXT,
  clipboard_text TEXT,
  desktop_env TEXT,
  display_server TEXT,
  captured_at INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_context_session_id ON context_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_context_captured_at ON context_snapshots(captured_at);

CREATE TABLE IF NOT EXISTS memory_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  kind TEXT NOT NULL,
  key TEXT,
  value TEXT NOT NULL,
  importance REAL DEFAULT 0.5,
  created_at INTEGER NOT NULL,
  expires_at INTEGER,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_session_id ON memory_items(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_items(kind);
CREATE INDEX IF NOT EXISTS idx_memory_expires_at ON memory_items(expires_at);
