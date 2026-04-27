# Resetting local data (legacy RAG, traces, chat)

Use this when migrating from an old install (for example legacy `e3n.db`), removing personal or stale embeddings, or clearing reasoning traces. **Back up first.** There is no unauthenticated HTTP “wipe everything” endpoint on the dev server by design.

## 1. Stop the app

Stop `uvicorn` (and any ingest workers using the same paths) so SQLite and Chroma are not open for write.

## 2. Back up paths

Resolve and copy:

| Resource | Typical location |
|----------|------------------|
| SQLite | `$DELTA_SQLITE_PATH` or `~/.local/share/deltai/delta.db` |
| ChromaDB | `$CHROMADB_PATH` or `~/.local/share/deltai/chromadb` |
| Knowledge files | `data/knowledge/` under the repo (if used) |

If you still use a legacy DB path, see the `e3n.db` note in the root [README.md](../README.md).

## 3. Clear in-app chat history (optional but recommended)

With the server running (or before deleting DB), you can clear the conversation table via the API:

```bash
curl -X DELETE http://127.0.0.1:8000/chat/history
```

## 4. Remove RAG / ingested vectors

- **Targeted:** `DELETE /memory/files/{path}` for files under the knowledge directory (see API docs in [CLAUDE.md](../CLAUDE.md)).
- **Full Chroma reset:** stop the app, then remove or empty the Chroma directory you configured (`CHROMADB_PATH`). On next start, re-ingest as needed.

## 5. Clear SQLite analytics / traces (without deleting the file)

Tables used for chat persistence and learning signals (see `project/persistence.py`):

- `conversation_history`
- `reasoning_traces`
- `quality_scores`
- `routing_feedback`
- `knowledge_gaps`
- `budget_daily` (cloud spend — only delete if you want a fresh budget ledger)

Example (adjust path):

```bash
sqlite3 ~/.local/share/deltai/delta.db "
DELETE FROM reasoning_traces;
DELETE FROM quality_scores;
DELETE FROM routing_feedback;
DELETE FROM knowledge_gaps;
DELETE FROM conversation_history;
VACUUM;
"
```

To reset the daily cloud budget table as well, add `DELETE FROM budget_daily;`.

## 6. Scripted reset (confirmation required)

From the repo root:

```bash
python scripts/reset_deltai_data.py --i-understand-this-deletes-data \
  --wipe-chromadb \
  --wipe-sqlite-analytics
```

Inspect `python scripts/reset_deltai_data.py --help` for options (`--chromadb-path`, `--sqlite-path`, `--include-budget`).

## 7. Rebuild Ollama tags after modelfile edits

```bash
ollama create deltai-qwen14b -f modelfiles/deltai-qwen14b.modelfile
ollama create deltai-qwen3b -f modelfiles/deltai-qwen3b.modelfile
# plus backup tags if you use them
```
