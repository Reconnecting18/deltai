"""
deltai Memory System — ChromaDB + Ollama Embeddings
Handles ingestion, chunking, storage, and retrieval of knowledge documents.
"""

import os
import re
import hashlib
import json
import logging
import time as _time
from collections import defaultdict
import httpx
import chromadb
from chromadb.config import Settings

logger = logging.getLogger("deltai.memory")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("DELTAI_EMBED_MODEL", "nomic-embed-text")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", r"~/deltai/data\chromadb")
KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", r"~/deltai/data\knowledge")

# ── CHUNKING ────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB — skip files larger than this
CHUNK_SIZE = 512       # chars per chunk (smaller = more precise retrieval)
CHUNK_OVERLAP = 64     # overlap between chunks for context continuity

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".log", ".bat", ".ps1", ".sh",
    ".c", ".cpp", ".h", ".rs", ".go", ".java",
}

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if len(text) <= CHUNK_SIZE:
        chunks.append({
            "text": text.strip(),
            "source": source,
            "chunk_index": 0,
            "total_chunks": 1,
        })
        return chunks

    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at a newline or sentence boundary
        if end < len(text):
            # Look for a good break point in the last 20% of the chunk
            search_start = start + int(CHUNK_SIZE * 0.8)
            best_break = -1
            for br in ["\n\n", "\n", ". ", "? ", "! ", "; ", ", "]:
                pos = text.rfind(br, search_start, end)
                if pos > 0:
                    best_break = pos + len(br)
                    break
            if best_break > 0:
                end = best_break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "source": source,
                "chunk_index": idx,
            })
            idx += 1

        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    # Set total chunks count
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


def file_hash(path: str) -> str:
    """SHA-256 hash of file contents for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ── EMBEDDING ───────────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama's embedding API."""
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": texts}
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
        logger.error(f"Embedding request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise


# ── CHROMADB CLIENT ─────────────────────────────────────────────────────

_client = None
_collection = None

def get_collection():
    """Get or create the ChromaDB collection (singleton)."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(
        path=CHROMADB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name="deltai_knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ── HIERARCHICAL MEMORY (Hot/Warm/Cold) ──────────────────────────────────
# Three-tier storage: hot (in-memory, <5min), warm (ChromaDB persistent, 5min-24hr),
# cold (SQLite compressed, >24hr). Automatic demotion based on age.

import sqlite3
import zlib

_COLD_DB_PATH = os.getenv("COLD_MEMORY_DB",
    os.path.join(os.path.dirname(CHROMADB_PATH), "cold_memory.db"))
_WARM_TO_COLD_AGE = int(os.getenv("WARM_TO_COLD_AGE_SEC", str(24 * 3600)))  # 24h
_COLD_SEARCH_THRESHOLD = 3  # search cold only if hot/warm returns fewer than this

_cold_db_initialized = False


def _init_cold_db():
    """Initialize the cold tier SQLite database."""
    global _cold_db_initialized
    if _cold_db_initialized:
        return
    os.makedirs(os.path.dirname(_COLD_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_COLD_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cold_chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            text_compressed BLOB NOT NULL,
            embedding_compressed BLOB NOT NULL,
            metadata_json TEXT,
            ingested_at REAL,
            demoted_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cold_source ON cold_chunks(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cold_ingested ON cold_chunks(ingested_at)")
    conn.commit()
    conn.close()
    _cold_db_initialized = True


def _demote_to_cold(chunk_ids: list[str], collection) -> int:
    """
    Move aged chunks from warm (ChromaDB) to cold (SQLite).
    Returns number of chunks demoted.
    """
    if not chunk_ids:
        return 0
    _init_cold_db()
    try:
        # Fetch full data from ChromaDB
        data = collection.get(ids=chunk_ids, include=["documents", "embeddings", "metadatas"])
        if not data["ids"]:
            return 0

        conn = sqlite3.connect(_COLD_DB_PATH)
        now = _time.time()
        demoted = 0
        for i, cid in enumerate(data["ids"]):
            text = data["documents"][i] if data["documents"] else ""
            emb = data["embeddings"][i] if data["embeddings"] else []
            meta = data["metadatas"][i] if data["metadatas"] else {}

            # Compress text and embedding for storage
            text_compressed = zlib.compress(text.encode("utf-8"), level=6)
            emb_bytes = json.dumps(emb).encode("utf-8")
            emb_compressed = zlib.compress(emb_bytes, level=6)

            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cold_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cid, meta.get("source", ""),
                     text_compressed, emb_compressed,
                     json.dumps(meta), meta.get("ingested_at", 0), now)
                )
                demoted += 1
            except Exception:
                pass

        conn.commit()
        conn.close()

        # Remove from ChromaDB
        collection.delete(ids=chunk_ids)
        return demoted
    except Exception as e:
        logger.warning(f"Cold demotion failed: {e}")
        return 0


def _search_cold_tier(query_embedding: list[float], n_results: int = 3,
                      source_filter: str = None) -> list[dict]:
    """
    Search cold tier using brute-force cosine similarity.
    Only called when hot/warm results are insufficient.
    """
    _init_cold_db()
    try:
        conn = sqlite3.connect(_COLD_DB_PATH)
        if source_filter:
            rows = conn.execute(
                "SELECT id, source, text_compressed, embedding_compressed, metadata_json FROM cold_chunks WHERE source = ?",
                (source_filter,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source, text_compressed, embedding_compressed, metadata_json FROM cold_chunks"
            ).fetchall()
        conn.close()

        if not rows:
            return []

        import math
        def cosine_distance(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - (dot / (norm_a * norm_b))

        results = []
        for cid, source, text_c, emb_c, meta_json in rows:
            text = zlib.decompress(text_c).decode("utf-8")
            emb = json.loads(zlib.decompress(emb_c).decode("utf-8"))
            dist = cosine_distance(query_embedding, emb)
            if dist < 0.75:
                meta = json.loads(meta_json) if meta_json else {}
                results.append({
                    "text": text,
                    "source": source,
                    "distance": round(dist, 4),
                    "chunk_index": meta.get("chunk_index", 0),
                    "tier": "cold",
                })
        results.sort(key=lambda x: x["distance"])
        return results[:n_results]
    except Exception as e:
        logger.debug(f"Cold tier search failed: {e}")
        return []


def compact_warm_to_cold() -> dict:
    """
    Background job: move chunks older than WARM_TO_COLD_AGE from ChromaDB to cold SQLite.
    Returns {"demoted": N, "checked": N}.
    """
    collection = get_collection()
    now = _time.time()
    cutoff = now - _WARM_TO_COLD_AGE

    try:
        # Find old ingested entries
        old_data = collection.get(
            where={"ingested_at": {"$lt": cutoff}},
            include=["metadatas"],
        )
        if not old_data["ids"]:
            return {"demoted": 0, "checked": 0}

        demoted = _demote_to_cold(old_data["ids"], collection)
        return {"demoted": demoted, "checked": len(old_data["ids"])}
    except Exception as e:
        logger.warning(f"Warm-to-cold compaction failed: {e}")
        return {"demoted": 0, "checked": 0, "error": str(e)}


def get_cold_stats() -> dict:
    """Get cold tier storage stats."""
    _init_cold_db()
    try:
        conn = sqlite3.connect(_COLD_DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM cold_chunks").fetchone()[0]
        conn.close()
        db_size = os.path.getsize(_COLD_DB_PATH) if os.path.exists(_COLD_DB_PATH) else 0
        return {"chunks": count, "size_mb": round(db_size / 1e6, 2)}
    except Exception:
        return {"chunks": 0, "size_mb": 0}


# ── INGESTION ───────────────────────────────────────────────────────────

def ingest_file(filepath: str) -> dict:
    """
    Read, chunk, embed, and store a file in ChromaDB.
    Returns {"status": "ok"/"skipped"/"error", "chunks": N, ...}
    """
    filepath = os.path.normpath(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {"status": "skipped", "reason": f"unsupported extension: {ext}"}

    try:
        fsize = os.path.getsize(filepath)
        if fsize > MAX_FILE_SIZE:
            return {"status": "skipped", "reason": f"file too large ({fsize} bytes)"}
    except OSError as e:
        return {"status": "error", "reason": str(e)}

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if not content.strip():
        return {"status": "skipped", "reason": "empty file"}

    # Check if file has changed since last ingestion
    current_hash = file_hash(filepath)
    collection = get_collection()
    rel_path = os.path.relpath(filepath, KNOWLEDGE_PATH)

    # Delete old chunks for this file
    try:
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            # Check if content changed
            old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            if old_meta.get("file_hash") == current_hash:
                return {"status": "skipped", "reason": "unchanged"}
            # Delete old chunks
            collection.delete(ids=existing["ids"])
    except Exception as e:
        logger.warning(f"Change detection check failed for {rel_path}: {e}")

    # Chunk the content
    chunks = chunk_text(content, rel_path)
    if not chunks:
        return {"status": "skipped", "reason": "no chunks produced"}

    # Get embeddings
    texts = [c["text"] for c in chunks]
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        return {"status": "error", "reason": f"embedding failed: {e}"}

    # Store in ChromaDB
    ids = [f"{rel_path}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": rel_path,
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "file_hash": current_hash,
            "char_count": len(c["text"]),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {"status": "ok", "chunks": len(chunks), "source": rel_path}


def remove_file(filepath: str) -> dict:
    """Remove all chunks for a file from ChromaDB."""
    filepath = os.path.normpath(filepath)
    rel_path = os.path.relpath(filepath, KNOWLEDGE_PATH)
    collection = get_collection()

    try:
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            return {"status": "ok", "removed": len(existing["ids"]), "source": rel_path}
        return {"status": "skipped", "reason": "not found in store"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def ingest_all() -> list[dict]:
    """Ingest all files in the knowledge directory."""
    results = []
    for root, dirs, files in os.walk(KNOWLEDGE_PATH):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            result = ingest_file(fpath)
            result["file"] = fname
            results.append(result)
    return results


# ── QUERY EXPANSION ─────────────────────────────────────────────────────

# Common synonyms/related terms for query expansion (domain-relevant)
_SYNONYMS = {
    "router": ["routing", "model selection", "tier", "VRAM"],
    "routing": ["router", "model selection", "tier"],
    "memory": ["RAG", "ChromaDB", "knowledge", "embeddings"],
    "rag": ["memory", "ChromaDB", "retrieval", "knowledge"],
    "model": ["LLM", "Ollama", "inference", "Qwen"],
    "tool": ["function", "executor", "capability"],
    "tools": ["functions", "executor", "capabilities"],
    "backup": ["emergency", "failover", "fallback"],
    "emergency": ["backup", "failover", "fallback"],
    "vram": ["GPU", "memory", "tier"],
    "gpu": ["VRAM", "CUDA", "graphics"],
    "sim": ["simulator", "racing", "LMU", "game"],
    "race": ["racing", "telemetry", "sim", "LMU"],
    "telemetry": ["race", "data", "ingest", "metrics"],
    "ingest": ["import", "connector", "push", "context"],
    "voice": ["STT", "TTS", "Whisper", "Kokoro", "speech"],
    "config": ["configuration", "settings", "env", "parameters"],
    "error": ["exception", "failure", "bug", "crash"],
    "file": ["document", "knowledge", "source"],
    "search": ["query", "find", "lookup", "retrieve"],
}

# Stop words to filter out during keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "into", "about", "between", "through",
    "during", "before", "after", "above", "below", "up", "down", "out",
    "off", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "if", "or", "and", "but", "nor", "because", "while", "although",
    "tell", "show", "explain", "describe", "give", "get", "make",
    "know", "think", "want", "let", "say", "go", "work", "use",
})


def _expand_query(query: str) -> list[str]:
    """
    Generate 2-3 search variants from the original query using keyword
    extraction and synonym expansion. No LLM needed — pure string manipulation.

    Returns a list of query strings (always includes the original).
    """
    queries = [query]

    # Extract meaningful keywords (lowercase, strip punctuation)
    words = re.findall(r"[a-zA-Z0-9_\-]+", query.lower())
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    if not keywords:
        return queries

    # Variant 1: keyword-dense version (just the extracted keywords)
    keyword_query = " ".join(keywords)
    if keyword_query.lower() != query.lower().strip():
        queries.append(keyword_query)

    # Variant 2: keywords + synonym expansions
    expanded_terms = list(keywords)
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in _SYNONYMS:
            # Add up to 2 synonyms per keyword
            for syn in _SYNONYMS[kw_lower][:2]:
                if syn.lower() not in expanded_terms:
                    expanded_terms.append(syn)

    expanded_query = " ".join(expanded_terms)
    if expanded_query.lower() != keyword_query.lower():
        queries.append(expanded_query)

    return queries[:3]  # Cap at 3 variants


# ── RERANKING ──────────────────────────────────────────────────────────

# Source grouping boost: documents with multiple matching chunks are more relevant
_SOURCE_GROUP_BOOST = 0.03   # distance reduction per extra chunk from same source
_RECENCY_BOOST = 0.05        # distance reduction for recently ingested context
_RECENCY_WINDOW = 300        # seconds (5 minutes)


def _rerank_results(matches: list[dict], boost_recent: bool = True) -> list[dict]:
    """
    Re-rank search results with two boosts:

    1. Source grouping — if multiple chunks from the same source matched,
       reduce their distance slightly. A doc with 3 matching chunks is more
       relevant than 3 separate docs with 1 match each.

    2. Recency bias — ingested context (from /ingest) with recent `ingested_at`
       metadata gets a slight relevance boost (lower distance).

    Returns a new list sorted by adjusted distance.
    """
    if not matches:
        return matches

    # Count how many chunks matched per source
    source_counts = defaultdict(int)
    for m in matches:
        source_counts[m["source"]] += 1

    now = _time.time()
    reranked = []

    for m in matches:
        adjusted_distance = m["distance"]

        # Source grouping boost: extra chunks from same source reduce distance
        extra_chunks = source_counts[m["source"]] - 1
        if extra_chunks > 0:
            # Diminishing returns: first extra chunk gives full boost, then half, etc.
            group_boost = 0.0
            for i in range(min(extra_chunks, 3)):  # cap at 3 extra
                group_boost += _SOURCE_GROUP_BOOST / (i + 1)
            adjusted_distance = max(0.0, adjusted_distance - group_boost)

        # Recency bias for ingested context
        if boost_recent:
            ingested_at = m.get("_ingested_at", 0)
            if ingested_at > 0:
                age = now - ingested_at
                if age < _RECENCY_WINDOW:
                    # Linear decay: full boost at t=0, zero boost at t=RECENCY_WINDOW
                    recency_factor = 1.0 - (age / _RECENCY_WINDOW)
                    adjusted_distance = max(
                        0.0, adjusted_distance - (_RECENCY_BOOST * recency_factor)
                    )

        reranked.append({**m, "distance": round(adjusted_distance, 4)})

    # Sort by adjusted distance (lower = more relevant)
    reranked.sort(key=lambda x: x["distance"])

    # Strip internal metadata before returning
    for m in reranked:
        m.pop("_ingested_at", None)
        m.pop("_id", None)

    return reranked


# ── QUERY ───────────────────────────────────────────────────────────────

def query_knowledge(query: str, n_results: int = 5, threshold: float = 0.75,
                    boost_recent: bool = True, source_filter: str = None,
                    max_age_sec: float = None) -> list[dict]:
    """
    Search the knowledge base for relevant chunks using multi-query expansion,
    source-grouped reranking, and optional recency bias.

    Returns list of {"text", "source", "distance", "chunk_index"}.
    threshold: max cosine distance (lower = more relevant). 0.75 for nomic-embed-text.
    boost_recent: if True, recently ingested context gets a slight relevance boost.
    source_filter: if provided, only return results from this source.
    max_age_sec: if provided, filter out results older than this many seconds (based on ingested_at).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Expand query into 1-3 search variants
    expanded_queries = _expand_query(query)

    # Get embeddings for all query variants in one batch
    try:
        query_embeddings = get_embeddings(expanded_queries)
    except Exception:
        return []

    # Build optional where clause for source filtering
    where_clause = None
    if source_filter:
        where_clause = {"source": source_filter}

    # Run each variant and collect results, deduplicating by chunk ID
    seen_ids = set()
    all_matches = []
    fetch_n = min(n_results * 2, collection.count())  # fetch extra for dedup headroom

    for emb in query_embeddings:
        query_kwargs = {
            "query_embeddings": [emb],
            "n_results": fetch_n,
            "include": ["documents", "distances", "metadatas"],
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        results = collection.query(**query_kwargs)

        if not results["ids"] or not results["ids"][0]:
            continue

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            dist = results["distances"][0][i]
            if dist > threshold:
                continue

            meta = results["metadatas"][0][i]

            # Age filtering: skip entries older than max_age_sec
            if max_age_sec is not None:
                ingested_at = meta.get("ingested_at", 0)
                if ingested_at > 0 and (_time.time() - ingested_at) > max_age_sec:
                    continue  # Skip old entries

            all_matches.append({
                "text": results["documents"][0][i],
                "source": meta.get("source", "?"),
                "distance": round(dist, 4),
                "chunk_index": meta.get("chunk_index", 0),
                # Internal fields for reranking (stripped before return)
                "_ingested_at": meta.get("ingested_at", 0),
                "_id": chunk_id,
            })

    # Rerank with source grouping + recency bias
    reranked = _rerank_results(all_matches, boost_recent=boost_recent)

    # ── Cold tier fallback: search compressed archive if hot/warm insufficient ──
    if len(reranked) < _COLD_SEARCH_THRESHOLD:
        try:
            cold_results = _search_cold_tier(
                query_embeddings[0], n_results=n_results - len(reranked),
                source_filter=source_filter)
            reranked.extend(cold_results)
        except Exception:
            pass

    # Return top n_results
    return reranked[:n_results]


# ── ITERATIVE RAG (Retrieval-Augmented Reasoning) ────────────────────────
# Two-round retrieval: first round standard, second round with model-directed
# sub-queries to fill knowledge gaps. Dramatically better for complex queries.

_RAR_ENABLED = os.getenv("RAR_ENABLED", "true").lower() in ("true", "1", "yes")
_RAR_MIN_RESULTS_FOR_SKIP = int(os.getenv("RAR_MIN_RESULTS", "3"))


def iterative_query_knowledge(query: str, sub_queries: list[str] | None = None,
                               n_results: int = 5, threshold: float = 0.75,
                               **kwargs) -> list[dict]:
    """
    Two-round retrieval: standard query + follow-up with refined sub-queries.

    Args:
        query: Original user query
        sub_queries: Model-generated follow-up queries (from first round analysis).
                     If None, only runs standard single-round retrieval.
        n_results: Max results to return
        threshold: Max cosine distance
        **kwargs: Passed to query_knowledge (boost_recent, source_filter, max_age_sec)

    Returns same format as query_knowledge.
    """
    # Round 1: Standard retrieval
    round1 = query_knowledge(query, n_results=n_results, threshold=threshold, **kwargs)

    if not _RAR_ENABLED or not sub_queries:
        return round1

    # Round 2: Refined sub-query retrieval
    seen_ids = {r.get("_id", r.get("text", "")[:50]) for r in round1}
    additional = []

    for sq in sub_queries[:3]:  # max 3 sub-queries to bound latency
        sq_results = query_knowledge(sq, n_results=3, threshold=threshold, **kwargs)
        for r in sq_results:
            rid = r.get("_id", r.get("text", "")[:50])
            if rid not in seen_ids:
                seen_ids.add(rid)
                additional.append(r)

    # Merge and re-sort by distance
    combined = round1 + additional
    combined.sort(key=lambda x: x.get("distance", 1.0))
    return combined[:n_results]


def generate_sub_queries(query: str, initial_results: list[dict]) -> list[str]:
    """
    Generate refined sub-queries based on initial retrieval results.
    Uses keyword extraction from gaps between query and results — no LLM needed.

    Returns up to 3 sub-queries.
    """
    # Extract key terms from the query
    query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))

    # Extract terms from results
    result_terms = set()
    for r in initial_results:
        result_terms.update(re.findall(r'\b[a-zA-Z]{3,}\b', r.get("text", "").lower()[:500]))

    # Find related terms in results that weren't in the query (context expansion)
    expansion_terms = result_terms - query_terms
    # Filter to meaningful terms (not too common)
    _common = {"the", "and", "for", "that", "this", "with", "from", "have", "been",
               "will", "are", "was", "were", "has", "had", "not", "but", "what",
               "all", "can", "her", "his", "how", "its", "may", "our", "out",
               "you", "also", "into", "just", "more", "most", "much", "only",
               "over", "some", "such", "than", "them", "then", "very", "when"}
    expansion_terms -= _common

    sub_queries = []
    # Sub-query 1: Original + most relevant expansion terms
    if expansion_terms:
        top_expansion = list(expansion_terms)[:5]
        sub_queries.append(f"{query} {' '.join(top_expansion)}")

    # Sub-query 2: Key noun phrases from query with different framing
    if len(query_terms) > 2:
        key_terms = [t for t in query_terms if len(t) > 4][:4]
        if key_terms:
            sub_queries.append(" ".join(key_terms))

    # Sub-query 3: If results mention specific sources, query for those
    sources = set()
    for r in initial_results:
        src = r.get("source", "")
        if src and not src.startswith("ingest:"):
            sources.add(src)
    if sources:
        sub_queries.append(f"{query} {' '.join(list(sources)[:2])}")

    return sub_queries[:3]


_file_details_cache = {"data": None, "ts": 0}

def get_file_details() -> list[dict]:
    """Get per-file details from ChromaDB (source, chunk count). Cached for 30s."""
    now = _time.time()
    if _file_details_cache["data"] is not None and (now - _file_details_cache["ts"]) < 30:
        return _file_details_cache["data"]

    collection = get_collection()
    if collection.count() == 0:
        return []

    try:
        all_meta = collection.get(include=["metadatas"])
        file_chunks = {}
        for m in all_meta["metadatas"]:
            src = m.get("source", "?")
            if src not in file_chunks:
                file_chunks[src] = 0
            file_chunks[src] += 1
        result = [{"source": s, "chunks": c} for s, c in sorted(file_chunks.items())]
        _file_details_cache["data"] = result
        _file_details_cache["ts"] = now
        return result
    except Exception:
        return []


_stats_cache = {"data": None, "ts": 0}
_STATS_CACHE_TTL = 30  # seconds

def get_memory_stats() -> dict:
    """Get stats about the knowledge base (cached for 30s)."""
    now = _time.time()
    if _stats_cache["data"] and (now - _stats_cache["ts"]) < _STATS_CACHE_TTL:
        return _stats_cache["data"]

    collection = get_collection()
    count = collection.count()

    # Reuse get_file_details() cache for source counting instead of a separate full scan
    file_details = get_file_details()
    sources = {f["source"] for f in file_details}

    # ChromaDB disk size
    total_bytes = 0
    try:
        for dp, dn, fn in os.walk(CHROMADB_PATH):
            for f in fn:
                try:
                    total_bytes += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    except OSError as e:
        logger.debug(f"ChromaDB disk size scan failed: {e}")

    result = {
        "total_chunks": count,
        "total_files": len(sources),
        "sources": sorted(sources),
        "disk_mb": round(total_bytes / 1e6, 2),
    }
    _stats_cache["data"] = result
    _stats_cache["ts"] = now
    return result


# ── SEMANTIC DEDUPLICATION ────────────────────────────────────────────────
# Before ingesting new content, check if a near-duplicate already exists.
# If so, update metadata (freshen timestamp) instead of adding a duplicate.

_DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.15"))  # cosine distance — lower = stricter

def _check_semantic_duplicate(collection, embedding, source: str) -> str | None:
    """
    Check if a near-duplicate chunk exists in ChromaDB.
    Returns the existing chunk ID if a duplicate is found, None otherwise.
    """
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances", "metadatas"],
            where={"source": f"ingest:{source}"} if source else None,
        )
        if (results["ids"] and results["ids"][0] and
                results["distances"][0][0] < _DEDUP_THRESHOLD):
            return results["ids"][0][0]
    except Exception:
        pass
    return None


def _freshen_metadata(collection, chunk_id: str, tags: list[str] | None = None,
                      ttl: int = 0):
    """Update the timestamp and optionally TTL/tags on an existing chunk."""
    try:
        now = _time.time()
        update_meta = {"ingested_at": now}
        if ttl > 0:
            update_meta["expires_at"] = now + ttl
        if tags:
            update_meta["tags"] = json.dumps(tags)
        collection.update(ids=[chunk_id], metadatas=[update_meta])
    except Exception as e:
        logger.debug(f"Freshen metadata failed for {chunk_id}: {e}")


# ── INGEST CONNECTOR ─────────────────────────────────────────────────────
# Allows external services to push structured context into deltai's RAG memory.
# Each entry gets a source tag, timestamp, and optional TTL for auto-expiry.

def ingest_context(source: str, context: str, ttl: int = 0,
                   tags: list[str] | None = None) -> dict:
    """
    Ingest structured context from an external service into ChromaDB.

    Args:
        source: Identifier for the sender (e.g., "lmu-telemetry")
        context: Human-readable content to embed
        ttl: Seconds until auto-expiry (0 = permanent)
        tags: Optional filtering tags

    Returns:
        {"status": "ok", "chunks": N, "source": "..."}
    """
    if not context or not context.strip():
        return {"status": "error", "reason": "empty context"}

    collection = get_collection()
    now = _time.time()
    tags = tags or []

    # Chunk the context (reuse existing chunker)
    chunks = chunk_text(context, source)
    if not chunks:
        return {"status": "error", "reason": "no chunks produced"}

    # Get embeddings
    texts = [c["text"] for c in chunks]
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        return {"status": "error", "reason": f"embedding failed: {e}"}

    # ── Semantic deduplication: skip chunks that are near-duplicates ──
    new_texts = []
    new_embeddings = []
    new_chunks = []
    dedup_count = 0
    for i, emb in enumerate(embeddings):
        dup_id = _check_semantic_duplicate(collection, emb, source)
        if dup_id:
            # Near-duplicate found — just freshen its metadata
            _freshen_metadata(collection, dup_id, tags=tags, ttl=ttl)
            dedup_count += 1
        else:
            new_texts.append(texts[i])
            new_embeddings.append(emb)
            new_chunks.append(chunks[i])

    if not new_texts:
        return {"status": "ok", "chunks": 0, "deduplicated": dedup_count,
                "source": source, "expires_at": (now + ttl) if ttl > 0 else None}

    # Build IDs with timestamp to allow multiple ingests from same source
    ts_str = str(int(now * 1000))
    ids = [f"ingest::{source}::{ts_str}::chunk_{i}" for i in range(len(new_chunks))]

    # Metadata includes source, timestamp, TTL expiry, and tags
    expires_at = now + ttl if ttl > 0 else 0  # 0 = never expires
    metadatas = [
        {
            "source": f"ingest:{source}",
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "ingested_at": now,
            "expires_at": expires_at,
            "tags": json.dumps(tags),
            "char_count": len(c["text"]),
        }
        for c in new_chunks
    ]

    collection.add(
        ids=ids,
        documents=new_texts,
        embeddings=new_embeddings,
        metadatas=metadatas,
    )

    return {"status": "ok", "chunks": len(new_chunks), "deduplicated": dedup_count,
            "source": source, "expires_at": expires_at if expires_at > 0 else None}


def ingest_context_batch(items: list[dict]) -> dict:
    """
    Batch ingest multiple context items in a single embedding + ChromaDB call.
    Each item: {"source": str, "context": str, "ttl": int, "tags": list[str]}
    Max 100 items per batch.

    Returns {"status": "ok", "ingested": N, "chunks": N}
    """
    if not items:
        return {"status": "error", "reason": "empty batch"}
    if len(items) > 100:
        return {"status": "error", "reason": f"batch too large ({len(items)} > 100 max)"}

    collection = get_collection()
    now = _time.time()

    all_texts = []
    all_ids = []
    all_metadatas = []
    items_processed = 0

    for item_idx, item in enumerate(items):
        source = item.get("source", "").strip()
        context = item.get("context", "").strip()
        ttl = item.get("ttl", 0)
        tags = item.get("tags", [])

        if not source or not context:
            continue
        items_processed += 1

        chunks = chunk_text(context, source)
        expires_at = now + ttl if ttl > 0 else 0
        ts_str = str(int(now * 1000)) + f"_{item_idx}"

        for c in chunks:
            all_texts.append(c["text"])
            all_ids.append(f"ingest::{source}::{ts_str}::chunk_{c['chunk_index']}")
            all_metadatas.append({
                "source": f"ingest:{source}",
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "ingested_at": now,
                "expires_at": expires_at,
                "tags": json.dumps(tags),
                "char_count": len(c["text"]),
            })

    if not all_texts:
        return {"status": "error", "reason": "no valid items in batch"}

    try:
        embeddings = get_embeddings(all_texts)
    except Exception as e:
        return {"status": "error", "reason": f"embedding failed: {e}"}

    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    return {
        "status": "ok",
        "ingested": items_processed,
        "chunks": len(all_texts),
    }


_cleanup_last_run = 0.0
_CLEANUP_MIN_INTERVAL = 10  # seconds — avoid hammering ChromaDB on rapid chat requests

def cleanup_expired() -> dict:
    """
    Remove TTL-expired ingested entries from ChromaDB.
    Call periodically (e.g., on each /chat request or via background task).
    Rate-limited to run at most once every 10 seconds.

    Returns:
        {"removed": N, "checked": N}
    """
    global _cleanup_last_run
    now = _time.time()

    # Rate-limit: skip if we ran recently
    if now - _cleanup_last_run < _CLEANUP_MIN_INTERVAL:
        return {"removed": 0, "checked": 0, "skipped": "rate-limited"}
    _cleanup_last_run = now

    collection = get_collection()

    try:
        # Only query ingested entries (source starts with "ingest:") that have TTL
        # Using $gt filter on expires_at to only fetch entries with active TTL
        ttl_data = collection.get(
            where={"$and": [
                {"expires_at": {"$gt": 0}},
                {"expires_at": {"$lte": now}},
            ]},
            include=["metadatas"],
        )

        expired_ids = ttl_data["ids"] if ttl_data["ids"] else []
        checked = len(expired_ids)

        if expired_ids:
            collection.delete(ids=expired_ids)

        return {"removed": len(expired_ids), "checked": checked}

    except Exception as e:
        logger.warning(f"TTL cleanup primary method failed, trying fallback: {e}")
        # Fallback: if $and filter fails (older ChromaDB), use the old approach
        # but only fetch ingested entries, not knowledge files
        try:
            ingest_data = collection.get(
                where={"expires_at": {"$gt": 0}},
                include=["metadatas"],
            )
            expired_ids = []
            for i, meta in enumerate(ingest_data["metadatas"]):
                expires_at = meta.get("expires_at", 0)
                if isinstance(expires_at, (int, float)) and expires_at > 0 and now >= expires_at:
                    expired_ids.append(ingest_data["ids"][i])
            if expired_ids:
                collection.delete(ids=expired_ids)
            return {"removed": len(expired_ids), "checked": len(ingest_data["ids"])}
        except Exception as e2:
            logger.warning(f"TTL cleanup failed (both methods): {e2}")
            return {"removed": 0, "checked": 0, "error": str(e2)}
