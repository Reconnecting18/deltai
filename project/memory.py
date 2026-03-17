"""
E3N Memory System — ChromaDB + Ollama Embeddings
Handles ingestion, chunking, storage, and retrieval of knowledge documents.
"""

import os
import re
import hashlib
import json
import time as _time
from collections import defaultdict
import httpx
import chromadb
from chromadb.config import Settings

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("E3N_EMBED_MODEL", "nomic-embed-text")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", r"C:\e3n\data\chromadb")
KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", r"C:\e3n\data\knowledge")

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
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts}
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]


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
        name="e3n_knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


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
    except Exception:
        pass

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

    # Return top n_results
    return reranked[:n_results]


def get_file_details() -> list[dict]:
    """Get per-file details from ChromaDB (source, chunk count)."""
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
        return [{"source": s, "chunks": c} for s, c in sorted(file_chunks.items())]
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

    # Count unique sources
    sources = set()
    if count > 0:
        try:
            all_meta = collection.get(include=["metadatas"])
            for m in all_meta["metadatas"]:
                sources.add(m.get("source", "?"))
        except Exception:
            pass

    # ChromaDB disk size
    total_bytes = 0
    for dp, dn, fn in os.walk(CHROMADB_PATH):
        for f in fn:
            total_bytes += os.path.getsize(os.path.join(dp, f))

    result = {
        "total_chunks": count,
        "total_files": len(sources),
        "sources": sorted(sources),
        "disk_mb": round(total_bytes / 1e6, 2),
    }
    _stats_cache["data"] = result
    _stats_cache["ts"] = now
    return result


# ── INGEST CONNECTOR ─────────────────────────────────────────────────────
# Allows external services to push structured context into E3N's RAG memory.
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

    # Build IDs with timestamp to allow multiple ingests from same source
    ts_str = str(int(now * 1000))
    ids = [f"ingest::{source}::{ts_str}::chunk_{i}" for i in range(len(chunks))]

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
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {"status": "ok", "chunks": len(chunks), "source": source,
            "expires_at": expires_at if expires_at > 0 else None}


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


def cleanup_expired() -> dict:
    """
    Remove TTL-expired ingested entries from ChromaDB.
    Call periodically (e.g., on each /chat request or via background task).

    Returns:
        {"removed": N, "checked": N}
    """
    collection = get_collection()
    now = _time.time()

    try:
        # Get all entries that have an expires_at metadata field > 0
        # ChromaDB where filters only support simple comparisons,
        # so we fetch all ingested entries and filter in Python
        all_data = collection.get(
            where={"source": {"$ne": ""}},
            include=["metadatas"],
        )

        expired_ids = []
        checked = 0
        for i, meta in enumerate(all_data["metadatas"]):
            expires_at = meta.get("expires_at", 0)
            if isinstance(expires_at, (int, float)) and expires_at > 0:
                checked += 1
                if now >= expires_at:
                    expired_ids.append(all_data["ids"][i])

        if expired_ids:
            collection.delete(ids=expired_ids)

        return {"removed": len(expired_ids), "checked": checked}

    except Exception as e:
        return {"removed": 0, "checked": 0, "error": str(e)}
