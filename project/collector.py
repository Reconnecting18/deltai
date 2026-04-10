"""
deltai Web Training Data Collector
Pulls training examples from Wikipedia, arXiv, OpenF1, Semantic Scholar,
and motorsport web pages. All content is deduplicated, formatted as
instruction/output pairs, and saved to domain JSONL datasets for fine-tuning.

Designed to run as Phase 0.5 of the nightly training cycle, before QLoRA
training so newly collected examples are available the same night.

Sources:
  wikipedia   — HuggingFace datasets streaming (6.7M articles, checkpointed)
  arxiv       — arXiv XML API (cs.LG, physics, math.NA, eess.SP, cond-mat)
  openf1      — OpenF1 REST API (race sessions, laps, pit stops, tire strategy)
  papers      — Semantic Scholar API (acoustics, FEA, CFD, vibration abstracts)
  motorsport  — trafilatura + DuckDuckGo (motorsport engineering web articles)

Output datasets:
  deltai-general-knowledge   → reasoning domain
  deltai-science-knowledge   → engineering domain
  deltai-arxiv-papers        → engineering domain
  deltai-openf1-strategy     → racing domain
  deltai-web-motorsport      → racing domain
"""

import os
import json
import time
import logging
import hashlib
import sqlite3
import re
import threading

import httpx

logger = logging.getLogger("deltai.collector")

# ── Paths ────────────────────────────────────────────────────────────────────

TRAINING_PATH = os.getenv("TRAINING_PATH", r"~/deltai/data\training")
DATASETS_PATH = os.path.join(TRAINING_PATH, "datasets")
CHECKPOINTS_PATH = os.path.join(TRAINING_PATH, "collect_checkpoints")

for _p in [DATASETS_PATH, CHECKPOINTS_PATH]:
    os.makedirs(_p, exist_ok=True)

_DEDUP_DB = os.path.join(CHECKPOINTS_PATH, "collect_dedup.db")
_WIKI_OFFSET_FILE = os.path.join(CHECKPOINTS_PATH, "wikipedia_offset.txt")

# ── Config from .env ─────────────────────────────────────────────────────────

WEB_COLLECT_ENABLED    = os.getenv("WEB_COLLECT_ENABLED", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_WIKIPEDIA  = os.getenv("WEB_COLLECT_WIKIPEDIA", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_ARXIV      = os.getenv("WEB_COLLECT_ARXIV", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_OPENF1     = os.getenv("WEB_COLLECT_OPENF1", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_MOTORSPORT = os.getenv("WEB_COLLECT_MOTORSPORT", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_PAPERS     = os.getenv("WEB_COLLECT_PAPERS", "true").lower() in ("true", "1", "yes")
WEB_COLLECT_WIKI_BATCH = int(os.getenv("WEB_COLLECT_WIKIPEDIA_BATCH", "2000"))
WEB_COLLECT_MAX_SOURCE = int(os.getenv("WEB_COLLECT_MAX_PER_SOURCE", "200"))

# ── Deduplication DB ─────────────────────────────────────────────────────────

_dedup_lock = threading.Lock()


def _get_dedup_db() -> sqlite3.Connection:
    db = sqlite3.connect(_DEDUP_DB, timeout=10)
    db.execute(
        "CREATE TABLE IF NOT EXISTS seen_hashes "
        "(hash TEXT PRIMARY KEY, source TEXT, added INTEGER)"
    )
    db.commit()
    return db


def _is_duplicate(content: str, source: str = "") -> bool:
    h = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
    with _dedup_lock:
        db = _get_dedup_db()
        try:
            row = db.execute("SELECT 1 FROM seen_hashes WHERE hash=?", (h,)).fetchone()
            if row:
                return True
            db.execute(
                "INSERT INTO seen_hashes (hash, source, added) VALUES (?,?,?)",
                (h, source, int(time.time()))
            )
            db.commit()
            return False
        finally:
            db.close()


# ── Dataset writing ───────────────────────────────────────────────────────────

def _dataset_path(name: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", name)
    return os.path.normpath(os.path.join(DATASETS_PATH, f"{safe}.jsonl"))


def _ensure_dataset(name: str):
    path = _dataset_path(name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass


def _write_example(dataset_name: str, instruction: str, output: str, category: str = "web") -> bool:
    """Append one example. Returns True if written, False if duplicate or error."""
    if not instruction.strip() or not output.strip():
        return False
    if _is_duplicate(instruction + output, dataset_name):
        return False
    path = _dataset_path(dataset_name)
    _ensure_dataset(dataset_name)
    example = {
        "input": instruction.strip(),
        "output": output.strip(),
        "category": category,
        "created": int(time.time()),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logger.warning(f"Failed to write example to {dataset_name}: {e}")
        return False


# ── Wikipedia checkpoint ──────────────────────────────────────────────────────

def _read_wiki_offset() -> int:
    if os.path.exists(_WIKI_OFFSET_FILE):
        try:
            return int(open(_WIKI_OFFSET_FILE, "r").read().strip())
        except (ValueError, OSError):
            pass
    return 0


def _write_wiki_offset(offset: int):
    try:
        with open(_WIKI_OFFSET_FILE, "w") as f:
            f.write(str(offset))
    except OSError as e:
        logger.warning(f"Could not save Wikipedia offset: {e}")


# ── Domain detection ──────────────────────────────────────────────────────────

_SCIENCE_KEYWORDS = re.compile(
    r"\b(physics|chemistry|biology|mathematics|calculus|algebra|thermodynamic|"
    r"quantum|molecule|atom|element|reaction|equation|electr|magnetic|optic|"
    r"mechanical|engineering|material|force|energy|wave|radiation|nuclear|"
    r"aerospace|fluid|dynamics|mechanics|simulation|FEA|CFD|structural|"
    r"metallurgy|polymer|semiconductor|algorithm|computation|neural|signal)\b",
    re.IGNORECASE
)


def _detect_wiki_domain(title: str, text: str) -> str:
    """Return 'science' or 'general' based on article content."""
    probe = (title + " " + text[:500])
    if _SCIENCE_KEYWORDS.search(probe):
        return "science"
    return "general"


# ── Instruction templates ─────────────────────────────────────────────────────

_WIKI_TEMPLATES = [
    lambda title, _: f"Explain the concept of {title}.",
    lambda title, _: f"What is {title}? Provide a comprehensive explanation.",
    lambda title, _: f"Describe the key principles and significance of {title}.",
    lambda title, text: f"Summarize the following topic: {title}.",
    lambda title, _: f"Give a detailed overview of {title}, including its history, applications, and importance.",
]


def _chunk_text(text: str, max_tokens: int = 900) -> list[str]:
    """Split text into ~max_tokens word chunks (proxy: 1 token ≈ 0.75 words)."""
    max_words = int(max_tokens * 0.75)
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


# ── SOURCE: Wikipedia ─────────────────────────────────────────────────────────

def collect_wikipedia_batch(batch_size: int = 2000, dry_run: bool = False) -> dict:
    """
    Stream Wikipedia articles via HuggingFace datasets, starting from the
    stored checkpoint offset. Saves instruction/output pairs to:
      deltai-general-knowledge  (broad topics)
      deltai-science-knowledge  (science/engineering topics)
    Returns a stats dict.
    """
    result = {"source": "wikipedia", "written": 0, "skipped": 0, "errors": 0, "offset_start": 0, "offset_end": 0}

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        result["error"] = "datasets library not installed — run: pip install datasets"
        logger.warning(result["error"])
        return result

    offset = _read_wiki_offset()
    result["offset_start"] = offset

    if dry_run:
        result["status"] = "dry_run"
        result["offset_end"] = offset
        return result

    try:
        logger.info(f"Wikipedia: streaming from offset {offset}, batch={batch_size}")
        ds = load_dataset(
            "wikipedia", "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.skip(offset)

        processed = 0
        for article in ds:
            if processed >= batch_size:
                break

            title = (article.get("title") or "").strip()
            text = (article.get("text") or "").strip()

            if not title or len(text) < 100:
                processed += 1
                continue

            domain = _detect_wiki_domain(title, text)
            dataset_name = "deltai-science-knowledge" if domain == "science" else "deltai-general-knowledge"

            # Use first chunk only for output (keep examples focused)
            chunks = _chunk_text(text, max_tokens=900)
            if not chunks:
                processed += 1
                continue

            output_text = chunks[0]
            # Rotate instruction templates based on offset for variety
            template = _WIKI_TEMPLATES[(offset + processed) % len(_WIKI_TEMPLATES)]
            instruction = template(title, text)
            category = f"wikipedia-{domain}"

            if _write_example(dataset_name, instruction, output_text, category):
                result["written"] += 1
            else:
                result["skipped"] += 1

            processed += 1

        new_offset = offset + processed
        _write_wiki_offset(new_offset)
        result["offset_end"] = new_offset
        result["status"] = "ok"
        logger.info(f"Wikipedia: wrote {result['written']}, skipped {result['skipped']}, new offset={new_offset}")

    except Exception as e:
        result["errors"] += 1
        result["error"] = str(e)
        result["status"] = "error"
        logger.error(f"Wikipedia collection error: {e}")

    return result


# ── SOURCE: arXiv ─────────────────────────────────────────────────────────────

_ARXIV_CATEGORIES = [
    ("cs.LG", "Machine learning and AI methods"),
    ("physics.class-ph", "Classical physics and mechanics"),
    ("math.NA", "Numerical analysis and computational mathematics"),
    ("eess.SP", "Signal processing and audio analysis"),
    ("cond-mat.mtrl-sci", "Materials science and metallurgy"),
    ("physics.flu-dyn", "Fluid dynamics and aerodynamics"),
    ("cs.CE", "Computational engineering and simulation"),
    ("math.OC", "Optimization and control theory"),
]

_ARXIV_URL = "http://export.arxiv.org/api/query"


def collect_arxiv_batch(max_per_cat: int = 25, dry_run: bool = False) -> dict:
    """
    Query the arXiv XML API for recent papers across engineering/physics/math
    categories. Formats abstract as instruction/output pair.
    Saves to deltai-arxiv-papers (engineering domain).
    """
    result = {"source": "arxiv", "written": 0, "skipped": 0, "errors": 0}

    if dry_run:
        result["status"] = "dry_run"
        return result

    import xml.etree.ElementTree as ET  # stdlib

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    for cat_id, cat_desc in _ARXIV_CATEGORIES:
        try:
            resp = httpx.get(
                _ARXIV_URL,
                params={
                    "search_query": f"cat:{cat_id}",
                    "start": 0,
                    "max_results": max_per_cat,
                    "sortBy": "lastUpdatedDate",
                    "sortOrder": "descending",
                },
                timeout=30.0,
                headers={"User-Agent": "deltai/1.0 (training data collector)"},
            )
            if resp.status_code != 200:
                result["errors"] += 1
                continue

            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)

            for entry in entries:
                title_el = entry.find("atom:title", ns)
                abstract_el = entry.find("atom:summary", ns)
                if title_el is None or abstract_el is None:
                    continue

                title = re.sub(r"\s+", " ", title_el.text or "").strip()
                abstract = re.sub(r"\s+", " ", abstract_el.text or "").strip()

                if not title or len(abstract) < 80:
                    result["skipped"] += 1
                    continue

                instruction = f"Explain the research paper titled '{title}' and its key contributions."
                output = f"This paper falls under {cat_desc}. {abstract}"

                if _write_example("deltai-arxiv-papers", instruction, output, f"arxiv-{cat_id}"):
                    result["written"] += 1
                else:
                    result["skipped"] += 1

            time.sleep(0.5)  # be polite to arXiv

        except Exception as e:
            result["errors"] += 1
            logger.warning(f"arXiv error for {cat_id}: {e}")

    result["status"] = "ok" if result["errors"] == 0 else "partial"
    logger.info(f"arXiv: wrote {result['written']}, skipped {result['skipped']}, errors {result['errors']}")
    return result


# ── SOURCE: OpenF1 ────────────────────────────────────────────────────────────

_OPENF1_BASE = "https://api.openf1.org/v1"


def collect_openf1_batch(seasons: list[int] = None, dry_run: bool = False) -> dict:
    """
    Fetch race session data from OpenF1 API. Formats pit strategy, stint
    analysis, and race results as instruction/output pairs.
    Saves to deltai-openf1-strategy (racing domain).
    """
    result = {"source": "openf1", "written": 0, "skipped": 0, "errors": 0}

    if seasons is None:
        seasons = [2023, 2024]

    if dry_run:
        result["status"] = "dry_run"
        return result

    try:
        for season in seasons:
            # Get all race sessions for the season
            resp = httpx.get(
                f"{_OPENF1_BASE}/sessions",
                params={"year": season, "session_type": "Race"},
                timeout=20.0,
                headers={"User-Agent": "deltai/1.0 (training data collector)"},
            )
            if resp.status_code != 200:
                result["errors"] += 1
                continue

            sessions = resp.json()
            if not sessions:
                continue

            for session in sessions[:WEB_COLLECT_MAX_SOURCE]:
                session_key = session.get("session_key")
                circuit = session.get("circuit_short_name", "Unknown")
                gp_name = session.get("meeting_name", "Unknown GP")
                date_str = (session.get("date_start") or "")[:10]

                if not session_key:
                    continue

                # Get pit stops
                try:
                    pit_resp = httpx.get(
                        f"{_OPENF1_BASE}/pit",
                        params={"session_key": session_key},
                        timeout=15.0,
                    )
                    pits = pit_resp.json() if pit_resp.status_code == 200 else []
                except Exception:
                    pits = []

                # Get stints
                try:
                    stint_resp = httpx.get(
                        f"{_OPENF1_BASE}/stints",
                        params={"session_key": session_key},
                        timeout=15.0,
                    )
                    stints = stint_resp.json() if stint_resp.status_code == 200 else []
                except Exception:
                    stints = []

                if not pits and not stints:
                    result["skipped"] += 1
                    continue

                # Format strategy overview
                pit_count = len(pits)
                compound_seq = list({s.get("compound", "UNKNOWN") for s in stints if s.get("compound")})
                avg_lap_at_stop = "unknown"
                if pits:
                    lap_numbers = [p.get("lap_number") for p in pits if p.get("lap_number")]
                    if lap_numbers:
                        avg_lap_at_stop = f"{sum(lap_numbers)/len(lap_numbers):.1f}"

                instruction = (
                    f"Analyze the race strategy for the {season} {gp_name} at {circuit}. "
                    f"What were the key strategic decisions made?"
                )
                output = (
                    f"The {season} {gp_name} ({circuit}, {date_str}) featured {pit_count} pit stops across all drivers. "
                    f"Compounds used: {', '.join(compound_seq) if compound_seq else 'data unavailable'}. "
                    f"Average pit stop lap: {avg_lap_at_stop}. "
                    f"Teams balanced tire degradation against track position, with undercut/overcut windows "
                    f"influenced by compound performance and safety car periods."
                )

                if _write_example("deltai-openf1-strategy", instruction, output, f"openf1-{season}"):
                    result["written"] += 1
                else:
                    result["skipped"] += 1

                time.sleep(0.3)  # rate limit

    except Exception as e:
        result["errors"] += 1
        result["error"] = str(e)
        logger.error(f"OpenF1 collection error: {e}")

    result["status"] = "ok" if result["errors"] == 0 else "partial"
    logger.info(f"OpenF1: wrote {result['written']}, skipped {result['skipped']}, errors {result['errors']}")
    return result


# ── SOURCE: Semantic Scholar ──────────────────────────────────────────────────

_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_SCHOLAR_QUERIES = [
    ("engine knock detection acoustic", "audio"),
    ("turbocharger noise vibration analysis", "audio"),
    ("brake squeal finite element analysis", "audio"),
    ("racing tire thermal degradation model", "engineering"),
    ("CFD aerodynamics race car downforce", "engineering"),
    ("telemetry data anomaly detection motorsport", "engineering"),
    ("FEA fatigue analysis suspension components", "engineering"),
    ("modal analysis vibration signal processing", "audio"),
    ("acoustic emission bearing fault diagnosis", "audio"),
    ("heat transfer simulation automotive", "engineering"),
]


def collect_papers_batch(max_per_query: int = 10, dry_run: bool = False) -> dict:
    """
    Query Semantic Scholar for domain-specific paper abstracts.
    Saves to deltai-arxiv-papers (engineering domain).
    """
    result = {"source": "semantic_scholar", "written": 0, "skipped": 0, "errors": 0}

    if dry_run:
        result["status"] = "dry_run"
        return result

    for query, domain_hint in _SCHOLAR_QUERIES:
        try:
            resp = httpx.get(
                f"{_SCHOLAR_BASE}/paper/search",
                params={
                    "query": query,
                    "fields": "title,abstract,year,venue",
                    "limit": max_per_query,
                },
                timeout=20.0,
                headers={
                    "User-Agent": "deltai/1.0 (training data collector)",
                },
            )

            if resp.status_code == 429:
                logger.warning("Semantic Scholar rate limited — sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                result["errors"] += 1
                continue

            data = resp.json()
            papers = data.get("data", [])

            for paper in papers:
                title = (paper.get("title") or "").strip()
                abstract = (paper.get("abstract") or "").strip()
                year = paper.get("year") or ""
                venue = (paper.get("venue") or "").strip()

                if not title or len(abstract) < 80:
                    result["skipped"] += 1
                    continue

                instruction = f"Summarize the research on: {title}"
                output_parts = []
                if year:
                    output_parts.append(f"Published {year}")
                if venue:
                    output_parts.append(f"in {venue}")
                output_parts.append(f". {abstract}")

                output = " ".join(output_parts)

                if _write_example("deltai-arxiv-papers", instruction, output, f"scholar-{domain_hint}"):
                    result["written"] += 1
                else:
                    result["skipped"] += 1

            time.sleep(1.0)  # Semantic Scholar: 100 req/day without key

        except Exception as e:
            result["errors"] += 1
            logger.warning(f"Semantic Scholar error for '{query}': {e}")

    result["status"] = "ok" if result["errors"] == 0 else "partial"
    logger.info(f"Semantic Scholar: wrote {result['written']}, skipped {result['skipped']}")
    return result


# ── SOURCE: Motorsport Web ────────────────────────────────────────────────────

_MOTORSPORT_QUERIES = [
    "motorsport engineering tire degradation analysis",
    "F1 aerodynamics floor design 2024",
    "GT3 race car setup balance oversteer understeer",
    "endurance racing fuel strategy Le Mans",
    "iRacing sim racing telemetry coaching technique",
    "racing differential setup corner exit",
    "brake bias adjustment race car performance",
    "racing data analysis sector time improvement",
    "pit stop strategy tire compound selection",
    "engine mapping race fuel consumption optimization",
]

_DDG_LITE = "https://lite.duckduckgo.com/lite/"


def _ddg_search_urls(query: str, max_results: int = 3) -> list[str]:
    """Return top URL list from DuckDuckGo Lite."""
    try:
        resp = httpx.get(
            _DDG_LITE,
            params={"q": query, "kl": "us-en"},
            headers={"User-Agent": "deltai/1.0 (training collector)"},
            timeout=10.0,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        from urllib.parse import parse_qs, urlparse, unquote
        link_pat = re.compile(r'<a[^>]*href="([^"]*uddg=[^"]+)"', re.DOTALL)
        urls = []
        for raw in link_pat.findall(resp.text):
            try:
                parsed = parse_qs(urlparse(raw).query)
                actual = unquote(parsed.get("uddg", [raw])[0])
                if actual.startswith("http") and "duckduckgo" not in actual:
                    urls.append(actual)
                    if len(urls) >= max_results:
                        break
            except Exception:
                continue
        return urls
    except Exception:
        return []


def _fetch_page_text(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract clean article text from a URL using trafilatura."""
    try:
        import trafilatura  # type: ignore
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return (text or "")[:max_chars].strip()
    except ImportError:
        # Fallback: basic httpx + strip tags
        try:
            resp = httpx.get(url, timeout=10.0, headers={"User-Agent": "deltai/1.0"}, follow_redirects=True)
            if resp.status_code != 200:
                return ""
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception:
            return ""
    except Exception:
        return ""


def collect_motorsport_batch(max_pages: int = 20, dry_run: bool = False) -> dict:
    """
    Search DuckDuckGo for motorsport engineering articles, fetch page content
    with trafilatura, and save as instruction/output pairs.
    Saves to deltai-web-motorsport (racing domain).
    """
    result = {"source": "motorsport_web", "written": 0, "skipped": 0, "errors": 0}

    if dry_run:
        result["status"] = "dry_run"
        return result

    pages_fetched = 0
    for query in _MOTORSPORT_QUERIES:
        if pages_fetched >= max_pages:
            break

        urls = _ddg_search_urls(query, max_results=2)
        for url in urls:
            if pages_fetched >= max_pages:
                break

            text = _fetch_page_text(url, max_chars=4000)
            if len(text) < 200:
                result["skipped"] += 1
                continue

            instruction = f"What does motorsport engineering research say about: {query}?"
            output = text

            if _write_example("deltai-web-motorsport", instruction, output, "web-motorsport"):
                result["written"] += 1
                pages_fetched += 1
            else:
                result["skipped"] += 1

            time.sleep(1.0)

    result["status"] = "ok"
    logger.info(f"Motorsport web: wrote {result['written']}, skipped {result['skipped']}, errors {result['errors']}")
    return result


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_collection_cycle(
    dry_run: bool = False,
    sources: list[str] | None = None,
    wikipedia_batch: int | None = None,
    max_per_source: int | None = None,
) -> dict:
    """
    Orchestrate all enabled collectors. Called by run_daily_cycle() as Phase 0.5,
    and directly by scripts/collect_training_data.py.

    Args:
        dry_run: Log actions without writing any data.
        sources: List of source names to run, e.g. ['wikipedia', 'arxiv'].
                 None = run all enabled sources from .env.
        wikipedia_batch: Override WEB_COLLECT_WIKIPEDIA_BATCH.
        max_per_source: Override WEB_COLLECT_MAX_PER_SOURCE.

    Returns:
        Structured report dict compatible with run_daily_cycle() phases format.
    """
    import datetime

    report: dict = {
        "started_at": datetime.datetime.now().isoformat(),
        "dry_run": dry_run,
        "sources": {},
        "total_written": 0,
        "total_skipped": 0,
        "total_errors": 0,
        "status": "ok",
    }

    wiki_batch = wikipedia_batch if wikipedia_batch is not None else WEB_COLLECT_WIKI_BATCH
    max_src = max_per_source if max_per_source is not None else WEB_COLLECT_MAX_SOURCE

    # Determine which sources to run
    all_sources = {
        "wikipedia": WEB_COLLECT_WIKIPEDIA,
        "arxiv": WEB_COLLECT_ARXIV,
        "openf1": WEB_COLLECT_OPENF1,
        "papers": WEB_COLLECT_PAPERS,
        "motorsport": WEB_COLLECT_MOTORSPORT,
    }

    if sources:
        run_sources = {s: True for s in sources if s in all_sources}
    else:
        run_sources = {k: v for k, v in all_sources.items() if v}

    logger.info(f"Collection cycle: sources={list(run_sources.keys())}, dry_run={dry_run}")

    if "wikipedia" in run_sources:
        r = collect_wikipedia_batch(batch_size=wiki_batch, dry_run=dry_run)
        report["sources"]["wikipedia"] = r
        report["total_written"] += r.get("written", 0)
        report["total_skipped"] += r.get("skipped", 0)
        report["total_errors"] += r.get("errors", 0)

    if "arxiv" in run_sources:
        per_cat = max(5, max_src // len(_ARXIV_CATEGORIES))
        r = collect_arxiv_batch(max_per_cat=per_cat, dry_run=dry_run)
        report["sources"]["arxiv"] = r
        report["total_written"] += r.get("written", 0)
        report["total_skipped"] += r.get("skipped", 0)
        report["total_errors"] += r.get("errors", 0)

    if "openf1" in run_sources:
        r = collect_openf1_batch(dry_run=dry_run)
        report["sources"]["openf1"] = r
        report["total_written"] += r.get("written", 0)
        report["total_skipped"] += r.get("skipped", 0)
        report["total_errors"] += r.get("errors", 0)

    if "papers" in run_sources:
        per_q = max(3, max_src // len(_SCHOLAR_QUERIES))
        r = collect_papers_batch(max_per_query=per_q, dry_run=dry_run)
        report["sources"]["papers"] = r
        report["total_written"] += r.get("written", 0)
        report["total_skipped"] += r.get("skipped", 0)
        report["total_errors"] += r.get("errors", 0)

    if "motorsport" in run_sources:
        r = collect_motorsport_batch(max_pages=min(max_src, 20), dry_run=dry_run)
        report["sources"]["motorsport"] = r
        report["total_written"] += r.get("written", 0)
        report["total_skipped"] += r.get("skipped", 0)
        report["total_errors"] += r.get("errors", 0)

    if report["total_errors"] > 0 and report["total_written"] == 0:
        report["status"] = "error"
    elif report["total_errors"] > 0:
        report["status"] = "partial"

    report["finished_at"] = datetime.datetime.now().isoformat()
    logger.info(
        f"Collection cycle complete: written={report['total_written']}, "
        f"skipped={report['total_skipped']}, errors={report['total_errors']}, "
        f"status={report['status']}"
    )
    return report
