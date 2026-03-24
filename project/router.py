"""
E3N Smart Router — decides which model handles each message.

Architecture:
  LOCAL TIER (VRAM-aware):
    - Tier A: e3n-qwen14b (Qwen2.5-14B Q4, ~8.5GB) → GPU free, no sim running
    - Tier B: e3n-qwen3b (Qwen2.5-3B Q4, ~2.5GB)   → sim running, GPU contested
    - Tier C: e3n-qwen3b on CPU (0 VRAM)             → GPU maxed, emergency fallback
  CLOUD TIER (dormant until API key is set):
    - Sonnet             → moderate complexity
    - Opus               → heavy reasoning, race strategy, engineering

Routing signals:
  1. VRAM free   — determines which local model fits
  2. Sim running — auto-detect Le Mans Ultimate process
  3. Complexity  — classify Tier 1/2/3
  4. Connectivity — is Anthropic API reachable?
  5. User override — force local, force cloud, or auto
  6. Split workload — prep data locally, reason in cloud
  7. Cloud budget — stay within daily spend limit

Model cascade:
  GPU free:     14B (Tier 1) → Sonnet (Tier 2) → Opus (Tier 3)
  Sim running:  3B  (Tier 1) → Sonnet (Tier 2) → Opus (Tier 3)
  GPU maxed:    3B-CPU (Tier 1) → Cloud fallback (Tier 2/3)
  No cloud:     Best available local model for everything

Emergency backup (last resort only):
  If primary model fails after BACKUP_MAX_RETRIES attempts,
  the system engages a backup model as a lifeline.
  e3n-qwen14b → e3n-nemo → e3n → system down
  e3n-qwen3b  → e3n      → system down
"""

import os
import re
import time
import logging

logger = logging.getLogger("e3n.router")

# ── CONFIGURATION ───────────────────────────────────────────────────────

def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower().strip()
    return val in ("true", "1", "yes")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ── DYNAMIC QUANTIZATION TIERS ───────────────────────────────────────────
# Multiple quantization levels for smooth performance degradation.
# Each tier maps VRAM availability to a model variant with appropriate quant.
# Models must be pre-registered in Ollama (e.g. ollama create e3n-qwen14b-q6 -f ...)

_QUANT_TIERS_14B = [
    # (min_vram_mb, model_name, quant_label)
    (10000, os.getenv("E3N_STRONG_MODEL_Q6", ""), "Q6_K"),       # Best quality
    (8500,  os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b"), "Q4_K_M"),  # Current default
    (6500,  os.getenv("E3N_STRONG_MODEL_Q3", ""), "Q3_K_M"),    # Good quality, less VRAM
]

_QUANT_TIERS_3B = [
    (2500, os.getenv("E3N_MODEL", "e3n-qwen3b"), "Q4_K_M"),     # Current default
    (1200, os.getenv("E3N_MODEL_Q2", ""), "Q2_K"),               # Emergency minimal
]


def _pick_best_quant(model_tiers: list, vram_free: int) -> tuple[str, str] | None:
    """
    Pick the best quantization variant that fits in available VRAM.
    Returns (model_name, quant_label) or None if no model fits or no variants configured.
    """
    for min_vram, model_name, quant_label in model_tiers:
        if model_name and vram_free >= min_vram:
            return model_name, quant_label
    return None


# ── GPU / VRAM DETECTION ─────────────────────────────────────────────────

_nvml_initialized = False

def _ensure_nvml():
    global _nvml_initialized
    if not _nvml_initialized:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True


def get_gpu_utilization() -> int:
    """Returns GPU utilization percentage (0-100), or 0 if unavailable."""
    try:
        import pynvml
        _ensure_nvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except Exception:
        return 0


def _get_vram_info() -> tuple[int, int, int]:
    """Returns (used_mb, total_mb, free_mb) in a single pynvml call, or (0,0,0) if unavailable."""
    try:
        import pynvml
        _ensure_nvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = round(mem.used / 1e6)
        total = round(mem.total / 1e6)
        return used, total, max(0, total - used)
    except Exception:
        return 0, 0, 0


def get_vram_used_mb() -> int:
    """Returns VRAM used in MB, or 0 if unavailable."""
    return _get_vram_info()[0]


def get_vram_total_mb() -> int:
    """Returns total VRAM in MB, or 0 if unavailable."""
    return _get_vram_info()[1]


def get_vram_free_mb() -> int:
    """Returns free VRAM in MB."""
    return _get_vram_info()[2]


def is_gpu_loaded() -> bool:
    """Check if GPU is under heavy load (gaming, rendering, etc)."""
    threshold = _env_int("GPU_THRESHOLD", 70)
    usage = get_gpu_utilization()
    loaded = usage >= threshold
    if loaded:
        logger.info(f"GPU loaded: {usage}% (threshold: {threshold}%)")
    return loaded


# ── SIM DETECTION ────────────────────────────────────────────────────────

_SIM_PROCESS_NAMES = [
    "lemansultimate",      # Le Mans Ultimate (main executable)
    "le mans ultimate",
    "rfactor2",            # rFactor 2 (same engine)
]
# NOTE: Removed bare "lmu" — too broad, false-matches on other processes.
# LMU's actual process name is "Le Mans Ultimate.exe" → normalized to "lemansultimate"

_last_sim_check = 0
_last_sim_result = False
_SIM_CHECK_INTERVAL = 10  # seconds


def is_sim_running() -> bool:
    """
    Check if Le Mans Ultimate (or compatible sim) is running.
    Cached for 10 seconds to avoid hammering psutil.
    """
    global _last_sim_check, _last_sim_result

    now = time.time()
    if now - _last_sim_check < _SIM_CHECK_INTERVAL:
        return _last_sim_result

    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info["name"] or "").lower().replace(".exe", "").replace("_", "").replace(" ", "")
                for sim_name in _SIM_PROCESS_NAMES:
                    if sim_name.replace(" ", "") in name:
                        _last_sim_check = now
                        _last_sim_result = True
                        logger.info(f"Sim detected: {proc.info['name']}")
                        return True
            except Exception:
                continue
    except Exception:
        pass

    _last_sim_check = now
    _last_sim_result = False
    return False


# ── SESSION MODE (GPU protection for active racing) ────────────────────
_session_active = False
_session_started_at = 0.0
_session_last_ingest = 0.0
_SESSION_TIMEOUT = _env_int("SESSION_TIMEOUT_SEC", 60)
_SESSION_SOURCE_PATTERN = os.getenv("SESSION_SOURCE_PATTERN", "lmu").lower()
_SESSION_GPU_PROTECT = _env_bool("SESSION_GPU_PROTECT", True)
_SESSION_FORCE_CLOUD = _env_bool("SESSION_FORCE_CLOUD", True)


def is_session_active() -> bool:
    """Check if a racing session is active (manual or auto-detected via ingest)."""
    global _session_active
    if not _session_active:
        return False
    # Auto-deactivate after timeout without ingest
    if _session_last_ingest > 0 and (time.time() - _session_last_ingest) > _SESSION_TIMEOUT:
        _session_active = False
        logger.info("Session auto-deactivated (ingest timeout)")
        return False
    return True


def activate_session(manual: bool = False):
    """Manually activate session mode."""
    global _session_active, _session_started_at
    _session_active = True
    _session_started_at = time.time()
    logger.info(f"Session activated (manual={manual})")


def deactivate_session():
    """Manually deactivate session mode."""
    global _session_active
    _session_active = False
    logger.info("Session deactivated")


def activate_session_from_ingest(source: str):
    """Auto-activate session when ingest source matches pattern."""
    global _session_active, _session_started_at, _session_last_ingest
    if not _SESSION_GPU_PROTECT:
        return
    if _SESSION_SOURCE_PATTERN and _SESSION_SOURCE_PATTERN in source.lower():
        now = time.time()
        _session_last_ingest = now
        if not _session_active:
            _session_active = True
            _session_started_at = now
            logger.info(f"Session auto-activated from ingest source: {source}")


def get_session_status() -> dict:
    """Return session state for the /session/status endpoint."""
    return {
        "active": is_session_active(),
        "started_at": _session_started_at if _session_active else None,
        "last_ingest": _session_last_ingest if _session_last_ingest > 0 else None,
        "timeout_sec": _SESSION_TIMEOUT,
        "gpu_protect": _SESSION_GPU_PROTECT,
        "force_cloud": _SESSION_FORCE_CLOUD,
        "source_pattern": _SESSION_SOURCE_PATTERN,
    }


# ── TASK COMPLEXITY CLASSIFICATION ──────────────────────────────────────

# Tier 3 — needs Opus: deep reasoning, engineering, real-time strategy
TIER3_PATTERNS = [
    # Engineering / physics / math
    r"\b(derive|derivat|equation|integral|differential|simulation|simulate)\b",
    r"\b(stress|strain|torque|thermodynamic|fluid\s*dynamic|structural)\b",
    r"\b(free\s*body|moment of inertia|reynolds|navier.stokes|bernoulli)\b",
    r"\b(finite element|FEA|CFD|heat transfer|beam deflection)\b",
    r"\b(calculate|compute|solve).{0,30}(force|pressure|velocity|acceleration|energy)\b",
    # Race strategy (real-time decisions)
    r"\b(pit strategy|tire deg|undercut|overcut|fuel strat|race strat)\b",
    r"\b(gap analysis|stint plan|weather strat|safety car|VSC)\b",
    r"\b(optimal lap|racing line|sector analysis|telemetry analysis)\b",
    # Complex code / architecture
    r"\b(architect|refactor entire|system design|design pattern)\b",
    r"\b(debug.*complex|optimize.*algorithm|concurrency|distributed)\b",
    # Deep analysis
    r"\b(analyze.*in\s*depth|comprehensive.*review|detailed.*breakdown)\b",
    r"\b(compare.*tradeoffs|evaluate.*approach|pros.*cons.*detailed)\b",
    # Explicit deep request
    r"\b(use opus|deep think|think hard|heavy lifting|cloud mode)\b",
]

# Tier 2 — Sonnet handles well: moderate code, summaries, analysis
TIER2_PATTERNS = [
    # Code generation
    r"\b(write|create|build|implement)\b.{0,20}\b(code|script|function|class|module|component|endpoint)\b",
    r"\b(fix|debug|refactor|optimize)\b",
    # Analysis (non-deep)
    r"\b(summarize|explain|compare|review|analyze)\b",
    # Telemetry review (post-session, not real-time)
    r"\b(lap time|telemetry|sector|corner speed|braking point)\b",
    r"\b(review.*lap|session.*analysis|post.*race)\b",
    # Writing / documents
    r"\b(write|draft|compose|outline)\b.{0,20}\b(email|doc|report|essay|plan)\b",
    # Multi-step tasks
    r"\b(step.by.step|walk.*through|guide.*me|how (do|would|should))\b",
]

# Split workload patterns — tasks where local preps data, cloud reasons
SPLIT_PATTERNS = [
    # Telemetry + strategy (read data locally, analyze in cloud)
    r"\b(analyze|review).{0,20}(telemetry|lap|session|stint)\b",
    # Engineering with data (read specs locally, derive in cloud)
    r"\b(calculate|derive|simulate).{0,20}(from|using|with|based on)\b",
    # Code review with file reading (read file locally, review in cloud)
    r"\b(review|audit|improve).{0,20}(code|file|module|codebase)\b",
]


# ── ADAPTIVE ROUTING FEEDBACK ────────────────────────────────────────────
# Learns from quality scores which tier/model works for which domain.
# Soft adjustment: ±1 tier based on recent routing outcomes.

_routing_adjustment_cache: dict = {}  # {domain: (adjustment, cached_at)}
_ROUTING_CACHE_TTL = 600  # 10 minutes


def get_routing_adjustment(domain: str) -> int:
    """
    Get a routing tier adjustment based on recent quality feedback.
    Returns -1 (downgrade — model handles it), 0 (no change), +1 (upgrade needed).
    Cached for 10 minutes to avoid SQLite on every chat.
    """
    if not domain:
        return 0

    now = time.time()
    cached = _routing_adjustment_cache.get(domain)
    if cached and (now - cached[1]) < _ROUTING_CACHE_TTL:
        return cached[0]

    try:
        from persistence import get_routing_stats
        stats = get_routing_stats(domain, limit=100)
        if len(stats) < 10:
            _routing_adjustment_cache[domain] = (0, now)
            return 0  # not enough data

        # Analyze quality at each tier
        tier_scores = {}
        for s in stats:
            t = s["tier"]
            if t not in tier_scores:
                tier_scores[t] = []
            tier_scores[t].append(s["score"])

        adjustment = 0
        # If Tier 1 consistently bad → suggest upgrade
        if 1 in tier_scores and len(tier_scores[1]) >= 5:
            avg_t1 = sum(tier_scores[1]) / len(tier_scores[1])
            if avg_t1 < 0.5:
                adjustment = 1

        # If Tier 2 consistently great with local model → suggest downgrade
        if 2 in tier_scores and len(tier_scores[2]) >= 5:
            avg_t2 = sum(tier_scores[2]) / len(tier_scores[2])
            local_t2 = [s for s in stats if s["tier"] == 2 and "qwen" in (s["model"] or "").lower()]
            if local_t2 and avg_t2 > 0.8:
                adjustment = -1

        _routing_adjustment_cache[domain] = (adjustment, now)
        return adjustment
    except Exception:
        _routing_adjustment_cache[domain] = (0, now)
        return 0


def classify_complexity(message: str, domain: str = None) -> int:
    """
    Classify message complexity.
    Returns 1 (local), 2 (sonnet), or 3 (opus).
    Accepts optional domain for adaptive adjustment from feedback.
    """
    text = message.lower().strip()

    # Short simple messages
    if len(text) < 20 and not any(re.search(p, text) for p in TIER3_PATTERNS):
        return 1

    # Check Tier 3 first
    for pattern in TIER3_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 3

    # Check Tier 2
    for pattern in TIER2_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            base_tier = 2
            # Apply adaptive adjustment from feedback
            adjustment = get_routing_adjustment(domain) if domain else 0
            return max(1, min(3, base_tier + adjustment))

    # Long complex messages
    if len(text) > 300 and ("?" in text or "how" in text or "why" in text):
        return 2

    return 1


def is_split_workload(message: str) -> bool:
    """
    Check if this task benefits from split workload:
    local model preps/reads data, cloud model reasons over it.
    Only relevant when cloud is available.
    """
    text = message.lower().strip()
    for pattern in SPLIT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ── TELEMETRY QUERY CLASSIFICATION ─────────────────────────────────────

TELEMETRY_PATTERNS = {
    "telemetry_lookup": [
        r"\b(fuel remaining|fuel left|how much fuel)\b",
        r"\b(tire temps?|tyre temps?|tire temperature|tyre temperature)\b",
        r"\b(tire pressure|tyre pressure|tire wear|tyre wear)\b",
        r"\b(current lap|last lap|best lap|lap time)\b",
        r"\b(position|standings|gap to|gap ahead|gap behind)\b",
        r"\b(sector time|sector \d|s1|s2|s3)\b",
        r"\b(speed trap|top speed|vmax)\b",
        r"\b(brake temp|brake bias|brake balance)\b",
        r"\b(water temp|oil temp|engine temp)\b",
        r"\b(drs|ers|battery|energy)\b",
        r"\b(how many laps)\b(?!.{0,15}(tire|tyre|fuel))",
        r"\b(remaining laps|laps to go)\b",
    ],
    "telemetry_coaching": [
        r"\b(why am i slow|where am i losing|losing time)\b",
        r"\b(improve|faster|quicker|better)\b.{0,20}\b(lap|sector|corner|time)\b",
        r"\b(braking point|turn.in|apex|exit speed|racing line)\b",
        r"\b(oversteer|understeer|snap|loose|tight)\b",
        r"\b(compare.*lap|delta|diff.*best)\b",
        r"\b(coach|coaching|driving tip|technique)\b",
    ],
    "telemetry_strategy": [
        r"\b(pit now|should i pit|pit window|pit stop)\b",
        r"\b(tire strategy|tyre strategy|compound|stint)\b",
        r"\b(fuel strategy|fuel save|fuel target|lift.and.coast)\b",
        r"\b(undercut|overcut|gap for pit)\b",
        r"\b(weather change|weather changing|rain|dry|inters|wets|intermediates)\b",
        r"\b(safety car|vsc|yellow flag)\b",
        r"\b(damage|hit.*wall|wing damage|puncture|crashed)\b",
        r"\b(how many laps|laps remaining|laps left).{0,15}(tire|tyre|fuel|stint)\b",
        r"\b(what tire|what tyre|which compound|which tire)\b",
    ],
    "telemetry_debrief": [
        r"\b(race analysis|race review|race debrief|post.?race)\b",
        r"\b(session analysis|session review|session summary)\b",
        r"\b(full analysis|complete review|detailed breakdown)\b",
        r"\b(stint comparison|stint analysis|race pace)\b",
    ],
}


def classify_telemetry_category(message: str) -> str | None:
    """
    Classify if a message is telemetry-related and what category.
    Returns category string or None for non-telemetry queries.
    Only meaningful when session is active or recent telemetry data exists.

    Check order: debrief → strategy → coaching → lookup (most specific first).
    """
    text = message.lower().strip()
    # Check in priority order: more specific categories first
    for category in ("telemetry_debrief", "telemetry_strategy", "telemetry_coaching", "telemetry_lookup"):
        patterns = TELEMETRY_PATTERNS.get(category, [])
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return category
    return None


# ── ADAPTER DOMAIN CLASSIFIER ──────────────────────────────────────────
# Maps queries to augmentation slot domains for adapter-aware routing.

ADAPTER_DOMAIN_PATTERNS = {
    "racing": [
        r"\b(racing|race|lap\s*time|tire|tyre|pit\s*stop|stint|telemetry)\b",
        r"\b(overtake|undercut|overcut|draft|slipstream|DRS)\b",
        r"\b(qualifying|pole|grid|safety\s*car|yellow\s*flag)\b",
        r"\b(compound|soft|medium|hard|intermediate|wet)\b",
        r"\b(sector|split|delta|gap|interval)\b",
        r"\b(downforce|aero|wing|ride\s*height|rake)\b",
        r"\b(LMU|Le\s*Mans|endurance|prototype|GT[34]?)\b",
        r"\b(fuel\s*load|fuel\s*save|lift.and.coast)\b",
        r"\b(setup|spring\s*rate|damper|ARB|anti.roll)\b",
    ],
    "engineering": [
        r"\b(derive|equation|integral|differential|simulation)\b",
        r"\b(stress|strain|torque|thermodynamic|fluid\s*dynamic)\b",
        r"\b(free\s*body|FBD|moment\s*of\s*inertia|reynolds|bernoulli)\b",
        r"\b(finite\s*element|FEA|CFD|heat\s*transfer)\b",
        r"\b(calculate|compute|solve).{0,30}(force|pressure|velocity|temperature|mass)\b",
        r"\b(engineering|physics|calculus|statics|thermo|materials)\b",
        r"\b(Newton|Euler|Lagrange|Hooke|Fourier|Navier.Stokes)\b",
        r"\b(beam|truss|shear|bending|deflection|buckling)\b",
        r"\b(matrix|eigenvalue|determinant|linear\s*algebra)\b",
        r"\b(Carnot|Rankine|entropy|enthalpy|specific\s*heat)\b",
    ],
    "reasoning": [
        r"\b(analyze|compare|evaluate|reason|decompose|break\s*down)\b",
        r"\b(step.by.step|chain.of.thought|think.through|logical)\b",
        r"\b(pro.*con|tradeoff|trade.off|decision|weigh.option)\b",
        r"\b(summarize|synthesize|critique|assess|review)\b",
        r"\b(what\s+are\s+the\s+implications|how\s+does\s+this\s+affect)\b",
    ],
    # "personality" is always-on (blended into base), not query-triggered
}

def classify_adapter_domain(message: str) -> str | None:
    """
    Classify a query into an adapter domain for augmentation slot routing.
    Returns primary domain string or None for general queries.
    """
    domains = classify_adapter_domains(message)
    return domains[0] if domains else None


def classify_adapter_domains(message: str) -> list[str]:
    """
    Classify a query into ALL matching adapter domains, sorted by relevance.
    Supports cross-domain queries (e.g., racing + engineering).
    Returns list of domain strings (highest score first), or empty list.
    """
    text = message.lower().strip()
    scores = {}
    for domain, patterns in ADAPTER_DOMAIN_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[domain] = score
    if not scores:
        return []
    # Return all matching domains sorted by score (highest first)
    return sorted(scores, key=scores.get, reverse=True)


# ── MIXTURE-OF-LORA ROUTING ──────────────────────────────────────────────
# Instead of always using the TIES-merged model, dynamically select which
# LoRA adapter to apply at inference time for best per-domain quality.

_MIXTURE_LORA_ENABLED = _env_bool("MIXTURE_LORA_ENABLED", False)


def resolve_adapter_model(base_model: str, adapter_domain: str | None) -> tuple[str, str | None]:
    """
    Resolve the best model to use given the adapter domain.

    If a domain-specific adapter is active in the registry AND Mixture-of-LoRA
    is enabled, returns the adapter's Ollama model name.
    Otherwise returns the base model unchanged.

    Returns (model_name, adapter_name_or_none).
    """
    if not _MIXTURE_LORA_ENABLED or not adapter_domain:
        return base_model, None

    try:
        from training import get_active_adapters, _load_registry
        active = get_active_adapters()
        adapter_name = active.get(adapter_domain)
        if not adapter_name:
            return base_model, None

        registry = _load_registry()
        adapter_entry = registry.get("adapters", {}).get(adapter_name, {})
        # Check if adapter has an Ollama model registered
        ollama_model = adapter_entry.get("ollama_model")
        if ollama_model:
            logger.info(f"MoLoRA: routing to domain adapter {adapter_name} ({ollama_model}) for {adapter_domain}")
            return ollama_model, adapter_name

        return base_model, None
    except Exception as e:
        logger.debug(f"MoLoRA resolution failed: {e}")
        return base_model, None


# ── CONNECTIVITY CHECK ──────────────────────────────────────────────────

_last_connectivity_check = 0
_last_connectivity_result = False

async def is_cloud_available() -> bool:
    """
    Check if the Anthropic API is reachable.
    Caches result for 30 seconds.
    """
    global _last_connectivity_check, _last_connectivity_result

    now = time.time()
    if now - _last_connectivity_check < 30:
        return _last_connectivity_result

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        _last_connectivity_check = now
        _last_connectivity_result = False
        return False

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
        reachable = resp.status_code in (200, 400, 401, 405, 429)
        _last_connectivity_check = now
        _last_connectivity_result = reachable
        if not reachable:
            logger.warning(f"Cloud check failed: HTTP {resp.status_code}")
        return reachable
    except Exception as e:
        logger.warning(f"Cloud unreachable: {e}")
        _last_connectivity_check = now
        _last_connectivity_result = False
        return False


def is_cloud_available_sync() -> bool:
    """Synchronous version for non-async contexts (e.g., /router/status)."""
    now = time.time()
    if now - _last_connectivity_check < 30:
        return _last_connectivity_result
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return bool(api_key) and _last_connectivity_result


def has_api_key() -> bool:
    """Check if an API key is configured (doesn't test validity)."""
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# ── CLOUD BUDGET TRACKING ───────────────────────────────────────────────

_daily_cloud_spend = 0.0
_daily_cloud_reset_date = ""


def init_budget_from_db():
    """Load today's spend from SQLite at startup. Safe to call if persistence unavailable."""
    global _daily_cloud_spend, _daily_cloud_reset_date
    try:
        from persistence import load_budget
        today = time.strftime("%Y-%m-%d")
        _daily_cloud_spend = load_budget(today)
        _daily_cloud_reset_date = today
        logger.info(f"Budget loaded from DB: ${_daily_cloud_spend:.4f} for {today}")
    except Exception as e:
        logger.warning(f"Could not load budget from DB: {e}")


def _check_budget() -> bool:
    """Returns True if cloud budget has remaining capacity."""
    global _daily_cloud_spend, _daily_cloud_reset_date

    today = time.strftime("%Y-%m-%d")
    if today != _daily_cloud_reset_date:
        _daily_cloud_spend = 0.0
        _daily_cloud_reset_date = today
        # Persist the new day's zero spend
        try:
            from persistence import save_budget
            save_budget(today, 0.0)
        except Exception:
            pass

    budget = _env_float("CLOUD_BUDGET_DAILY", 5.0)
    return _daily_cloud_spend < budget


def record_cloud_usage(input_tokens: int, output_tokens: int, model: str):
    """Record estimated cost of a cloud API call."""
    global _daily_cloud_spend

    # Approximate pricing (per 1M tokens)
    if "opus" in model.lower():
        cost = (input_tokens * 15.0 + output_tokens * 75.0) / 1_000_000
    else:  # sonnet
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    _daily_cloud_spend += cost
    logger.info(f"Cloud cost: ${cost:.4f} (daily total: ${_daily_cloud_spend:.4f})")

    # Persist to SQLite
    try:
        from persistence import save_budget
        save_budget(_daily_cloud_reset_date or time.strftime("%Y-%m-%d"), _daily_cloud_spend)
    except Exception as e:
        logger.warning(f"Failed to persist budget: {e}")


def get_budget_status() -> dict:
    """Return current budget status."""
    budget = _env_float("CLOUD_BUDGET_DAILY", 5.0)
    return {
        "daily_budget": budget,
        "daily_spent": round(_daily_cloud_spend, 4),
        "daily_remaining": round(max(0, budget - _daily_cloud_spend), 4),
        "budget_ok": _check_budget(),
    }


# ── EMERGENCY BACKUP SYSTEM ─────────────────────────────────────────────


def get_backup_model(primary: str) -> str | None:
    """
    Map a primary model to its emergency backup.
    Only called after primary has failed BACKUP_MAX_RETRIES attempts.
    Returns None if backups are disabled or no backup exists.

    Chain:
      e3n-qwen14b → e3n-nemo → e3n → None
      e3n-qwen3b  → e3n      → None
    """
    if not _env_bool("BACKUP_ENABLED", True):
        return None

    backup_strong = os.getenv("E3N_BACKUP_STRONG_MODEL", "e3n-nemo").strip()
    backup_default = os.getenv("E3N_BACKUP_MODEL", "e3n").strip()

    strong_model = os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b").strip()
    default_model = os.getenv("E3N_MODEL", "e3n-qwen3b").strip()

    mapping = {
        strong_model: backup_strong,     # e3n-qwen14b → e3n-nemo
        default_model: backup_default,   # e3n-qwen3b  → e3n
        backup_strong: backup_default,   # e3n-nemo    → e3n (second fallback)
    }
    return mapping.get(primary)


async def check_model_health(model: str) -> bool:
    """
    Minimal 1-token ping to verify a model is functional.
    Used by the hourly background health check only.
    """
    try:
        import httpx
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            return "message" in data
    except Exception as e:
        logger.warning(f"Health check failed for {model}: {e}")
        return False


async def check_model_exists(model: str) -> bool:
    """
    Check if a model exists in Ollama's registry (does not load it).
    Used at startup only.
    """
    try:
        import httpx
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(model in m or m.startswith(model + ":") for m in models)
    except Exception as e:
        logger.warning(f"Model existence check failed for {model}: {e}")
        return False


# ── MAIN ROUTER ─────────────────────────────────────────────────────────

class RouteDecision:
    """Describes which model to use and why."""
    def __init__(self, backend: str, model: str, tier: int, reason: str,
                 split: bool = False, gpu_loaded: bool = False,
                 sim_running: bool = False, cpu_only: bool = False,
                 backup_model: str | None = None,
                 query_category: str = "general",
                 adapter_domain: str | None = None,
                 num_gpu: int | None = None,
                 num_ctx: int | None = None):
        self.backend = backend      # "ollama" or "anthropic"
        self.model = model          # model name/id
        self.tier = tier            # 1, 2, or 3
        self.reason = reason        # human-readable explanation
        self.split = split          # whether to use split workload
        self.gpu_loaded = gpu_loaded
        self.sim_running = sim_running
        self.cpu_only = cpu_only
        self.backup_model = backup_model  # emergency fallback (None = no backup)
        self.query_category = query_category
        self.adapter_domain = adapter_domain  # augmentation slot domain (racing/engineering/etc)
        self.num_gpu = num_gpu      # number of GPU layers (None = Ollama default / all)
        self.num_ctx = num_ctx      # context window size (None = model default)

    def to_dict(self) -> dict:
        """Serialize for frontend. backup_model excluded — invisible infrastructure."""
        d = {
            "backend": self.backend,
            "model": self.model,
            "tier": self.tier,
            "reason": self.reason,
            "split": self.split,
            "gpu_loaded": self.gpu_loaded,
            "sim_running": self.sim_running,
            "cpu_only": self.cpu_only,
            "query_category": self.query_category,
            "adapter_domain": self.adapter_domain,
        }
        if self.num_gpu is not None:
            d["num_gpu"] = self.num_gpu
        if self.num_ctx is not None:
            d["num_ctx"] = self.num_ctx
        return d

    def __repr__(self):
        tag = f"Route({self.backend}/{self.model}, tier={self.tier}"
        if self.split:
            tag += ", SPLIT"
        if self.sim_running:
            tag += ", SIM"
        if self.cpu_only:
            tag += ", CPU"
        if self.query_category != "general":
            tag += f", cat={self.query_category}"
        return tag + f", reason={self.reason})"


def _calc_num_gpu(model_name: str, vram_free: int) -> int | None:
    """
    Calculate optimal number of GPU layers for partial offloading.
    Returns None for full GPU, 0 for full CPU, or N for partial offload.

    14B Q4_K_M has ~40 layers, ~210MB per layer on GPU.
    3B Q4_K_M has ~32 layers, ~78MB per layer on GPU.
    """
    if "14b" in model_name.lower():
        total_layers = 40
        mb_per_layer = 210
    elif "3b" in model_name.lower():
        total_layers = 32
        mb_per_layer = 78
    else:
        return None  # unknown model, let Ollama decide

    # Reserve 500MB for KV cache and overhead
    usable_vram = max(0, vram_free - 500)
    max_layers = min(total_layers, int(usable_vram / mb_per_layer))

    if max_layers >= total_layers:
        return None  # full GPU — no need to specify
    if max_layers <= 0:
        return 0     # full CPU
    return max_layers


def _calc_num_ctx(vram_free: int, sim_active: bool) -> int | None:
    """
    Dynamically adjust context window based on VRAM pressure.
    Smaller context = less KV cache VRAM usage.
    Returns None for default (model decides), or explicit value.
    """
    if sim_active and vram_free < 4000:
        return 1024  # minimal context during racing
    if vram_free < 2000:
        return 1024
    if vram_free < 4000:
        return 2048
    return None  # default (typically 4096)


def _pick_local_model() -> tuple[str, bool, str | None, int | None, int | None]:
    """
    Pick the best local model based on VRAM availability and sim status.
    Returns (model_name, cpu_only, backup_model, num_gpu, num_ctx).

    Strategy:
      VRAM free > TIER_A_MIN (9GB)  → strong model (14B), full GPU
      VRAM free > TIER_AB_MIN (5GB) → strong model (14B), partial GPU offload
      VRAM free > TIER_B_MIN (3GB)  → sim model (3B on GPU)
      VRAM free < TIER_B_MIN        → sim model (3B on CPU)

    num_gpu: number of GPU layers for partial offloading (None = all layers).
    num_ctx: context window size override (None = model default).
    backup_model is the emergency fallback if the primary fails.
    """
    strong_model = os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b")
    default_model = os.getenv("E3N_MODEL", "e3n-qwen3b")
    sim_model = os.getenv("E3N_SIM_MODEL", default_model)

    tier_a_min = _env_int("VRAM_TIER_A_MIN_MB", 9000)
    tier_ab_min = _env_int("VRAM_TIER_AB_MIN_MB", 5000)  # partial offload zone
    tier_b_min = _env_int("VRAM_TIER_B_MIN_MB", 3000)

    vram_free = get_vram_free_mb()
    sim_active = is_sim_running()
    num_ctx = _calc_num_ctx(vram_free, sim_active)

    # If sim is running, always prefer the small model to save VRAM
    if sim_active:
        if vram_free >= tier_b_min:
            return sim_model, False, get_backup_model(sim_model), None, num_ctx
        else:
            return sim_model, True, get_backup_model(sim_model), 0, num_ctx

    # No sim — pick based on available VRAM with dynamic quantization
    # Try higher-quality quant tiers first (14B variants)
    quant_14b = _pick_best_quant(_QUANT_TIERS_14B, vram_free)
    if quant_14b and vram_free >= tier_ab_min:
        quant_model, quant_label = quant_14b
        if vram_free >= tier_a_min:
            return quant_model, False, get_backup_model(quant_model), None, num_ctx
        else:
            # Partial offload for mid-VRAM
            ngpu = _calc_num_gpu(quant_model, vram_free)
            return quant_model, False, get_backup_model(quant_model), ngpu, num_ctx

    # Fall back to 3B tiers
    quant_3b = _pick_best_quant(_QUANT_TIERS_3B, vram_free)
    if quant_3b and vram_free >= tier_b_min:
        quant_model, quant_label = quant_3b
        return quant_model, False, get_backup_model(quant_model), None, num_ctx

    # CPU fallback
    return default_model, True, get_backup_model(default_model), 0, num_ctx


async def route(message: str, force_cloud: bool = False, force_local: bool = False) -> RouteDecision:
    """
    Decide which model should handle this message.

    Args:
        message: The user's input
        force_cloud: User explicitly requested cloud (deep mode toggle)
        force_local: User explicitly disabled cloud

    Returns:
        RouteDecision with backend, model, tier, reason, split flag
    """
    cloud_enabled = _env_bool("CLOUD_ENABLED", True) and not force_local
    sonnet_model = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514")
    opus_model = os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-20250514")

    # Pre-classify adapter domain for routing feedback
    adapter_domain = classify_adapter_domain(message)

    tier = classify_complexity(message, domain=adapter_domain)
    local_model, cpu_only, backup, num_gpu, num_ctx = _pick_local_model()
    sim_active = is_sim_running()
    gpu_loaded = is_gpu_loaded()
    split = is_split_workload(message)

    query_cat = "general"
    # Classify telemetry queries when session is active or sim is running
    if is_session_active() or is_sim_running():
        tel_cat = classify_telemetry_category(message)
        if tel_cat:
            query_cat = tel_cat

    # Mixture-of-LoRA: resolve domain-specific adapter model (adapter_domain classified above)
    resolved_model, active_adapter = resolve_adapter_model(local_model, adapter_domain)
    if active_adapter:
        local_model = resolved_model

    cloud_ready = (cloud_enabled and has_api_key()
                   and await is_cloud_available() and _check_budget())

    vram_free = get_vram_free_mb()

    # Helper: common kwargs for local (ollama) decisions
    _local_kw = dict(num_gpu=num_gpu, num_ctx=num_ctx)

    # Partial offload label for reason strings
    _offload_note = ""
    if num_gpu is not None and num_gpu > 0:
        _offload_note = f" ({num_gpu}L GPU)"
    elif num_gpu == 0 or cpu_only:
        _offload_note = " (CPU)"

    # ── Force cloud ──
    if force_cloud:
        if cloud_ready:
            model = opus_model if tier == 3 else sonnet_model
            return RouteDecision("anthropic", model, tier,
                                 "user requested cloud", split=split,
                                 sim_running=sim_active, backup_model=backup,
                                 query_category=query_cat,
                                 adapter_domain=adapter_domain)
        return RouteDecision("ollama", local_model, tier,
                             "cloud requested but unavailable — local fallback",
                             gpu_loaded=gpu_loaded, sim_running=sim_active,
                             cpu_only=cpu_only, backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    # ── Force local ──
    if force_local:
        return RouteDecision("ollama", local_model, tier,
                             f"user forced local — {local_model}{_offload_note}"
                             f"{' (sim active)' if sim_active else ''}",
                             gpu_loaded=gpu_loaded, sim_running=sim_active,
                             cpu_only=cpu_only, backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    # ── Sim running — VRAM is precious ──
    if sim_active:
        if tier >= 2 and cloud_ready:
            model = opus_model if tier == 3 else sonnet_model
            return RouteDecision("anthropic", model, tier,
                                 f"sim running, VRAM {vram_free}MB free — "
                                 f"complex task routed to cloud",
                                 split=split, sim_running=True,
                                 gpu_loaded=gpu_loaded, backup_model=backup,
                                 query_category=query_cat,
                                 adapter_domain=adapter_domain)
        return RouteDecision("ollama", local_model, tier,
                             f"sim running, VRAM {vram_free}MB free — "
                             f"{local_model}{_offload_note}",
                             sim_running=True, gpu_loaded=gpu_loaded,
                             cpu_only=cpu_only, backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    # ── GPU loaded (no sim, but something else using GPU) ──
    if gpu_loaded:
        if cloud_ready:
            model = opus_model if tier == 3 else sonnet_model
            return RouteDecision("anthropic", model, tier,
                                 f"GPU loaded ({get_gpu_utilization()}%) — routing to cloud",
                                 split=split, gpu_loaded=True, backup_model=backup,
                                 query_category=query_cat,
                                 adapter_domain=adapter_domain)
        return RouteDecision("ollama", local_model, tier,
                             f"GPU loaded ({get_gpu_utilization()}%) — "
                             f"using {local_model}{_offload_note}",
                             gpu_loaded=True, cpu_only=cpu_only, backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    # ── Normal routing by tier ──
    if tier == 1:
        return RouteDecision("ollama", local_model, 1,
                             f"simple task — {local_model}{_offload_note}",
                             backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    if tier == 2:
        if cloud_ready:
            return RouteDecision("anthropic", sonnet_model, 2,
                                 "moderate complexity — Sonnet",
                                 split=split, backup_model=backup,
                                 query_category=query_cat,
                                 adapter_domain=adapter_domain)
        return RouteDecision("ollama", local_model, 2,
                             f"moderate complexity — no cloud, using {local_model}{_offload_note}",
                             backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    if tier == 3:
        if cloud_ready:
            return RouteDecision("anthropic", opus_model, 3,
                                 "heavy reasoning — Opus",
                                 split=split, backup_model=backup,
                                 query_category=query_cat,
                                 adapter_domain=adapter_domain)
        return RouteDecision("ollama", local_model, 3,
                             f"heavy reasoning — no cloud, using {local_model}{_offload_note} (best effort)",
                             backup_model=backup,
                             query_category=query_cat,
                             adapter_domain=adapter_domain, **_local_kw)

    return RouteDecision("ollama", local_model, 1,
                         f"default — {local_model}{_offload_note}",
                         backup_model=backup,
                         query_category=query_cat,
                         adapter_domain=adapter_domain, **_local_kw)
