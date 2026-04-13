"""
deltai Response Quality Scoring Engine

Heuristic-based quality scorer for chat responses. Produces a 0.0-1.0 score
from 6 weighted signals. Used to drive smart auto-capture, adaptive routing
feedback, and knowledge gap detection.

No LLM calls — all signals are computed from string analysis and metadata.
"""

import logging
import re

logger = logging.getLogger("deltai.quality")

# ── Signal Weights ───────────────────────────────────────────────────────
# Must sum to 1.0
_WEIGHTS = {
    "length_appropriateness": 0.20,
    "tool_success_rate": 0.20,
    "specificity": 0.20,
    "no_error_indicators": 0.15,
    "structural_match": 0.15,
    "no_repeat": 0.10,
}

# ── Error / hedging patterns ────────────────────────────────────────────
_ERROR_PATTERNS = [
    r"^error:",
    r"^exception:",
    r"^sorry,?\s+i\s+(can't|cannot|couldn't)",
    r"^i\s+(don't|do not)\s+know",
    r"^i'm\s+not\s+sure",
    r"^unfortunately,?\s+i",
    r"^i\s+apologize",
]
_ERROR_RE = [re.compile(p, re.IGNORECASE) for p in _ERROR_PATTERNS]

_HEDGING_PHRASES = [
    "i think",
    "i believe",
    "it might be",
    "it could be",
    "possibly",
    "not entirely sure",
    "i'm not certain",
    "take this with",
    "don't quote me",
    "i may be wrong",
]

# ── Specificity indicators ──────────────────────────────────────────────
_NUMBER_RE = re.compile(
    r"\b\d+\.?\d*\s*(?:MB|GB|KB|ms|s|min|hr|km/h|mph|kg|N|Pa|°C|°F|psi|bar|rpm|kW|HP|Nm|%)\b",
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")

# ── Engineering scaffolds ───────────────────────────────────────────────
_ENG_SCAFFOLDS = [
    "given:",
    "find:",
    "solve:",
    "solution:",
    "approach:",
    "step 1",
    "step 2",
    "therefore",
    "substituting",
    "result:",
]
_RACING_INDICATORS = [
    "lap",
    "stint",
    "tire",
    "tyre",
    "compound",
    "pit",
    "fuel",
    "sector",
    "delta",
    "gap",
    "position",
    "brake",
    "throttle",
    "speed",
    "corner",
]


def _score_length(response: str, tier: int) -> float:
    """
    Score response length appropriateness.
    Too short = not helpful. Too long = rambling.
    Expected length varies by complexity tier.
    """
    length = len(response)
    # Expected ranges by tier
    ranges = {
        1: (30, 800),  # simple: 30-800 chars
        2: (100, 2000),  # moderate: 100-2000 chars
        3: (200, 4000),  # complex: 200-4000 chars
    }
    min_len, max_len = ranges.get(tier, (50, 2000))

    if length < min_len:
        return max(0.0, length / min_len)
    if length > max_len * 1.5:
        # Gradually penalize excessive length
        overshoot = (length - max_len) / max_len
        return max(0.2, 1.0 - overshoot * 0.3)
    return 1.0


def _score_tool_success(metadata: dict) -> float:
    """Score based on tool call success rate."""
    tool_results = metadata.get("tool_results", [])
    if not tool_results:
        return 0.8  # no tools used — neutral-positive

    successes = sum(1 for r in tool_results if r.get("success", True))
    total = len(tool_results)
    if total == 0:
        return 0.8
    return successes / total


def _score_specificity(response: str) -> float:
    """Score presence of concrete, specific content."""
    score = 0.0
    # Numbers with units
    numbers = _NUMBER_RE.findall(response)
    score += min(0.4, len(numbers) * 0.1)

    # Code blocks
    code_blocks = _CODE_BLOCK_RE.findall(response)
    score += min(0.3, len(code_blocks) * 0.15)

    # Proper nouns (indicates specific references)
    proper_nouns = _PROPER_NOUN_RE.findall(response)
    score += min(0.2, len(proper_nouns) * 0.04)

    # Lists / bullet points
    bullets = response.count("\n- ") + response.count("\n* ") + response.count("\n1.")
    score += min(0.1, bullets * 0.03)

    return min(1.0, score)


def _score_no_errors(response: str) -> float:
    """Score absence of error indicators and hedging."""
    score = 1.0

    # Check error patterns (heavy penalty)
    lower = response.lower().strip()
    for pattern in _ERROR_RE:
        if pattern.search(lower):
            score -= 0.5
            break

    # Check hedging phrases (mild penalty)
    hedge_count = sum(1 for phrase in _HEDGING_PHRASES if phrase in lower)
    score -= min(0.3, hedge_count * 0.1)

    return max(0.0, score)


def _score_structural_match(response: str, domain: str) -> float:
    """Score whether response uses appropriate domain structure."""
    lower = response.lower()

    if domain == "engineering":
        # Look for engineering scaffolds
        matches = sum(1 for s in _ENG_SCAFFOLDS if s in lower)
        return min(1.0, matches * 0.2)

    if domain == "racing":
        # Look for data-driven racing language
        matches = sum(1 for s in _RACING_INDICATORS if s in lower)
        return min(1.0, matches * 0.15)

    if domain == "reasoning":
        # Look for structured reasoning
        reasoning_markers = [
            "because",
            "therefore",
            "however",
            "first",
            "second",
            "in conclusion",
            "the key",
            "this means",
        ]
        matches = sum(1 for s in reasoning_markers if s in lower)
        return min(1.0, matches * 0.15)

    # General: reward any structure
    has_structure = (
        "\n" in response  # multi-line
        and (response.count(". ") > 2 or "- " in response)  # multiple sentences or bullets
    )
    return 0.7 if has_structure else 0.4


def _score_no_repeat(response: str, recent_responses: list[str] | None = None) -> float:
    """
    Score novelty — is this response different from recent captures?
    Uses simple word overlap ratio as a proxy for semantic similarity.
    """
    if not recent_responses:
        return 1.0  # no comparison data

    response_words = set(response.lower().split())
    if not response_words:
        return 1.0

    max_overlap = 0.0
    for prev in recent_responses[-20:]:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        overlap = len(response_words & prev_words) / max(len(response_words), len(prev_words))
        max_overlap = max(max_overlap, overlap)

    # High overlap = low novelty
    if max_overlap > 0.8:
        return 0.1  # near-duplicate
    if max_overlap > 0.6:
        return 0.5  # similar
    return 1.0


# ── Recent response cache for dedup ─────────────────────────────────────
_recent_responses: list[str] = []
_RECENT_CACHE_SIZE = 50


def score_response(user_msg: str, assistant_msg: str, metadata: dict | None = None) -> dict:
    """
    Compute a quality score for a chat response.

    Args:
        user_msg: The user's query
        assistant_msg: deltai's response
        metadata: Optional dict with:
            - tool_calls: list of tool names called
            - tool_results: list of {"name": str, "success": bool}
            - tier: int (1/2/3 complexity)
            - domain: str (racing/engineering/reasoning/general)
            - react_used: bool

    Returns:
        {
            "score": 0.0-1.0,
            "signals": {signal_name: score, ...},
            "capture_decision": bool  (score >= threshold)
        }
    """
    global _recent_responses

    if metadata is None:
        metadata = {}

    tier = metadata.get("tier", 1)
    domain = metadata.get("domain") or "general"

    # Compute individual signals
    signals = {
        "length_appropriateness": _score_length(assistant_msg, tier),
        "tool_success_rate": _score_tool_success(metadata),
        "specificity": _score_specificity(assistant_msg),
        "no_error_indicators": _score_no_errors(assistant_msg),
        "structural_match": _score_structural_match(assistant_msg, domain),
        "no_repeat": _score_no_repeat(assistant_msg, _recent_responses),
    }

    # Weighted sum
    total = sum(signals[k] * _WEIGHTS[k] for k in signals)
    total = round(min(1.0, max(0.0, total)), 3)

    # Update recent cache
    _recent_responses.append(assistant_msg)
    if len(_recent_responses) > _RECENT_CACHE_SIZE:
        _recent_responses = _recent_responses[-_RECENT_CACHE_SIZE:]

    import os

    threshold = float(os.getenv("QUALITY_CAPTURE_THRESHOLD", "0.6"))

    return {
        "score": total,
        "signals": {k: round(v, 3) for k, v in signals.items()},
        "capture_decision": total >= threshold,
    }
