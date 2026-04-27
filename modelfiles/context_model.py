"""
deltai · Mythos context model
=============================
Generalized task-session state for the Mythos-style reason-act pipeline.
Designed to be backend-agnostic: swap Ollama for any inference provider,
swap SQLite for any persistence layer — the model stays the same.

Fork-friendly surface:
  - All phase dataclasses are standalone and independently replaceable.
  - TaskContext is the root; pass it end-to-end through FastAPI route handlers.
  - `extensions: dict` on every phase lets forks attach arbitrary fields
    without modifying the core schema.
  - ModelConfig is the only place provider-specific logic lives.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Lifecycle state of the top-level task."""
    PENDING    = "pending"       # Created, surface mapping not yet run
    MAPPING    = "mapping"       # Phase 1: ingesting context
    PLANNING   = "planning"      # Phase 2: generating/ranking hypotheses
    RUNNING    = "running"       # Phase 3: reason-act loop active
    VERIFYING  = "verifying"     # Phase 4: self-correction pass
    DONE       = "done"          # Phase 5: synthesis complete
    FAILED     = "failed"        # Unrecoverable error
    ABORTED    = "aborted"       # Cancelled by caller


class ConfidenceGrade(str, Enum):
    """
    Human-readable confidence band.
    Maps to numeric thresholds in ModelConfig.confidence_threshold.
    Forks can add grades (e.g. SPECULATIVE) without changing the pipeline.
    """
    LOW      = "low"       # < 0.4
    MEDIUM   = "medium"    # 0.4 – 0.74
    HIGH     = "high"      # 0.75 – 0.94
    VERIFIED = "verified"  # >= 0.95 (passed adversarial critique)


class ModelBackend(str, Enum):
    """
    Which inference backend handled this call.
    Add new values here when integrating additional providers.
    """
    OLLAMA    = "ollama"
    ANTHROPIC = "anthropic"
    LOCAL_API = "local_api"   # Generic escape hatch for forks
    MOCK      = "mock"        # Unit tests / dry runs


# ---------------------------------------------------------------------------
# Configuration  (provider-specific, lives outside TaskContext)
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """
    Runtime configuration injected at task creation.
    Nothing in the pipeline hardcodes a model name or endpoint.

    Fork guidance
    -------------
    Override primary_model / fallback_model to point at any Ollama tag,
    OpenAI-compatible endpoint, or Anthropic model string.
    confidence_threshold controls when Phase 4 forces a re-loop.
    max_loop_iterations is a hard safety cap on Phase 3.
    """
    primary_model:        str   = "qwen2.5:14b"
    fallback_model:       str   = "claude-sonnet-4-20250514"
    reasoning_model:      str   = "deepseek-r1:32b"     # Phase 2 heavy reasoning
    primary_backend:      ModelBackend = ModelBackend.OLLAMA
    fallback_backend:     ModelBackend = ModelBackend.ANTHROPIC
    ollama_base_url:      str   = "http://localhost:11434"
    confidence_threshold: float = Field(0.75, ge=0.0, le=1.0)
    max_loop_iterations:  int   = Field(6, ge=1, le=20)
    context_window:       int   = 8192
    temperature:          float = Field(0.2, ge=0.0, le=2.0)
    extensions:           dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1 — Surface Mapping
# ---------------------------------------------------------------------------

class ContextEntry(BaseModel):
    """
    A single unit of ingested context (file, memory chunk, peer message, etc).
    Source is intentionally untyped so forks can pass any identifier.
    """
    source:      str                  # e.g. "sqlite:sessions/42", "file:/home/user/notes.md"
    content:     str
    relevance:   float = Field(1.0, ge=0.0, le=1.0)
    tags:        list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extensions:  dict[str, Any] = Field(default_factory=dict)


class SurfaceMap(BaseModel):
    """
    Phase 1 output.
    Represents everything the model knows before it starts reasoning.
    """
    task_summary:       str                          # One-sentence restatement of the goal
    ingested_context:   list[ContextEntry] = Field(default_factory=list)
    component_graph:    dict[str, list[str]] = Field(default_factory=dict)
    # ^ adjacency map: component → [related components]. Forks can populate
    #   this from code ASTs, file dependency graphs, network topology, etc.
    token_budget_used:  int  = 0
    completed:          bool = False
    extensions:         dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 2 — Hypothesis Generation & Ranking
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    """
    A single candidate approach, scored before execution begins.
    The pipeline executes hypotheses in descending score order.
    """
    id:              str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description:     str
    score:           float = Field(..., ge=1.0, le=5.0)  # Mythos 1–5 scale
    rationale:       str   = ""
    test_cases:      list[str] = Field(default_factory=list)
    # test_cases: what would confirm or deny this hypothesis.
    # Used in Phase 4 to auto-generate verification prompts.
    rejected:        bool  = False
    reject_reason:   str   = ""
    extensions:      dict[str, Any] = Field(default_factory=dict)


class HypothesisTree(BaseModel):
    """
    Phase 2 output. Ordered list of hypotheses + the chosen path.
    Persisted to SQLite so Phase 4 can reference it during critique.
    """
    hypotheses:     list[Hypothesis] = Field(default_factory=list)
    chosen_id:      str | None       = None   # ID of the hypothesis currently being executed
    completed:      bool             = False
    extensions:     dict[str, Any]   = Field(default_factory=dict)

    def ranked(self) -> list[Hypothesis]:
        """Return non-rejected hypotheses sorted by score descending."""
        return sorted(
            [h for h in self.hypotheses if not h.rejected],
            key=lambda h: h.score,
            reverse=True,
        )

    def active(self) -> Hypothesis | None:
        if not self.chosen_id:
            return None
        return next((h for h in self.hypotheses if h.id == self.chosen_id), None)


# ---------------------------------------------------------------------------
# Phase 3 — Reason-Act Loop
# ---------------------------------------------------------------------------

class LoopIteration(BaseModel):
    """
    Single pass through the reason-act loop.
    One LoopIteration = one tool/model call + one reflection call.
    """
    iteration:       int
    backend_used:    ModelBackend
    action_prompt:   str                         # What was sent to the model
    action_response: str                         # Raw model output
    observation:     str                         # Parsed / summarized result
    reasoning:       str                         # Extended thinking: what to do next
    confidence:      float = Field(0.0, ge=0.0, le=1.0)
    should_continue: bool  = True               # False = loop halts after this iteration
    fallback_used:   bool  = False
    latency_ms:      int   = 0
    timestamp:       datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extensions:      dict[str, Any] = Field(default_factory=dict)


class ReasonActState(BaseModel):
    """
    Phase 3 mutable state. Grows one iteration per loop pass.
    The pipeline reads `current_iteration` to decide whether to continue.
    """
    iterations:        list[LoopIteration] = Field(default_factory=list)
    current_iteration: int   = 0
    halted:            bool  = False
    halt_reason:       str   = ""    # "confidence_met" | "max_iterations" | "error" | custom
    extensions:        dict[str, Any] = Field(default_factory=dict)

    def last(self) -> LoopIteration | None:
        return self.iterations[-1] if self.iterations else None

    def average_confidence(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(i.confidence for i in self.iterations) / len(self.iterations)


# ---------------------------------------------------------------------------
# Phase 4 — Self-Correction & Verification
# ---------------------------------------------------------------------------

class CritiqueNote(BaseModel):
    """
    A single adversarial observation produced during self-critique.
    severity: "info" | "warning" | "blocking"
    blocking notes force a re-loop back to Phase 3.
    """
    severity:    Literal["info", "warning", "blocking"]
    observation: str
    addressed:   bool = False
    extensions:  dict[str, Any] = Field(default_factory=dict)


class SelfCorrectionPass(BaseModel):
    """
    Phase 4 output. One pass = one adversarial critique prompt + response.
    Multiple passes allowed if blocking notes are found.
    """
    pass_number:       int
    critique_prompt:   str
    critique_response: str
    notes:             list[CritiqueNote] = Field(default_factory=list)
    confidence_after:  float = Field(0.0, ge=0.0, le=1.0)
    grade:             ConfidenceGrade = ConfidenceGrade.LOW
    requires_reloop:   bool = False
    completed:         bool = False
    timestamp:         datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extensions:        dict[str, Any] = Field(default_factory=dict)

    def blocking_notes(self) -> list[CritiqueNote]:
        return [n for n in self.notes if n.severity == "blocking" and not n.addressed]


# ---------------------------------------------------------------------------
# Phase 5 — Final Synthesis
# ---------------------------------------------------------------------------

class RemediationStep(BaseModel):
    """
    A single actionable step in the remediation plan.
    priority: 1 = highest. Forks can extend with owner, deadline, etc.
    """
    priority:   int
    action:     str
    rationale:  str = ""
    extensions: dict[str, Any] = Field(default_factory=dict)


class SynthesisReport(BaseModel):
    """
    Phase 5 output. The auditable, structured response that leaves deltai.
    This is what gets written to SQLite and optionally returned over the API.

    Fork guidance
    -------------
    result is the primary answer. Everything else is metadata.
    proof_of_concept and root_cause_analysis are optional — set them to ""
    if your use case doesn't require deep explanation.
    """
    result:               str
    root_cause_analysis:  str  = ""
    proof_of_concept:     str  = ""   # Reproduction steps / evidence
    remediation:          list[RemediationStep] = Field(default_factory=list)
    hypotheses_tried:     int  = 0
    loop_iterations_used: int  = 0
    final_confidence:     float = Field(0.0, ge=0.0, le=1.0)
    final_grade:          ConfidenceGrade = ConfidenceGrade.LOW
    backend_used:         ModelBackend = ModelBackend.OLLAMA
    fallback_triggered:   bool = False
    completed:            bool = False
    generated_at:         datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extensions:           dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root: TaskContext
# ---------------------------------------------------------------------------

class TaskContext(BaseModel):
    """
    The complete state of a single Mythos-pipeline task session.

    Pass this object end-to-end through your FastAPI route handlers.
    Serialize it to SQLite (json column or separate columns per phase)
    between loop iterations so the session survives restarts.

    Lifecycle
    ---------
    1. Create via TaskContext.new(goal, config)
    2. Populate surface_map    (Phase 1)
    3. Populate hypothesis_tree (Phase 2)
    4. Append to reason_act_state.iterations in a loop (Phase 3)
    5. Append to correction_passes (Phase 4); re-trigger Phase 3 if blocking
    6. Populate synthesis_report (Phase 5)
    7. Persist finalized context to SQLite; set status = DONE

    Fork guidance
    -------------
    peer_node_id: set this if the task originated from a deltai peer on
    the local network rather than directly from the user.
    user_id: optional; useful if you add multi-user support later.
    extensions on the root level is the widest escape hatch — anything that
    doesn't fit the phase models goes here.
    """
    task_id:           str      = Field(default_factory=lambda: str(uuid.uuid4()))
    goal:              str      # The raw user/system request, unmodified
    status:            TaskStatus = TaskStatus.PENDING
    config:            ModelConfig = Field(default_factory=ModelConfig)

    # Per-phase state — all optional until that phase runs
    surface_map:       SurfaceMap | None        = None
    hypothesis_tree:   HypothesisTree | None    = None
    reason_act_state:  ReasonActState           = Field(default_factory=ReasonActState)
    correction_passes: list[SelfCorrectionPass] = Field(default_factory=list)
    synthesis_report:  SynthesisReport | None   = None

    # Metadata
    created_at:        datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:        datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    peer_node_id:      str | None = None   # Origin peer if task came over the network
    user_id:           str | None = None
    tags:              list[str] = Field(default_factory=list)
    error_log:         list[str] = Field(default_factory=list)
    extensions:        dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, goal: str, config: ModelConfig | None = None) -> "TaskContext":
        """
        Standard entry point. Creates a pending task with sane defaults.
        Override config to change models, thresholds, or loop limits.
        """
        return cls(
            goal=goal,
            config=config or ModelConfig(),
        )

    # ------------------------------------------------------------------
    # Phase transition helpers
    # ------------------------------------------------------------------

    def advance(self, status: TaskStatus) -> None:
        """Move to the next lifecycle status and stamp updated_at."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def log_error(self, msg: str) -> None:
        self.error_log.append(f"{datetime.now(timezone.utc).isoformat()} — {msg}")
        self.updated_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def current_confidence(self) -> float:
        """Best available confidence estimate across whichever phase ran last."""
        if self.synthesis_report:
            return self.synthesis_report.final_confidence
        if self.correction_passes:
            return self.correction_passes[-1].confidence_after
        return self.reason_act_state.average_confidence()

    def should_fallback(self) -> bool:
        """
        True when confidence is below threshold after the last loop iteration.
        The pipeline calls this to decide whether to switch to the fallback model.
        """
        return self.current_confidence() < self.config.confidence_threshold

    def loop_exhausted(self) -> bool:
        return self.reason_act_state.current_iteration >= self.config.max_loop_iterations

    def to_sqlite_dict(self) -> dict[str, Any]:
        """
        Flat dict suitable for a single SQLite row.
        Phases are stored as JSON blobs — adjust column names to match your schema.
        """
        return {
            "task_id":          self.task_id,
            "goal":             self.goal,
            "status":           self.status.value,
            "surface_map":      self.surface_map.model_dump_json() if self.surface_map else None,
            "hypothesis_tree":  self.hypothesis_tree.model_dump_json() if self.hypothesis_tree else None,
            "reason_act_state": self.reason_act_state.model_dump_json(),
            "correction_passes":self.model_dump_json(include={"correction_passes"}),
            "synthesis_report": self.synthesis_report.model_dump_json() if self.synthesis_report else None,
            "confidence":       self.current_confidence(),
            "created_at":       self.created_at.isoformat(),
            "updated_at":       self.updated_at.isoformat(),
            "peer_node_id":     self.peer_node_id,
            "user_id":          self.user_id,
            "tags":             ",".join(self.tags),
            "error_log":        "\n".join(self.error_log),
        }
