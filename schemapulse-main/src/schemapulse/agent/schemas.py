"""Pydantic schemas for model output.

The model is treated as **powerful but untrusted**. Everything it returns is
parsed into these types before any other part of the system sees it, and the
SQL inside is then re-validated against the PostgreSQL AST by
:mod:`schemapulse.safety.candidate_validator`.

Two rules shape this schema:

1. ``extra="forbid"`` — a model that invents fields is a model that misunderstood
   the task, and we would rather know than silently drop data.
2. No chain-of-thought.  ``reasoning_summary`` is a short *evidence* summary
   that a user can read and check, not an internal monologue. Models do not need
   to be asked for their hidden reasoning to be useful, and not displaying it
   keeps the demo honest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CandidateType = Literal["index"]


class OptimizationCandidate(BaseModel):
    """One optimization hypothesis.

    ``candidate_type`` is ``index`` only in the MVP: PostgreSQL's planner already
    does a good job of join ordering and method selection, so those are not where
    a language model adds value.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_type: CandidateType = "index"
    hypothesis: str = Field(
        min_length=8,
        max_length=400,
        description="One sentence: what is slow and what change should fix it.",
    )
    candidate_sql: str = Field(
        min_length=10,
        max_length=2000,
        description="A single CREATE INDEX statement. Nothing else.",
    )
    expected_plan_change: str = Field(
        default="",
        max_length=400,
        description="Which plan node should change, e.g. 'Seq Scan -> Index Scan'.",
    )
    reasoning_summary: str = Field(
        default="",
        max_length=1200,
        description="Evidence from the plan/schema that supports this hypothesis.",
    )
    risk_notes: str = Field(
        default="",
        max_length=600,
        description="Costs and trade-offs a human should know about.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence. Advisory only: never used to accept.",
    )

    @field_validator("candidate_sql")
    @classmethod
    def _strip_code_fences(cls, value: str) -> str:
        """Models wrap SQL in ```sql fences even when told not to. Unwrap once."""
        v = value.strip()
        if v.startswith("```"):
            lines = [ln for ln in v.splitlines() if not ln.strip().startswith("```")]
            v = "\n".join(lines).strip()
        return v

    @field_validator("hypothesis")
    @classmethod
    def _require_substance(cls, value: str) -> str:
        if len(value.strip()) < 8:
            raise ValueError("hypothesis is too short to be a real claim")
        return value


class CandidateSet(BaseModel):
    """The full model response: a summary plus ranked candidates."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_summary: str = Field(
        default="",
        max_length=2000,
        description="What the model concluded from the diagnostic context.",
    )
    root_cause: str = Field(
        default="",
        max_length=600,
        description="The single dominant reason the query is slow.",
    )
    candidates: list[OptimizationCandidate] = Field(
        default_factory=list, max_length=4, description="Ranked best-first."
    )


class ExperimentFeedback(BaseModel):
    """A measured experiment result, fed back to the model for one refinement.

    This is the feedback half of the closed loop. It is deliberately
    *measurement*, not opinion: median timings, plan shape and the verdict.
    """

    model_config = ConfigDict(extra="forbid")

    attempt: int
    candidate_sql: str
    verdict: Literal["verified", "rejected", "inconclusive", "failed", "blocked"]
    baseline_median_ms: float
    candidate_median_ms: float
    speedup: float
    min_speedup: float
    baseline_plan_shape: str
    candidate_plan_shape: str
    notes: str = ""
