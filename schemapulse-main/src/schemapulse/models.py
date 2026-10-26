"""Cross-cutting result models.

These are the objects a caller receives. They are serialisable so a diagnosis
can be stored, diffed or turned into a report without re-running anything.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from schemapulse.agent.schemas import CandidateSet
from schemapulse.plans.comparator import PlanComparison, Verdict
from schemapulse.plans.models import Hotspot, PlanSummary


class OptimizationCandidate(BaseModel):
    """One proposed optimization, as validated by the deterministic layer.

    Mirrors :class:`schemapulse.agent.schemas.OptimizationCandidate` but is the
    *post-validation* view: it carries the parsed statement that will actually
    be executed, not the raw text.

    ``statement``, ``index_name`` and ``columns`` are empty when the candidate
    was refused before it could be parsed.  A blocked candidate is still recorded
    -- with the SQL the model actually proposed -- because a report row that
    cannot be traced back to a proposal is not evidence of anything.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    candidate_sql: str
    statement: str = ""
    index_name: str = ""
    table: str = ""
    columns: list[str] = Field(default_factory=list)
    predicate: str | None = None
    unique: bool = False
    include_columns: list[str] = Field(default_factory=list)
    expected_plan_change: str = ""
    reasoning_summary: str = ""
    risk_notes: str = ""
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AttemptOutcome(BaseModel):
    """What happened to one candidate during one loop iteration."""

    model_config = ConfigDict(extra="forbid")

    attempt: int
    candidate: OptimizationCandidate | None = None
    verdict: Verdict
    reason: str
    speedup: float | None = None
    baseline_median_ms: float | None = None
    candidate_median_ms: float | None = None
    index_used: str | None = None
    safety_rejections: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    model_latency_ms: float | None = None
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class QueryInsight(BaseModel):
    """The deterministic, model-free part of the diagnosis.

    Produced without any model call — this is what ``schemapulse inspect``
    prints and what a judge can verify independently.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    database: str
    server_version: str
    tables: list[str] = Field(default_factory=list)
    is_select_only: bool = True
    plan: PlanSummary
    hotspots: list[Hotspot] = Field(default_factory=list)
    existing_indexes: list[str] = Field(default_factory=list)
    relation_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rows_returned: int | None = None
    captured_at: str = ""

    @property
    def execution_time_ms(self) -> float:
        return self.plan.execution_time_ms

    @property
    def planning_time_ms(self) -> float:
        return self.plan.planning_time_ms

    @property
    def rows_examined(self) -> int:
        return self.plan.total_rows_examined

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "database": self.database,
            "server_version": self.server_version,
            "tables": self.tables,
            "is_select_only": self.is_select_only,
            "execution_time_ms": round(self.execution_time_ms, 4),
            "planning_time_ms": round(self.planning_time_ms, 4),
            "rows_examined": self.rows_examined,
            "rows_returned": self.rows_returned,
            "plan": self.plan.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "existing_indexes": self.existing_indexes,
            "relation_stats": self.relation_stats,
            "captured_at": self.captured_at,
        }


class Recommendation(BaseModel):
    """A verified suggestion for a human to review and deploy themselves."""

    model_config = ConfigDict(extra="forbid")

    production_sql: str
    measured_speedup: float
    rationale: str
    warnings: list[str] = Field(default_factory=list)
    auto_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DiagnosisResult(BaseModel):
    """Everything one ``SchemaPulse.diagnose()`` call produced."""

    model_config = ConfigDict(extra="forbid")

    insight: QueryInsight
    candidate_set: CandidateSet | None = None
    attempts: list[AttemptOutcome] = Field(default_factory=list)
    best_attempt: int | None = None
    comparison: PlanComparison | None = None
    verified: OptimizationCandidate | None = None
    recommendation: Recommendation | None = None
    provider: str = ""
    provider_is_live: bool = False
    model_id: str = ""
    min_speedup: float = 1.2
    elapsed_ms: float = 0.0
    trace: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        if self.comparison is not None:
            return self.comparison.verdict
        return Verdict.FAILED

    @property
    def speedup(self) -> float | None:
        return self.comparison.speedup if self.comparison else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_is_live": self.provider_is_live,
            "model_id": self.model_id,
            "verdict": self.verdict.value,
            "speedup": None if self.comparison is None else round(self.comparison.speedup, 4),
            "min_speedup": self.min_speedup,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "insight": self.insight.to_dict(),
            "candidate_set": self.candidate_set.model_dump(mode="json")
            if self.candidate_set
            else None,
            "attempts": [a.to_dict() for a in self.attempts],
            "verified_candidate": self.verified.to_dict() if self.verified else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "trace": self.trace,
        }
