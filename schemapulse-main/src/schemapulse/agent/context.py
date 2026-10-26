"""The structured diagnostic context handed to Gemma.

This is the "INPUT → PREPROCESSING → GEMMA" step of the pipeline, and it is the
part that makes Gemma useful rather than decorative.  A language model cannot
out-think a missing statistic, so we do not hand it raw catalog dumps; we hand it
a compact, normalised, evidence-bearing package:

* the query and the tables/columns it actually touches
* only those relations' columns, existing indexes and ``pg_stats``
* the normalised execution-plan tree with estimated-vs-actual rows
* deterministic hotspots computed in Python

The context is a Pydantic model so it can be logged, diffed and asserted on in
tests, and so the prompt rendering is a pure function of structured data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from schemapulse.db.inspector import IndexInfo, SchemaSnapshot
from schemapulse.db.statistics import ColumnStats
from schemapulse.plans.models import Hotspot, PlanSummary
from schemapulse.safety.sql_guard import ParsedQuery


class RelationBrief(BaseModel):
    """Everything about one relation that could inform an index decision."""

    model_config = ConfigDict(extra="forbid")

    name: str
    row_estimate: int
    size_bytes: int
    primary_key: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    referenced_columns: list[str] = Field(default_factory=list)
    indexes: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def index_names(self) -> list[str]:
        return [i["name"] for i in self.indexes]


class DiagnosticContext(BaseModel):
    """The single input to the optimization agent."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    query: str
    database: str
    server_version: str
    query_shape: dict[str, Any]
    relations: list[RelationBrief]
    plan_summary: dict[str, Any]
    plan_text: str
    hotspots: list[dict[str, Any]]
    baseline_execution_ms: float
    baseline_planning_ms: float
    rows_examined: int
    rows_returned: int | None = None
    plan_node_types: list[str] = Field(default_factory=list)
    indexes_used: list[str] = Field(default_factory=list)
    # kept for the heuristic baseline and for tests; not rendered into the prompt
    parsed: ParsedQuery | None = None
    snapshot: SchemaSnapshot | None = None
    column_stats: dict[str, dict[str, ColumnStats]] = Field(default_factory=dict)

    def relation(self, name: str) -> RelationBrief | None:
        lowered = name.lower()
        for r in self.relations:
            if r.name.lower() == lowered:
                return r
        return None

    def to_prompt_dict(self) -> dict[str, Any]:
        """The exact dict rendered into the prompt (and asserted on in tests)."""
        return {
            "database": self.database,
            "server_version": self.server_version,
            "query": self.query,
            "query_shape": self.query_shape,
            "baseline": {
                "execution_time_ms": round(self.baseline_execution_ms, 3),
                "planning_time_ms": round(self.baseline_planning_ms, 3),
                "rows_examined": self.rows_examined,
                "rows_returned": self.rows_returned,
                "plan_node_types": self.plan_node_types,
                "indexes_used": self.indexes_used,
            },
            "execution_plan": self.plan_text,
            "plan_metrics": self.plan_summary,
            "deterministic_hotspots": self.hotspots,
            "relations": [r.model_dump(mode="json") for r in self.relations],
        }


def build_context(
    *,
    query: str,
    parsed: ParsedQuery,
    snapshot: SchemaSnapshot,
    stats: dict[str, dict[str, ColumnStats]],
    plan: PlanSummary,
    hotspots: tuple[Hotspot, ...],
    rows_returned: int | None = None,
    plan_text: str = "",
) -> DiagnosticContext:
    """Assemble a :class:`DiagnosticContext` from inspection + one plan capture.

    Only the relations the query touches are included, and within each relation
    only statistics for columns that matter (referenced by the query, part of an
    index, or a plausible key).  This keeps the prompt small enough to reason
    about, which is the whole point.
    """
    briefs: list[RelationBrief] = []
    for table_name in parsed.tables:
        info = snapshot.table(table_name)
        if info is None:
            briefs.append(RelationBrief(name=table_name, row_estimate=0, size_bytes=0))
            continue

        table_stats = stats.get(info.name, {})
        referenced: set[str] = set()
        for key, cols in parsed.referenced_columns.items():
            key_lower = key.lower()
            if key_lower in {t.lower() for t in parsed.tables} or key_lower in {
                a.lower() for a in parsed.aliases
            }:
                referenced.update(cols)

        index_cols = {c for idx in info.indexes for c in idx.columns}

        brief_stats: dict[str, dict[str, Any]] = {}
        for col_name, st in table_stats.items():
            if col_name in referenced or col_name in index_cols or st.is_low_selectivity_constant:
                brief_stats[col_name] = st.to_dict()

        briefs.append(
            RelationBrief(
                name=info.name,
                row_estimate=info.row_estimate,
                size_bytes=info.size_bytes,
                primary_key=list(info.primary_key),
                columns=list(info.column_names),
                column_types={c.name: c.data_type for c in info.columns},
                referenced_columns=sorted(referenced),
                indexes=[i.to_dict() for i in info.indexes],
                stats=brief_stats,
            )
        )

    return DiagnosticContext(
        query=query,
        database=snapshot.database,
        server_version=snapshot.server_version,
        query_shape=parsed.to_dict(),
        relations=briefs,
        plan_summary=plan.to_dict(),
        plan_text=plan_text,
        hotspots=[h.to_dict() for h in hotspots],
        baseline_execution_ms=plan.execution_time_ms,
        baseline_planning_ms=plan.planning_time_ms,
        rows_examined=plan.total_rows_examined,
        rows_returned=rows_returned,
        plan_node_types=list(plan.node_type_signature),
        indexes_used=list(plan.indexes_used),
        parsed=parsed,
        snapshot=snapshot,
        column_stats=stats,
    )


def existing_index_for(
    context: DiagnosticContext, table: str, columns: tuple[str, ...]
) -> IndexInfo | None:
    """Deterministic duplicate check used to reject redundant candidates."""
    if context.snapshot is None:
        return None
    info = context.snapshot.table(table)
    if info is None:
        return None
    for index in info.indexes:
        if index.columns[: len(columns)] == columns:
            return index
    return None
