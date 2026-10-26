"""The second deterministic gate: does this candidate make sense *for this schema*?

:mod:`schemapulse.safety.sql_guard` proves a statement is *safe*.  This module
proves it is *grounded*: that it targets a table that exists, uses columns that
exist, is not a duplicate of an index already present, and is not so wide it
would be absurd.

Model confidence is deliberately not an input here.  A 0.99-confidence candidate
referencing a nonexistent column is rejected exactly like a 0.1-confidence one,
because "how sure the model felt" and "is this real" are unrelated questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from schemapulse.agent.schemas import OptimizationCandidate
from schemapulse.db.inspector import SchemaSnapshot
from schemapulse.errors import UnsafeQueryError
from schemapulse.safety.sql_guard import (
    IndexCandidate,
    identifier_ok,
    parse_index_ddl,
    sanitize_index_sql,
)

if TYPE_CHECKING:  # avoids a circular import: agent.context needs the safety guard
    from schemapulse.agent.context import DiagnosticContext

#: Reject indexes wider than this many key columns as likely-unreasoned.
MAX_INDEX_COLUMNS = 4


@dataclass(slots=True)
class _Grounding:
    """Just enough catalog to answer "does this column/table actually exist?".

    Grounding is deliberately satisfiable from either a live
    :class:`SchemaSnapshot` or the already-captured
    :class:`DiagnosticContext`.  Requiring the snapshot meant that a caller who
    forgot to pass one silently skipped the check -- and a hallucinated column
    sailed straight through to ``CREATE INDEX`` on the shadow database.
    """

    known: bool
    columns: frozenset[str] = frozenset()
    indexes: tuple[dict[str, Any], ...] = ()

    def has_table(self, table: str) -> bool:
        # An unknown catalog ("we could not check") must not block the
        # candidate: the shadow database is the final arbiter.  A known catalog
        # only proves the table exists if we actually captured something about
        # it -- an empty column set with no indexes is a failed lookup, not an
        # empty table.
        return not self.known or bool(self.columns or self.indexes)

    def has_column(self, column: str) -> bool:
        if not self.known:
            return True
        return column.lower() in self.columns


def _grounding(
    table: str,
    *,
    snapshot: SchemaSnapshot | None,
    context: DiagnosticContext | None,
) -> _Grounding:
    if snapshot is not None:
        info = snapshot.table(table)
        if info is None:
            return _Grounding(known=True)
        return _Grounding(
            known=True,
            columns=frozenset(c.name.lower() for c in info.columns),
            indexes=tuple(
                {
                    "name": i.name,
                    "columns": tuple(i.columns),
                    "definition": i.definition,
                }
                for i in info.indexes
            ),
        )
    if context is not None:
        brief = context.relation(table)
        if brief is None:
            return _Grounding(known=True)
        return _Grounding(
            known=True,
            columns=frozenset(c.lower() for c in brief.columns)
            | frozenset(c.lower() for c in brief.column_types),
            indexes=tuple(brief.indexes),
        )
    return _Grounding(known=False)


@dataclass(slots=True)
class ValidationReport:
    """Outcome of grounding one model candidate."""

    ok: bool
    reasons: tuple[str, ...] = ()
    statement: IndexCandidate | None = None
    execution_sql: str = ""
    index_name: str = ""

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


@dataclass(slots=True)
class CandidateValidator:
    """Validates a candidate against the live catalog."""

    snapshot: SchemaSnapshot | None = None
    existing_identities: set[tuple[object, ...]] = field(default_factory=set)

    def validate(
        self,
        candidate: OptimizationCandidate,
        *,
        context: DiagnosticContext | None = None,
        snapshot: SchemaSnapshot | None = None,
        allow_partial: bool = True,
    ) -> ValidationReport:
        reasons: list[str] = []

        # ---- 1. syntax + allow-list ------------------------------------
        try:
            parsed = parse_index_ddl(candidate.candidate_sql)
        except UnsafeQueryError as exc:
            return ValidationReport(ok=False, reasons=(f"safety guard rejected it: {exc}",))

        # ---- 2. structural sanity --------------------------------------
        if len(parsed.columns) > MAX_INDEX_COLUMNS:
            reasons.append(f"index has {len(parsed.columns)} key columns (max {MAX_INDEX_COLUMNS})")
        if not parsed.index_name:
            parsed = IndexCandidate(
                sql=parsed.sql,
                index_name=parsed.suggest_name(),
                table=parsed.table,
                columns=parsed.columns,
                unique=parsed.unique,
                predicate=parsed.predicate,
                concurrently=parsed.concurrently,
                include_columns=parsed.include_columns,
                column_order=parsed.column_order,
            )
        elif not identifier_ok(parsed.index_name):
            reasons.append(f"unsafe index name {parsed.index_name!r}")

        # ---- 3. grounding in the catalog -------------------------------
        # Prefer a live snapshot, but fall back to the columns and indexes the
        # diagnostic context already captured.  Only when neither source knows
        # the table do we treat it as nonexistent.
        snap = snapshot or self.snapshot
        ground = _grounding(parsed.table, snapshot=snap, context=context)

        if not ground.has_table(parsed.table):
            reasons.append(f"table {parsed.table!r} does not exist in the analysed database")
        else:
            for column in (*parsed.columns, *parsed.include_columns):
                if not ground.has_column(column):
                    reasons.append(f"column {parsed.table}.{column} does not exist")

        if ground.known:
            for index in ground.indexes:
                name = str(index.get("name") or "")
                definition = str(index.get("definition") or "")
                existing_cols = tuple(str(c).lower() for c in (index.get("columns") or ()))
                if name and name.lower() == parsed.index_name.lower():
                    reasons.append(f"an index named {parsed.index_name!r} already exists")
                if existing_cols == tuple(c.lower() for c in parsed.columns) and (
                    not parsed.predicate or "WHERE" not in definition.upper()
                ):
                    reasons.append(
                        f"index {name!r} already covers leading columns "
                        f"{list(parsed.columns)} — this would be redundant"
                    )
                    break

        if not allow_partial and parsed.predicate:
            reasons.append("partial indexes are disabled for this run")

        if reasons:
            return ValidationReport(ok=False, reasons=tuple(reasons))

        execution_sql = sanitize_index_sql(parsed, parsed.index_name)
        return ValidationReport(
            ok=True,
            statement=parsed,
            execution_sql=execution_sql,
            index_name=parsed.index_name,
        )
